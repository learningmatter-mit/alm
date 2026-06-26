import os
import argparse
from tqdm import tqdm
import torch
import torch.distributed as dist
from torch.utils.data import random_split, Subset
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
from transformers import get_cosine_schedule_with_warmup

from utils import AtomisticLanguageDataset, custom_collate_fn, is_main_process, FullAtomisticLanguageDataset
from model import AtomisticLanguageModel
import wandb


def train(args):
    dist.init_process_group(backend='nccl', init_method='env://')
    local_rank = int(os.environ["LOCAL_RANK"])
    local_world_size = int(os.environ['WORLD_SIZE'])
    local_env_rank = int(os.environ['RANK'])
    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")
    main_process = is_main_process()
    use_wandb = main_process and not args.disable_wandb

    # Empty string => disabled, fall back to live OrbV3 encoding.
    print(f"cached_embs_parent_path: {args.cached_embs_parent_path}")
    if not args.cached_embs_parent_path:
        args.cached_embs_parent_path = None

    use_cached_embeddings = args.cached_embs_parent_path is not None
    print(f"use_cached_embeddings: {use_cached_embeddings}")
    model = AtomisticLanguageModel(
        llm_name=args.llm_name,
        atomistic_model_name=args.atomistic_model_name,
        atomistic_feature_dim=args.atomistic_feature_dim,
        device=device,
        use_cached_embeddings=use_cached_embeddings,
    )
    # Frozen LLM, but grad flows back to the projector; checkpoint to avoid OOM.
    model.llm.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.llm.enable_input_require_grads()
    model = model.to(device)

    model = DDP(model, device_ids=[local_rank], find_unused_parameters=False)

    trainable_params = sum([p.numel() for p in model.parameters() if p.requires_grad])
    total_params = sum([p.numel() for p in model.parameters()])
    if is_main_process():
        print(f"Trainable: {trainable_params:,} / {total_params:,} "
              f"({100*trainable_params/total_params:.2f}%)")


    if args.train_csv_path is not None:
        dataset = AtomisticLanguageDataset(
            tokenizer=model.module.tokenizer,
            db_path=args.db_path,
            csv_path=args.train_csv_path,
            thinking=args.thinking,
            max_num_tokens=args.max_num_tokens,
        )
        generator = torch.Generator().manual_seed(42)
        train_dataset, val_dataset = random_split(dataset, [0.8, 0.2], generator=generator)
    else:
        if is_main_process():
            msg = f"Training on full dataset from {args.data_parent_path}"
            if use_cached_embeddings:
                msg += f" with cached OrbV3 embeddings from {args.cached_embs_parent_path}"
            print(msg)
        train_dataset = FullAtomisticLanguageDataset(
            tokenizer=model.module.tokenizer,
            split='train',
            parent_folder=args.data_parent_path,
            thinking=args.thinking,
            max_num_tokens=args.max_num_tokens,
            cached_embs_parent_path=args.cached_embs_parent_path,
            atomistic_model_name=args.atomistic_model_name,
            atomistic_feature_dim=args.atomistic_feature_dim,
        )
        val_dataset = FullAtomisticLanguageDataset(
            tokenizer=model.module.tokenizer,
            split='validation',
            parent_folder=args.data_parent_path,
            thinking=args.thinking,
            max_num_tokens=args.max_num_tokens,
            cached_embs_parent_path=args.cached_embs_parent_path,
            atomistic_model_name=args.atomistic_model_name,
            atomistic_feature_dim=args.atomistic_feature_dim,
        )

    if args.val_subset_fraction and args.val_subset_fraction < 1.0:
        n = max(1, int(args.val_subset_fraction * len(val_dataset)))
        g = torch.Generator().manual_seed(42)
        idx = torch.randperm(len(val_dataset), generator=g)[:n].tolist()
        val_dataset = Subset(val_dataset, idx)
        if is_main_process():
            print(f"Subsampled validation to {len(val_dataset)} examples "
                  f"({args.val_subset_fraction:.3f} of full).")

    sampler = DistributedSampler(train_dataset, shuffle=True)
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        sampler=sampler,
        num_workers=args.num_workers,
        persistent_workers=args.num_workers > 0,
        pin_memory=True,
        collate_fn=custom_collate_fn,
    )
    # Shuffle val so each rank gets a mix; datasets are concatenated alphabetically, else step time is imbalanced.
    val_sampler = DistributedSampler(val_dataset, shuffle=True, drop_last=True, seed=42)
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        sampler=val_sampler,
        num_workers=args.num_workers,
        persistent_workers=args.num_workers > 0,
        pin_memory=True,
        collate_fn=custom_collate_fn,
    )

    optim = torch.optim.AdamW(
        model.module.projector.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
        betas=(args.beta1, args.beta2),
    )

    num_epochs = args.num_epochs
    total_steps = num_epochs * len(train_dataloader)
    warmup_steps = min(2000, int(0.03 * total_steps))
    scheduler = get_cosine_schedule_with_warmup(
        optim,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )

    start_epoch = args.start_epoch
    scheduler_state_loaded = False
    if args.resume_from_checkpoint is not None:
        checkpoint = torch.load(args.resume_from_checkpoint, map_location=device)
        if isinstance(checkpoint, dict) and "projector_state_dict" in checkpoint:
            model.module.projector.load_state_dict(checkpoint["projector_state_dict"])
            if "optimizer_state_dict" in checkpoint:
                optim.load_state_dict(checkpoint["optimizer_state_dict"])
            if "scheduler_state_dict" in checkpoint:
                scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
                scheduler_state_loaded = True
            if args.start_epoch == 0:
                start_epoch = int(checkpoint.get("epoch", 0))
            if main_process:
                print(
                    f"Resumed training state from {args.resume_from_checkpoint} "
                    f"at epoch {start_epoch}."
                )
        else:
            # Backward compat: projector-only checkpoints.
            model.module.projector.load_state_dict(checkpoint)
            if main_process:
                print(
                    "Loaded projector-only checkpoint. "
                    "Optimizer/scheduler state not found; restarting those states."
                )
        dist.barrier()

    # Resuming without scheduler state: fast-forward the LR schedule so we don't restart warmup.
    target_global_step = start_epoch * len(train_dataloader) + args.start_step
    if not scheduler_state_loaded and target_global_step > 0:
        import warnings
        with warnings.catch_warnings():
            # Advancing the schedule without an optimizer step is intentional here.
            warnings.filterwarnings("ignore", category=UserWarning)
            for _ in range(target_global_step):
                scheduler.step()
        if main_process:
            print(
                f"Fast-forwarded scheduler by {target_global_step} steps "
                f"(epoch={start_epoch}, step={args.start_step}). "
                f"Current LR: {scheduler.get_last_lr()[0]:.6f}"
            )

    if use_wandb:
        wandb.init(
            project=args.wandb_project,
            config=vars(args),
        )

    for epoch in tqdm(
        range(start_epoch, num_epochs),
        desc="Training",
        disable=not main_process,
        dynamic_ncols=True,
        position=0,
    ):
        sampler.set_epoch(epoch)
        model.train()
        # Don't .eval() the LLM: HF gates grad-checkpointing on self.training, and Qwen3 has no dropout.
        if model.module.atomistic_model is not None:
            model.module.atomistic_model.eval()
        optim.zero_grad(set_to_none=True)
        initial_step = args.start_step if epoch == start_epoch else 0
        for step, batch in tqdm(
            enumerate(train_dataloader),
            total=len(train_dataloader),
            desc=f"Train Epoch {epoch}",
            disable=not main_process,
            dynamic_ncols=True,
            leave=False,
            position=1,
        ):
            if step < initial_step:
                continue
            # Early-exit for the scaling-law sweep's tight step budget.
            if args.max_steps is not None and (epoch * len(train_dataloader) + step) >= args.max_steps:
                break
            row_batch = batch.get('atom_rows')
            atom_embeds = batch.get('atom_embeds')
            input_ids = [ids.to(device) for ids in batch["input_ids"]]
            labels = [lab.to(device) for lab in batch["labels"]]
            attention_mask = [mask.to(device) for mask in batch["attention_mask"]]

            with torch.cuda.amp.autocast(dtype=torch.bfloat16):
                outputs = model(input_ids, attention_mask, labels,
                                row_batch=row_batch, atom_embeds=atom_embeds)
                loss = outputs.loss
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.module.projector.parameters(), max_norm=1.0)
            optim.step()
            scheduler.step()
            optim.zero_grad(set_to_none=True)
            
            if step % 100 == 0 and main_process:
                print(f"Step {step}/{total_steps}, Loss: {loss.item():.4f}, "
                      f"LR: {scheduler.get_last_lr()[0]:.6f}")
            global_step = epoch * len(train_dataloader) + step
            if use_wandb and global_step % args.log_every == 0:
                wandb.log(
                    {
                        "train/loss": loss.item(),
                        "train/lr": scheduler.get_last_lr()[0],
                        "epoch": epoch,
                        "global_step": global_step,
                    }
                )

            if step % args.eval_every == 0:
                model.eval()
                model.module.llm.eval()
                if model.module.atomistic_model is not None:
                    model.module.atomistic_model.eval()

                val_loss = 0
                for val_step, batch in tqdm(
                    enumerate(val_dataloader),
                    desc="Validation",
                    total=len(val_dataloader),
                    disable=not main_process,
                    dynamic_ncols=True,
                    leave=False,
                    position=1,
                ):
                    row_batch = batch.get('atom_rows')
                    atom_embeds = batch.get('atom_embeds')
                    input_ids = [ids.to(device) for ids in batch["input_ids"]]
                    labels = [lab.to(device) for lab in batch["labels"]]
                    attention_mask = [mask.to(device) for mask in batch["attention_mask"]]

                    with torch.no_grad():
                        with torch.cuda.amp.autocast(dtype=torch.bfloat16):
                            outputs = model(input_ids, attention_mask, labels,
                                            row_batch=row_batch, atom_embeds=atom_embeds)
                            val_loss += outputs.loss.item()

                print(f"Epoch {epoch}, Validation Loss: {(val_loss / len(val_dataloader)):.4f}")
                avg_val_loss = val_loss / len(val_dataloader)
                if use_wandb:
                    wandb.log(
                        {
                            "val/loss": avg_val_loss,
                            "epoch": epoch,
                        }
                    )

                if is_main_process():
                    torch.save(
                        {
                            "projector_state_dict": model.module.projector.state_dict(),
                            "optimizer_state_dict": optim.state_dict(),
                            "scheduler_state_dict": scheduler.state_dict(),
                            "epoch": epoch,
                            "step": step,
                            "global_step": epoch * len(train_dataloader) + step,
                        },
                        args.model_save_path.replace(".pt", f"_step={step}.pt"),
                    )
                    if args.checkpoint_save_path is not None:
                        checkpoint_path = args.checkpoint_save_path.format(epoch=epoch + 1)
                        torch.save(
                            {
                                "projector_state_dict": model.module.projector.state_dict(),
                                "optimizer_state_dict": optim.state_dict(),
                                "scheduler_state_dict": scheduler.state_dict(),
                                "epoch": epoch + 1,
                                    "global_step": (epoch + 1) * len(train_dataloader),
                                },
                                checkpoint_path,
                            )
            

    # Final save when --max_steps caps the run before the eval-cadence save fires.
    if args.max_steps is not None and is_main_process():
        final_path = args.model_save_path.replace(".pt", f"_step={args.max_steps}.pt")
        torch.save(
            {
                "projector_state_dict": model.module.projector.state_dict(),
                "optimizer_state_dict": optim.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "epoch": start_epoch,
                "step": args.max_steps,
                "global_step": args.max_steps,
            },
            final_path,
        )
        print(f"[stage1-exit] saved final projector → {final_path}")

    if use_wandb:
        wandb.finish()
    dist.destroy_process_group()
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--db_path", type=str, default=None)
    parser.add_argument("--train_csv_path", type=str, default=None)
    parser.add_argument("--model_save_path", type=str, default="runs/stage1/checkpoint_model.pt")
    parser.add_argument("--data_parent_path", type=str, default='/tmp/LLM4Mat-Bench/')
    parser.add_argument("--cached_embs_parent_path", type=str, default=None,
                        help="Parent of {dataset}/embeddings/{model}_{split}_atom.flat.bin pre-cached OrbV3 features. "
                             "Set to empty string to force live OrbV3 encoding.")
    parser.add_argument("--learning_rate", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=0)
    parser.add_argument("--beta1", type=float, default=0.9)
    parser.add_argument("--beta2", type=float, default=0.999)
    parser.add_argument("--eval_every", type=int, default=1000)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--num_epochs", type=int, default=5)
    parser.add_argument("--max_steps", type=int, default=None,
                        help="Optional step cap for sweeps (early-exits the inner loop). "
                             "Default None = run num_epochs to completion.")
    parser.add_argument("--thinking", action="store_true")
    parser.add_argument("--log_every", type=int, default=10)
    parser.add_argument("--disable_wandb", action="store_true")
    parser.add_argument("--wandb_project", type=str, default="alm-pretrain")
    parser.add_argument("--max_num_tokens", type=int, default=2048)
    parser.add_argument("--val_subset_fraction", type=float, default=None,
                        help="If set in (0, 1), validate on a fixed-seed random fraction of the val set.")
    parser.add_argument("--start_epoch", type=int, default=0)
    parser.add_argument("--start_step", type=int, default=0,
                        help="Skip the first N batches of the starting epoch. "
                             "Only applies to the first epoch of the resumed run.")
    parser.add_argument("--resume_from_checkpoint", type=str, default=None)
    parser.add_argument("--checkpoint_save_path", type=str, default=None)
    parser.add_argument("--atomistic_model_name", type=str, default="orb_v3_direct_20_omat",
                        help="Encoder tag baked into cached-embedding filenames "
                             "({atomistic_model_name}_{split}_atom.flat.bin). Default OrbV3.")
    parser.add_argument("--atomistic_feature_dim", type=int, default=256,
                        help="Per-atom feature dim of the cached embeddings. OrbV3=256, UMA=128, "
                             "PET-MAD variable.")
    parser.add_argument("--llm_name", type=str, default="Qwen/Qwen3-8B",
                        help="HuggingFace model id; used by the scaling-law sweep to "
                             "swap the base LLM (e.g. Qwen/Qwen3-0.6B, -1.7B, -4B, ...).")
    parser.add_argument("--num_workers", type=int, default=0,
                        help="DataLoader workers per rank. 0 runs the pipeline in the main process.")
    args = parser.parse_args()
    train(args)

