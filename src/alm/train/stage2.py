"""Stage 2: LoRA on Qwen3-8B + continued projector training, run via torchrun -m alm.train.stage2."""

import argparse
import os
from contextlib import nullcontext
from pathlib import Path

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import ConcatDataset, DataLoader, Dataset, Subset, DistributedSampler
from transformers import get_cosine_schedule_with_warmup

from peft import LoraConfig, get_peft_model

from model import AtomisticLanguageModel
from samplers import BucketedDistributedSampler
from utils import (
    ArxivAbstractDataset,
    AtomisticLanguageDataset,
    CamelAIDataset,
    FullAtomisticLanguageDataset,
    GPTNarrativeDataset,
    MaScQADataset,
    applications_tasks_for_narrative,
    custom_collate_fn,
    describe_tasks_for_dataset,
    describe_tasks_for_narrative,
    is_main_process,
    property_tasks_for_dataset,
)

import wandb


NARRATIVE_PARQUET_NAMES = ["dft_3d", "mp_3d_2020", "aflow2", "oqmd"]
LORA_TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj",
                       "gate_proj", "up_proj", "down_proj"]
BUCKET_NAMES = ["describe", "property_apps", "arxiv", "camel", "mascqa", "matterchat"]


class _EmptyDataset(Dataset):
    """Len-0 placeholder for zero-weight buckets (ConcatDataset rejects empty lists; never sampled)."""
    def __len__(self):
        return 0

    def __getitem__(self, idx):
        raise IndexError(
            f"_EmptyDataset has no items (idx={idx}); this bucket should have "
            "weight=0 and never be sampled. If you see this, the sampler is "
            "drawing from a zero-weight bucket — check --bucket_weights.")


def _narrative_subsets(tokenizer, parquet_dir, cache_dir, max_num_tokens, task_fn,
                       atomistic_model_name: str = "orb_v3_direct_20_omat",
                       atomistic_feature_dim: int = 256):
    out = []
    for name in NARRATIVE_PARQUET_NAMES:
        parquet = parquet_dir / f"{name}_gpt_narratives.parquet"
        cache = cache_dir / name / "embeddings" / f"{atomistic_model_name}_atom.flat.bin"
        if not parquet.exists() or not cache.exists():
            if is_main_process():
                print(f"[stage2] skip {name}: parquet={parquet.exists()} cache={cache.exists()}")
            continue
        out.append(GPTNarrativeDataset(
            tokenizer=tokenizer, parquet_path=parquet, cached_embs_path=cache,
            thinking=False, max_num_tokens=max_num_tokens,
            dataset_name=name, tasks=task_fn(name),
            atomistic_feature_dim=atomistic_feature_dim,
        ))
    return out


def build_stage2_datasets(args, tokenizer, weights=None):
    """Build the 6-bucket Stage 2 mixture; zero-weight buckets become len-0 placeholders to save per-rank CPU RAM."""
    parquet_dir = Path(args.narrative_parquet_dir)
    cache_dir = Path(args.narrative_cache_dir)

    skip = set()
    if weights is not None:
        for name, w in zip(BUCKET_NAMES, weights):
            if float(w) == 0.0:
                skip.add(name)
        if skip and is_main_process():
            print(f"[stage2] zero-weight buckets — skipping eager load: {sorted(skip)}")

    if "describe" in skip:
        describe_bucket = _EmptyDataset()
    else:
        describe_bucket = ConcatDataset([
            FullAtomisticLanguageDataset(
                tokenizer=tokenizer, split="train", parent_folder=args.data_parent_path,
                thinking=False, max_num_tokens=args.max_num_tokens,
                cached_embs_parent_path=args.cached_embs_parent_path,
                tasks=describe_tasks_for_dataset,
                atomistic_model_name=args.atomistic_model_name,
                atomistic_feature_dim=args.atomistic_feature_dim,
            ),
            *_narrative_subsets(tokenizer, parquet_dir, cache_dir, args.max_num_tokens,
                                describe_tasks_for_narrative,
                                atomistic_model_name=args.atomistic_model_name,
                                atomistic_feature_dim=args.atomistic_feature_dim),
        ])

    if "property_apps" in skip:
        # Still appended to below if matterchat lives inside this bucket.
        property_subsets = []
    else:
        property_subsets = [
            FullAtomisticLanguageDataset(
                tokenizer=tokenizer, split="train", parent_folder=args.data_parent_path,
                thinking=False, max_num_tokens=args.max_num_tokens,
                cached_embs_parent_path=args.cached_embs_parent_path,
                tasks=property_tasks_for_dataset,
                atomistic_model_name=args.atomistic_model_name,
                atomistic_feature_dim=args.atomistic_feature_dim,
            ),
            *_narrative_subsets(tokenizer, parquet_dir, cache_dir, args.max_num_tokens,
                                applications_tasks_for_narrative,
                                atomistic_model_name=args.atomistic_model_name,
                                atomistic_feature_dim=args.atomistic_feature_dim),
        ]
    # MatterChat NL tasks on the same MP rows; either inside property_apps (default) or its own bucket.
    mc_needed = (
        ("matterchat" not in skip and not args.matterchat_in_property_apps) or
        ("property_apps" not in skip and args.matterchat_in_property_apps)
    )
    mc_csv = Path(args.matterchat_train_csv)
    mc_cache = Path(args.matterchat_train_cache)
    # .idx.json sibling derived via with_suffix, matching the constructor.
    mc_cache_ok = mc_cache.exists() and mc_cache.with_suffix(".idx.json").exists()
    matterchat_dataset = None
    if mc_needed and mc_csv.exists() and mc_cache_ok:
        matterchat_dataset = AtomisticLanguageDataset(
            tokenizer=tokenizer, db_path=None, csv_path=str(mc_csv),
            thinking=False, max_num_tokens=args.max_num_tokens,
            dataset_name="matterchat_mp", cached_embs_path=str(mc_cache),
            tasks=property_tasks_for_dataset("matterchat_mp"),
            atomistic_feature_dim=args.atomistic_feature_dim,
        )
    elif mc_needed and is_main_process():
        print(f"[stage2] skip matterchat_mp: csv={mc_csv.exists()} cache_ok={mc_cache_ok}")

    if args.matterchat_in_property_apps:
        if matterchat_dataset is not None:
            property_subsets.append(matterchat_dataset)
        matterchat_bucket = _EmptyDataset()
    else:
        matterchat_bucket = matterchat_dataset if matterchat_dataset is not None else _EmptyDataset()
    property_bucket = ConcatDataset(property_subsets) if property_subsets else _EmptyDataset()

    if "arxiv" in skip:
        arxiv_bucket = _EmptyDataset()
    else:
        arxiv_bucket = ArxivAbstractDataset(tokenizer, args.arxiv_parquet, args.max_num_tokens,
                                            atomistic_feature_dim=args.atomistic_feature_dim)

    if "camel" in skip:
        camel_bucket = _EmptyDataset()
    else:
        camel_bucket = CamelAIDataset(tokenizer, args.camel_jsonl, thinking=False,
                                      max_num_tokens=args.max_num_tokens,
                                      atomistic_feature_dim=args.atomistic_feature_dim)

    if "mascqa" in skip:
        mascqa_bucket = _EmptyDataset()
    else:
        mascqa_bucket = MaScQADataset(tokenizer, args.mascqa_json, args.mascqa_xlsx,
                                      thinking=False, max_num_tokens=args.max_num_tokens,
                                      atomistic_feature_dim=args.atomistic_feature_dim)

    bucket_map = {
        "describe": describe_bucket, "property_apps": property_bucket,
        "arxiv": arxiv_bucket, "camel": camel_bucket, "mascqa": mascqa_bucket,
        "matterchat": matterchat_bucket,
    }
    buckets = list(bucket_map.values())
    lengths = [len(b) for b in buckets]
    offsets, off = [], 0
    for n in lengths:
        offsets.append(off); off += n
    return ConcatDataset(buckets), offsets, lengths, bucket_map


def _print_coverage(lengths, weights, total_optim_steps, effective_batch):
    total = total_optim_steps * effective_batch
    print(f"[stage2] total_optim_steps={total_optim_steps} effective_batch={effective_batch} "
          f"-> {total:,} total samples seen over the run")
    print(f"{'bucket':<16}{'size':>12}{'weight':>10}{'visits':>14}{'visits/size':>14}")
    for name, n, w in zip(BUCKET_NAMES, lengths, weights):
        v = int(total * w)
        print(f"{name:<16}{n:>12,}{w:>10.3f}{v:>14,}{v/max(1,n):>14.2f}x")


def _log_mem(label, main_process):
    """Log rank-0 RSS + cgroup usage/limit at every level up the hierarchy."""
    if not main_process:
        return
    try:
        with open(f"/proc/{os.getpid()}/status") as f:
            rss_kb = next(int(l.split()[1]) for l in f if l.startswith("VmRSS:"))
        cg = next(l.split(":")[2].strip() for l in open(f"/proc/{os.getpid()}/cgroup")
                  if "memory" in l or l.split(":")[1] == "")
        print(f"[mem] {label}: rss={rss_kb/1e6:.2f}GB", flush=True)
        # Walk up so the binding ancestor limit is visible.
        path = cg
        while path and path != "/":
            for usage_p, limit_p in [
                (f"/sys/fs/cgroup/memory{path}/memory.usage_in_bytes",
                 f"/sys/fs/cgroup/memory{path}/memory.limit_in_bytes"),
                (f"/sys/fs/cgroup{path}/memory.current",
                 f"/sys/fs/cgroup{path}/memory.max"),
            ]:
                if os.path.exists(usage_p) and os.path.exists(limit_p):
                    u = int(open(usage_p).read())
                    raw = open(limit_p).read().strip()
                    lim = "unlimited" if raw == "max" or int(raw) > 1 << 60 else f"{int(raw)/1e9:.2f}GB"
                    print(f"[mem]   {path}: usage={u/1e9:.2f}GB limit={lim}", flush=True)
                    break
            path = os.path.dirname(path)
    except Exception as e:
        print(f"[mem] {label}: error ({e})", flush=True)


class _MultiOpt(torch.optim.Optimizer):
    """Two optimizers stepped in lockstep under one scheduler (Muon path); subclasses Optimizer so LambdaLR's isinstance check passes."""
    def __init__(self, optimizers):
        super().__init__([torch.nn.Parameter(torch.zeros(1))], {})
        self.optimizers = optimizers
        self.param_groups = [g for o in optimizers for g in o.param_groups]
    def zero_grad(self, set_to_none=True):
        for o in self.optimizers:
            o.zero_grad(set_to_none=set_to_none)
    def step(self, closure=None):
        for o in self.optimizers:
            o.step()
    def state_dict(self):
        return {"opts": [o.state_dict() for o in self.optimizers]}
    def load_state_dict(self, sd):
        for o, s in zip(self.optimizers, sd["opts"]):
            o.load_state_dict(s)


def _build_optimizer(args, lora_params, projector_params):
    if args.optimizer == "adamw":
        return torch.optim.AdamW(
            [{"params": lora_params,      "lr": args.lora_lr},
             {"params": projector_params, "lr": args.projector_lr}],
            betas=(0.9, 0.95), weight_decay=0.01,
        )
    if args.optimizer == "muon":
        from muon import Muon
        muon_p  = [p for p in lora_params if p.ndim >= 2]
        adam_p  = [p for p in lora_params if p.ndim < 2] + projector_params
        return _MultiOpt([
            Muon(muon_p, lr=args.lora_lr, momentum=0.95),
            torch.optim.AdamW(
                [{"params": adam_p, "lr": args.projector_lr}],
                betas=(0.9, 0.95), weight_decay=0.01,
            ),
        ])
    raise ValueError(f"unknown optimizer: {args.optimizer}")


def train(args):
    # Pin NCCL to the LOCAL device before init; without device_id it guesses from global rank and hangs on multi-node.
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    rank = int(os.environ["RANK"])
    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")
    dist.init_process_group(backend="nccl", init_method="env://", device_id=device)
    main_process = is_main_process()
    _log_mem("startup", main_process)
    use_wandb = main_process and not args.disable_wandb

    weights = [float(x) for x in args.bucket_weights.split(",")]
    # Backward-compat: pad a legacy 5-entry string with 0.0 for the matterchat bucket.
    if len(weights) == len(BUCKET_NAMES) - 1:
        weights = weights + [0.0]
    assert len(weights) == len(BUCKET_NAMES), \
        f"expected {len(BUCKET_NAMES)} weights ({BUCKET_NAMES}), got {weights}"

    # Guard: nonzero matterchat weight while it lives in property_apps means the bucket is empty and the sampler crashes.
    mc_idx = BUCKET_NAMES.index("matterchat")
    if args.matterchat_in_property_apps and weights[mc_idx] > 0:
        raise ValueError(
            f"--matterchat_in_property_apps is True (default) but weights[{mc_idx}] "
            f"(matterchat) = {weights[mc_idx]}. The matterchat bucket is empty in "
            "this mode (matterchat data lives inside property_apps). Either pass "
            "--no-matterchat_in_property_apps to extract it as a separate bucket, "
            "or set the matterchat weight to 0.")

    model = AtomisticLanguageModel(
        llm_name=args.llm_name,
        atomistic_model_name=args.atomistic_model_name,
        atomistic_feature_dim=args.atomistic_feature_dim,
        device=device,
        use_cached_embeddings=True,   # Stage 2 is cached-only across all atomistic buckets
        max_atoms=max(1, args.max_num_tokens - 256),
        atom_bidirectional_attention=args.atom_bidirectional_attention,
    )

    # Wrap fresh then overwrite on resume; PeftModel.from_pretrained hits a version mismatch we bypass.
    lora_cfg = LoraConfig(
        r=args.lora_rank, lora_alpha=args.lora_alpha, lora_dropout=0.05,
        bias="none", task_type="CAUSAL_LM", target_modules=LORA_TARGET_MODULES,
    )
    model.llm = get_peft_model(model.llm, lora_cfg)
    if args.resume_from_stage2:
        from safetensors.torch import load_file
        adapter_dir = Path(args.resume_from_stage2) / "lora_adapter"
        sd = load_file(str(adapter_dir / "adapter_model.safetensors"))
        # save_pretrained strips the "default" adapter name from keys; re-insert it.
        sd = {k.replace(".lora_A.weight", ".lora_A.default.weight")
               .replace(".lora_B.weight", ".lora_B.default.weight"): v
              for k, v in sd.items()}
        # Vocab-resize migration: copy old rows into the matching prefix when the current model grew.
        cur_sd = model.llm.state_dict()
        for k in list(sd.keys()):
            if k in cur_sd and sd[k].shape != cur_sd[k].shape:
                old, cur = sd[k], cur_sd[k]
                if old.ndim == cur.ndim and all(o <= c for o, c in zip(old.shape, cur.shape)):
                    new = cur.clone()
                    new[tuple(slice(0, s) for s in old.shape)] = old.to(new.dtype)
                    sd[k] = new
                    if main_process:
                        print(f"  resized {k}: {tuple(old.shape)} → {tuple(new.shape)}")
        _, unexpected = model.llm.load_state_dict(sd, strict=False)
        assert not unexpected, f"unexpected keys after rename: {unexpected[:5]}..."
        if main_process:
            print(f"Loaded Stage 2 LoRA weights from {adapter_dir}")

    # Non-reentrant grad checkpointing; disabling it OOMs on worst-case long-sequence batches.
    model.llm.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.llm.enable_input_require_grads()

    if args.resume_from_stage1:
        ckpt = torch.load(args.resume_from_stage1, map_location=device)
        proj_state = ckpt.get("projector_state_dict", ckpt) if isinstance(ckpt, dict) else ckpt
        model.projector.load_state_dict(proj_state)
        if main_process:
            print(f"Loaded Stage 1 projector from {args.resume_from_stage1}")
    elif args.resume_from_stage2:
        state = torch.load(Path(args.resume_from_stage2) / "projector_and_state.pt",
                           map_location=device)
        model.projector.load_state_dict(state["projector_state_dict"])

    model = model.to(device)
    model = DDP(model, device_ids=[local_rank], find_unused_parameters=True)

    if main_process:
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in model.parameters())
        print(f"Trainable: {trainable:,} / {total:,} ({100 * trainable / total:.2f}%)")
    _log_mem("after model+lora+ddp", main_process)

    train_dataset, bucket_offsets, bucket_lengths, bucket_map = build_stage2_datasets(
        args, model.module.tokenizer, weights=weights)
    _log_mem("after train datasets", main_process)
    effective_batch = args.batch_size * args.grad_accum_steps * world_size
    if main_process:
        _print_coverage(bucket_lengths, weights, args.total_optim_steps, effective_batch)

    # Total per-sample indices = optim_steps x grad_accum x batch_size x world_size.
    num_samples_total = (args.total_optim_steps * args.grad_accum_steps
                         * args.batch_size * world_size)
    train_sampler = BucketedDistributedSampler(
        bucket_lengths=bucket_lengths, bucket_offsets=bucket_offsets, weights=weights,
        num_microbatches=num_samples_total, num_replicas=world_size, rank=rank, seed=42,
    )
    train_sampler.set_epoch(0)
    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, sampler=train_sampler,
        num_workers=args.num_workers, persistent_workers=args.num_workers > 0,
        pin_memory=True, collate_fn=custom_collate_fn,
    )

    def _make_val_loader(task_fn):
        if not Path(args.data_parent_path).exists():
            return None
        ds = FullAtomisticLanguageDataset(
            tokenizer=model.module.tokenizer, split="validation",
            parent_folder=args.data_parent_path,
            thinking=False, max_num_tokens=args.max_num_tokens,
            cached_embs_parent_path=args.cached_embs_parent_path,
            tasks=task_fn,
            atomistic_model_name=args.atomistic_model_name,
            atomistic_feature_dim=args.atomistic_feature_dim,
        )
        if args.val_subset_fraction and args.val_subset_fraction < 1.0:
            n = max(1, int(args.val_subset_fraction * len(ds)))
            g = torch.Generator().manual_seed(42)
            ds = Subset(ds, torch.randperm(len(ds), generator=g)[:n].tolist())
        sampler = DistributedSampler(ds, shuffle=True, drop_last=True, seed=42)
        return DataLoader(
            ds, batch_size=args.batch_size, sampler=sampler,
            num_workers=args.num_workers, persistent_workers=args.num_workers > 0,
            pin_memory=True, collate_fn=custom_collate_fn,
        )

    # Text-only val: held-out split="validation" slice (deterministic split_seed=42).
    def _text_val_loader(name):
        tok = model.module.tokenizer
        if name == "arxiv":
            ds = ArxivAbstractDataset(tok, args.arxiv_parquet, args.max_num_tokens,
                                      split="validation",
                                      atomistic_feature_dim=args.atomistic_feature_dim)
        elif name == "camel":
            ds = CamelAIDataset(tok, args.camel_jsonl, thinking=False,
                                max_num_tokens=args.max_num_tokens, split="validation",
                                atomistic_feature_dim=args.atomistic_feature_dim)
        elif name == "mascqa":
            ds = MaScQADataset(tok, args.mascqa_json, args.mascqa_xlsx,
                               thinking=False, max_num_tokens=args.max_num_tokens,
                               split="validation",
                               atomistic_feature_dim=args.atomistic_feature_dim)
        else:
            raise ValueError(name)
        sampler = DistributedSampler(ds, shuffle=False, drop_last=True, seed=42)
        return DataLoader(
            ds, batch_size=args.batch_size, sampler=sampler,
            num_workers=args.num_workers, persistent_workers=args.num_workers > 0,
            pin_memory=True, collate_fn=custom_collate_fn,
        )

    # Separate MatterChat val: NL A/B/C prompts vs LLM4Mat's JSON, tracked apart to disambiguate format drift from regression.
    def _matterchat_val_loader():
        mc_csv = Path(args.matterchat_val_csv)
        mc_cache = Path(args.matterchat_val_cache)
        mc_cache_ok = mc_cache.exists() and mc_cache.with_suffix(".idx.json").exists()
        if not (mc_csv.exists() and mc_cache_ok):
            if is_main_process():
                print(f"[stage2] skip matterchat val: csv={mc_csv.exists()} cache_ok={mc_cache_ok}")
            return None
        ds = AtomisticLanguageDataset(
            tokenizer=model.module.tokenizer, db_path=None, csv_path=str(mc_csv),
            thinking=False, max_num_tokens=args.max_num_tokens,
            dataset_name="matterchat_mp", cached_embs_path=str(mc_cache),
            tasks=property_tasks_for_dataset("matterchat_mp"),
            atomistic_feature_dim=args.atomistic_feature_dim,
        )
        if args.val_subset_fraction and args.val_subset_fraction < 1.0:
            n = max(1, int(args.val_subset_fraction * len(ds)))
            g = torch.Generator().manual_seed(42)
            ds = Subset(ds, torch.randperm(len(ds), generator=g)[:n].tolist())
        sampler = DistributedSampler(ds, shuffle=True, drop_last=True, seed=42)
        return DataLoader(
            ds, batch_size=args.batch_size, sampler=sampler,
            num_workers=args.num_workers, persistent_workers=args.num_workers > 0,
            pin_memory=True, collate_fn=custom_collate_fn,
        )

    # Skip val loaders for zero-weight buckets to save per-rank CPU RAM.
    weights_by_name = dict(zip(BUCKET_NAMES, weights))
    def _vl_if_active(name, builder):
        if weights_by_name.get(name, 0.0) == 0.0:
            return None
        return builder()
    val_loaders = {
        "describe":      _vl_if_active("describe",      lambda: _make_val_loader(describe_tasks_for_dataset)),
        "property_apps": _vl_if_active("property_apps", lambda: _make_val_loader(property_tasks_for_dataset)),
        "matterchat":    _vl_if_active("matterchat",    _matterchat_val_loader),
        "arxiv":         _vl_if_active("arxiv",         lambda: _text_val_loader("arxiv")),
        "camel":         _vl_if_active("camel",         lambda: _text_val_loader("camel")),
        "mascqa":        _vl_if_active("mascqa",        lambda: _text_val_loader("mascqa")),
    }
    val_loaders = {k: v for k, v in val_loaders.items() if v is not None}
    _log_mem("after val loaders", main_process)

    lora_params, projector_params = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if "projector" in name:
            projector_params.append(p)
        else:
            lora_params.append(p)
    optim = _build_optimizer(args, lora_params, projector_params)
    warmup_steps = min(2000, int(0.03 * args.total_optim_steps))
    scheduler = get_cosine_schedule_with_warmup(
        optim, num_warmup_steps=warmup_steps, num_training_steps=args.total_optim_steps,
    )

    global_opt_step = 0
    if args.resume_from_stage2:
        state = torch.load(Path(args.resume_from_stage2) / "projector_and_state.pt",
                           map_location=device)
        if args.reset_optim_on_resume:
            # Keep weights but start a fresh optimizer/scheduler/step-0 for continuation runs.
            if main_process:
                print(f"[stage2] reset_optim_on_resume=True: ignoring optimizer/scheduler/global_opt_step "
                      f"in {args.resume_from_stage2}; starting fresh schedule.")
            global_opt_step = 0
        else:
            if "optimizer_state_dict" in state:
                optim.load_state_dict(state["optimizer_state_dict"])
            if "scheduler_state_dict" in state:
                scheduler.load_state_dict(state["scheduler_state_dict"])
            global_opt_step = int(state.get("global_opt_step", 0))
        dist.barrier()
    _log_mem("after optim+resume", main_process)

    if use_wandb:
        # Resume by ID (with step=global_opt_step on every log) so curves stay on one plot across resumes.
        init_kwargs = {"project": args.wandb_project, "config": vars(args)}
        if args.wandb_run_id:
            init_kwargs["id"] = args.wandb_run_id
            init_kwargs["resume"] = "allow"
        wandb.init(**init_kwargs)

    save_root = Path(args.save_dir)
    save_root.mkdir(parents=True, exist_ok=True)

    # Single-pass loop; the sampler delivers exactly total_optim_steps x grad_accum_steps microbatches per rank.
    model.train()
    model.module.llm.train()
    if model.module.atomistic_model is not None:
        model.module.atomistic_model.eval()
    optim.zero_grad(set_to_none=True)

    micro_i = 0
    resume_micro = global_opt_step * args.grad_accum_steps
    for batch in train_loader:
        if micro_i < resume_micro:
            micro_i += 1
            continue
        row_batch = batch.get("atom_rows")
        atom_embeds = batch.get("atom_embeds")
        input_ids = [t.to(device) for t in batch["input_ids"]]
        labels = [t.to(device) for t in batch["labels"]]
        attention_mask = [t.to(device) for t in batch["attention_mask"]]

        is_accum = ((micro_i + 1) % args.grad_accum_steps) != 0
        sync_ctx = model.no_sync() if is_accum else nullcontext()
        with sync_ctx, torch.cuda.amp.autocast(dtype=torch.bfloat16):
            outputs = model(input_ids, attention_mask, labels,
                            row_batch=row_batch, atom_embeds=atom_embeds)
            loss = outputs.loss / args.grad_accum_steps
        loss.backward()

        micro_i += 1
        if is_accum:
            continue

        total_norm = torch.nn.utils.clip_grad_norm_(
            [p for p in model.parameters() if p.requires_grad], max_norm=1.0,
        )
        # All-reduce a finite flag so all ranks skip together; one NaN batch otherwise poisons AdamW state.
        is_finite = torch.isfinite(total_norm).to(torch.float32)
        dist.all_reduce(is_finite, op=dist.ReduceOp.MIN)
        if is_finite.item() == 1.0:
            optim.step()
            scheduler.step()
        elif main_process:
            print(f"[skip] nan/inf grads at opt_step {global_opt_step + 1}; "
                  f"local total_norm={total_norm.item():.3e}")
        optim.zero_grad(set_to_none=True)
        global_opt_step += 1

        if main_process and global_opt_step % 10 == 0:
            print(f"opt_step {global_opt_step}/{args.total_optim_steps} "
                  f"loss={loss.item() * args.grad_accum_steps:.4f} "
                  f"lr_lora={optim.param_groups[0]['lr']:.2e} "
                  f"lr_proj={optim.param_groups[1]['lr']:.2e}")
        if use_wandb and global_opt_step % args.log_every == 0:
            wandb.log({
                "train/loss": loss.item() * args.grad_accum_steps,
                "train/lr_lora": optim.param_groups[0]["lr"],
                "train/lr_projector": optim.param_groups[1]["lr"],
                "global_opt_step": global_opt_step,
            }, step=global_opt_step)

        if val_loaders and global_opt_step % args.eval_every == 0:
            run_validation(model, val_loaders, device, global_opt_step, main_process, use_wandb)
            save_checkpoint(model, optim, scheduler, global_opt_step, save_root, main_process)

        if global_opt_step >= args.total_optim_steps:
            break

    save_checkpoint(model, optim, scheduler, global_opt_step, save_root, main_process)
    if use_wandb:
        wandb.finish()
    dist.destroy_process_group()


def run_validation(model, val_loaders, device, global_opt_step, main_process, use_wandb):
    model.eval()
    per_bucket = {}
    for name, loader in val_loaders.items():
        total_loss, n_batches = 0.0, 0
        for batch in loader:
            row_batch = batch.get("atom_rows")
            atom_embeds = batch.get("atom_embeds")
            input_ids = [t.to(device) for t in batch["input_ids"]]
            labels = [t.to(device) for t in batch["labels"]]
            attention_mask = [t.to(device) for t in batch["attention_mask"]]
            with torch.no_grad(), torch.cuda.amp.autocast(dtype=torch.bfloat16):
                out = model(input_ids, attention_mask, labels,
                            row_batch=row_batch, atom_embeds=atom_embeds)
                total_loss += out.loss.item()
                n_batches += 1
        per_bucket[name] = total_loss / max(1, n_batches)
    model.train()
    model.module.llm.train()
    if model.module.atomistic_model is not None:
        model.module.atomistic_model.eval()
    if main_process:
        for name, v in per_bucket.items():
            print(f"val_loss/{name} (opt_step {global_opt_step}): {v:.4f}")
    if use_wandb:
        log = {f"val/loss_{n}": v for n, v in per_bucket.items()}
        log["val/loss"] = sum(per_bucket.values()) / max(1, len(per_bucket))
        log["global_opt_step"] = global_opt_step
        wandb.log(log, step=global_opt_step)


def save_checkpoint(model, optim, scheduler, global_opt_step, save_root, main_process):
    if not main_process:
        return
    ckpt_dir = save_root / f"step={global_opt_step}"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    model.module.llm.save_pretrained(ckpt_dir / "lora_adapter")
    torch.save(
        {
            "projector_state_dict": model.module.projector.state_dict(),
            "optimizer_state_dict": optim.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "global_opt_step": global_opt_step,
        },
        ckpt_dir / "projector_and_state.pt",
    )
    print(f"Saved Stage 2 checkpoint → {ckpt_dir}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--data_parent_path", type=str, default="/tmp/LLM4Mat-Bench")
    p.add_argument("--cached_embs_parent_path", type=str, default="/tmp/cached_embs")
    p.add_argument("--narrative_parquet_dir", type=str, default="/tmp/GPT-Narratives-for-Materials")
    p.add_argument("--narrative_cache_dir", type=str, default="/tmp/cached_embs_narratives")
    p.add_argument("--matterchat_train_csv", type=str,
                   default="data/matterchat/train.csv",
                   help="MatterChat MP train CSV (128k rows). Skipped if missing.")
    p.add_argument("--matterchat_train_cache", type=str,
                   default="data/matterchat/cached_embs/orb_v3_direct_20_omat_train_atom.flat.bin",
                   help="OrbV3 cache for matterchat MP train. Build via cache_embeddings_atomistic_orbv3.py.")
    p.add_argument("--matterchat_val_csv", type=str,
                   default="data/matterchat/val.csv",
                   help="MatterChat MP val CSV (~14k rows). Held-out, used only for val/loss_matterchat.")
    p.add_argument("--matterchat_val_cache", type=str,
                   default="data/matterchat/cached_embs/orb_v3_direct_20_omat_validation_atom.flat.bin",
                   help="OrbV3 cache for matterchat MP val.")
    p.add_argument("--mascqa_json", type=str, default="/tmp/MaScQA/mascqa-eval.json")
    p.add_argument("--mascqa_xlsx", type=str, default="/tmp/MaScQA/scoresheets/all_questions.xlsx")
    p.add_argument("--arxiv_parquet", type=str, default="/tmp/jarvis_arxiv.parquet")
    p.add_argument("--camel_jsonl",   type=str, default="/tmp/camel_ai.jsonl")
    p.add_argument("--max_num_tokens", type=int, default=2048)
    p.add_argument("--bucket_weights", type=str, default="0.408,0.408,0.14,0.04,0,0",
                   help="order: describe, property_apps, arxiv, camel, mascqa, matterchat (0 = skip). "
                        "Legacy 5-entry strings (no matterchat) are accepted and padded with 0.")
    p.add_argument("--resume_from_stage1", type=str, default=None)
    p.add_argument("--resume_from_stage2", type=str, default=None)
    p.add_argument("--reset_optim_on_resume", action="store_true",
                   help="On --resume_from_stage2, ignore the saved optimizer + scheduler + "
                        "global_opt_step and start a fresh schedule with the current "
                        "--lora_lr / --projector_lr / --total_optim_steps. Use this for "
                        "continuation runs where the resumed cosine has already decayed to ~0.")
    p.add_argument("--matterchat_in_property_apps", action=argparse.BooleanOptionalAction, default=True,
                   help="(default True) Include matterchat_mp data inside the property_apps bucket. "
                        "Pass --no-matterchat_in_property_apps "
                        "to extract matterchat into its own (6th) bucket so its sampling weight can be "
                        "set independently via --bucket_weights. When True, the matterchat bucket weight "
                        "must be 0; when False, set the matterchat weight to a positive value.")
    p.add_argument("--atomistic_model_name", type=str, default="orb_v3_direct_20_omat",
                   help="Encoder tag baked into cached-embedding filenames "
                        "({atomistic_model_name}_{split}_atom.flat.bin). Default is OrbV3.")
    p.add_argument("--atomistic_feature_dim", type=int, default=256,
                   help="Per-atom feature dim of the cached embeddings. OrbV3=256, UMA=128, "
                        "PET-MAD variable.")
    p.add_argument("--atom_bidirectional_attention", action="store_true",
                   help="When set, atoms inside the <atoms>-spliced block attend BIDIRECTIONALLY "
                        "to each other (instead of the default causal mask). Forces the LLM's "
                        "attn_implementation to 'sdpa' (flash_attn_2 requires strictly causal). "
                        "OrbV3 outputs atoms in arbitrary ASE-index order — causal attention "
                        "imposes a meaningless ordering on a fundamentally unordered set; "
                        "bidirectional-within-block fixes that. Other tokens (text, output-side "
                        "[atoms_i]) remain causal. Slower (~10-15%%) than flash_attn_2 but "
                        "principled.")
    p.add_argument("--llm_name", type=str, default="Qwen/Qwen3-8B",
                   help="HuggingFace model id; used by the scaling-law sweep to swap the "
                        "base LLM (Qwen/Qwen3-0.6B, -1.7B, -4B, -8B, -14B, -32B).")
    p.add_argument("--lora_rank", type=int, default=64)
    p.add_argument("--lora_alpha", type=int, default=128)
    p.add_argument("--optimizer", choices=["adamw", "muon"], default="adamw",
                   help="muon: Newton-Schulz orthogonalized SGD on LoRA 2D mats, AdamW for the rest")
    p.add_argument("--lora_lr", type=float, default=2e-4)
    p.add_argument("--projector_lr", type=float, default=2e-5)
    p.add_argument("--batch_size", type=int, default=12)
    p.add_argument("--grad_accum_steps", type=int, default=4)
    p.add_argument("--total_optim_steps", type=int, default=12000,
                   help="LLaVA-1.5 style total optimizer-step budget (~1-3 logical passes of the mixture).")
    p.add_argument("--eval_every", type=int, default=500)
    p.add_argument("--val_subset_fraction", type=float, default=0.01)
    p.add_argument("--log_every", type=int, default=10)
    p.add_argument("--disable_wandb", action="store_true")
    p.add_argument("--wandb_project", type=str, default="alm-stage2")
    p.add_argument("--wandb_run_id", type=str, default=None,
                   help="Resume a specific wandb run by id (e.g. 'vc5cfy32' to keep "
                        "curves on one plot when changing node count or batch size).")
    p.add_argument("--save_dir", type=str, default="runs/stage2")
    p.add_argument("--num_workers", type=int, default=4)
    args = p.parse_args()
    train(args)
