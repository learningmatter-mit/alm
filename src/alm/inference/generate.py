#!/usr/bin/env python
"""Arbitrary-prompt inference CLI for the Atomistic Language Model: `understand` (text + optional structure -> text) and `generate` (text -> crystal CIF via the bridge)."""

import argparse

import torch

from loader import load_alm
from text_generation import generate_batch


def _build_understand_batch(tokenizer, prompt, atoms=None):
    """Build a one-sample generate_batch input; with `atoms`, splice an `<atoms>` placeholder (LLaVA-style input path)."""
    content = ("<atoms>\n" + prompt) if atoms is not None else prompt
    ids = tokenizer.apply_chat_template(
        [{"role": "user", "content": content}],
        add_generation_prompt=True, enable_thinking=False, return_tensors="pt",
    )[0]
    batch = {
        "input_ids": [ids],
        "labels": [torch.full_like(ids, -100)],   # whole sequence is prompt
        "attention_mask": [torch.ones_like(ids)],
        "id": ["prompt0"],
    }
    if atoms is not None:
        batch["atom_rows"] = [atoms]               # live OrbV3 encode in generate_batch
    return batch


def cmd_understand(args):
    model, tokenizer = load_alm(checkpoint=args.alm_checkpoint, merge_lora=True,
                                attn_implementation=args.attn_implementation)
    model.eval()

    atoms = None
    if args.structure:
        from ase.io import read as ase_read
        atoms = ase_read(args.structure)
        print(f"[understand] loaded structure {args.structure} "
              f"({len(atoms)} atoms, {atoms.get_chemical_formula()})")

    batch = _build_understand_batch(tokenizer, args.prompt, atoms=atoms)
    with torch.no_grad():
        out = generate_batch(model, batch, max_new_tokens=args.max_new_tokens,
                             atomistic=atoms is not None)
    print("\n=== ALM ===\n" + out[0].strip() + "\n")


def cmd_generate(args):
    from generate_stage3 import load_alm_and_pl_module, generate_for_prompts

    prompts = [args.prompt] if args.prompt else [
        ln.strip() for ln in open(args.prompts_file) if ln.strip()
    ]
    alm, tokenizer, pl_module, _K = load_alm_and_pl_module(
        alm_checkpoint=args.alm_checkpoint,
        atoms_mapper=args.atoms_mapper,
        mattergen_pretrained=args.mattergen_pretrained,
        model_path=args.mattergen_model_path,
    )
    n_batches = max(1, (args.num_samples + args.batch_size - 1) // args.batch_size)
    generate_for_prompts(
        prompts=prompts,
        alm=alm, tokenizer=tokenizer, pl_module=pl_module,
        out_root=args.out_dir,
        batch_size=args.batch_size,
        num_batches=n_batches,
        diffusion_guidance_factor=args.guidance_factor,
        diffusion_seed=args.seed,
    )
    print(f"[generate] {len(prompts)} prompt(s) x {args.num_samples} sample(s) -> {args.out_dir}")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="mode", required=True)

    u = sub.add_parser("understand", help="text (+ optional structure) -> text answer")
    u.add_argument("--alm_checkpoint", required=True, help="Stage-2 step=N dir")
    u.add_argument("--prompt", required=True)
    u.add_argument("--structure", default=None, help="optional CIF/POSCAR to condition on (atoms->text)")
    u.add_argument("--max_new_tokens", type=int, default=512)
    u.add_argument("--attn_implementation", default="flash_attention_2",
                   choices=["flash_attention_2", "sdpa", "eager"],
                   help="sdpa/eager need no flash-attn build (correctness-equivalent, slightly slower)")
    u.set_defaults(func=cmd_understand)

    g = sub.add_parser("generate", help="text description -> crystal structure (CIF)")
    g.add_argument("--alm_checkpoint", required=True, help="Stage-2 step=N dir")
    g.add_argument("--atoms_mapper", required=True, help="Stage-3 step=N/atoms_mapper.pt")
    g.add_argument("--mattergen_pretrained", default="mattergen_base",
                   help="HF backbone name (DNG mode); ignored if --mattergen_model_path is set")
    g.add_argument("--mattergen_model_path", default=None,
                   help="from-scratch CSP-mode backbone dir (e.g. csp_backbone) for structure-conditioned gen")
    g.add_argument("--prompt", default=None, help="single description")
    g.add_argument("--prompts_file", default=None, help="one description per line (alternative to --prompt)")
    g.add_argument("--num_samples", type=int, default=4)
    g.add_argument("--batch_size", type=int, default=4)
    g.add_argument("--guidance_factor", type=float, default=1.0)
    g.add_argument("--seed", type=int, default=0)
    g.add_argument("--out_dir", default="./gen_out")
    g.set_defaults(func=cmd_generate)

    args = ap.parse_args()
    if args.mode == "generate" and not (args.prompt or args.prompts_file):
        ap.error("generate: pass --prompt or --prompts_file")
    args.func(args)


if __name__ == "__main__":
    main()
