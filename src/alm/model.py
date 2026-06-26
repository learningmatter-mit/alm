import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer

from orb_models.forcefield import atomic_system, pretrained
from orb_models.forcefield.base import batch_graphs

from utils import is_main_process

class AtomisticLanguageModel(nn.Module):
    def __init__(self, llm_name='Qwen/Qwen3-8B', atomistic_model_name='orb_v3_direct_20_omat', device=None,
                 attn_implementation="flash_attention_2", use_cached_embeddings=False, max_atoms=None,
                 num_output_atom_tokens: int = 8, atomistic_feature_dim: int = 256,
                 use_last_prompt_token: bool = False,
                 bridge_source: str = 'atoms_tokens',
                 qformer_n_context: int = 128,
                 qformer_input_atoms: int = 0,
                 init_atoms_tokens_from_eos: bool = False,
                 atom_bidirectional_attention: bool = False):
        super().__init__()
        self.device = device if device is not None else torch.device("cuda")
        self.use_cached_embeddings = use_cached_embeddings
        # Bidirectional attention within the spliced atom block; OrbV3 emits atoms
        # in arbitrary order so causal masking imposes a meaningless ordering on a set.
        self.atom_bidirectional_attention = bool(atom_bidirectional_attention)
        if self.atom_bidirectional_attention and attn_implementation == "flash_attention_2":
            print(f"[ALM] atom_bidirectional_attention=True forces "
                  f"attn_implementation='sdpa' (flash_attn_2 requires strict causal)")
            attn_implementation = "sdpa"
        # Pin all ranks to mem-efficient SDPA: shape-based kernel dispatch otherwise diverges across multi-node DDP ranks and times out NCCL.
        if self.atom_bidirectional_attention:
            # No function-local torch import: it would shadow torch for the whole __init__ and UnboundLocalError earlier refs.
            torch.backends.cuda.enable_flash_sdp(False)
            torch.backends.cuda.enable_math_sdp(False)
            torch.backends.cuda.enable_mem_efficient_sdp(True)
            print(f"[ALM] pinned SDPA backend → mem_efficient only "
                  f"(disables per-call dispatch divergence under multi-node DDP)")
        self.use_last_prompt_token = bool(use_last_prompt_token)
        if bridge_source not in ('atoms_tokens', 'last_k_prompt', 'context_plus_atoms'):
            raise ValueError(
                f"bridge_source must be 'atoms_tokens', 'last_k_prompt', or "
                f"'context_plus_atoms'; got {bridge_source!r}"
            )
        self.bridge_source = bridge_source
        # Context hidden states gathered before [atoms_0] for the Q-Former bridge; source len S = N + K.
        self.qformer_n_context = int(qformer_n_context)
        # When >0, prepend L_in input-side <atoms> states so the input structure is always in the source regardless of prompt length.
        self.qformer_input_atoms = int(qformer_input_atoms)
        # Must match the cap in AtomisticLanguageDataset.prepare_sample so live mode doesn't splice uncapped atoms.
        self.max_atoms = max_atoms

        self.llm = AutoModelForCausalLM.from_pretrained(
            llm_name,
            torch_dtype=torch.bfloat16,
            attn_implementation=attn_implementation
        )
        self.tokenizer = AutoTokenizer.from_pretrained(llm_name)
        self.llm_hidden_dim = self.llm.config.hidden_size

        for param in self.llm.parameters():
            param.requires_grad = False

        # Frozen encoder; skipped under cached embeddings to avoid the 7 GB load + per-step graph build.
        if use_cached_embeddings:
            self.atomistic_model = None
        else:
            model = getattr(pretrained, atomistic_model_name)
            orbff = model(
                device=self.device,
                precision="float32-high",
            )
            self.atomistic_model = orbff
            for param in self.atomistic_model.parameters():
                param.requires_grad = False

        # Only trainable part; atomistic_feature_dim is encoder-specific (256 OrbV3, 128 UMA).
        llm_dim = self.llm_hidden_dim
        self.atomistic_feature_dim = atomistic_feature_dim
        self.projector = nn.Sequential(
            nn.Linear(atomistic_feature_dim, llm_dim),
            nn.GELU(),
            nn.Linear(llm_dim,  llm_dim),
        )
        self.atom_token = '<atoms>'
        self.tokenizer.add_tokens([self.atom_token])
        self.atoms_token_id = self.tokenizer.convert_tokens_to_ids(self.atom_token)

        # K output-side tokens whose final hidden states feed AtomsMapper. Seed/restore RNG so the frozen init rows are deterministic across cache + inference.
        self.num_output_atom_tokens = num_output_atom_tokens
        self.output_atom_tokens = [f'[atoms_{i}]' for i in range(num_output_atom_tokens)]
        self.tokenizer.add_tokens(self.output_atom_tokens, special_tokens=True)
        self.output_atom_token_ids = self.tokenizer.convert_tokens_to_ids(self.output_atom_tokens)
        rng_state = torch.get_rng_state()
        torch.manual_seed(42)
        self.llm.resize_token_embeddings(len(self.tokenizer))
        torch.set_rng_state(rng_state)

        # Seed the K [atoms_i] rows from the <|im_end|> embedding (id 151645) so they start with end-of-turn semantics rather than random.
        self.init_atoms_tokens_from_eos = bool(init_atoms_tokens_from_eos)
        if self.init_atoms_tokens_from_eos:
            try:
                eos_id = self.tokenizer.convert_tokens_to_ids("<|im_end|>")
                if eos_id is None or eos_id == self.tokenizer.unk_token_id:
                    eos_id = self.tokenizer.eos_token_id
            except Exception:
                eos_id = self.tokenizer.eos_token_id
            if eos_id is None:
                raise ValueError("init_atoms_tokens_from_eos=True but tokenizer has no EOS / <|im_end|> token")
            embed = self.llm.get_input_embeddings()
            with torch.no_grad():
                src = embed.weight.data[eos_id].clone()
                for tid in self.output_atom_token_ids:
                    embed.weight.data[tid] = src
            if is_main_process():
                print(f"[ALM] init_atoms_tokens_from_eos=True: copied <|im_end|> "
                      f"embedding (token_id={eos_id}) into K={len(self.output_atom_token_ids)} "
                      f"[atoms_i] rows.")

    def encode_atoms(self, row_batch):
        with torch.no_grad():
            # Accepts ASE db rows (.toatoms()) or raw ASE Atoms.
            atoms_list = [r.toatoms() if hasattr(r, "toatoms") else r for r in row_batch]
            if self.max_atoms is not None:
                atoms_list = [a[:self.max_atoms] for a in atoms_list]
            batch = [
                atomic_system.ase_atoms_to_atom_graphs(
                    atoms,
                    self.atomistic_model.system_config,
                    device=self.device,
                )
                for atoms in atoms_list
            ]
            graph = batch_graphs(batch)
            results = self.atomistic_model.model(graph)
            node_features = results["node_features"]
        out = self.projector(node_features)
        n_atoms = tuple(graph.n_node.tolist())
        return out, n_atoms

    def encode_cached_atoms(self, atom_embeds):
        # atom_embeds: list of (N_i, atomistic_feature_dim) tensors; concat then project. Same return contract as encode_atoms.
        n_atoms = tuple(a.shape[0] for a in atom_embeds)
        projector_dtype = next(self.projector.parameters()).dtype
        stacked = torch.cat(
            [a.to(device=self.device, dtype=projector_dtype, non_blocking=True) for a in atom_embeds],
            dim=0,
        )
        out = self.projector(stacked)
        return out, n_atoms


    def forward(self, input_ids, attention_mask, labels=None, row_batch=None,
                atom_embeds=None, output_atoms_hidden_states: bool = False):
        """LM-loss forward (labels required), or (B, K, hidden_dim) at the [atoms_i] positions when output_atoms_hidden_states=True. Text-only prompts pass zero-row atom_embeds to skip the splice."""
        if atom_embeds is not None:
            atomistic_features, n_atoms = self.encode_cached_atoms(atom_embeds)
        else:
            atomistic_features, n_atoms = self.encode_atoms(row_batch)
        atomistic_features = torch.split(atomistic_features, n_atoms)

        embed_layer = self.llm.get_input_embeddings()
        text_embeds = [embed_layer(sample_input_ids) for sample_input_ids in input_ids]

        if output_atoms_hidden_states:
            K = len(self.output_atom_token_ids)
            # Compute LM-CE in this same forward; a second forward breaks multi-node DDP's reducer (NCCL mismatch).
            _compute_lm = labels is not None
            _lbl_in = labels if _compute_lm else [torch.zeros_like(sids) for sids in input_ids]
            new_embeds, new_labels, new_attention_mask, atom_ranges = self._merge_embeddings(
                text_embeds, atomistic_features, input_ids, _lbl_in, attention_mask
            )
            # Map [atoms_i] positions to post-splice indices: each <atoms> shifts later tokens by (n_atoms[b] - 1).
            output_atom_set = set(self.output_atom_token_ids)
            positions_per_sample = []
            for b in range(len(input_ids)):
                ids = input_ids[b].tolist()
                running_offset = 0
                sample_positions = []
                for orig_idx, tid in enumerate(ids):
                    if tid == self.atoms_token_id:
                        running_offset += n_atoms[b] - 1
                    elif tid in output_atom_set:
                        sample_positions.append(orig_idx + running_offset)
                assert len(sample_positions) == K, (
                    f"Sample {b}: expected exactly {K} output atom tokens, "
                    f"found {len(sample_positions)} in input_ids of len {len(ids)}."
                )
                positions_per_sample.append(sample_positions)
            # Also make the K output [atoms_i] block bidirectional: causal would build [atoms_0] without [atoms_1..7], breaking the set the bridge consumes. Verify contiguity first.
            if self.atom_bidirectional_attention:
                for b, sp in enumerate(positions_per_sample):
                    if len(sp) == K and sp[-1] - sp[0] == K - 1:
                        atom_ranges[b].append((sp[0], sp[-1] + 1))
            attn_in = new_attention_mask
            if self.atom_bidirectional_attention:
                attn_in = self._build_atom_bidir_4d_mask(
                    new_attention_mask, atom_ranges, dtype=new_embeds.dtype,
                )
            outputs = self.llm(
                inputs_embeds=new_embeds,
                attention_mask=attn_in,
                labels=(new_labels if _compute_lm else None),
                output_hidden_states=True,
                return_dict=True,
            )
            last_hidden = outputs.hidden_states[-1]  # (B, max_len, hidden_dim)
            if self.use_last_prompt_token:
                # Broadcast the state before the first [atoms_i] K times.
                out_stack = []
                for b in range(len(input_ids)):
                    first_atoms_pos = positions_per_sample[b][0]
                    last_prompt_pos = max(0, first_atoms_pos - 1)
                    h = last_hidden[b, last_prompt_pos, :]
                    out_stack.append(h.unsqueeze(0).expand(K, -1))
                hidden_stack = torch.stack(out_stack, dim=0)  # (B, K, hidden_dim)
                return (hidden_stack, outputs.loss) if _compute_lm else hidden_stack
            if self.bridge_source == 'last_k_prompt':
                # K states at [first_atoms_pos - K : first_atoms_pos]; left-pad short prompts.
                out_stack = []
                for b in range(len(input_ids)):
                    first_atoms_pos = positions_per_sample[b][0]
                    start = max(0, first_atoms_pos - K)
                    sl = last_hidden[b, start:first_atoms_pos, :]
                    if sl.shape[0] < K:
                        pad = sl[:1].expand(K - sl.shape[0], -1)
                        sl = torch.cat([pad, sl], dim=0)
                    out_stack.append(sl)
                hidden_stack = torch.stack(out_stack, dim=0)  # (B, K, hidden_dim)
                return (hidden_stack, outputs.loss) if _compute_lm else hidden_stack
            if self.bridge_source == 'context_plus_atoms':
                # Q-Former source: N context states before [atoms_0] (left-padded) ++ K [atoms_i] states. Returns (B, L_in+N+K, hidden).
                N = self.qformer_n_context
                L_in = self.qformer_input_atoms
                out_stack = []
                for b in range(len(input_ids)):
                    first_atoms_pos = positions_per_sample[b][0]
                    start = max(0, first_atoms_pos - N)
                    ctx = last_hidden[b, start:first_atoms_pos, :]  # (<=N, hidden)
                    if ctx.shape[0] < N:
                        pad_src = ctx[:1] if ctx.shape[0] > 0 else last_hidden[b, :1, :]
                        pad = pad_src.expand(N - ctx.shape[0], -1)
                        ctx = torch.cat([pad, ctx], dim=0)  # (N, hidden)
                    atoms = last_hidden[b, positions_per_sample[b], :]  # (K, hidden)
                    if L_in > 0:
                        # Prepend input-side <atoms> states, fixed to L_in, so the input structure is always in the source.
                        in_segs = [last_hidden[b, s:e, :] for (s, e) in atom_ranges[b]]
                        inp = torch.cat(in_segs, dim=0) if in_segs else last_hidden[b, :0, :]
                        if inp.shape[0] >= L_in:
                            inp = inp[-L_in:]
                        else:
                            pad_src = inp[:1] if inp.shape[0] > 0 else last_hidden[b, :1, :]
                            pad = pad_src.expand(L_in - inp.shape[0], -1)
                            inp = torch.cat([pad, inp], dim=0)  # (L_in, hidden)
                        out_stack.append(torch.cat([inp, ctx, atoms], dim=0))
                    else:
                        out_stack.append(torch.cat([ctx, atoms], dim=0))
                hidden_stack = torch.stack(out_stack, dim=0)  # (B, L_in + N + K, hidden_dim)
                return (hidden_stack, outputs.loss) if _compute_lm else hidden_stack
            hidden_stack = torch.stack(
                [last_hidden[b, positions_per_sample[b], :] for b in range(len(input_ids))],
                dim=0,
            )  # (B, K, hidden_dim)
            return (hidden_stack, outputs.loss) if _compute_lm else hidden_stack

        new_embeds, new_labels, new_attention_mask, atom_ranges = self._merge_embeddings(
            text_embeds, atomistic_features, input_ids, labels, attention_mask
        )
        attn_in = new_attention_mask
        if self.atom_bidirectional_attention:
            attn_in = self._build_atom_bidir_4d_mask(
                new_attention_mask, atom_ranges, dtype=new_embeds.dtype,
            )
        outputs = self.llm(
            inputs_embeds=new_embeds,
            attention_mask=attn_in,
            labels=new_labels,
            return_dict=True,
        )
        return outputs

    def extract_atoms_hidden_states(self, input_ids, attention_mask,
                                    row_batch=None, atom_embeds=None):
        """Inference-only alias for forward(..., output_atoms_hidden_states=True); in DDP call alm(...) directly so sync hooks fire."""
        return self.forward(
            input_ids, attention_mask, labels=None,
            row_batch=row_batch, atom_embeds=atom_embeds,
            output_atoms_hidden_states=True,
        )

    def _merge_embeddings(self, text_embeds, atomistic_features, input_ids, labels, attention_mask):
        batch_size = len(text_embeds)

        new_embeds = []
        new_labels = []
        new_attention_mask = []
        # Per-sample atom-block (start, end_exclusive) ranges for _build_atom_bidir_4d_mask; empty for text-only samples.
        atom_ranges_per_sample: list[list[tuple[int, int]]] = []

        for b in range(batch_size):
            atom_token_embed = atomistic_features[b].to(
                dtype=text_embeds[b].dtype,
                device=text_embeds[b].device,
            )
            num_atomistic_tokens = atom_token_embed.shape[0]
            atoms_positions = (input_ids[b] == self.atoms_token_id).nonzero(as_tuple=True)[0]

            if len(atoms_positions) == 0:
                new_embeds.append(text_embeds[b])
                new_labels.append(labels[b])
                new_attention_mask.append(attention_mask[b])
                atom_ranges_per_sample.append([])
                continue

            cur_labels = labels[b]
            cur_embs = text_embeds[b]
            curr_attn_mask = attention_mask[b]
            ranges_for_sample: list[tuple[int, int]] = []
            for position in atoms_positions:
                position_idx = int(position.item())
                emb_before = cur_embs[:position_idx]
                emb_after = cur_embs[position_idx + 1:]

                cur_embs = torch.cat([emb_before, atom_token_embed, emb_after], dim=0)
                ranges_for_sample.append(
                    (position_idx, position_idx + num_atomistic_tokens)
                )

                before_labels = cur_labels[:position_idx]
                atom_labels = torch.full(
                    (num_atomistic_tokens,), -100,
                    dtype=cur_labels.dtype, device=cur_labels.device
                )
                after_labels = cur_labels[position_idx + 1:]
                cur_labels = torch.cat([before_labels, atom_labels, after_labels])

                before_mask = curr_attn_mask[:position_idx]
                atom_mask = torch.ones(
                    num_atomistic_tokens,
                    dtype=curr_attn_mask.dtype, device=curr_attn_mask.device
                )
                after_mask = curr_attn_mask[position_idx + 1:]
                curr_attn_mask = torch.cat([before_mask, atom_mask, after_mask])

            new_embeds.append(cur_embs)
            new_labels.append(cur_labels)
            new_attention_mask.append(curr_attn_mask)
            atom_ranges_per_sample.append(ranges_for_sample)

        max_len = max(len(emb) for emb in new_embeds)
        embed_dim = text_embeds[0].shape[-1]
        padded_embeds = torch.zeros(
            batch_size,
            max_len,
            embed_dim,
            dtype=text_embeds[0].dtype,
            device=text_embeds[0].device,
        )
        padded_labels = torch.full(
            (batch_size, max_len),
            -100,
            dtype=new_labels[0].dtype,
            device=new_labels[0].device,
        )
        padded_mask = torch.zeros(
            batch_size,
            max_len,
            dtype=new_attention_mask[0].dtype,
            device=new_attention_mask[0].device,
        )

        for b in range(batch_size):
            cur_len = len(new_embeds[b])
            padded_embeds[b, :cur_len] = new_embeds[b]
            padded_labels[b, :cur_len] = new_labels[b]
            padded_mask[b, :cur_len] = new_attention_mask[b]

        return padded_embeds, padded_labels, padded_mask, atom_ranges_per_sample

    def _build_atom_bidir_4d_mask(self, padding_mask, atom_ranges_per_sample, dtype):
        """Additive (B, 1, L, L) mask, causal by default but bidirectional within each atom block (0.0=allowed, -inf=blocked)."""
        B, L = padding_mask.shape
        device = padding_mask.device
        causal = torch.zeros(L, L, dtype=dtype, device=device)
        causal.masked_fill_(
            torch.triu(torch.ones(L, L, dtype=torch.bool, device=device),
                       diagonal=1),
            float("-inf"),
        )
        mask_4d = causal.unsqueeze(0).expand(B, L, L).clone()
        # Open up each atom block (Python loop: 1-2 small blocks per sample).
        for b in range(B):
            for (start, end) in atom_ranges_per_sample[b]:
                mask_4d[b, start:end, start:end] = 0.0
        pad_block = (1.0 - padding_mask.to(dtype)).unsqueeze(1) * float("-inf")
        # torch.where, not multiply: 0 * -inf = NaN.
        pad_block = torch.where(
            padding_mask.to(torch.bool).unsqueeze(1),
            torch.zeros_like(pad_block),
            torch.full_like(pad_block, float("-inf")),
        )
        mask_4d = mask_4d + pad_block
        return mask_4d.unsqueeze(1)  # (B, 1, L, L)
