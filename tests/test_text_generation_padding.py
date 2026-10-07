"""generate_batch must pass left-padded prompts to generate()."""
import torch

import alm  # noqa: F401  (puts the flat alm module namespace on sys.path)
from text_generation import generate_batch


class _FakeLLM:
    def __init__(self, dim):
        self.emb = torch.nn.Embedding(50, dim)
        self.seen = {}

    def get_input_embeddings(self):
        return self.emb

    def generate(self, inputs_embeds, attention_mask, **kw):
        self.seen["embeds"], self.seen["mask"] = inputs_embeds, attention_mask
        return torch.zeros(inputs_embeds.shape[0], 1, dtype=torch.long)


class _FakeTok:
    eos_token_id = pad_token_id = 0

    def decode(self, ids, skip_special_tokens=True):
        return ""


class _FakeALM:
    device = torch.device("cpu")

    def __init__(self, dim=4):
        self.llm, self.tokenizer = _FakeLLM(dim), _FakeTok()

    def _merge_embeddings(self, text_embeds, atom_features, input_ids, labels, attention_mask):
        # Right-pads, like AtomisticLanguageModel._merge_embeddings.
        L = max(len(e) for e in text_embeds)
        emb = torch.zeros(len(text_embeds), L, text_embeds[0].shape[-1])
        mask = torch.zeros(len(text_embeds), L, dtype=torch.long)
        for b, e in enumerate(text_embeds):
            emb[b, :len(e)], mask[b, :len(e)] = e, 1
        return emb, None, mask, None


def _batch(lengths):
    ids = [torch.arange(1, n + 1) for n in lengths]
    return {"input_ids": ids, "labels": [torch.full_like(i, -100) for i in ids],
            "attention_mask": [torch.ones_like(i) for i in ids]}


def test_prompts_are_left_padded():
    alm_model = _FakeALM()
    generate_batch(alm_model, _batch([3, 7, 5]), atomistic=False)
    emb, mask = alm_model.llm.seen["embeds"], alm_model.llm.seen["mask"]
    assert mask[:, -1].tolist() == [1, 1, 1]  # every row ends on a real prompt token
    assert mask.sum(1).tolist() == [3, 7, 5]
    want = alm_model.llm.emb(torch.arange(1, 4))
    assert torch.allclose(emb[0, -3:], want) and torch.all(emb[0, :-3] == 0)
