"""Preprocess the Stage 2 text-only buckets: JARVIS arXiv JSON to Parquet, CAMEL-AI JSONs to one JSONL."""

import argparse
import json
import os
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
from tqdm import tqdm

from paths import DATA_ROOT


_KEEP_FIELDS = ("id", "title", "categories", "abstract")


def _stream_json_array(path, buf_bytes=1 << 20):
    """Stream dicts from a top-level JSON array without loading the whole file."""
    dec = json.JSONDecoder()
    with open(path, "r", encoding="utf-8") as f:
        ch = f.read(1)
        while ch and ch.isspace():
            ch = f.read(1)
        if ch != "[":
            raise ValueError(f"expected top-level JSON array in {path}, got {ch!r}")
        buf = ""
        while True:
            chunk = f.read(buf_bytes)
            if not chunk and not buf:
                break
            buf += chunk
            while True:
                i = 0
                while i < len(buf) and buf[i] in " \t\n\r,":
                    i += 1
                if i:
                    buf = buf[i:]
                if not buf or buf[0] == "]":
                    return
                try:
                    obj, end = dec.raw_decode(buf)
                except json.JSONDecodeError:
                    break   # need more data
                buf = buf[end:]
                yield obj


def preprocess_arxiv(src, dst, rows_per_chunk=50_000):
    src = Path(src)
    dst = Path(dst)
    if dst.exists() and dst.stat().st_mtime >= src.stat().st_mtime:
        print(f"[arxiv] up-to-date: {dst}")
        return
    dst.parent.mkdir(parents=True, exist_ok=True)

    schema = pa.schema([(f, pa.string()) for f in _KEEP_FIELDS])
    writer = pq.ParquetWriter(dst, schema, compression="zstd")
    buf = {f: [] for f in _KEEP_FIELDS}
    n_written = 0
    pbar = tqdm(desc="arxiv", unit="rec")
    for rec in _stream_json_array(src):
        for f in _KEEP_FIELDS:
            v = rec.get(f)
            buf[f].append(v if v is None else str(v))
        if len(buf["id"]) >= rows_per_chunk:
            writer.write_table(pa.table(buf, schema=schema))
            n_written += len(buf["id"])
            pbar.update(len(buf["id"]))
            buf = {f: [] for f in _KEEP_FIELDS}
    if buf["id"]:
        writer.write_table(pa.table(buf, schema=schema))
        n_written += len(buf["id"])
        pbar.update(len(buf["id"]))
    writer.close()
    pbar.close()
    size_gb = dst.stat().st_size / 1e9
    print(f"[arxiv] wrote {n_written:,} rows → {dst} ({size_gb:.2f} GB)")


def preprocess_camel(root, dst):
    root = Path(root)
    dst = Path(dst)
    dst.parent.mkdir(parents=True, exist_ok=True)
    files = []
    for sub in ("chemistry", "physics"):
        d = root / sub
        if d.exists():
            files.extend(sorted(d.iterdir()))
    if not files:
        raise FileNotFoundError(f"no CAMEL-AI JSONs under {root}")
    if dst.exists() and dst.stat().st_mtime >= max(p.stat().st_mtime for p in files):
        print(f"[camel] up-to-date: {dst}")
        return

    n = 0
    with open(dst, "w", encoding="utf-8") as out:
        for p in tqdm(files, desc="camel", unit="file"):
            with open(p) as f:
                rec = json.load(f)
            # Raw files have a typo key "topic;" — normalize on the way in.
            row = {
                "topic": rec.get("topic;") or rec.get("topic"),
                "sub_topic": rec.get("sub_topic"),
                "message_1": rec.get("message_1"),
                "message_2": rec.get("message_2"),
            }
            out.write(json.dumps(row, ensure_ascii=False) + "\n")
            n += 1
    print(f"[camel] wrote {n:,} rows → {dst}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--arxiv_json", default=os.path.join(DATA_ROOT, "jarvis/arXivdataset.json"))
    p.add_argument("--arxiv_out",  default="/tmp/jarvis_arxiv.parquet")
    p.add_argument("--camel_root", default=os.path.join(DATA_ROOT, "camel-ai"))
    p.add_argument("--camel_out",  default="/tmp/camel_ai.jsonl")
    p.add_argument("--skip_arxiv", action="store_true")
    p.add_argument("--skip_camel", action="store_true")
    args = p.parse_args()
    if not args.skip_arxiv:
        preprocess_arxiv(args.arxiv_json, args.arxiv_out)
    if not args.skip_camel:
        preprocess_camel(args.camel_root, args.camel_out)
