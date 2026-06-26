import os
import torch
import argparse

from paths import DATA_ROOT
from tqdm import tqdm
from itertools import islice

from orb_models.forcefield import atomic_system, pretrained
from orb_models.forcefield.base import batch_graphs
from ase.db import connect

def get_batch(batch_size: int, db):
    iterator = db.select()
    while True:
        batch = list(islice(iterator, batch_size))
        if not batch:
            break
        yield batch

def main(args):
    db = connect(args.data_path)
    print('~~The length of the db is', len(db), flush=True)
    batch_size = args.batch_size
    embeddings = {}
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = getattr(pretrained, args.model_name)
    orbff = model(
        device=device,
        precision="float32-high",
    )
    embedding_dim = 256  # OrbV3 node-feature dim
    torch_embeddings = []
    batch_idx = 0
    for batch in tqdm(get_batch(batch_size, db), total=len(db) // batch_size, desc="Outer loop"):
        smiles = [row.data[args.id_key] for row in batch]
        atoms = [row.toatoms() for row in batch]
        if batch_idx % 100 == 0:
            print(smiles)
            print(len(list(embeddings.keys())))
        batch = [atomic_system.ase_atoms_to_atom_graphs(row.toatoms(), orbff.system_config, device=device) for row in batch]
        graph = batch_graphs(batch)
        results = orbff.model(graph)
        node_features = results["node_features"]
        node_features = torch.split(node_features, tuple(graph.n_node.tolist()))
        for i, result in enumerate(node_features):
            if args.save_atom_embeddings:
                torch_embeddings.append(result.cpu().detach())
            else:
                torch_embeddings.append(result.mean(dim=0).cpu().detach())
            embeddings[smiles[i]] = torch_embeddings[-1].numpy()
        batch_idx += 1
    
    emb_dir = os.path.join(DATA_ROOT, "cached_embs", args.dataset_name, "embeddings")
    os.makedirs(emb_dir, exist_ok=True)
    if args.save_atom_embeddings:
        torch.save(embeddings, os.path.join(emb_dir, f"{args.model_name}{args.postfix}.pt"))
    else:
        torch_embeddings = torch.stack(torch_embeddings, dim=0)
        torch.save(torch_embeddings, os.path.join(emb_dir, f"{args.model_name}{args.postfix}.pt"))

    with open(os.path.join(emb_dir, f"{args.model_name}{args.postfix}_keys.txt"), "w") as f:
        f.write("\n".join(list(embeddings.keys())))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Cache embeddings for dataset.")
    parser.add_argument("--model_name", type=str, default='orb_v3_direct_20_omat', help="Exact OrbV3 name of the model to use for embedding, with dashes")
    parser.add_argument("--data_path", type=str, required=True, help="Absolute path to QM9 ASE DB")
    parser.add_argument('--dataset_name', type=str, default='dataset name')
    parser.add_argument('--batch_size', type=int, default=10)
    parser.add_argument('--save_atom_embeddings', action='store_true', default=False)
    parser.add_argument('--postfix', type=str, default='', help="Postfix to add to the embedding path")
    parser.add_argument('--id_key', type=str, default='smiles',
                        help="row.data field used as per-sample id ('smiles' for LLM4Mat, 'material_id' for MatterChat)")
    args = parser.parse_args()
    main(args)