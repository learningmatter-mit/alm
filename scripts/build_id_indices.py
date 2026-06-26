from ase.db import connect
from tqdm import tqdm
import json
from pathlib import Path
import argparse

def build_id_indices(db_path):
    db = connect(db_path)
    index = {}
    for row in tqdm(db.select(), total=len(db), desc=f"Building index for dataset {db_path}"):
        index[str(row.data['smiles'])] = row.id
    with open(db_path.with_suffix('.id_index.json'), 'w') as f:
        json.dump(index, f)
    return index

def main(args):
    for path in Path(args.parent).iterdir():
        if path.is_dir():
            build_id_indices(path / 'train.db')
            build_id_indices(path / 'validation.db')

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--parent", type=str, required=True)
    args = parser.parse_args()
    main(args)