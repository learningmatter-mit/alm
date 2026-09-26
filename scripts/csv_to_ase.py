"""Convert LLM4Mat-Bench train/validation CSVs (CIF column) into ASE databases next to each CSV."""
import argparse
from io import StringIO
from pathlib import Path

import polars as pl
from ase.db import connect
from ase.io import read
from tqdm import tqdm


def convert_split(dataset_path: Path, split: str) -> None:
    df = pl.read_csv(dataset_path / f'{split}.csv')
    df = df.drop("Unnamed: 0", strict=False)
    id_name = [column for column in df.columns if column.endswith('_id')][0]
    db = connect(dataset_path / f'{split}.db')
    for row in tqdm(range(len(df)), total=len(df), desc=f'Processing {split} data for {dataset_path}'):
        ase_atoms = read(StringIO(df[row]['cif_structure'][0]), format='cif')
        data = {k: df[row][k][0] for k in df[row].columns}
        data['smiles'] = str(df[row][id_name][0])
        db.write(
            ase_atoms,
            data=data,
        )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--data_dir', type=Path, required=True,
                    help="LLM4Mat-Bench root with one subdirectory per dataset.")
    ap.add_argument('--datasets', nargs='*', default=None,
                    help="Subset of dataset subdirectory names to convert (default: all).")
    args = ap.parse_args()

    for dataset_path in sorted(args.data_dir.iterdir()):
        if not dataset_path.is_dir():
            continue
        if args.datasets and dataset_path.name not in args.datasets:
            continue
        for split in ('train', 'validation'):
            convert_split(dataset_path, split)


if __name__ == '__main__':
    main()
