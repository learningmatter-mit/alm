from pathlib import Path
import polars as pl
from ase.db import connect
from tqdm import tqdm
from ase.io import read
from io import StringIO

data_dir = Path('/tmp/LLM4Mat-Bench')
for dataset_path in data_dir.iterdir():
    if dataset_path.is_dir():
        if 'snumat' not in str(dataset_path):
            continue
        df = pl.read_csv(dataset_path / 'train.csv')
        df = df.drop("Unnamed: 0", strict=False)
        id_name = [column for column in df.columns if column.endswith('_id')][0]
        db = connect(dataset_path / 'train.db')
        for row in tqdm(range(len(df)), total=len(df), desc=f'Processing train data for {dataset_path}'):
            ase_atoms = read(StringIO(df[row]['cif_structure'][0]), format='cif')
            data = {k: df[row][k][0] for k in df[row].columns}
            data['smiles'] = str(df[row][id_name][0])
            db.write(
                ase_atoms,
                data=data,
            )
        df = pl.read_csv(dataset_path / 'validation.csv')
        df = df.drop("Unnamed: 0", strict=False)
        db = connect(dataset_path / 'validation.db')
        for row in tqdm(range(len(df)), total=len(df), desc=f'Processing validation data for {dataset_path}'):
            ase_atoms = read(StringIO(df[row]['cif_structure'][0]), format='cif')
            data = {k: df[row][k][0] for k in df[row].columns}
            data['smiles'] = str(df[row][id_name][0])
            db.write(
                ase_atoms,
                data=data,
            )
