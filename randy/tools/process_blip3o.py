import re
import shutil
import hashlib
import filetype
import jsonlines
import pandas as pd
import webdataset as wds
from tqdm import tqdm
from pathlib import Path
from loguru import logger
from datasets import load_dataset
from multiprocessing import Pool, cpu_count


DATA_ROOTS = [
    '/nas_user/app.e0031982/datasets/Amshaker/Mobile-O-Pre-Train',
    '/nas_user/app.e0031982/datasets/BLIP3o/BLIP3o-Pretrain-Long-Caption',
]

OUTPUT_ROOT = Path('/nas_user/app.e0016372/datasets/BLIP3o')
ERROR_LOG = OUTPUT_ROOT / 'error.log'


def compute_hash(x):
    return hashlib.md5(str(x).encode()).hexdigest()


def save_image(image_bytes, image_dir, image_name):
    suffix = filetype.guess_extension(image_bytes) or 'png'
    image_path = image_dir / f'{image_name}.{suffix}'
    image_path.write_bytes(image_bytes)
    return str(image_path)


def process_sample(row, parquet_path, idx, image_dir):
    if not row['txt'] or row['jpg'] is None:
        raise ValueError('empty caption or image')

    sample_id = f'{parquet_path}_{idx}'
    image_name = compute_hash(sample_id)
    image_path = save_image(row['jpg'], image_dir, image_name)

    return {
        'id': sample_id,
        'messages': [
            {'role': 'user', 'content': '<image>'},
            {'role': 'assistant', 'content': row['txt']}
        ],
        'images': [image_path]
    }


def process_parquet(args):
    parquet_path, dataset_name = args

    image_dir = OUTPUT_ROOT / dataset_name / 'images'
    jsonl_dir = OUTPUT_ROOT / dataset_name / 'jsonl'

    image_dir.mkdir(parents=True, exist_ok=True)
    jsonl_dir.mkdir(parents=True, exist_ok=True)

    jsonl_path = jsonl_dir / f'{compute_hash(parquet_path)}.jsonl'

    try:
        ds = wds.WebDataset(str(parquet_path), shardshuffle=False).decode()
        # ds = load_dataset('webdataset', data_files=parquet_path, split='train')

        with jsonlines.open(jsonl_path, 'w') as writer:
            for idx, row in enumerate(ds):
                try:
                    record = process_sample(row, parquet_path, idx, image_dir)
                    writer.write(record)
                except Exception as e:
                    logger.error(f'{parquet_path} | row={idx} | {repr(e)}')
    except Exception as e:
        logger.error(f'{parquet_path} | parquet_error | {repr(e)}')


def main():
    shutil.rmtree(OUTPUT_ROOT, ignore_errors=True)
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

    logger.remove()
    logger.add(ERROR_LOG, level='ERROR', enqueue=True)

    tasks = []
    for root in DATA_ROOTS:
        root = Path(root)
        dataset_name = root.stem
        for p in root.glob('**/*.tar'):
            tasks.append((p, dataset_name))

    print(len(tasks))
    if not tasks: return

    n_workers = min(cpu_count() // 2, len(tasks))
    with Pool(n_workers, maxtasksperchild=10) as pool:
        list(tqdm(
            pool.imap_unordered(process_parquet, tasks),
            total=len(tasks)
        ))


if __name__ == '__main__':
    main()
