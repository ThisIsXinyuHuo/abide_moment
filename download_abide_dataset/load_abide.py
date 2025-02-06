import datasets
from datasets import load_dataset
import torch
import numpy as np
import random
from torch.utils.data import DataLoader
import os

def transform_images(batch):
    time_series_lst = [np.loadtxt(
        time_series_path, dtype=np.float32
    ) for time_series_path in batch['time_series_path']] # bs x sequence_length x num_input_channels

    bs = len(time_series_lst)
    #print(bs)
    sequence_length = 512
    num_input_channels = time_series_lst[0].shape[-1]

    mask = np.zeros((bs, sequence_length, num_input_channels), dtype=np.bool_)

    for i in range(len(time_series_lst)):
        time_series = time_series_lst[i]
        # truncate
        if time_series.shape[0] > sequence_length:
            time_series = time_series[:sequence_length]
        # mask
        mask[i, :time_series.shape[0]] = 1
        # pad
        time_series_lst[i] = np.pad(
            time_series, ((0, sequence_length - time_series.shape[0]), (0, 0))
        )
    time_series_lst = np.stack(time_series_lst, axis=0).transpose(0,2,1)
    #print(time_series_lst.shape)
    mask = mask.transpose(0,2,1)[:, 0, :]

    batch['time_series'] = torch.from_numpy(time_series_lst)
    batch['mask'] = torch.from_numpy(mask)
    return batch

# if set num_workers > 0 in dataloader
def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

def set_seed(seed: int = 42):
    """Function to control randomness in the code."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def get_dataset(dataset_path, data_dir, batch_size=32, trust_remote_code = True, validation = False, seed = 42):
    set_seed(seed)
    ds = load_dataset(
        path=dataset_path,  # data path
        data_dir=data_dir,
        split='train',
        trust_remote_code=trust_remote_code
    ).train_test_split(
        test_size=.2,
        stratify_by_column='label',
        seed=seed,
    )

    # train,val,test split
    if validation:
        ds_train_val = ds['train']
        ds_test = ds['test']
        ds_train_val = ds_train_val.train_test_split(
            test_size=.2,
            stratify_by_column='label',
            seed=seed,
        )
        ds_train = ds_train_val['train']
        ds_val = ds_train_val['test']
        dataset = datasets.DatasetDict({
            'train': ds_train,
            'validation': ds_val,
            'test': ds_test
        })
    else:
        ds_train = ds['train']
        ds_test = ds['test']
        dataset = datasets.DatasetDict({
            'train': ds_train,
            'test': ds_test
        })
    dataset.set_transform(transform_images)
    return dataset

def get_dataloader(dataset_path, data_dir, batch_size=32, trust_remote_code = True, validation = False, seed = 42):
    set_seed(seed)
    # load huggingface dataset
    ds = load_dataset(
        path=dataset_path,  # data path
        data_dir=data_dir,
        split='train',
        trust_remote_code=trust_remote_code
    ).train_test_split(
        test_size=.2,
        stratify_by_column='label',
        seed=seed,
    )

    # train,val,test split
    if validation:
        ds_train_val = ds['train']
        ds_test = ds['test']
        ds_train_val = ds_train_val.train_test_split(
            test_size=.2,
            stratify_by_column='label',
            seed=seed,
        )
        ds_train = ds_train_val['train']
        ds_val = ds_train_val['test']
        dataset = datasets.DatasetDict({
            'train': ds_train,
            'validation': ds_val,
            'test': ds_test
        })
    else:
        ds_train = ds['train']
        ds_test = ds['test']
        dataset = datasets.DatasetDict({
            'train': ds_train,
            'test': ds_test
        })
    dataset.set_transform(transform_images)

    # create data loader
    generator = torch.Generator()
    generator.manual_seed(seed)
    train_dataloader = DataLoader(dataset['train'], batch_size=batch_size, shuffle=True, generator=generator)
    val_dataloader = DataLoader(dataset['validation'], batch_size=batch_size, shuffle=True, generator=generator) if validation else None
    test_dataloader = DataLoader(dataset['test'], batch_size=batch_size, shuffle=True, generator=generator)
    return train_dataloader, val_dataloader, test_dataloader



    





