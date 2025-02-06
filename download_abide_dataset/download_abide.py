"""
Taken and modified from https://github.com/huggingface/datasets/blob/3.2.0/templates/new_dataset_script.py
"""

import datasets
from nilearn.datasets import fetch_abide_pcp
import numpy as np
import pandas as pd


class ABIDEDataset(datasets.GeneratorBasedBuilder):
    def _info(self):
        features = datasets.Features(
            {
                'time_series_path': datasets.Value('string'),
                'label': datasets.ClassLabel(names=['hc', 'asd'])
            }
        )

        return datasets.DatasetInfo(
            features=features,
        )

    def _split_generators(self, dl_manager):
        data_dir = self.config.data_dir
        data = fetch_abide_pcp(
            data_dir=data_dir,
            pipeline='cpac',
            band_pass_filtering=True,
            global_signal_regression=False,
            derivatives=['func_preproc'],
            verbose=0,
            n_subjects=100
        )
        image_path_lst = data['func_preproc']
        time_series_path_lst = [image_path.replace(
            'func_preproc.nii.gz', 'rois_cc200.1D'
        ) for image_path in image_path_lst]

        phenotypics_df: pd.DataFrame = data['phenotypic']
        label_lst = phenotypics_df['DX_GROUP'].map(
            {2: 'hc', 1: 'asd'}
        ).to_list()

        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={
                    'time_series_path_lst': time_series_path_lst,
                    'label_lst': label_lst
                }
            )
        ]

    def _generate_examples(
        self,
        time_series_path_lst: list[str],
        label_lst: list[str]
    ):
        for i in range(len(time_series_path_lst)):
            time_series_path = time_series_path_lst[i]
            label = label_lst[i]
            yield i, {
                'time_series_path': time_series_path,
                'label': label
            }
