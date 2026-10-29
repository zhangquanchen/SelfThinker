# Copyright 2025 Bytedance Ltd. and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0

import os
from .interleave_datasets import UnifiedEditIterableDataset, AuditPassGenIterableDataset
_CODE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
_ROOT = os.environ.get(
    'SELFTHINKER_ROOT',
    _CODE_DIR if (os.path.isdir(os.path.join(_CODE_DIR, 'models')) or os.path.isdir(os.path.join(_CODE_DIR, 'data'))) else os.path.dirname(_CODE_DIR),
)
from .t2i_dataset import T2IIterableDataset
from .vlm_dataset import SftJSONLIterableDataset


DATASET_REGISTRY = {
    't2i_pretrain': T2IIterableDataset,
    'vlm_sft': SftJSONLIterableDataset,
    'unified_edit': UnifiedEditIterableDataset,
    'audit_pass_gen': AuditPassGenIterableDataset,
}


DATASET_INFO = {
    'unified_edit':{
        'seedxedit_multi': {
            'data_dir': 'your_data_path/bagel_example/editing/seedxedit_multi',
            'num_files': 10,
            'num_total_samples': 1000,
            "parquet_info_path": 'your_data_path/bagel_example/editing/parquet_info/seedxedit_multi_nas.json', # information of the parquet files
		},
    },
    'vlm_sft': {
        'llava_ov': {
			'data_dir': 'your_data_path/bagel_example/vlm/images',
			'jsonl_path': 'your_data_path/bagel_example/vlm/llava_ov_si.jsonl',
			'num_total_samples': 1000
		},
    },
    'audit_pass_gen': {
        'Spatial_Navigation': {
            'data_dir': os.path.join(_ROOT, 'data/audit_pass_nav/Spatial_Navigation/data'),
            'num_files': 2,
            'num_total_samples': 1576,
        },
        'Spatial_Navigation_r2': {
            'data_dir': os.path.join(_ROOT, 'data/audit_pass_rejsft_nav/Spatial_Navigation/data'),
            'num_files': 1,
            'num_total_samples': 408,
        },
        # Held-out diag excluded (train-split audit-pass only).
        'Spatial_Navigation_trainonly': {
            'data_dir': os.path.join(_ROOT, 'data/audit_pass_nav_trainonly/Spatial_Navigation/data'),
            'num_files': 1,
            'num_total_samples': 1179,
        },
        # L2 self-evolution: nav RejSFT generator, train-split, NEW dir (not r3).
        'Spatial_Navigation_l2_trainonly': {
            'data_dir': os.path.join(_ROOT, 'data/audit_pass_nav_l2_trainonly/Spatial_Navigation/data'),
            'num_files': 1,
            'num_total_samples': 1219,
        },
        # Matched no-audit baseline: same pids/count, random same-item generated VT.
        'Spatial_Navigation_random_vt_trainonly': {
            'data_dir': os.path.join(_ROOT, 'data/random_vt_nav_trainonly/Spatial_Navigation/data'),
            'num_files': 1,
            'num_total_samples': 1179,
        },
        # L1 consensus audit-pass on fresh Visual Search train split (no diag).
        'Visual_Search_l1_trainonly': {
            'data_dir': os.path.join(_ROOT, 'data/audit_pass_l1_vsearch_trainonly/Visual_Search/data'),
            'num_files': 1,
            'num_total_samples': 1074,
        },
        # L0 GT-direction audit-pass on the same VS train split (boxed preds were empty).
        'Visual_Search_l0_trainonly': {
            'data_dir': os.path.join(_ROOT, 'data/audit_pass_l0_vsearch_trainonly/Visual_Search/data'),
            'num_files': 1,
            'num_total_samples': 1092,
        },
    },
}