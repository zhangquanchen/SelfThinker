# Copyright 2025 Bytedance Ltd. and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0
"""Audit-pass VT as generation target: problem image + question -> VT (MSE)."""

from __future__ import annotations

import io

import pyarrow.parquet as pq
from PIL import Image, ImageFile, PngImagePlugin

from .interleave_t2i_dataset import InterleavedBaseIterableDataset, ParquetStandardIterableDataset
from ..data_utils import pil_img2rgb
from ..parquet_utils import get_parquet_data_paths

Image.MAX_IMAGE_PIXELS = 200_000_000
ImageFile.LOAD_TRUNCATED_IMAGES = True
PngImagePlugin.MAX_TEXT_CHUNK = 1024 * (2**20)


def _open_image_cell(cell):
    if cell is None:
        return None
    if isinstance(cell, dict):
        b = cell.get("bytes") or cell.get(b"bytes")
        if b:
            return Image.open(io.BytesIO(b)).convert("RGB")
        path = cell.get("path") or cell.get(b"path")
        if path:
            return Image.open(str(path)).convert("RGB")
        return None
    if isinstance(cell, (bytes, bytearray, memoryview)):
        return Image.open(io.BytesIO(bytes(cell))).convert("RGB")
    return None


class AuditPassGenIterableDataset(InterleavedBaseIterableDataset, ParquetStandardIterableDataset):
    """Condition on (problem image, question); MSE-supervise the audit-pass VT."""

    def __init__(
        self,
        dataset_name,
        transform,
        tokenizer,
        vit_transform,
        data_dir_list,
        num_used_data,
        parquet_info=None,
        local_rank=0,
        world_size=1,
        num_workers=8,
        data_status=None,
    ):
        super().__init__(
            dataset_name,
            transform,
            tokenizer,
            vit_transform,
            data_dir_list,
            num_used_data,
            parquet_info or {},
            local_rank=local_rank,
            world_size=world_size,
            num_workers=num_workers,
            data_status=data_status,
        )

    def get_data_paths(self, data_dir_list, num_used_data, parquet_info=None):
        row_groups = []
        for data_dir, num_data_path in zip(data_dir_list, num_used_data):
            data_paths = get_parquet_data_paths([data_dir], [num_data_path])
            for data_path in data_paths:
                try:
                    n_rg = pq.ParquetFile(data_path).num_row_groups
                except Exception as e:
                    print(f"skip parquet {data_path}: {e}")
                    continue
                for rg_idx in range(n_rg):
                    row_groups.append((data_path, rg_idx))
        return row_groups

    def parse_row(self, row):
        try:
            problem = _open_image_cell(row.get("problem_image_0"))
            vt = _open_image_cell(row.get("reasoning_image_0"))
        except Exception:
            return {}
        if problem is None or vt is None:
            return {}
        q = row.get("question") or ""
        ans = row.get("answer") or ""
        data = self._init_data()
        data = self._add_image(
            data, pil_img2rgb(problem), need_loss=False, need_vae=True, need_vit=True
        )
        data = self._add_text(data, str(q), need_loss=False)
        data = self._add_image(
            data, pil_img2rgb(vt), need_loss=True, need_vae=True, need_vit=True
        )
        if ans:
            data = self._add_text(data, str(ans), need_loss=True)
        return data
