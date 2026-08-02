"""内置样例数据加载

平台内置每个评测集的少量样例数据 (data/ 下的 jsonl), 让用户无需联网下载
即可跑通流程、验证接入。要做严肃全量评测时, 把对应全集 jsonl 放到 data/
同名文件即可 (格式见各 benchmark 的 load_samples)。

真实评测建议:
- MMLU/MMLU-Pro/GPQA/C-Eval/CMMLU: 从 HuggingFace 下载, 转 costar jsonl
- GSM8K/MATH: HF
- HumanEval/MBPP: HF / GitHub
- IFEval/MT-Bench/AlpacaEval/Arena-Hard: 各自官方 repo
"""
from __future__ import annotations

import os
from typing import List

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")


def data_path(name: str) -> str:
    return os.path.join(DATA_DIR, name)
