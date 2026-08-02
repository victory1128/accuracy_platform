"""BFCL: 函数调用评测 (Berkeley Function Calling Leaderboard)

给用户问题 + 可用函数列表 (含 name/description/parameters schema), 模型需生成
正确的函数调用 (函数名 + 参数)。ground_truth: [{func_name: {param: [可接受值列表]}}]。

评分: 解析模型输出的函数调用 JSON, 比对函数名一致 + 每个参数值在可接受列表内。
模型输出格式要求: `[{"func_name": {"param1": val1, "param2": val2}}]` (JSON 数组)。
"""
from __future__ import annotations

import json
import os
import re
from typing import Any, List, Optional

from ..models import BenchmarkMeta, Sample, SampleResult, Stage, TaskType
from .base import Benchmark
from .registry import register
from ._data import data_path

BFCL_HEADER = (
    "You have access to the following functions. Call the appropriate function(s) to answer "
    "the user's question. Output ONLY the function call as a JSON array, e.g. "
    '[{"function_name": {"param1": value1, "param2": value2}}]. Do not include any other text.'
)


@register
class BFCL(Benchmark):
    META = BenchmarkMeta(
        name="bfcl",
        display_name="BFCL (Function Calling)",
        stage=Stage.POSTTRAIN,
        task_type=TaskType.GEN,
        description="函数调用能力评测, 模型按 schema 生成正确函数调用",
        tags=["modern", "tool_use", "function_calling"],
        num_fewshot=0,
        source="Patil et al. 2024 (Gorilla/Berkeley)",
    )

    def load_samples(self, limit: Optional[int] = None, seed: int = 42) -> List[Sample]:
        path = data_path("bfcl.jsonl")
        if not os.path.exists(path):
            return []
        rows = self._load_jsonl(path)
        samples = []
        for r in rows:
            sid = r.get("sample_id") or ""
            samples.append(
                Sample(
                    sample_id=sid,
                    prompt="",
                    question=r.get("question", ""),
                    gold=r.get("gold", []),
                    meta={"functions": r.get("functions", [])},
                )
            )
        return self._maybe_limit(samples, limit, seed)

    def build_prompt(self, sample: Sample) -> str:
        funcs = sample.meta.get("functions", [])
        funcs_str = json.dumps(funcs, ensure_ascii=False, indent=2)
        return (
            f"{BFCL_HEADER}\n\n"
            f"Functions:\n{funcs_str}\n\n"
            f"User: {sample.question}\n\n"
            "Function call:"
        )

    def parse_params(self) -> dict:
        # function calling: 思维链模型先思考再输出 JSON, 512 不够, 给 8192。
        return {"temperature": 0.0, "max_tokens": 8192, "stop": None}

    def evaluate(
        self,
        sample: Sample,
        response: str,
        judge_client=None,
    ) -> SampleResult:
        pred_calls = _parse_function_calls(response)
        gold_calls = sample.gold if isinstance(sample.gold, list) else []
        correct = _bfcl_match(pred_calls, gold_calls)
        extracted = json.dumps(pred_calls, ensure_ascii=False)[:200] if pred_calls else ""
        return SampleResult(
            sample_id=sample.sample_id,
            response=response,
            extracted=extracted,
            correct=correct,
            score=1.0 if correct else 0.0,
        )


def _parse_function_calls(text: str) -> List[dict]:
    """从模型输出解析函数调用。

    支持格式:
    - [{func: {params}}]  (BFCL 标准)
    - {func: {params}}    (单个对象)
    - func(arg1=val1)     (文本式, 兜底)
    返回 [{func_name: {param: value}}] 列表。
    """
    if not text:
        return []
    # 找第一个 JSON 数组或对象
    for pat in (r"\[.*\]", r"\{.*\}"):
        m = re.search(pat, text, re.DOTALL)
        if m:
            try:
                obj = json.loads(m.group(0))
                if isinstance(obj, list):
                    return [o for o in obj if isinstance(o, dict) and o]
                if isinstance(obj, dict) and obj:
                    return [obj]
            except json.JSONDecodeError:
                continue
    return []


def _bfcl_match(pred_calls: List[dict], gold_calls: List[dict]) -> bool:
    """比对函数调用: 函数名一致 + 每个参数值在可接受列表内。

    gold 格式: [{"func_name": {"param": [可接受值列表], ...}}]
    pred 格式: [{"func_name": {"param": value, ...}}]
    任一 gold call 被某个 pred call 匹配即算对。
    """
    if not gold_calls:
        return False
    for gold in gold_calls:
        if not isinstance(gold, dict):
            continue
        for gname, gparams in gold.items():
            for pred in pred_calls:
                if not isinstance(pred, dict):
                    continue
                for pname, pparams in pred.items():
                    if pname != gname:
                        continue
                    if not isinstance(gparams, dict):
                        continue
                    if _params_match(pparams or {}, gparams):
                        return True
    return False


def _params_match(pred_params: dict, gold_params: dict) -> bool:
    """每个 gold 参数的可接受列表里, 有 pred 提供的值。
    gold 缺的参数 pred 也不给 (或给的可接受)。"""
    for gkey, gvals in gold_params.items():
        if not isinstance(gvals, list):
            gvals = [gvals]
        pval = pred_params.get(gkey)
        if not _value_in_list(pval, gvals):
            return False
    return True


def _value_in_list(val: Any, accept: List[Any]) -> bool:
    """val 是否在可接受值列表里 (容忍类型: 10 vs "10"; val 可能是 list/dict)。"""
    if val is None:
        return False
    # val 可能是 list/dict (不可哈希), 用字符串比对兜底
    val_str = str(val)
    for a in accept:
        if a == val or str(a) == val_str:
            return True
        try:
            if float(a) == float(val):
                return True
        except (TypeError, ValueError):
            pass
    # val 是 list 时, 任一元素匹配即可 (多值参数)
    if isinstance(val, list):
        for v in val:
            if _value_in_list(v, accept):
                return True
    return False
