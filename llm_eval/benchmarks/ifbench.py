"""IFBench: 可验证指令遵循泛化评测 (allenai/IFBench_test, 300题)

Allen AI (Ai2) 发布, 与 IFEval (Google) 平行独立。核心: 当前模型在 IFEval 等已有 IF
benchmark 的小约束集上严重过拟合, IFBench 用 58 个新的 OOD 可验证约束测泛化能力。

与 IFEval 的关系: 平行/补充, 非 IFEval v2。IFEval 测通用约束 (25 类), IFBench 测未见过
的新约束 (58 类, 与 IFEval 零重叠)。

评分: IFBench 的 58 类约束与平台 IFEval 规则引擎 (check_ifeval, 仅支持 IFEval 25 类)
完全不兼容, 故不走路基类 _eval_rule。改为 vendor 了 allenai/IFBench 官方 verifier
(llm_eval/scoring/ifbench_verifier/, 含 instructions.py 59 个 checker 类 +
instructions_registry.py 映射 + instructions_util.py 工具函数), 本类重写 _eval_rule
直接调官方 checker.check_following(response) —— 与官方 evaluation_lib 的 strict 模式一致
(逐约束实例化 → build_description(**kwargs) → 需 prompt 时注入 → check_following)。

依赖: nltk / emoji / syllapy (官方 instructions.py 顶层导入, 已加 requirements)。
数据字段: question (含约束的指令), ifbench_constraints ([{instruction_id, kwargs}, ...])。
"""
from __future__ import annotations

import os
from typing import Any, Dict, List, Optional

from ..models import BenchmarkMeta, Sample, SampleResult, Stage, TaskType
from .base import Benchmark
from .registry import register
from ._data import data_path
# 官方 IFBench verifier (vendored): instruction_id -> checker 类
from ..scoring.ifbench_verifier import INSTRUCTION_DICT


@register
class IFBench(Benchmark):
    META = BenchmarkMeta(
        name="ifbench",
        display_name="IFBench",
        stage=Stage.POSTTRAIN,
        task_type=TaskType.RULE,
        description="可验证指令遵循泛化评测, 300题, 58个OOD新约束, 官方verifier评分 (与IFEval平行独立)",
        tags=["modern", "instruction_following", "generalization"],
        num_fewshot=0,
        needs_judge=False,
        source="Pyatkin et al. 2025 (Allen AI)",
    )

    def load_samples(self, limit: Optional[int] = None, seed: int = 42) -> List[Sample]:
        path = data_path("ifbench.jsonl")
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
                    meta={"ifbench_constraints": r.get("ifbench_constraints", [])},
                )
            )
        return self._maybe_limit(samples, limit, seed)

    def build_prompt(self, sample: Sample) -> str:
        return sample.question

    def parse_params(self) -> dict:
        # 思维链模型需足预算让推理+答案完整输出, 否则规则检查必失败。
        return {"temperature": 0.7, "max_tokens": 8192, "stop": None}

    def _eval_rule(self, sample: Sample, response: str) -> SampleResult:
        """调官方 IFBench verifier 逐约束检查 (strict 模式)。

        与官方 evaluation_lib.test_instruction_following_strict 一致:
        逐 constraint 实例化 checker → build_description(**kwargs) → 若 checker 需 prompt
        则注入 (get_instruction_args 含 'prompt') → check_following(response)。
        空/纯空白响应判不通过 (同官方)。
        """
        constraints = sample.meta.get("ifbench_constraints", [])
        details = []
        satisfied = 0
        resp_ok = bool(response and response.strip())
        for c in constraints:
            inst_id = c.get("instruction_id", "")
            kwargs = c.get("kwargs", {}) or {}
            ok = False
            note = ""
            if resp_ok:
                cls = INSTRUCTION_DICT.get(inst_id)
                if cls is None:
                    note = f"未知约束类型: {inst_id}"
                else:
                    try:
                        inst = cls(inst_id)
                        inst.build_description(**kwargs)
                        # 部分 checker 需原始 prompt (如 repeat_prompt), 按官方注入
                        iargs = inst.get_instruction_args()
                        if iargs and "prompt" in iargs:
                            inst.build_description(prompt=sample.question)
                        ok = bool(inst.check_following(response))
                    except Exception as e:  # noqa: BLE001
                        note = f"checker 异常: {type(e).__name__}: {e}"
            else:
                note = "空响应"
            details.append({"instruction_id": inst_id, "satisfied": ok, "note": note})
            if ok:
                satisfied += 1
        total = len(constraints) or 1
        rate = round(satisfied / total, 4)
        return SampleResult(
            sample_id=sample.sample_id,
            response=response,
            score=rate,
            correct=rate == 1.0,
            analysis={"ifbench": {"satisfied": satisfied, "total": len(constraints), "rate": rate, "details": details}},
        )
