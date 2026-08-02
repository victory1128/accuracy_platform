"""SWE-bench Verified: 代码 Agent 修复真实 GitHub issue

500 题, 每题给一个真实 Python 项目的 GitHub issue (problem_statement), 模型需生成
补丁 (unified diff) 解决问题。评测: 在该 repo 的 base_commit 检出代码, 应用模型补丁,
跑 FAIL_TO_PASS 测试 (修复后应通过) + PASS_TO_PASS 测试 (不应回归)。

执行依赖官方 swebench 包 + Docker (每题一个预构建镜像, 首次自动拉取/构建, 耗时较长)。
资源需求: ~120GB 存储 / 16GB RAM。仅 docker 沙箱可用且资源充足时提交。

注: 这是逐题评测模式 —— 每题单独生成 prediction 文件, 调
swebench.harness.run_evaluation --instance_ids 跑单题, 解析结果日志。
"""
from __future__ import annotations

import json
import os
import re
import subprocess
import tempfile
from typing import List, Optional

from ..models import BenchmarkMeta, Sample, SampleResult, Stage, TaskType
from .base import Benchmark
from .registry import register
from ._data import data_path

SWEBENCH_HEADER = (
    "You are a software engineer resolving a GitHub issue. Based on the issue description "
    "below, generate a patch (unified diff format) that fixes the problem. "
    "Output the patch in a ```diff code block. Only modify what's necessary to fix the issue.\n\n"
    "The patch should be a valid unified diff that can be applied with `git apply`."
)


@register
class SWEBench(Benchmark):
    META = BenchmarkMeta(
        name="swebench",
        display_name="SWE-bench Verified",
        stage=Stage.PRETRAIN,
        task_type=TaskType.CODE,
        description="代码Agent, 修复真实GitHub issue, 500题, 需Docker+swebench包",
        tags=["modern", "code", "agent", "hard"],
        num_fewshot=0,
        source="Jimenez et al. 2024 (Princeton)",
    )

    def load_samples(self, limit: Optional[int] = None, seed: int = 42) -> List[Sample]:
        path = data_path("swebench.jsonl")
        if not os.path.exists(path):
            return []
        rows = self._load_jsonl(path)
        samples = []
        for r in rows:
            sid = r.get("sample_id") or r.get("instance_id", "")
            samples.append(
                Sample(
                    sample_id=sid,
                    prompt="",
                    question=r.get("problem_statement", ""),
                    gold=r.get("patch", ""),
                    test_code=r.get("test_patch", ""),
                    meta={
                        "repo": r.get("repo", ""),
                        "base_commit": r.get("base_commit", ""),
                        "fail_to_pass": r.get("fail_to_pass", ""),
                        "pass_to_pass": r.get("pass_to_pass", ""),
                        "version": r.get("version", ""),
                        "instance_id": r.get("sample_id", ""),
                    },
                )
            )
        return self._maybe_limit(samples, limit, seed)

    def build_prompt(self, sample: Sample) -> str:
        repo = sample.meta.get("repo", "")
        instance_id = sample.meta.get("instance_id", "") or sample.sample_id
        hints = sample.meta.get("hints_text", "")
        hints_part = f"\n\nAdditional context (issue comments):\n{hints}" if hints else ""
        return (
            f"{SWEBENCH_HEADER}\n\n"
            f"Repository: {repo}\n"
            f"Instance: {instance_id}\n\n"
            f"Issue:\n{sample.question}{hints_part}\n\n"
            "```diff\n"
        )

    def parse_params(self) -> dict:
        # 生成补丁需较长推理; 不设 stop。
        return {"temperature": 0.0, "max_tokens": 8192, "stop": None}

    def evaluate(
        self,
        sample: Sample,
        response: str,
        judge_client=None,
    ) -> SampleResult:
        patch = _extract_diff(response)
        instance_id = sample.meta.get("instance_id", "") or sample.sample_id

        if not patch:
            return SampleResult(
                sample_id=sample.sample_id,
                response=response,
                extracted="",
                correct=False,
                score=0.0,
                error="模型未生成有效 diff 补丁",
            )

        passed, err = _run_swebench_single(instance_id, patch)
        return SampleResult(
            sample_id=sample.sample_id,
            response=response,
            extracted=patch[:200],
            correct=passed,
            score=1.0 if passed else 0.0,
            error=err or None,
        )


def _extract_diff(response: str) -> str:
    """从模型输出抽取 unified diff 补丁。

    优先取 ```diff ... ``` 代码块; 否则找 'diff --git' 开头的部分。
    """
    if not response:
        return ""
    m = re.search(r"```(?:diff|patch)?\s*\n(.*?)```", response, re.DOTALL)
    if m:
        return m.group(1).strip()
    # 兜底: 找 diff --git 开头到结尾
    idx = response.find("diff --git")
    if idx >= 0:
        return response[idx:].strip()
    return ""


def _run_swebench_single(instance_id: str, patch: str) -> tuple:
    """对单题运行 swebench 评测, 返回 (是否通过, 错误信息)。

    生成 prediction 文件 (单行 JSONL), 调 run_evaluation --instance_ids。
    需要 docker + 预构建镜像 (首次自动拉取, 耗时)。
    """
    try:
        from swebench.harness.run_evaluation import main as run_eval
    except Exception as e:
        return False, f"swebench 包不可用: {e}"

    with tempfile.TemporaryDirectory() as tmpdir:
        pred_path = os.path.join(tmpdir, "pred.jsonl")
        with open(pred_path, "w", encoding="utf-8") as f:
            f.write(json.dumps({
                "instance_id": instance_id,
                "model_name_or_path": "eval-platform",
                "model_patch": patch,
            }) + "\n")

        run_id = f"eval_{instance_id}"
        try:
            run_eval(
                dataset_name="princeton-nlp/SWE-bench_Verified",
                split="test",
                instance_ids=[instance_id],
                predictions_path=pred_path,
                max_workers=1,
                run_id=run_id,
                timeout=1800,
            )
        except SystemExit:
            pass  # run_eval 可能 sys.exit
        except Exception as e:
            return False, f"run_evaluation 异常: {e}"

        # 解析结果: logs/run_evaluation/<run_id>/<instance_id>/result.json
        return _parse_swebench_result(run_id, instance_id)


def _parse_swebench_result(run_id: str, instance_id: str) -> tuple:
    """从 swebench 结果日志解析该 instance 是否通过。"""
    # 结果可能在 logs/ 或 evaluation_results/
    candidates = [
        os.path.join("logs", "run_evaluation", run_id, instance_id, "result.json"),
        os.path.join("logs", "run_evaluation", run_id, "report", "results.json"),
        os.path.join("evaluation_results", run_id, "results.json"),
    ]
    for path in candidates:
        if os.path.exists(path):
            try:
                with open(path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                # results.json 格式: {instance_id: {"resolved": bool, ...}}
                entry = data.get(instance_id, data) if isinstance(data, dict) else data
                if isinstance(entry, dict):
                    resolved = entry.get("resolved", False)
                    return bool(resolved), "" if resolved else "未通过测试"
            except (json.JSONDecodeError, OSError):
                continue
    return False, "未找到评测结果 (可能 docker 镜像未就绪或超时)"
