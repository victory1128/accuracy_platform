"""评分模块

不同任务类型用不同评分器:
- exact_match: MCQ 选项精确匹配 (归一化后)
- numeric_match: 数学题, 从输出抽取最后的数字/\\boxed{}
- pass_at_k: 代码执行通过率
- judge_score: LLM 裁判打分
- rule_score: 规则可验证 (IFEval 指令约束)
"""
from .extract import extract_choice, extract_number, normalize_answer, numbers_equal
from .code_exec import run_code_tests, compute_pass_at_k, extract_code, pass_at_1
from .judging import judge_single, judge_pair
from .rule_check import check_ifeval

__all__ = [
    "extract_choice",
    "extract_number",
    "normalize_answer",
    "numbers_equal",
    "run_code_tests",
    "compute_pass_at_k",
    "extract_code",
    "pass_at_1",
    "judge_single",
    "judge_pair",
    "check_ifeval",
]
