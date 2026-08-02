"""核心数据模型 (dataclasses, 兼容 Python 3.9)"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import Any, Dict, List, Optional


class Stage(str, Enum):
    """评测集所属训练阶段"""
    PRETRAIN = "pretrain"     # 预训练阶段: 知识/推理/代码能力
    POSTTRAIN = "posttrain"   # 后训练阶段: 指令遵循/对齐/多轮对话


class TaskType(str, Enum):
    """评测集任务类型, 决定如何构造 prompt 与如何评分"""
    MCQ = "mcq"               # 多项选择 (MMLU, ARC, ...), 评分=exact match 选项
    GEN = "gen"               # 自由生成, 答案可被抽取匹配 (GSM8K, MATH)
    CODE = "code"             # 代码生成, 执行用例 pass@k (HumanEval, MBPP)
    JUDGE = "judge"           # LLM 裁判打分 (MT-Bench, AlpacaEval, Arena-Hard)
    RULE = "rule"             # 规则可验证 (IFEval: 指令遵循格式约束)


@dataclass
class ModelConfig:
    """被测模型配置 (OpenAI 兼容)"""
    name: str
    base_url: str
    api_key: str
    model: str
    # 可选的默认生成参数, 会被评测集按需覆盖
    temperature: float = 0.0
    max_tokens: int = 2048
    extra: Dict[str, Any] = field(default_factory=dict)

    def __repr__(self) -> str:
        # 永不暴露 api_key (避免进日志/报告/异常栈)
        masked = (self.api_key[:3] + "***") if self.api_key else "(空)"
        return f"ModelConfig(name={self.name!r}, model={self.model!r}, base_url={self.base_url!r}, api_key={masked!r})"
    __str__ = __repr__


@dataclass
class Sample:
    """单条评测样本 (基准无关的统一表示)"""
    sample_id: str
    # 输入
    prompt: str                       # 最终拼好的 prompt (含 few-shot)
    question: str = ""                # 原始题面 (报告展示用)
    choices: Optional[List[str]] = None     # MCQ 选项
    gold: Optional[str] = None             # 标准答案 (选项字母 / 数字 / 代码)
    # 代码评测专用
    test_code: Optional[str] = None        # 完整测试代码 (HumanEval/MBPP)
    # 通用元信息
    meta: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SampleResult:
    """单条样本的评测结果 (一条 = 一次 API 请求)"""
    sample_id: str
    response: str                     # 模型原始输出 (content)
    raw_response: Optional[str] = None  # 完整未截断输出 (含 reasoning 等)
    reasoning_content: Optional[str] = None  # 思维链全文 (思维链模型才有)
    extracted: Optional[str] = None      # 从输出中抽取的答案
    reference: Optional[str] = None      # 参考答案 (JUDGE 类: 参考模型输出; MCQ/GEN: 同 gold 的可读版)
    correct: Optional[bool] = None       # 是否正确 (None=未判定, 如 judge 类给分)
    score: Optional[float] = None        # 0~1 或 0~10 的分数
    error: Optional[str] = None          # 出错信息
    latency_ms: Optional[float] = None
    usage: Optional[Dict[str, int]] = None  # token 用量 + finish_reason + 速度
    analysis: Optional[Dict[str, Any]] = None  # 乱码分析结果
    # —— 可追溯/可搜索 字段 ——
    request_hash: Optional[str] = None   # 该请求的唯一 hash (搜索用)
    benchmark: Optional[str] = None      # 所属评测集名 (跨集搜索用)
    prompt: Optional[str] = None         # 发给模型的完整 prompt (输入)
    system_prompt: Optional[str] = None  # system 消息 (若有)
    question: Optional[str] = None       # 原始题面 (展示用)
    gold: Optional[str] = None           # 标准答案
    gen_params: Optional[Dict[str, Any]] = None  # 生成参数 (temperature/max_tokens/stop)
    # —— 速度/长度指标 (从 usage 派生, 单独存便于聚合) ——
    tokens_per_sec: Optional[float] = None    # 输出速度 = completion_tokens / 端到端秒数
    ttft_ms: Optional[float] = None           # 首字延迟: 流式=真实值, 非流式=端到端近似
    streaming: Optional[bool] = None          # 是否用了流式调用 (决定 ttft/tpot 是否为真实值)
    real_stream: Optional[bool] = None        # 是否真流式 (伪流式时 TPOT 无意义, 不显示)
    tpot_ms: Optional[float] = None           # 每个输出token生成时间 (仅真流式有真实值)
    gen_time_ms: Optional[float] = None       # 生成阶段耗时 (仅真流式: 末chunk-首chunk)
    prompt_tokens: Optional[int] = None       # 输入 token 数
    completion_tokens: Optional[int] = None   # 输出 token 数 (含思维链)
    reasoning_tokens: Optional[int] = None    # 思维链 token 数 (若有)
    prompt_chars: Optional[int] = None        # 输入字符数
    response_chars: Optional[int] = None      # 输出字符数


@dataclass
class BenchmarkMeta:
    """评测集元信息 (由各插件提供)"""
    name: str                         # 内部名 (如 mmlu)
    display_name: str                 # 展示名 (如 MMLU)
    stage: Stage
    task_type: TaskType
    description: str = ""
    tags: List[str] = field(default_factory=list)   # modern / classic / knowledge / math / code / ...
    num_fewshot: int = 0
    # 是否需要裁判模型
    needs_judge: bool = False
    # 数据来源说明 (供报告展示)
    source: str = ""


@dataclass
class RunResult:
    """一次评测运行的完整结果"""
    run_id: str
    model_name: str
    benchmark: str
    benchmark_meta: BenchmarkMeta
    num_samples: int
    results: List[SampleResult] = field(default_factory=list)
    aggregate: Dict[str, Any] = field(default_factory=dict)   # 汇总指标
    started_at: str = ""
    finished_at: str = ""
    fake_stream: bool = False   # 是否检测到伪流式 (报告据此提醒并退化非流式口径)

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        # stage/task_type 转 字符串
        d["benchmark_meta"]["stage"] = self.benchmark_meta.stage.value
        d["benchmark_meta"]["task_type"] = self.benchmark_meta.task_type.value
        return d
