"""精度测试平台 (LLM Accuracy Evaluation Platform)

一个自研轻量大模型精度评测平台:
- 统一 OpenAI 兼容 API 客户端 (DeepSeek / Kimi / Qwen / GLM / OpenAI 等)
- 预训练阶段评测集 (MMLU, GPQA, GSM8K, HumanEval ...) + 后训练阶段评测集 (IFEval, MT-Bench, AlpacaEval ...)
- 自动评分 + 乱码分析 + JSON/HTML 报告
- CLI + Web 控制台
"""

__version__ = "0.1.0"
