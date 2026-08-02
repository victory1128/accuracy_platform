"""精度测试平台 (LLM Accuracy Evaluation Platform)

一个自研轻量大模型精度评测平台:
- 统一 OpenAI 兼容 API 客户端 (DeepSeek / Kimi / Qwen / GLM / OpenAI 等)
- 预训练阶段评测集 (MMLU, GPQA, GSM8K, HumanEval ...) + 后训练阶段评测集 (IFEval, MT-Bench, AlpacaEval ...)
- 自动评分 + 乱码分析 + JSON/HTML 报告
- CLI + Web 控制台
"""

# 版本号是全项目唯一数据源: CLI / FastAPI app / /api/server-info 都引用它。
# 版本管理规范见 CHANGELOG.md 顶部。语义化版本: MAJOR.MINOR.PATCH。
__version__ = "1.0.0"
