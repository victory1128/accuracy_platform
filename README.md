# 大模型精度测试平台

输入大模型 API, 对 DeepSeek V4 / Kimi K3 / Qwen / GLM / GPT 等主流大模型做精度评测。覆盖**预训练阶段**与**后训练阶段**的主流评测集, 测试后给出分数 + **乱码/异常输出分析** + 可视化报告。

> 📌 **新服务器部署** 请看 [DEPLOY.md](DEPLOY.md) —— 从零上线的完整指南(含配置、数据集下载、Docker 沙箱、systemd 守护、故障排查)。

## 特性

- **统一 API 客户端**: 一套 OpenAI 兼容客户端覆盖 DeepSeek / Kimi(Moonshot) / Qwen / GLM / OpenAI 等, 不依赖厂商 SDK
- **18 个评测集** (可扩展), 按训练阶段分类:
  - **预训练** (知识/推理/代码): MMLU, MMLU-Pro, GPQA-Diamond, C-Eval, CMMLU, ARC, HellaSwag, WinoGrande, TruthfulQA, GSM8K, MATH-500, AIME, HumanEval, MBPP
  - **后训练** (指令遵循/对齐): IFEval, MT-Bench, AlpacaEval 2.0, Arena-Hard
- **多种评分方式**: 选项精确匹配 / 数字抽取匹配 / 代码执行 pass@1 / LLM 裁判打分 / 规则可验证
- **乱码分析**: 检测 mojibake 编码乱码、重复退化(repetition degeneration)、控制字符、异常字符集、语言不一致、截断、空输出, 给出健康度等级
- **双入口**: 命令行 (CLI) 自动化批跑 + 本地 Web 控制台可视化配置与查看
- **报告**: JSON 完整结果 + HTML 仪表板

## 快速开始

### 1. 安装

```bash
# 需要 Python 3.9+
python3.9 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 2. 配置 API

```bash
cp config.example.yaml config.yaml
# 编辑 config.yaml, 填入你的 API Key
```

`config.yaml` 关键字段:
```yaml
models:
  - name: "deepseek-v4"
    base_url: "https://api.deepseek.com/v1"
    api_key: "sk-xxx"
    model: "deepseek-chat"

judge:          # 裁判模型 (用于 MT-Bench/AlpacaEval/Arena-Hard), 可选
  name: "gpt-4o-judge"
  base_url: "https://api.openai.com/v1"
  api_key: "sk-xxx"
  model: "gpt-4o"
```

> 所有模型只需 `base_url` + `api_key` + `model` 三个字段。各厂商 OpenAI 兼容端点见文末附录。

### 3a. 命令行使用

```bash
# 列出所有评测集
python -m llm_eval.cli list

# 演练模式 (不调用真实API, 用模拟数据跑通全流程, 验证平台是否正常)
python -m llm_eval.cli run -m deepseek-v4 -b mmlu gsm8k ifeval --limit 5 --dry-run

# 正式评测: 指定模型 + 评测集
python -m llm_eval.cli run -m deepseek-v4 -b mmlu mmlu_pro gsm8k math500 humaneval ifeval

# 用配置里的默认 benchmarks, 每个评测集采样 20 条 (快速测试)
python -m llm_eval.cli run --limit 20
```

报告输出到 `results/` 目录 (JSON + HTML)。

### 3b. Web 控制台 (单机调试, 遗留)

```bash
python -m llm_eval.web
# 浏览器打开 http://127.0.0.1:8765
```

单用户本地调试用。多人在线平台请用下面的 **3c**。

### 3c. 在线平台 (多用户, 客户端/服务端架构)

把评测平台升级为多用户在线服务: 用户注册登录、提交/查看/克隆/删除自己的评测任务, 管理员可监控全平台。一个进程同时提供 REST API 和前端 SPA。

```bash
# 1. 在 config.yaml 的 server 段填管理员引导账号 + 选代码沙箱策略
#    (见下方 "配置说明")
# 2. 启动平台
python -m llm_eval.server
# 浏览器打开 http://127.0.0.1:8765  (或 server.host/port)
```

**功能**:
- **用户**: 注册/登录 (HttpOnly Cookie 会话, 7 天有效)。首个注册用户或 `config.yaml → server.admin` 引导的用户自动成为管理员。API Key 仅在任务运行期间驻留内存, **绝不落库**, 克隆任务时需重填。
- **任务**: 提交评测 (正式 / dry-run / 快速试跑 三种模式) → 实时进度 (SSE 日志/进度流) → 查看 HTML 报告 → 历史列表 / 克隆修改 / 删除。报告按用户隔离存 `results/user_<id>/`。
- **管理员后台**: 用户管理 (角色升降/启停)、查看所有用户任务、平台统计监控 (用户数/任务状态分布/运行中数)。
- **开发模式**: dry-run 跑通流程、快速小样本试跑、评测集热加载 (改 `benchmarks/` 后不重启即生效)、平台自检、任务级调试日志。
- **代码沙箱**: HumanEval/MBPP/BigCodeBench/LiveCodeBench 执行模型生成的代码, **默认禁用** (最安全)。仅在 `config.yaml → server.code_exec.sandbox` 设为 `subprocess`/`docker` 且**仅管理员**可提交含此类评测集的任务。BigCodeBench 任务依赖 pandas/numpy/matplotlib 等数据科学库, docker 模式需用本仓库 `Dockerfile.code-sandbox` 构建的预装镜像 (运行时禁网无法 pip install)。

**构建代码沙箱镜像 (跑 BigCodeBench 必需)**:

```bash
docker build -f Dockerfile.code-sandbox -t llm-eval-sandbox:latest .
# 然后在 config.yaml 设: server.code_exec.image: "llm-eval-sandbox:latest"
# 仅跑 HumanEval/MBPP/LiveCodeBench (标准库) 可直接用 python:3.11-slim
```

**配置 (`config.yaml` 的 `server` 段)**:

```yaml
server:
  host: "0.0.0.0"
  port: 8765
  db_path: "data/platform.db"     # 用户/任务/会话/日志 (SQLite)
  results_dir: "results"
  admin:                          # 首次启动引导管理员 (之后改密码这里不再覆盖)
    username: "admin"
    password: "change-me"
  code_exec:
    sandbox: "disabled"           # disabled | subprocess | docker
  cors_origins: []                # 跨域来源, 空=同源
```

**API 概览** (详见 `/docs`):
- `POST /api/auth/{register,login,logout}` · `GET /api/auth/me`
- `GET /api/benchmarks` · `POST /api/tasks` · `GET /api/tasks` · `GET/DELETE /api/tasks/{id}` · `POST /api/tasks/{id}/{clone,cancel}` · `GET /api/tasks/{id}/{logs,report,events}`
- `POST /api/dev/{dry-run,quick-sample,reload-benchmarks}` · `GET /api/dev/health`
- `GET /api/admin/{users,tasks,stats}` · `PUT /api/admin/users/{id}/{role,active}`

## 评测集说明

| 名称 | 阶段 | 类型 | 说明 |
|------|------|------|------|
| mmlu | 预训练 | MCQ | 57学科知识问答 |
| mmlu_pro | 预训练 | MCQ | MMLU增强版, 更难 |
| mmlu_redux | 预训练 | MCQ | MMLU去噪重标版, 修正标注错误 |
| supergpqa | 预训练 | MCQ | 研究生级多学科, 最高10选项 |
| gpqa | 预训练 | MCQ | 研究生级科学问答 |
| ceval | 预训练 | MCQ | 中文多学科 |
| cmmlu | 预训练 | MCQ | 中文多任务 |
| arc | 预训练 | MCQ | 小学科学推理 |
| hellaswag | 预训练 | MCQ | 常识续写 |
| winogrande | 预训练 | MCQ | 共指消解 |
| truthfulqa | 预训练 | MCQ | 真实性问答 |
| agieval | 预训练 | MCQ | 高考/LSAT/SAT等标准化考试(中英) |
| bbh | 预训练 | GEN | 23项难推理任务 |
| gsm8k | 预训练 | GEN | 小学数学 |
| math500 | 预训练 | GEN | 竞赛数学 |
| aime | 预训练 | GEN | 美国数学邀请赛 |
| hmmt_feb_2025 | 预训练 | GEN | 哈佛-MIT数学锦标赛2025年2月赛, 30题 |
| imo_answerbench | 预训练 | GEN | 国际数学奥林匹克短答案, 400题 |
| hle | 预训练 | GEN | 人类终极考试, 极难专家级问题 |
| drop | 预训练 | GEN | 阅读理解+离散推理 |
| longbench_v2 | 预训练 | MCQ | 长文本理解, 4选项长上下文 |
| mrcr | 预训练 | GEN | 超长多轮上下文检索(needle) |
| corpusqa | 预训练 | GEN | 百万字词语料问答 |
| humaneval | 预训练 | CODE | 函数级代码补全 pass@1 |
| mbpp | 预训练 | CODE | 基础编程 pass@1 |
| evalplus | 预训练 | CODE | HumanEval增强版, 更严测试 |
| bigcodebench | 预训练 | CODE | 实用代码生成, 需数据科学库 |
| livecodebench | 预训练 | CODE | 竞赛代码生成(LeetCode风格) pass@1 |
| livecodebench_v6 | 预训练 | CODE | LiveCodeBench release_v6全量(1054题: 函数式444 class Solution + stdin610 AtCoder/Codeforces) pass@1 |
| ds1000 | 预训练 | CODE | 数据科学代码(pandas/numpy) |
| swebench | 预训练 | CODE | 代码Agent, 修复真实GitHub issue |
| simpleqa | 后训练 | GEN | 短答案事实性问答 |
| simpleqa_verified | 后训练 | GEN | SimpleQA人工核验修订版, LLM裁判三分类 |
| bfcl | 后训练 | GEN | 函数调用(tool use)能力评测 |
| ifeval | 后训练 | RULE | 指令遵循(可验证) |
| ifbench | 后训练 | RULE | 指令遵循泛化, 58个OOD新约束(与IFEval平行独立) |
| mt_bench | 后训练 | JUDGE | 多轮对话, 裁判打分 |
| alpaca_eval | 后训练 | JUDGE | 回答质量胜率 |
| arena_hard | 后训练 | JUDGE | 高难度指令配对胜率 |

> 标注 JUDGE 的评测集需要配置裁判模型 (`config.yaml` 的 `judge`)。
> 标注 CODE 的评测集需要启用代码沙箱 (见下节); BigCodeBench/DS-1000 需数据科学库镜像, SWE-bench 需 swebench 包+Docker+预构建镜像。
> 标注长文本 (longbench_v2/mrcr/corpusqa) 的上下文可达百万字符, 需模型支持超长上下文窗口。

## 乱码分析

平台对每条模型输出自动分析, 检测:

- **编码乱码 (mojibake)**: UTF-8 被错误解码的典型字符 (ï¿½, Ã¢ 等)
- **重复退化**: n-gram 重复率过高 (模型卡死复读)
- **控制字符 / 非打印字符**
- **字符集失衡**: 非预期脚本 (西里尔/泰文等) 占比过高
- **语言一致性**: 中文题却全英文输出等
- **截断**: 疑似因 max_tokens 截断
- **空输出 / 过短输出**

汇总给出可疑率 + 健康度等级 (A 无异常 / B 基本正常 / C 少量异常 / D 异常较多 / F 异常严重), 并在报告中展示可疑样本示例。

## 数据集说明

平台**内置每个评测集的少量样例数据** (`llm_eval/benchmarks/data/*.jsonl`), 让你无需联网下载即可跑通流程、验证接入 (用 `--dry-run` 或小 `--limit`)。

要做**严肃全量评测**时, 把对应全集数据转成同名 jsonl 放到 `llm_eval/benchmarks/data/` 即可, 各评测集期望的字段格式见对应 benchmark 文件的 `load_samples`。数据来源建议:

- MMLU / MMLU-Pro / GPQA / C-Eval / CMMLU / GSM8K / MATH / HumanEval / MBPP / IFEval: HuggingFace
- MT-Bench / AlpacaEval / Arena-Hard: 各自官方 GitHub repo

## 扩展: 新增评测集

在 `llm_eval/benchmarks/` 下新建 `.py`, 继承 `Benchmark` (或更具体的 `_MCQBenchmark` / `_MathBenchmark` / `_CodeBenchmark` / `_JudgeBenchmark`), 用 `@register` 装饰, 提供数据 jsonl 即可被自动发现:

```python
from .registry import register
from .pretrain_mcq import _MCQBenchmark
from ..models import BenchmarkMeta, Stage, TaskType

@register
class MyBenchmark(_MCQBenchmark):
    DATA_FILE = "my_bench.jsonl"
    META = BenchmarkMeta(
        name="my_bench", display_name="MyBench",
        stage=Stage.PRETRAIN, task_type=TaskType.MCQ,
        description="我的评测集", tags=["custom"],
    )
```

## 项目结构

```
精度测试平台/
├── config.example.yaml          # 配置模板
├── requirements.txt
├── llm_eval/
│   ├── client.py                # 统一 OpenAI 兼容客户端 (重试/并发)
│   ├── models.py                # 数据模型
│   ├── config.py                # 配置加载
│   ├── runner.py                # 评测运行器 (编排)
│   ├── report.py                # JSON/HTML 报告生成
│   ├── cli.py                   # 命令行
│   ├── analysis/gibberish.py    # 乱码分析器
│   ├── scoring/                 # 评分器 (extract/code_exec/judging/rule_check)
│   ├── benchmarks/              # 评测集插件
│   │   ├── base.py              # 基类
│   │   ├── registry.py          # 自动注册
│   │   ├── pretrain_mcq.py      # MMLU 等 MCQ 评测集
│   │   ├── pretrain_gen.py      # GSM8K 等数学评测集
│   │   ├── pretrain_code.py     # HumanEval 等代码评测集
│   │   ├── posttrain.py         # IFEval/MT-Bench 等后训练评测集
│   │   └── data/*.jsonl         # 内置样例数据
│   ├── web/                     # FastAPI Web 控制台 (单机调试, 遗留)
│   │   ├── app.py
│   │   └── templates/
│   └── server/                  # 多用户在线平台 (生产)
│       ├── app.py               # FastAPI 工厂 (路由+静态SPA+admin引导)
│       ├── db.py                # SQLite 持久层 (用户/任务/会话/日志)
│       ├── auth.py              # 注册/登录/会话/角色 (pbkdf2)
│       ├── schemas.py           # Pydantic 请求/响应模型
│       ├── taskman.py           # 任务队列+worker+SSE事件总线
│       ├── server_config.py     # server 段配置加载
│       ├── routes/              # auth/task/dev/admin 路由
│       └── static/              # 前端 SPA (index.html/app.js/style.css, 原生JS)
└── results/                     # 输出报告 (平台按用户子目录隔离)
```

## 附录: 各厂商 OpenAI 兼容端点

| 厂商 | base_url | model 示例 |
|------|----------|-----------|
| DeepSeek | `https://api.deepseek.com/v1` | `deepseek-chat` |
| Kimi (Moonshot) | `https://api.moonshot.cn/v1` | `moonshot-v1-auto` |
| Qwen (通义) | `https://dashscope.aliyuncs.com/compatible-mode/v1` | `qwen-plus` |
| GLM (智谱) | `https://open.bigmodel.cn/api/paas/v4` | `glm-4` |
| OpenAI | `https://api.openai.com/v1` | `gpt-4o` |

## 安全说明

- 代码评测 (`HumanEval`/`MBPP`) 会执行模型生成的代码, 用子进程 + 超时限制做轻量隔离。**生产环境请勿在不可信网络/共享主机上直接跑**, 建议放进 Docker 容器内执行。
- API Key 只存在本地 `config.yaml`, 平台不会上传任何凭证。
