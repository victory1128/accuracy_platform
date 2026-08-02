# 部署指南 (Linux 服务器)

本文档指导在一台全新的 Linux 服务器上从零部署「大模型精度测试平台」。
按顺序执行即可;每步都标注了「为什么」,便于排错。

> 仓库地址:`https://github.com/victory1128/accuracy_platform`
> 技术栈:Python 3.9+ / FastAPI / SQLite / 原生 JS SPA。无需数据库服务、无需 Node 构建。

---

## 0. 部署架构速览

```
git clone → 装依赖(venv) → 填 config.yaml → (可选)下数据集 → (可选)建 Docker 沙箱镜像 → 启动
```

| 组件 | 是否必需 | 说明 |
|---|---|---|
| Python 3.9+ venv + 依赖 | ✅ 必需 | 平台本体 |
| `config.yaml` | ✅ 必需 | 含 API Key / 管理员 / 沙箱配置 (不入 git,需手建) |
| 评测数据集 `benchmarks/data/` | ⚠️ 可选 | 缺失时平台正常启动,该评测集显示 0 条样本;要跑全量评测才需下载 |
| Docker 沙箱镜像 | ⚠️ 可选 | 仅跑 HumanEval/MBPP/BigCodeBench/LiveCodeBench 等**代码类**评测集时需要 |
| SQLite 数据库 | ✅ 自动 | 首次启动自动建表 (`data/platform.db`),无需手动初始化 |

**关键:代码仓库里不含数据集和 config.yaml**,这两样是新服务器部署时必须自己准备的(下文详述)。

---

## 1. 系统准备

### 1.1 安装基础工具

```bash
# Ubuntu / Debian
sudo apt update && sudo apt install -y git python3 python3-venv python3-pip curl

# CentOS / RHEL
sudo yum install -y git python3 python3-pip curl
```

确认 Python 版本 ≥ 3.9:

```bash
python3 --version   # 应输出 Python 3.9.x 或更高
```

### 1.2 (可选)安装 Docker —— 仅跑代码类评测集需要

如果只跑 MMLU/GSM8K/IFEval 等**非代码**评测集,跳过本步。
要跑 HumanEval/MBPP/BigCodeBench/LiveCodeBench(执行模型生成的 Python 代码),需 Docker 做沙箱隔离:

```bash
# Ubuntu
sudo apt install -y docker.io
sudo systemctl enable --now docker
sudo usermod -aG docker $USER   # 让当前用户免 sudo 调 docker, 需重新登录生效
```

验证:`docker ps` 能正常输出(无权限报错则重新登录或 `newgrp docker`)。

---

## 2. 获取代码 + 安装依赖

### 2.1 克隆仓库

```bash
cd /opt   # 或任意部署目录
sudo mkdir -p accuracy_platform && sudo chown $USER accuracy_platform
git clone https://github.com/victory1128/accuracy_platform.git accuracy_platform
cd accuracy_platform
```

> 私有仓库需先配 SSH key 或用 token;公开仓库直接 clone。

### 2.2 创建虚拟环境 + 装依赖

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

**国内服务器加速 pip**:

```bash
pip install -r requirements.txt -i https://mirrors.aliyun.com/pypi/simple/
```

> `requirements.txt` 含 `datasets`/`huggingface_hub`(下载评测数据集用)。
> 若不打算下载全集数据集,这两步装失败不影响平台启动,但建议装上。

验证依赖完整:

```bash
python -c "import fastapi, uvicorn, requests, yaml; print('核心依赖 OK')"
```

---

## 3. 配置 `config.yaml` (必需)

仓库不含 `config.yaml`(含 API Key,不入 git),需从模板创建:

```bash
cp config.example.yaml config.yaml
```

然后编辑 `config.yaml`,**必须修改**以下三处:

### 3.1 被测模型 API(顶部 `models` 段)

填入你要评测的模型的 OpenAI 兼容端点:

```yaml
models:
  - name: "glm-5.2-fp8"                          # 任意显示名
    base_url: "http://your-model-host:port/v1"   # 模型的 /v1 端点
    api_key: "sk-your-real-api-key"              # ← 必填真实 Key
    model: "glm-5.2-fp8"                         # 实际模型名
```

支持同时配多个模型对比。所有遵循 OpenAI 协议的模型(DeepSeek/Kimi/Qwen/GLM/OpenAI 等)均可。

### 3.2 管理员账号(`server.admin` 段)

首次启动时,平台会按这里的配置自动创建超级管理员:

```yaml
server:
  host: "0.0.0.0"            # 0.0.0.0 = 监听所有网卡(外网可访问); 仅本机用 127.0.0.1
  port: 8765
  db_path: "data/platform.db"
  results_dir: "results"
  admin:
    username: "admin"
    password: "改成强密码"    # ← 首次启动用它建管理员, 之后在网页后台改密码这里不再覆盖
```

> ⚠️ `password` 务必改成强密码。该密码仅在**首次启动且库中无该用户**时用于建管理员;
> 之后在网页「修改密码」改过的密码以数据库为准,改这里的配置不会覆盖。

### 3.3 代码沙箱策略(`server.code_exec` 段)

按需选择,**默认 `disabled` 最安全**:

```yaml
  code_exec:
    sandbox: "disabled"     # disabled=禁用代码执行 | subprocess=子进程(弱隔离) | docker=容器(强隔离)
```

| 取值 | 适用 | 说明 |
|---|---|---|
| `disabled` | 不跑代码类评测集 | HumanEval/MBPP 等提交时被拒(非 admin)/仅判语法 |
| `subprocess` | 可信内网环境 | 子进程跑,隔离弱,仅开发用 |
| `docker` | ✅ 生产推荐 | 一次性容器:禁网/只读根/资源上限,用完即删。需先建镜像(见第 5 步) |

选 `docker` 还要配 `image`(见第 5 步):

```yaml
    sandbox: "docker"
    image: "llm-eval-sandbox:latest"   # 或 python:3.11-slim(仅标准库题)
    memory: "512m"
    cpus: "1.0"
    timeout: 30.0
```

### 3.4 (可选)裁判模型(`judge` 段)

跑 MT-Bench/AlpacaEval/Arena-Hard 等 LLM 裁判评测集需要,建议配一个比被测模型更强的:

```yaml
judge:
  name: "gpt-4o-judge"
  base_url: "https://api.openai.com/v1"
  api_key: "sk-your-openai-key"
  model: "gpt-4o"
```

不跑裁判类评测集可留占位符。

---

## 4. (可选)下载评测数据集

仓库不含数据集(3.3G,可重新下载)。**不下载也能启动平台**,只是评测集显示 0 条样本、无法实际跑评测。

### 4.1 按需下载

```bash
# 在仓库根目录, venv 已激活
python scripts/download_datasets.py --list          # 列出所有可下载数据集
python scripts/download_datasets.py mmlu gsm8k      # 只下指定集
python scripts/download_datasets.py                 # 下载全部(耗时较长, 3G+)
```

数据集下载到 `llm_eval/benchmarks/data/<name>.jsonl`。

### 4.2 选择性下载建议

全集很大,按需下:

| 评测集 | 体积 | 建议 |
|---|---|---|
| mmlu / gsm8k / humaneval / mbpp / math500 | <10MB | 必下(核心基线) |
| mmlu_pro / bbh / ceval / cmmlu | 3-9MB | 推荐下 |
| corpusqa.jsonl | **1.06GB** | 按需(单条很长, 全量极慢) |
| longbench_v2.jsonl | **465MB** | 按需 |

### 4.3 注意事项

- GPQA 是 gated 数据集:需先在 HuggingFace 页面同意条款,并配置环境变量 `export HF_TOKEN=hf_xxx`。
- 国内访问 HuggingFace 慢:可设代理 `export HF_ENDPOINT=https://hf-mirror.com` 用镜像站。
- 下载脚本依赖 `datasets`/`huggingface_hub`(已在 requirements.txt)。

---

## 5. (可选)构建 Docker 沙箱镜像

仅当第 3.3 步选了 `sandbox: "docker"` 时需要。

### 5.1 按评测集选镜像

| 跑哪些代码评测集 | 用哪个镜像 |
|---|---|
| 仅 HumanEval / MBPP / LiveCodeBench(只用标准库) | 官方 `python:3.11-slim`,**无需构建**,直接 `docker pull python:3.11-slim` |
| 含 BigCodeBench(需 pandas/numpy/matplotlib 等数据科学栈) | 用仓库 `Dockerfile.code-sandbox` 构建 |

### 5.2 构建 BigCodeBench 镜像

```bash
# 在仓库根目录
docker build -f Dockerfile.code-sandbox -t llm-eval-sandbox:latest .
```

> Dockerfile 已配国内镜像源(清华 apt + 阿里云 pip)加速构建。
> 镜像约 1.5GB(含完整数据科学栈),首次构建 10-20 分钟。

构建后,确认 `config.yaml` 的 `server.code_exec.image` 设为 `llm-eval-sandbox:latest`。

### 5.3 验证沙箱

```bash
docker images | grep -E "llm-eval-sandbox|python.*slim"   # 确认镜像在
```

---

## 6. 启动平台

### 6.1 首次启动(前台验证)

```bash
source .venv/bin/activate
python -m llm_eval.server
```

看到如下输出即成功:

```
INFO:     Application startup complete.
INFO:     Uvicorn running on http://0.0.0.0:8765 (Press CTRL+C to quit)
```

**首次启动会自动**:
- 创建 `data/platform.db` 并建表(无需手动初始化数据库)
- 按 `config.yaml → server.admin` 创建超级管理员
- 清理回收站里超过 30 天的任务
- 注入代码沙箱配置

浏览器打开 `http://<服务器IP>:8765`,用第 3.2 步配的 admin 账号登录。

### 6.2 验证服务正常

```bash
# 服务端信息(无需登录)
curl http://127.0.0.1:8765/api/server-info
# 期望: {"version":"1.0.0","code_exec_enabled":true/false,"code_exec_sandbox":"...","benchmarks_count":34}

# 评测集目录(看数据集是否到位, num_samples>0 即已下载)
curl http://127.0.0.1:8765/api/benchmarks | python -m json.tool | head
```

若 `benchmarks_count` 为 0,检查 `llm_eval/benchmarks/` 下 .py 是否齐全(应是代码问题)。

### 6.3 生产环境:用 systemd 守护进程

前台运行关闭终端即停。生产环境用 systemd 托管(开机自启 + 崩溃重启):

```bash
sudo tee /etc/systemd/system/accuracy-platform.service > /dev/null <<'EOF'
[Unit]
Description=LLM Accuracy Platform
After=network.target docker.service

[Service]
Type=simple
User=YOUR_USERNAME
WorkingDirectory=/opt/accuracy_platform
# 用 venv 的 python, 不依赖 shell 激活
ExecStart=/opt/accuracy_platform/.venv/bin/python -m llm_eval.server
Restart=on-failure
RestartSec=5
# 日志输出到 journalctl
StandardOutput=journal
StandardError=journal

[Install]
WantedBy=multi-user.target
EOF
```

> 把 `User=` 和 `WorkingDirectory=` / `ExecStart=` 路径改成你的实际值。

启动并设为开机自启:

```bash
sudo systemctl daemon-reload
sudo systemctl enable accuracy-platform
sudo systemctl start accuracy-platform

# 查看状态
sudo systemctl status accuracy-platform
# 查看日志
sudo journalctl -u accuracy-platform -f   # -f 实时跟踪
```

### 6.4 防火墙 / 端口放行

```bash
# Ubuntu (ufw)
sudo ufw allow 8765/tcp

# CentOS (firewalld)
sudo firewall-cmd --permanent --add-port=8765/tcp
sudo firewall-cmd --reload
```

云服务器还需在**安全组**放行 8765 端口。

> 生产建议:不要直接暴露 8765 到公网,用 Nginx 反向代理 + HTTPS。
> 最低限度也应在 `config.yaml` 把 `host` 改成 `127.0.0.1`,由 Nginx 转发。

---

## 7. 常用运维操作

```bash
# 重启服务(改了代码或 config.yaml 后)
sudo systemctl restart accuracy-platform

# 查看实时日志
sudo journalctl -u accuracy-platform -f

# 备份数据库 + 结果
cp data/platform.db data/platform.db.bak.$(date +%Y%m%d)
tar czf results_backup_$(date +%Y%m%d).tar.gz results/

# 更新代码
cd /opt/accuracy_platform
git pull
source .venv/bin/activate
pip install -r requirements.txt   # 依赖若有更新
sudo systemctl restart accuracy-platform
```

---

## 8. 故障排查

| 现象 | 原因 / 解决 |
|---|---|
| `ModuleNotFoundError: No module named 'fastapi'` | 没激活 venv,或用了系统 python。务必用 `.venv/bin/python` 或 `source .venv/bin/activate` |
| 启动报 `No module named 'llm_eval'` | 工作目录不对,`python -m llm_eval.server` 必须在仓库根目录(含 `llm_eval/` 子目录)执行 |
| 登录提示密码错误 | admin 密码以数据库为准;首次启动才读 config。改过密码后改 config 不生效,需在网页改或直接重建库 |
| 评测集显示 0 条样本 | 数据集未下载,见第 4 步 `scripts/download_datasets.py` |
| 代码类评测集提交被拒 | `code_exec.sandbox` 为 `disabled`,或非管理员提交(code 类仅 admin 可提交) |
| docker 模式报 `docker 未安装或不在 PATH` | docker 未装/未启动/当前用户无 docker 权限(重新登录或 `newgrp docker`) |
| docker 模式报镜像不存在 | `docker images` 看镜像名是否与 `config.yaml → code_exec.image` 一致 |
| 任务卡住不结束 | 思维链模型生成慢;检查 `config.yaml → run.timeout`(默认 1200s);任务详情页可手动取消 |
| 外网访问不到 | ① `host` 是否 `0.0.0.0` ② 防火墙/安全组是否放行 8765 ③ 云服务器是否有公网 IP |
| `git pull` 后前端没更新 | 浏览器缓存;app.js 带了 mtime 版本号应自动刷新,强刷(Ctrl+Shift+R)即可 |

---

## 9. 数据与目录说明

```
accuracy_platform/
├── config.yaml              # ← 你手建的, 含 API Key (不入 git)
├── .venv/                   # ← 虚拟环境 (不入 git)
├── data/
│   └── platform.db          # ← SQLite 数据库, 首次启动自动建 (不入 git, 各环境独立)
├── results/                 # ← 用户评测报告, 按用户子目录 (不入 git, 各环境独立)
│   └── user_<id>/
├── llm_eval/
│   ├── benchmarks/
│   │   ├── *.py             # 评测集代码 (入 git)
│   │   └── data/            # ← 数据集 jsonl (不入 git, 第4步下载)
│   ├── server/              # 平台后端 (入 git)
│   └── ...
└── ...
```

**三个不入 git 的运行时目录**,迁移/重建环境时各自独立:
- `config.yaml`:各环境的 API Key/管理员不同
- `data/platform.db`:各环境的用户/任务数据不同
- `results/`:各环境跑的评测报告不同
- `llm_eval/benchmarks/data/`:数据集可重新下载,不必随仓库迁移

---

## 10. 快速部署清单 (TL;DR)

新服务器上一口气部署(假设只跑非代码评测集、不需 Docker):

```bash
# 1. 装基础工具
sudo apt update && sudo apt install -y git python3 python3-venv

# 2. 克隆 + 依赖
cd /opt && git clone https://github.com/victory1128/accuracy_platform.git && cd accuracy_platform
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt -i https://mirrors.aliyun.com/pypi/simple/

# 3. 配置
cp config.example.yaml config.yaml
# 编辑 config.yaml: 填 models[].api_key、server.admin.password

# 4. 下数据集(按需)
python scripts/download_datasets.py mmlu gsm8k humaneval

# 5. 启动验证
python -m llm_eval.server
# 浏览器开 http://<IP>:8765 登录

# 6. (生产) 配 systemd 守护, 见第 6.3 步
```

部署问题先查第 8 步故障排查表。
