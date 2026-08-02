# 更新日志 (CHANGELOG)

本项目所有显著变更均记录于此。版本号遵循 [语义化版本 (SemVer)](https://semver.org/lang/zh-CN/),
格式参考 [Keep a Changelog](https://keepachangelog.com/zh-CN/1.1.0/)。

---

## 版本管理规范

### 版本号 `MAJOR.MINOR.PATCH`

| 段 | 何时升 | 举例 |
|---|---|---|
| **MAJOR**(大版本) | 有**不兼容的破坏性变更**:API/配置/数据格式不向后兼容,或整体架构重大调整、里程碑式大改 | `1.x.x → 2.0.0` |
| **MINOR**(小版本) | **向后兼容的新功能**:新增评测集、新增端点、新增配置项、显著特性增强 | `1.0.x → 1.1.0` |
| **PATCH**(补丁) | **向后兼容的缺陷修复**:修 bug、修评分误判、性能优化、文档完善,无新功能 | `1.0.0 → 1.0.1` |

### 升版时机

- **每完成一批改动**(修复若干 bug 或上线一个新功能)→ 发版,更新本文件 + `llm_eval/__init__.py` 的 `__version__`,提交并打 git tag。
- **大的内容更新**(新增评测集、新的大特性、架构调整)→ 升 MINOR 或 MAJOR。
- **定期发版**:即便无大改动,每隔一段时间(如每月)累积的小修复也可发一个 PATCH/MINOR。
- **紧急修复**:线上 bug 修复后立即发 PATCH。

### 变更类型

- `新增` — 新功能 / 新评测集 / 新端点
- `变更` — 对已有功能的修改(向后兼容)
- `修复` — Bug 修复
- `优化` — 性能 / 体验提升,无行为变化
- `文档` — 文档变更
- `已弃用` / `移除` — 即将移除 / 已移除的功能

### 发版流程

```bash
# 1. 改 llm_eval/__init__.py 的 __version__
# 2. 在本文件顶部新增版本条目 (倒序, 最新在上)
# 3. 提交 + 打 tag + 推送
git add -A
git commit -m "release: vX.Y.Z"
git tag vX.Y.Z
git push origin main --tags
```

---

## [1.0.0] — 2026-08-02

首个纳入 git 管理的正式版本。基于真实全量评测(13 万+样本,31 个评测集)打磨稳定,
覆盖预训练 + 后训练主流评测集,含自动评分、乱码分析、实时进度、多用户平台。

### 新增

- **平台核心**:FastAPI 多用户在线平台(注册/登录/HttpOnly 会话),REST API + 原生 JS SPA 前端,无构建步骤。
- **18+ 评测集**:预训练(MMLU/MMLU-Pro/GPQA/C-Eval/CMMLU/ARC/HellaSwag/WinoGrande/TruthfulQA/GSM8K/MATH-500/AIME/HumanEval/MBPP 等)+ 后训练(IFEval/MT-Bench/AlpacaEval/Arena-Hard),最终注册 34 个。
- **统一 API 客户端**:一套 OpenAI 兼容客户端覆盖 DeepSeek/Kimi/Qwen/GLM/OpenAI 等,不依赖厂商 SDK。
- **五种评分方式**:选项精确匹配 / 数字抽取匹配 / 代码执行 pass@1 / LLM 裁判打分 / 规则可验证。
- **乱码分析**:检测 mojibake 编码乱码、重复退化、控制字符、异常字符集、语言不一致、截断、空输出,给出健康度等级。
- **双入口**:命令行 CLI 自动化批跑 + Web 控制台可视化。
- **报告**:JSON 完整结果 + HTML 仪表板(每条请求明细、分页、搜索、展开看 prompt/思维链/输出)。
- **代码沙箱**:docker(禁网/只读根/资源上限,最强隔离)/ subprocess 双模式,执行模型生成的 Python 代码。
- **实时进度**:SSE 推送任务日志 + 数据集进度;单条样本跑完即可查看吐字明细,一集跑完即出评分(不必等全任务结束)。
- **增量落盘**:每跑完一个评测集即保存 JSON/HTML,中途异常不丢已完成结果。
- **任务管理**:提交/查看/克隆/删除/取消,回收站 30 天清理,仪表盘与管理后台分页(修 >200 截断)。
- **部署文档**:`DEPLOY.md` 新 Linux 服务器从零部署指南。

### 修复

- **MCQ 中文答案抽取漏判**:`经过分析，选A`/`故选C`/`应选 D` 抽不到字母导致全判错(影响 ceval/cmmlu/agieval)。加 `(?:故|应|因此|所以|则)?选` 模式 + valid_choices 过滤防误匹配。
- **思维链吃光 token 预算**:reasoning 过长致 content 空,旧逻辑把 reasoning 当 response 评分致开放 QA 全错。各评测集 max_tokens 统一提到 8192(AIME 竞赛题需 65536),保留 reasoning 回退兜底(MCQ/数学答案常在 reasoning 末尾)。
- **代码执行 NameError(两处)**:① MBPP test_code 用 `math.isclose` 却没 import → `run_code_tests` 加 `_ensure_stdlib_imports` 自动补标准库 import;② HumanEval prompt 里辅助函数(如 encode_shift)执行时丢失 → `_eval_code` 把含 `def` 的 prompt 作前缀拼到 completion 前。
- **乱码检测误报**:LaTeX 数学定界符 `$`、数学符号 `× ⁻ ·` 被归 "异常脚本" → `_script_of` 把 S 类符号归 "symbol";LaTeX `\text{ mile}` 反复触发 "高度重复" → ngram 30%-50% 降级为弱提示,≥50% 才报重复退化。
- **runner 单题慢吐卡死全局**:思维链模型持续慢吐 token 使 `requests.post(timeout)` 永不触发(该 timeout 是两次数据间间隔非总耗时),慢题占满并发槽致整个任务卡死。加请求级硬超时(`ThreadPoolExecutor` 包 post,`fut.result(timeout=remaining)` 超时强弃)。
- **系统代理吊死连接**:本机全局代理(http_proxy)致评测请求绕道代理,代理对大并发长连接"吊住"(ESTABLISHED 但收发队列全空,timeout 触发极慢)。`requests.Session(trust_env=False, proxies={http:None,https:None})` 强制直连。
- **取消任务不即时**:点取消后任务仍显示 running 几十分钟(等在途慢吐请求自然结束)。三层修复:流式 `iter_lines` 阻塞时用底层 socket `shutdown(SHUT_RDWR)` 中断(唯一可靠方式,`r.close()`/`os.close(fd)` 都无效);`run_concurrent` 改 `wait(timeout=0.5)` 轮询取消标志不等 in-flight;taskman 加 cancelled 终态判定。
- **livecodebench 驼峰函数名误判**:442 题中 440 个 entry_point 是驼峰(findPeaks/sumOfSquares),模型常写成蛇形(find_peaks),harness 按驼峰调用 → NameError 误判 fail。prompt 明示函数名 + 执行层 `_entry_point_alias` 自动补别名(驼峰↔蛇形归一化匹配,唯一候选才补)。
- **corpusqa 数据 context 全空**:HF 源语料在 `prompt` 消息列表的 user content 里,下载脚本误用 `context` 字段取不到 → 改遍历 `prompt` 取 role=user 的 content;配套改 corpusqa.py 支持多正确答案(任一匹配即对)。
- **报告分页跳转滚到网页顶**:`<tbody>` 的 `offsetTop` 在多数浏览器返回 0,改用 `#view-bench` 的 `getBoundingClientRect` 定位明细区开头。

### 优化

- **大任务明细页加载**:`/requests` 端点 lite 模式(不返回大文本,62MB→8.6MB)+ 点 Hash 按需拉单条完整明细 + 结果 JSON 缓存(按 mtime 失效)+ GZip 中间件(8.6MB→0.9MB)。万条级样本首屏不再卡死。
- **并发吞吐**:压测确定 32 并发为甜点(再高无收益),全量 13 万样本从 ~12 天压到 ~1 天。

### 已知限制

- 无断点续跑机制:任务中断后只能新建任务跑剩余集(增量落盘保已完成集不丢)。
- 思维链模型单题耗时长(AIME 最长 ~5 万 token),高并发长时间打满端点易触发 HTTP 429 限流,**一次只提交一个任务批次**。
- 超长上下文评测集(mrcr/corpusqa 单条达 132 万 token)超过模型输入上限会 ContextWindowExceeded,按模型能力判错(预期行为,反映真实能力边界)。
