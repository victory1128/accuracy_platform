"""代码评测: 执行模型生成的代码 + pass@k 计算

HumanEval / MBPP 风格: 给定 prompt (函数签名+docstring), 模型补全函数体,
然后把"补全代码 + 测试用例"拼在一起, 在隔离环境里执行, 看是否通过全部断言。

沙箱模式 (由 configure_sandbox 注入, 启动时从 server_config 读):
- subprocess: 子进程跑 (隔离弱, 仅开发/可信环境; 加资源限制)
- docker: 一次性 docker 容器跑 (最强隔离: 独立文件系统/禁网/资源上限, 用完即删)
"""
from __future__ import annotations

import math
import os
import re
import subprocess
import sys
import tempfile
from typing import List, Optional, Tuple

CODE_BLOCK_RE = re.compile(r"```(?:python)?\s*\n(.*?)```", re.DOTALL)

# 模块级沙箱配置 (app 启动时由 configure_sandbox 注入; 默认 subprocess)
_SANDBOX = {
    "mode": "subprocess",   # subprocess | docker
    "image": "python:3.11-slim",
    "memory": "256m",
    "cpus": "1.0",
    "timeout": 15.0,
}


def configure_sandbox(mode: str = "subprocess", image: str = "python:3.11-slim",
                      memory: str = "256m", cpus: str = "1.0", timeout: float = 15.0) -> None:
    """注入沙箱配置。由 server app 启动时调用 (从 server_config 读)。

    未调用则默认 subprocess 模式 (CLI 直接跑时)。
    """
    _SANDBOX["mode"] = mode or "subprocess"
    _SANDBOX["image"] = image
    _SANDBOX["memory"] = memory
    _SANDBOX["cpus"] = cpus
    _SANDBOX["timeout"] = float(timeout)


def extract_code(response: str, entry_point: Optional[str] = None) -> str:
    """从模型输出抽取 python 代码

    优先取 ```python ... ``` 代码块; 否则取整个输出 (假设模型直接输出代码)。
    若有 entry_point (函数名), 截到该函数定义之后的完整内容。
    """
    if not response:
        return ""
    blocks = CODE_BLOCK_RE.findall(response)
    if blocks:
        code = "\n\n".join(blocks)
    else:
        code = response
    # 去掉可能的解释性首行 (如 "Here is the code:")
    return code.strip()


def run_code_tests(
    completion: str,
    test_code: str,
    *,
    timeout: Optional[float] = None,
    entry_point: Optional[str] = None,
) -> Tuple[bool, str]:
    """执行 completion + test_code, 返回 (是否全部通过, stdout/stderr 摘要)

    test_code 应包含 assert 语句, 并可能依赖 completion 里定义的函数。
    沙箱模式由 configure_sandbox 注入: docker (容器隔离) / subprocess (子进程)。

    entry_point: harness 期望调用的函数名 (LiveCodeBench 等)。模型常把驼峰
    entry_point (findPeaks) 写成蛇形 (find_peaks), 导致 harness 按 entry_point
    调用时 NameError 误判 fail。传入后会在执行前补别名: 若 completion 里定义了
    别的函数名而 entry_point 不存在, 自动 `entry_point = 实际名`。
    """
    # 函数名别名: 挽救模型把 entry_point 写成不同命名风格的情况 (驼峰↔蛇形)。
    # 必须插在 completion (定义函数) 之后、test_code (调用函数) 之前, 否则别名
    # 在调用点之后才定义会 NameError。故先单独算 completion 的别名行, 再拼接。
    alias_line = _entry_point_alias(completion, entry_point) if entry_point else ""
    full = f"{completion}\n{alias_line}\n\n{test_code}\n"
    # MBPP/部分数据集的 test_code 用了标准库 (如 math.isclose) 却自身没 import,
    # 依赖模型代码补 import; 但模型常只补全函数体不写 import -> NameError 误判。
    # 兜底: 扫描合并代码里用到但未 import 的常见标准库, 自动补到顶部。
    full = _ensure_stdlib_imports(full)
    to = timeout if timeout is not None else _SANDBOX["timeout"]
    if _SANDBOX["mode"] == "docker":
        return _run_in_docker(full, to)
    return _run_in_subprocess(full, to)


# 常见标准库模块名 (仅安全的标准库; 第三方库不自动补, 需镜像预装)
_STDLIB_MODULES = {
    "math", "re", "string", "itertools", "functools", "collections",
    "operator", "decimal", "fractions", "statistics", "random", "json",
    "copy", "heapq", "bisect", "datetime", "time", "os", "sys", "io",
    "typing", "enum", "abc", "array", "struct", "hashlib", "base64",
}
# 匹配 模块.名字 或 import 模块
_MODULE_USE_RE = re.compile(r"\b([a-zA-Z_]\w*)\s*\.")


def _ensure_stdlib_imports(code: str) -> str:
    """扫描代码里以 `模块.` 形式使用但未 import 的标准库模块, 自动补 import。

    只补标准库 (安全, 容器内必有), 第三方库不补 (需镜像预装, 补了也 ImportError)。
    避免把变量名误当模块: 要求 形如 `xxx.yyy` 且 xxx 是已知标准库模块名。
    """
    if not code:
        return code
    # 已 import 的模块
    imported = set()
    for m in re.finditer(r"^\s*import\s+([a-zA-Z_]\w*)", code, re.MULTILINE):
        imported.add(m.group(1))
    for m in re.finditer(r"^\s*from\s+([a-zA-Z_]\w*)\s+import", code, re.MULTILINE):
        imported.add(m.group(1))
    # 以 `模块.` 使用的模块
    used = {m.group(1) for m in _MODULE_USE_RE.finditer(code)}
    missing = sorted((used & _STDLIB_MODULES) - imported)
    if not missing:
        return code
    prefix = "\n".join(f"import {m}" for m in missing) + "\n"
    return prefix + code


# 匹配顶层 `def 函数名(` 定义 (忽略缩进的嵌套 def, 如类方法)
_TOP_DEF_RE = re.compile(r"^[ \t]*def\s+([a-zA-Z_]\w*)\s*\(", re.MULTILINE)


def _entry_point_alias(completion: str, entry_point: str) -> str:
    """若 entry_point 未在 completion 里定义, 但 completion 定义了命名风格等价的函数,
    返回一行 `entry_point = 实际函数名` 别名 (供插在 completion 后、test_code 前)。

    典型场景: LiveCodeBench entry_point=findPeaks (驼峰), 模型写成 find_peaks (蛇形)。
    只在 entry_point 确实未定义、且能找到唯一候选时补; 否则返回空串 (避免误伤)。
    """
    if not completion or not entry_point:
        return ""
    defined = set(_TOP_DEF_RE.findall(completion))
    if entry_point in defined:
        return ""  # entry_point 已正确定义, 无需别名
    # entry_point 缺失: 找命名风格等价的候选 (驼峰↔蛇形互换后相等)
    ep_norm = _name_key(entry_point)
    candidates = [n for n in defined if _name_key(n) == ep_norm]
    if len(candidates) != 1:
        return ""  # 0 个或多个候选都无法确定, 不补 (避免猜错)
    actual = candidates[0]
    return f"# 函数名别名: 模型定义为 {actual}, 评测入口期望 {entry_point}\n{entry_point} = {actual}"


def _name_key(name: str) -> str:
    """把函数名归一化为比较键: 忽略驼峰/蛇形差异。
    findPeaks / find_peaks / FindPeaks → "findpeaks"。
    """
    return name.replace("_", "").lower()



def _run_in_subprocess(code: str, timeout: float) -> Tuple[bool, str]:
    """subprocess 模式: 子进程跑 (隔离弱, 仅可信环境)。"""
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".py", delete=False, encoding="utf-8"
    ) as f:
        f.write(code)
        path = f.name
    try:
        proc = subprocess.run(
            [sys.executable, path],
            capture_output=True,
            text=True,
            timeout=timeout,
            env={**os.environ, "PYTHONPATH": ""},
        )
        if proc.returncode == 0:
            return True, ""
        err = (proc.stderr or proc.stdout or "")[-800:]
        return False, err
    except subprocess.TimeoutExpired:
        return False, f"超时(>{timeout}s)"
    except Exception as e:  # noqa: BLE001
        return False, f"执行异常: {e}"
    finally:
        try:
            os.unlink(path)
        except OSError:
            pass


def _run_in_docker(code: str, timeout: float) -> Tuple[bool, str]:
    """docker 模式: 在一次性容器里跑 (最强隔离)。

    - --rm: 用完即删容器
    - --network none: 禁止联网 (防模型代码外联)
    - --read-only: 根文件系统只读, 仅 /tmp 可写 (防写恶意文件)
    - --memory/--cpus: 资源上限 (防内存/CPU 耗尽)
    - 代码通过 stdin 传入容器, 容器内 python 执行
    超时由外层 subprocess 控制 (docker run 没有可靠的内建超时)。
    """
    image = _SANDBOX["image"]
    cmd = [
        "docker", "run", "--rm",
        "--network", "none",
        "--read-only",
        "--mount", "type=tmpfs,destination=/tmp",
        "--memory", _SANDBOX["memory"],
        "--cpus", _SANDBOX["cpus"],
        "-i",                          # 用 stdin 传代码, 避免命令行长度/转义问题
        image,
        "python", "-",                 # 从 stdin 读代码执行
    ]
    try:
        proc = subprocess.run(
            cmd,
            input=code,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        if proc.returncode == 0:
            return True, ""
        err = (proc.stderr or proc.stdout or "")[-800:]
        return False, err
    except subprocess.TimeoutExpired:
        return False, f"超时(>{timeout}s, 容器已杀)"
    except FileNotFoundError:
        return False, "docker 未安装或不在 PATH (sandbox=docker 但无 docker)"
    except Exception as e:  # noqa: BLE001
        return False, f"docker 执行异常: {e}"


def compute_pass_at_k(n: int, c: int, k: int) -> float:
    """pass@k 的无偏估计 (HumanEval 论文公式)

    n: 总采样数, c: 通过数, k: k 值
    """
    if n - c < k:
        return 1.0
    return 1.0 - math.prod(1.0 - k / (n - i) for i in range(c))


def pass_at_1(passed: List[bool]) -> float:
    """单次采样 (n=1) 的 pass@1 = 通过率"""
    if not passed:
        return 0.0
    return sum(1 for p in passed if p) / len(passed)
