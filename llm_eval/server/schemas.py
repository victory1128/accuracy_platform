"""Pydantic 请求/响应模型

前端 <-> API 的数据契约。被测模型 API Key 在请求里携带, 但:
- 响应里绝不回显 key (ModelConfigOut 没有该字段)
- 落库时 key 不进 model_config (见 routes/task 提交处的 _strip_key)
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, ConfigDict, Field


# ----------------------------- 通用 -----------------------------
class Message(BaseModel):
    message: str
    detail: Optional[str] = None


# ----------------------------- 认证 -----------------------------
class RegisterIn(BaseModel):
    username: str = Field(..., min_length=3, max_length=32)
    password: str = Field(..., min_length=6)


class LoginIn(BaseModel):
    username: str
    password: str


class UserOut(BaseModel):
    id: int
    username: str
    role: str
    is_active: bool
    created_at: Optional[str] = None


# ----------------------------- 评测任务 -----------------------------
class ModelConfigIn(BaseModel):
    """提交任务时填的被测模型配置 (含 api_key, 仅内存用)。

    max_tokens/temperature 为 None 表示用各评测集自带值 (如 IFEval 8192),
    填了则强制覆盖所有评测集。
    """
    name: str
    base_url: str
    api_key: str
    model: str
    temperature: Optional[float] = None
    max_tokens: Optional[int] = None
    extra: Dict[str, Any] = Field(default_factory=dict)


class JudgeConfigIn(BaseModel):
    name: str = ""
    base_url: str = ""
    api_key: str = ""
    model: str = ""
    temperature: float = 0.0
    max_tokens: int = 256


class TaskCreateIn(BaseModel):
    """提交评测任务。api_key 仅在运行期间驻留内存, 不落库。

    注: 字段名用 model_cfg (Pydantic v2 保留 model_config), 序列化别名仍为
    model_config, 与 DB 列名/前端字段保持一致。
    """
    model_config = ConfigDict(populate_by_name=True)

    name: str = ""
    mode: str = "real"  # real | dry_run | quick
    model_cfg: ModelConfigIn = Field(alias="model_config")
    judge_config: Optional[JudgeConfigIn] = None
    use_judge: bool = False
    benchmarks: List[str]
    limit: Optional[int] = None
    concurrency: Optional[int] = None
    streaming: bool = False
    debug: bool = False  # 开发模式: 详细日志/单步追踪
    # 单请求硬超时(秒): 总耗时上限, 超过即放弃该题(防慢吐/stall 占满并发槽)。
    # None=用平台默认 1200(20min)。端点易 stall 时可调大给恢复时间, 或调小快速释放。
    timeout: Optional[int] = None


class TaskOut(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    id: int
    user_id: int
    name: str
    mode: str
    model_cfg: Dict[str, Any] = Field(alias="model_config")   # 不含 api_key
    judge_config: Optional[Dict[str, Any]] = None
    benchmarks: List[str]
    run_params: Dict[str, Any]
    status: str
    summary: Optional[List[Dict[str, Any]]] = None
    error: Optional[str] = None
    report_path: Optional[str] = None
    num_samples: int = 0
    progress: Optional[Dict[str, Any]] = None   # 各评测集进度 {bench: {done,total,pct,status}}
    created_at: str
    started_at: Optional[str] = None
    finished_at: Optional[str] = None


class PageOut(BaseModel):
    """分页响应信封。page_size=0 表示全部。"""
    items: List[Any]
    total: int
    page: int
    page_size: int


class CloneOut(BaseModel):
    """克隆任务: 返回可编辑的提交表单数据 (不含 key, 需用户重填)。"""
    model_config = ConfigDict(populate_by_name=True)

    name: str
    mode: str
    model_cfg: Dict[str, Any] = Field(alias="model_config")   # 不含 api_key
    judge_config: Optional[Dict[str, Any]] = None
    benchmarks: List[str]
    limit: Optional[int] = None
    concurrency: Optional[int] = None
    streaming: bool = False
    debug: bool = False
    timeout: Optional[int] = None


class LogLine(BaseModel):
    id: int
    task_id: int
    ts: str
    level: str
    message: str


# ----------------------------- 开发模式 -----------------------------
class QuickSampleIn(BaseModel):
    """快速小样本试跑。"""
    model_config = ConfigDict(populate_by_name=True)

    model_cfg: ModelConfigIn = Field(alias="model_config")
    judge_config: Optional[JudgeConfigIn] = None
    use_judge: bool = False
    benchmarks: List[str]
    limit: int = 5
    streaming: bool = False


class DryRunIn(BaseModel):
    """dry-run: 不调真实 API, 用模拟响应跑通流程。"""
    model_name: str = "dry-run"
    benchmarks: List[str]
    limit: Optional[int] = None


# ----------------------------- 管理员 -----------------------------
class RoleUpdateIn(BaseModel):
    role: str  # admin | user (super 不可被授/改; 由 can_change_role 校验)


class ActiveUpdateIn(BaseModel):
    is_active: bool


class PasswordChangeIn(BaseModel):
    """用户自己改密码: 需验证原密码。新密码前端输两次 (后端校验长度)。"""
    old_password: str = Field(..., min_length=1)
    new_password: str = Field(..., min_length=6)


class AdminPasswordResetIn(BaseModel):
    """管理员重置用户密码: 不需原密码。"""
    new_password: str = Field(..., min_length=6)


class PlatformStats(BaseModel):
    users: int
    tasks_total: int
    tasks_by_status: Dict[str, int]
    running_now: int
