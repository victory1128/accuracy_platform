"""管理员路由: 用户管理 / 全部任务 / 平台统计

所有路由都需 admin 角色 (Depends(auth.require_admin))。
"""
from __future__ import annotations

from typing import List, Optional

from fastapi import APIRouter, Depends, HTTPException

from .. import auth, db, taskman
from ..schemas import (
    ActiveUpdateIn, AdminPasswordResetIn, Message, PageOut, PlatformStats, RoleUpdateIn,
    TaskOut, UserOut,
)
from ..routes.task_routes import _task_out

router = APIRouter(prefix="/api/admin", tags=["admin"], dependencies=[Depends(auth.require_admin)])


# ----------------------------- 用户管理 -----------------------------
@router.get("/users", response_model=List[UserOut])
def list_users():
    return [auth._public_user(u) for u in db.list_users()]


@router.put("/users/{user_id}/role", response_model=UserOut)
def set_role(user_id: int, body: RoleUpdateIn, actor: dict = Depends(auth.current_user)):
    """升降级用户角色。仅 super 可操作; super 不可被改; 不可授 super。"""
    target = db.get_user_by_id(user_id)
    if not target:
        raise HTTPException(404, "用户不存在")
    if not auth.can_change_role(actor, target, body.role):
        raise HTTPException(403, "无权修改该用户角色 (仅超级管理员可升降级, 且不可改超级管理员/授超级管理员)")
    if not db.update_user_role(user_id, body.role):
        raise HTTPException(404, "用户不存在")
    return auth._public_user(db.get_user_by_id(user_id))


@router.put("/users/{user_id}/active", response_model=UserOut)
def set_active(user_id: int, body: ActiveUpdateIn, actor: dict = Depends(auth.current_user)):
    """启用/禁用用户。必须严格高于目标层级 (super 改 admin/user, admin 改 user)。"""
    target = db.get_user_by_id(user_id)
    if not target:
        raise HTTPException(404, "用户不存在")
    if not auth.can_manage(actor, target):
        raise HTTPException(403, "无权操作同级或更高层级用户")
    if not db.set_user_active(user_id, body.is_active):
        raise HTTPException(404, "用户不存在")
    return auth._public_user(db.get_user_by_id(user_id))


@router.post("/users/{user_id}/password", response_model=Message)
def reset_password(user_id: int, body: AdminPasswordResetIn, actor: dict = Depends(auth.current_user)):
    """重置下级用户密码 (不需原密码)。必须严格高于目标层级。"""
    target = db.get_user_by_id(user_id)
    if not target:
        raise HTTPException(404, "用户不存在")
    if not auth.can_manage(actor, target):
        raise HTTPException(403, "无权修改同级或更高层级用户的密码")
    auth.admin_reset_password(user_id, body.new_password)
    return Message(message="密码已重置")


# ----------------------------- 全部任务 -----------------------------
@router.get("/tasks", response_model=List[TaskOut])
def list_all_tasks(
    status: Optional[str] = None,
    user_id: Optional[int] = None,
    q: Optional[str] = None,
):
    """管理员查全部任务, 支持按 状态/用户/关键词(名称+评测集) 筛选。"""
    return [_task_out(t) for t in db.list_tasks(user_id=user_id, status=status, query=q)]


@router.get("/tasks/page", response_model=PageOut)
def list_all_tasks_page(
    status: Optional[str] = None,
    user_id: Optional[int] = None,
    q: Optional[str] = None,
    page: int = 1,
    page_size: int = 20,
):
    """管理员全部任务的分页列表 (供管理后台分页)。page_size=0 表示全部。"""
    page = max(1, page)
    limit = 0 if page_size == 0 else max(1, page_size)
    offset = 0 if limit == 0 else (page - 1) * limit
    total = db.count_tasks(user_id=user_id, status=status, query=q)
    items = [_task_out(t) for t in db.list_tasks(
        user_id=user_id, status=status, query=q, limit=limit, offset=offset,
    )]
    return PageOut(items=items, total=total, page=page, page_size=page_size)


@router.get("/tasks/{task_id}/logs")
def admin_logs(task_id: int, after_id: int = 0):
    """管理员查看任意任务日志。"""
    from ..schemas import LogLine
    if not db.get_task(task_id):
        raise HTTPException(404, "任务不存在")
    return db.list_logs(task_id, after_id=after_id)


# ----------------------------- 平台统计 -----------------------------
@router.get("/stats", response_model=PlatformStats)
def platform_stats():
    by_status = db.count_tasks_by_status()
    return PlatformStats(
        users=db.count_users(),
        tasks_total=sum(by_status.values()),
        tasks_by_status=by_status,
        running_now=len(taskman._INFLIGHT),
    )


# ----------------------------- 回收站 (仅管理岗) -----------------------------
def _purge_and_clean():
    """清除过期回收站任务 (30天) 并删除其报告文件。"""
    expired = db.purge_expired(days=30)
    for t in expired:
        rp = t.get("report_path")
        if rp:
            _delete_report_files(rp)


@router.get("/trash", response_model=List[TaskOut])
def list_trash():
    """列出回收站任务 (管理岗可见)。访问时顺带清理过期的。"""
    _purge_and_clean()
    return [_task_out(t) for t in db.list_trashed_tasks()]


@router.post("/trash/{task_id}/restore", response_model=Message)
def restore_task(task_id: int):
    """从回收站恢复任务。"""
    if not db.restore_task(task_id):
        raise HTTPException(404, "回收站中无此任务")
    return Message(message="已恢复")


def _delete_report_files(report_path: str) -> None:
    """删除某任务报告的 .html/.json 文件 (忽略缺失/出错)。"""
    from ..server_config import load_server_config
    import os
    scfg = load_server_config()
    for ext in (".html", ".json"):
        p = os.path.join(scfg.results_dir, report_path + ext)
        try:
            if os.path.exists(p):
                os.remove(p)
        except OSError:
            pass


@router.delete("/trash/{task_id}", response_model=Message)
def hard_delete_task(task_id: int):
    """彻底删除回收站任务 (永久清除, 含报告文件)。"""
    t = db.hard_delete_task(task_id)
    if not t:
        raise HTTPException(404, "回收站中无此任务")
    rp = t.get("report_path")
    if rp:
        _delete_report_files(rp)
    return Message(message="已彻底删除")


@router.delete("/trash", response_model=Message)
def empty_trash():
    """清空回收站: 永久清除全部已软删除任务 (含报告文件)。空回收站也返回成功。"""
    deleted = db.empty_trash()
    for t in deleted:
        rp = t.get("report_path")
        if rp:
            _delete_report_files(rp)
    return Message(message=f"已清空回收站 ({len(deleted)} 条)")
