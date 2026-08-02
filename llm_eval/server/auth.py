"""认证: 注册 / 登录 / 会话 / 角色

密码: pbkdf2-hmac-sha256 (标准库 hashlib, 零依赖, 比 bcrypt 慢但安全够用)。
会话: 随机 token (secrets.token_urlsafe), 存 sessions 表, 有过期时间。
     不用 JWT (无状态 token 撤销麻烦); 用数据库会话更可控。
角色: admin / user。admin 能进管理后台 + 提交含 code 类评测集 (沙箱)。
"""
from __future__ import annotations

import hashlib
import hmac
import secrets
from datetime import datetime, timedelta
from typing import Optional

from fastapi import Depends, HTTPException, Request, status

from . import db

# 会话有效期
SESSION_TTL_DAYS = 7
# pbkdf2 迭代次数 (NIST 建议 >= 600000 for sha256)
_PBKDF2_ITERS = 600_000
_SALT_BYTES = 16


# ----------------------------- 密码哈希 -----------------------------
def hash_password(password: str) -> str:
    """返回 'pbkdf2_sha256$iters$salt_hex$hash_hex' 格式的字符串。"""
    salt = secrets.token_bytes(_SALT_BYTES)
    dk = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, _PBKDF2_ITERS)
    return f"pbkdf2_sha256${_PBKDF2_ITERS}${salt.hex()}${dk.hex()}"


def verify_password(password: str, stored: str) -> bool:
    """校验密码是否匹配存储的哈希。常量时间比较防时序攻击。"""
    try:
        algo, iters, salt_hex, hash_hex = stored.split("$")
        if algo != "pbkdf2_sha256":
            return False
        salt = bytes.fromhex(salt_hex)
        iters = int(iters)
        dk = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, iters)
        return hmac.compare_digest(dk.hex(), hash_hex)
    except (ValueError, AttributeError):
        return False


# ----------------------------- 会话 token -----------------------------
def new_token() -> str:
    return secrets.token_urlsafe(32)


def _expires_at() -> str:
    return (datetime.now() + timedelta(days=SESSION_TTL_DAYS)).isoformat(timespec="seconds")


# ----------------------------- 当前用户依赖 -----------------------------
def _extract_token(request: Request) -> Optional[str]:
    """从 Cookie 或 Authorization 头取 token (Cookie 优先)。"""
    token = request.cookies.get("session")
    if token:
        return token
    auth = request.headers.get("authorization", "")
    if auth.lower().startswith("bearer "):
        return auth[7:].strip()
    return None


def current_user(request: Request) -> dict:
    """FastAPI 依赖: 返回当前登录用户 dict; 未登录 401。

    同时校验会话是否过期, 过期则删除并 401。
    """
    token = _extract_token(request)
    if not token:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="未登录")
    sess = db.get_session(token)
    if not sess:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="会话无效, 请重新登录")
    # 过期检查
    try:
        if datetime.fromisoformat(sess["expires_at"]) < datetime.now():
            db.delete_session(token)
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="会话已过期, 请重新登录")
    except ValueError:
        pass
    user = db.get_user_by_id(sess["user_id"])
    if not user or not user.get("is_active"):
        db.delete_session(token)
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="用户不可用")
    user["_token"] = token  # 供 logout 用 (不进响应)
    return user


# ----------------------------- 三级权限模型 -----------------------------
# 角色: super (超级管理员) > admin (管理员) > user (普通用户)
# 规则:
#   - 不能操作同级人 (super 不能改 super, admin 不能改 admin, user 不能改 user)
#   - super: 可改 admin/user 的密码; 可对 admin↔user 升降级; 不可被改/降级 (防锁死)
#   - admin: 只能改 user 的密码; 不能升降级任何人
#   - user: 不能操作他人
ROLE_LEVEL = {"super": 3, "admin": 2, "user": 1}


def role_level(role: str) -> int:
    """角色层级数字, 越大越高。未知角色按最低。"""
    return ROLE_LEVEL.get(role, 0)


def can_manage(actor: dict, target: dict) -> bool:
    """actor 能否管理 target (改密码/禁用等)? 必须严格高于 target 层级。"""
    if not actor or not target:
        return False
    if actor.get("id") == target.get("id"):
        return False  # 自己不算"管理他人"
    return role_level(actor.get("role")) > role_level(target.get("role"))


def can_change_role(actor: dict, target: dict, new_role: str) -> bool:
    """actor 能否把 target 的角色改成 new_role?

    - 仅 super 能升降级
    - super 不可被改 (target.role == super -> 拒)
    - new_role 不能是 super (super 不可被授予, 只能引导创建)
    - super 把 admin<->user 互转 OK
    """
    if not actor or not target:
        return False
    if actor.get("role") != "super":
        return False  # 只有 super 能改角色
    if target.get("role") == "super":
        return False  # super 不可被改
    if new_role == "super":
        return False  # super 不可被授予
    return new_role in ("admin", "user")


def require_staff(user: dict = Depends(current_user)) -> dict:
    """依赖: super 或 admin 可过 (管理岗), 否则 403。"""
    if role_level(user.get("role")) < role_level("admin"):
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="需要管理员权限")
    return user


def require_admin(user: dict = Depends(current_user)) -> dict:
    """依赖: super 或 admin 可过 (保留旧名兼容)。"""
    return require_staff(user)


# ----------------------------- 注册/登录/登出 -----------------------------
def register(username: str, password: str) -> dict:
    """注册新用户 (默认 user 角色)。返回用户 dict (不含密码哈希)。

    - 第一个注册的用户自动成为 super (超级管理员, 平台所有者)。
    - 之后注册的都是普通 user (要升 admin 需 super 在后台提权)。
    - 用户名 3-32 字符, 密码 >= 6 字符。
    """
    username = (username or "").strip()
    if not (3 <= len(username) <= 32):
        raise HTTPException(status_code=400, detail="用户名需 3-32 字符")
    if len(password) < 6:
        raise HTTPException(status_code=400, detail="密码至少 6 字符")
    if db.get_user_by_name(username):
        raise HTTPException(status_code=409, detail="用户名已存在")
    # 第一个用户自动 super (平台所有者); 其余默认 user
    role = "super" if db.count_users() == 0 else "user"
    uid = db.create_user(username, hash_password(password), role)
    u = db.get_user_by_id(uid)
    return _public_user(u)


def login(username: str, password: str) -> tuple[dict, str, str]:
    """登录。返回 (用户, token, expires_at)。失败 401。"""
    u = db.get_user_by_name((username or "").strip())
    if not u or not verify_password(password, u["password_hash"]):
        raise HTTPException(status_code=401, detail="用户名或密码错误")
    if not u.get("is_active"):
        raise HTTPException(status_code=403, detail="账号已被禁用")
    token = new_token()
    exp = _expires_at()
    db.create_session(u["id"], token, exp)
    return _public_user(u), token, exp


def logout(token: str) -> None:
    db.delete_session(token)


def change_password(user: dict, old_password: str, new_password: str) -> None:
    """用户自己改密码: 验证原密码, 新密码 >= 6 字符。

    改成功后不强制重登 (当前会话 token 仍有效)。
    """
    if len(new_password) < 6:
        raise HTTPException(status_code=400, detail="新密码至少 6 字符")
    u = db.get_user_by_id(user["id"])
    if not u or not verify_password(old_password, u["password_hash"]):
        raise HTTPException(status_code=401, detail="原密码错误")
    if old_password == new_password:
        raise HTTPException(status_code=400, detail="新密码不能与原密码相同")
    db.set_user_password(user["id"], hash_password(new_password))


def admin_reset_password(target_user_id: int, new_password: str) -> None:
    """管理员重置任意用户密码: 不需原密码。"""
    if len(new_password) < 6:
        raise HTTPException(status_code=400, detail="新密码至少 6 字符")
    u = db.get_user_by_id(target_user_id)
    if not u:
        raise HTTPException(status_code=404, detail="用户不存在")
    db.set_user_password(target_user_id, hash_password(new_password))


def _public_user(u: Optional[dict]) -> Optional[dict]:
    """去掉敏感字段后的用户 (给前端)。"""
    if not u:
        return None
    return {
        "id": u["id"],
        "username": u["username"],
        "role": u["role"],
        "is_active": bool(u.get("is_active")),
        "created_at": u.get("created_at"),
    }
