import ipaddress
import secrets
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, Set

import jwt
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC

from .core.exceptions import CloudBillingError


@dataclass
class SecurityPolicy:
    password_min_length: int = 12
    password_require_uppercase: bool = True
    password_require_lowercase: bool = True
    password_require_numbers: bool = True
    password_require_symbols: bool = True
    session_timeout_minutes: int = 60
    max_login_attempts: int = 5
    lockout_duration_minutes: int = 15
    require_mfa: bool = False
    allowed_ip_ranges: List[str] = field(default_factory=list)
    audit_log_retention_days: int = 90


@dataclass
class UserSession:
    user_id: str
    username: str
    roles: List[str]
    permissions: Set[str]
    created_at: datetime
    expires_at: datetime
    last_accessed: datetime
    ip_address: str
    user_agent: str
    session_token: str


@dataclass
class AuditLog:
    timestamp: datetime
    user_id: str
    action: str
    resource: str
    result: str
    details: Dict[str, Any]
    ip_address: str
    user_agent: str


class SecurityUtils:
    def __init__(
        self,
        config: Any,
        policy: Optional[SecurityPolicy] = None,
    ):
        self.config = config
        self.policy = policy or SecurityPolicy()
        self.active_sessions: Dict[str, UserSession] = {}
        self.failed_attempts: Dict[str, int] = {}
        self.locked_accounts: Dict[str, datetime] = {}
        self.audit_logs: List[AuditLog] = []
        self.encryption_key = self._generate_encryption_key()

        self._validate_policy()

    def _validate_policy(self) -> None:
        if self.policy.password_min_length < 8:
            raise CloudBillingError(
                "Password minimum length must be at least 8."
            )

        if self.policy.session_timeout_minutes <= 0:
            raise CloudBillingError(
                "Session timeout must be greater than zero."
            )

        if self.policy.max_login_attempts <= 0:
            raise CloudBillingError(
                "Maximum login attempts must be greater than zero."
            )

        if self.policy.lockout_duration_minutes <= 0:
            raise CloudBillingError(
                "Lockout duration must be greater than zero."
            )

        if self.policy.audit_log_retention_days <= 0:
            raise CloudBillingError(
                "Audit log retention must be greater than zero."
            )

    def _now(self) -> datetime:
        return datetime.now(timezone.utc)

    def _normalize_datetime(
        self,
        value: datetime,
    ) -> datetime:
        if value.tzinfo is None:
            return value.replace(tzinfo=timezone.utc)

        return value.astimezone(timezone.utc)

    def hash_password(
        self,
        password: str,
        salt: Optional[bytes] = None,
    ) -> tuple[str, bytes]:
        if not isinstance(password, str):
            raise CloudBillingError(
                "Password must be a string."
            )

        if not password:
            raise CloudBillingError(
                "Password cannot be empty."
            )

        if salt is None:
            salt = secrets.token_bytes(32)

        try:
            kdf = PBKDF2HMAC(
                algorithm=hashes.SHA256(),
                length=32,
                salt=salt,
                iterations=100000,
            )

            password_hash = kdf.derive(
                password.encode("utf-8")
            )

            return password_hash.hex(), salt

        except Exception as exc:
            raise CloudBillingError(
                f"Password hashing failed: {exc}"
            ) from exc

    def verify_password(
        self,
        password: str,
        stored_hash: str,
        salt: bytes,
    ) -> bool:
        if not isinstance(password, str):
            return False

        if not isinstance(stored_hash, str):
            return False

        if not isinstance(salt, bytes):
            return False

        try:
            kdf = PBKDF2HMAC(
                algorithm=hashes.SHA256(),
                length=32,
                salt=salt,
                iterations=100000,
            )

            kdf.verify(
                password.encode("utf-8"),
                bytes.fromhex(stored_hash),
            )

            return True

        except (ValueError, TypeError):
            return False
        except Exception:
            return False

    def validate_password_strength(
        self,
        password: str,
    ) -> tuple[bool, List[str]]:
        errors: List[str] = []

        if not isinstance(password, str):
            return False, ["Password must be a string."]

        if len(password) < self.policy.password_min_length:
            errors.append(
                "Password must be at least "
                f"{self.policy.password_min_length} characters long"
            )

        if (
            self.policy.password_require_uppercase
            and not any(char.isupper() for char in password)
        ):
            errors.append(
                "Password must contain at least one uppercase letter"
            )

        if (
            self.policy.password_require_lowercase
            and not any(char.islower() for char in password)
        ):
            errors.append(
                "Password must contain at least one lowercase letter"
            )

        if (
            self.policy.password_require_numbers
            and not any(char.isdigit() for char in password)
        ):
            errors.append(
                "Password must contain at least one number"
            )

        if (
            self.policy.password_require_symbols
            and not any(
                char in "!@#$%^&*()_+-=[]{}|;:,.<>?"
                for char in password
            )
        ):
            errors.append(
                "Password must contain at least one special character"
            )

        return len(errors) == 0, errors

    def generate_session_token(
        self,
        user_id: str,
        username: str,
        roles: List[str],
        ip_address: str,
        user_agent: str,
    ) -> str:
        if not user_id:
            raise CloudBillingError(
                "user_id is required."
            )

        if not username:
            raise CloudBillingError(
                "username is required."
            )

        if not isinstance(roles, list):
            raise CloudBillingError(
                "roles must be a list."
            )

        now = self._now()
        session_id = secrets.token_urlsafe(32)

        expires_at = now + timedelta(
            minutes=self.policy.session_timeout_minutes
        )

        permissions = self._get_permissions_for_roles(
            roles
        )

        session = UserSession(
            user_id=str(user_id),
            username=str(username),
            roles=list(roles),
            permissions=permissions,
            created_at=now,
            expires_at=expires_at,
            last_accessed=now,
            ip_address=str(ip_address or ""),
            user_agent=str(user_agent or ""),
            session_token=session_id,
        )

        self.active_sessions[session_id] = session

        payload = {
            "session_id": session_id,
            "user_id": str(user_id),
            "username": str(username),
            "roles": list(roles),
            "iat": int(now.timestamp()),
            "exp": int(expires_at.timestamp()),
        }

        token = jwt.encode(
            payload,
            self.encryption_key,
            algorithm="HS256",
        )

        self._log_audit_event(
            user_id=str(user_id),
            action="session_created",
            resource="session",
            result="success",
            details={
                "session_id": session_id,
            },
            ip_address=str(ip_address or ""),
            user_agent=str(user_agent or ""),
        )

        return token

    def validate_session_token(
        self,
        token: str,
        ip_address: str,
        user_agent: str,
    ) -> Optional[UserSession]:
        if not isinstance(token, str) or not token:
            return None

        try:
            payload = jwt.decode(
                token,
                self.encryption_key,
                algorithms=["HS256"],
            )

            session_id = payload.get("session_id")

            if not session_id:
                return None

            session = self.active_sessions.get(
                session_id
            )

            if session is None:
                return None

            now = self._now()
            expires_at = self._normalize_datetime(
                session.expires_at
            )

            if now >= expires_at:
                self._cleanup_session(
                    session_id
                )
                return None

            if self.policy.allowed_ip_ranges:
                if not self._is_ip_allowed(
                    ip_address
                ):
                    self._log_audit_event(
                        user_id=session.user_id,
                        action="session_denied",
                        resource="session",
                        result="denied",
                        details={
                            "reason": "ip_not_allowed",
                        },
                        ip_address=str(
                            ip_address or ""
                        ),
                        user_agent=str(
                            user_agent or ""
                        ),
                    )
                    return None

            session.last_accessed = now

            return session

        except jwt.ExpiredSignatureError:
            return None
        except jwt.InvalidTokenError:
            return None
        except Exception as exc:
            raise CloudBillingError(
                f"Session validation error: {exc}"
            ) from exc

    def check_permission(
        self,
        session: UserSession,
        permission: str,
    ) -> bool:
        if not session:
            return False

        if not permission:
            return False

        if "system:admin" in session.permissions:
            return True

        return permission in session.permissions

    def check_any_permission(
        self,
        session: UserSession,
        permissions: List[str],
    ) -> bool:
        if not session or not permissions:
            return False

        return any(
            self.check_permission(
                session,
                permission,
            )
            for permission in permissions
        )

    def check_all_permissions(
        self,
        session: UserSession,
        permissions: List[str],
    ) -> bool:
        if not session:
            return False

        return all(
            self.check_permission(
                session,
                permission,
            )
            for permission in permissions
        )

    def check_role(
        self,
        session: UserSession,
        role: str,
    ) -> bool:
        if not session or not role:
            return False

        if "admin" in session.roles:
            return True

        return role in session.roles

    def revoke_session(
        self,
        session_id: str,
        reason: str = "manual_revocation",
    ) -> bool:
        if session_id not in self.active_sessions:
            return False

        session = self.active_sessions[
            session_id
        ]

        self._log_audit_event(
            user_id=session.user_id,
            action="session_revoked",
            resource="session",
            result="success",
            details={
                "session_id": session_id,
                "reason": reason,
            },
            ip_address=session.ip_address,
            user_agent=session.user_agent,
        )

        del self.active_sessions[
            session_id
        ]

        return True

    def revoke_all_user_sessions(
        self,
        user_id: str,
        reason: str = "user_logout",
    ) -> int:
        session_ids = [
            session_id
            for session_id, session
            in self.active_sessions.items()
            if session.user_id == user_id
        ]

        revoked_count = 0

        for session_id in session_ids:
            if self.revoke_session(
                session_id,
                reason,
            ):
                revoked_count += 1

        return revoked_count

    def handle_failed_login(
        self,
        username: str,
        ip_address: str,
        user_agent: str,
    ) -> tuple[bool, str]:
        username = str(username).strip()

        if not username:
            return False, "Invalid username."

        now = self._now()

        if username in self.locked_accounts:
            lockout_time = self._normalize_datetime(
                self.locked_accounts[username]
            )

            if now < lockout_time:
                remaining_seconds = int(
                    (
                        lockout_time - now
                    ).total_seconds()
                )

                remaining_minutes = max(
                    1,
                    (remaining_seconds + 59) // 60,
                )

                return (
                    False,
                    "Account locked. "
                    f"Try again in {remaining_minutes} minutes.",
                )

            del self.locked_accounts[
                username
            ]

        attempts = (
            self.failed_attempts.get(
                username,
                0,
            )
            + 1
        )

        self.failed_attempts[
            username
        ] = attempts

        if attempts >= self.policy.max_login_attempts:
            lockout_time = now + timedelta(
                minutes=self.policy.lockout_duration_minutes
            )

            self.locked_accounts[
                username
            ] = lockout_time

            self._log_audit_event(
                user_id=username,
                action="account_locked",
                resource="account",
                result="failure",
                details={
                    "failed_attempts": attempts,
                    "lockout_duration_minutes": (
                        self.policy.lockout_duration_minutes
                    ),
                },
                ip_address=str(
                    ip_address or ""
                ),
                user_agent=str(
                    user_agent or ""
                ),
            )

            return (
                False,
                "Account locked due to too many "
                "failed attempts. "
                f"Try again in "
                f"{self.policy.lockout_duration_minutes} minutes.",
            )

        remaining = (
            self.policy.max_login_attempts
            - attempts
        )

        self._log_audit_event(
            user_id=username,
            action="login_failure",
            resource="account",
            result="failure",
            details={
                "failed_attempts": attempts,
                "attempts_remaining": remaining,
            },
            ip_address=str(
                ip_address or ""
            ),
            user_agent=str(
                user_agent or ""
            ),
        )

        return (
            False,
            f"Invalid credentials. "
            f"{remaining} attempts remaining.",
        )

    def handle_successful_login(
        self,
        username: str,
        user_id: str,
        ip_address: str,
        user_agent: str,
    ) -> None:
        username = str(username).strip()
        user_id = str(user_id).strip()

        self.failed_attempts.pop(
            username,
            None,
        )

        self.locked_accounts.pop(
            username,
            None,
        )

        self._log_audit_event(
            user_id=user_id,
            action="login_success",
            resource="account",
            result="success",
            details={
                "username": username,
            },
            ip_address=str(
                ip_address or ""
            ),
            user_agent=str(
                user_agent or ""
            ),
        )

    def _get_permissions_for_roles(
        self,
        roles: List[str],
    ) -> Set[str]:
        role_permissions = {
            "admin": {
                "read:all",
                "write:all",
                "delete:all",
                "manage:users",
                "manage:credentials",
                "manage:config",
                "manage:alerts",
                "manage:reports",
                "system:admin",
            },
            "operator": {
                "read:all",
                "write:costs",
                "write:alerts",
                "manage:alerts",
                "manage:reports",
                "read:credentials",
                "automation:execute",
                "automation:approve",
            },
            "viewer": {
                "read:all",
                "read:costs",
                "read:alerts",
                "read:reports",
            },
            "billing_manager": {
                "read:all",
                "write:costs",
                "write:budget",
                "manage:alerts",
                "manage:reports",
                "read:credentials",
            },
            "devops": {
                "read:all",
                "write:costs",
                "write:alerts",
                "manage:alerts",
                "manage:credentials",
                "read:config",
                "automation:execute",
            },
        }

        permissions: Set[str] = set()

        for role in roles:
            normalized_role = str(
                role
            ).strip().lower()

            permissions.update(
                role_permissions.get(
                    normalized_role,
                    set(),
                )
            )

        return permissions

    def _is_ip_allowed(
        self,
        ip_address: str,
    ) -> bool:
        if not self.policy.allowed_ip_ranges:
            return True

        try:
            ip = ipaddress.ip_address(
                ip_address
            )

            for allowed_range in (
                self.policy.allowed_ip_ranges
            ):
                allowed_range = str(
                    allowed_range
                ).strip()

                if not allowed_range:
                    continue

                if "/" in allowed_range:
                    network = ipaddress.ip_network(
                        allowed_range,
                        strict=False,
                    )

                    if ip in network:
                        return True

                elif ip == ipaddress.ip_address(
                    allowed_range
                ):
                    return True

            return False

        except ValueError:
            return False

    def _log_audit_event(
        self,
        user_id: str,
        action: str,
        resource: str,
        result: str,
        details: Dict[str, Any],
        ip_address: str,
        user_agent: str,
    ) -> None:
        audit_log = AuditLog(
            timestamp=self._now(),
            user_id=str(user_id),
            action=str(action),
            resource=str(resource),
            result=str(result),
            details=dict(details or {}),
            ip_address=str(ip_address or ""),
            user_agent=str(user_agent or ""),
        )

        self.audit_logs.append(
            audit_log
        )

        self._cleanup_audit_logs()

    def _cleanup_audit_logs(self) -> None:
        cutoff_date = (
            self._now()
            - timedelta(
                days=self.policy.audit_log_retention_days
            )
        )

        self.audit_logs = [
            log
            for log in self.audit_logs
            if self._normalize_datetime(
                log.timestamp
            ) >= cutoff_date
        ]

    def _cleanup_session(
        self,
        session_id: str,
    ) -> None:
        session = self.active_sessions.get(
            session_id
        )

        if session is None:
            return

        self._log_audit_event(
            user_id=session.user_id,
            action="session_expired",
            resource="session",
            result="success",
            details={
                "session_id": session_id,
            },
            ip_address=session.ip_address,
            user_agent=session.user_agent,
        )

        self.active_sessions.pop(
            session_id,
            None,
        )

    def _generate_encryption_key(
        self,
    ) -> bytes:
        return Fernet.generate_key()

    def get_active_sessions(
        self,
    ) -> List[UserSession]:
        now = self._now()

        expired_sessions = [
            session_id
            for session_id, session
            in self.active_sessions.items()
            if now >= self._normalize_datetime(
                session.expires_at
            )
        ]

        for session_id in expired_sessions:
            self._cleanup_session(
                session_id
            )

        return list(
            self.active_sessions.values()
        )

    def get_audit_logs(
        self,
        days: int = 30,
        user_id: Optional[str] = None,
        action: Optional[str] = None,
    ) -> List[AuditLog]:
        if days < 0:
            raise CloudBillingError(
                "days must be non-negative."
            )

        cutoff_date = (
            self._now()
            - timedelta(days=days)
        )

        logs = [
            log
            for log in self.audit_logs
            if self._normalize_datetime(
                log.timestamp
            ) >= cutoff_date
        ]

        if user_id:
            logs = [
                log
                for log in logs
                if log.user_id == user_id
            ]

        if action:
            logs = [
                log
                for log in logs
                if log.action == action
            ]

        return logs

    def get_security_metrics(
        self,
    ) -> Dict[str, Any]:
        active_sessions = (
            self.get_active_sessions()
        )

        cutoff = (
            self._now()
            - timedelta(hours=24)
        )

        recent_security_events = [
            log
            for log in self.audit_logs
            if self._normalize_datetime(
                log.timestamp
            ) >= cutoff
        ]

        successful_logins = sum(
            1
            for log in recent_security_events
            if log.action == "login_success"
        )

        failed_logins = sum(
            1
            for log in recent_security_events
            if log.action == "login_failure"
        )

        denied_events = sum(
            1
            for log in recent_security_events
            if log.result == "denied"
        )

        return {
            "active_sessions": len(
                active_sessions
            ),
            "locked_accounts": len(
                self.locked_accounts
            ),
            "failed_attempts": dict(
                self.failed_attempts
            ),
            "total_audit_logs": len(
                self.audit_logs
            ),
            "recent_logins": successful_logins,
            "recent_failed_logins": failed_logins,
            "recent_denied_events": denied_events,
            "policy": {
                "session_timeout_minutes": (
                    self.policy.session_timeout_minutes
                ),
                "max_login_attempts": (
                    self.policy.max_login_attempts
                ),
                "lockout_duration_minutes": (
                    self.policy.lockout_duration_minutes
                ),
                "require_mfa": (
                    self.policy.require_mfa
                ),
                "audit_log_retention_days": (
                    self.policy.audit_log_retention_days
                ),
            },
        }