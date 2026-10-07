"""
CloudOpt AI — Enterprise Gmail Authentication & Real-Time Email Alert Engine.

Dispatches instant email notifications to the signed-in Gmail account whenever:
1. A cloud account is authenticated / connected (AWS, Azure, GCP).
2. An AI optimization recommendation is approved and dispatched.
3. Post-optimization health surveillance triggers a rollback or permanent commit.
"""
from __future__ import annotations

import json
import os
import smtplib
import uuid
from datetime import datetime, timezone
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from pathlib import Path
from typing import Any, Dict, List, Optional

ROOT = Path(__file__).resolve().parents[2]
LIVE_DIR = ROOT / "outputs" / "live"
LIVE_DIR.mkdir(parents=True, exist_ok=True)
EMAILS_DIR = LIVE_DIR / "emails"
EMAILS_DIR.mkdir(parents=True, exist_ok=True)
SESSION_FILE = LIVE_DIR / "user_session.json"
ALERTS_LOG = LIVE_DIR / "email_alerts.json"

if not ALERTS_LOG.exists():
    ALERTS_LOG.write_text("[]", encoding="utf-8")


SMTP_CONFIG_FILE = LIVE_DIR / "smtp_config.json"


class GmailAlertService:
    """Manages Gmail authentication session and automated email dispatching."""

    def __init__(self):
        self._ensure_files()

    def _ensure_files(self) -> None:
        if not ALERTS_LOG.exists():
            ALERTS_LOG.write_text("[]", encoding="utf-8")

    # =========================================================================
    # User Session Management (Email-Only Input)
    # =========================================================================
    def get_session(self) -> Dict[str, Any]:
        """Returns the current active user session."""
        if SESSION_FILE.exists():
            try:
                data = json.loads(SESSION_FILE.read_text(encoding="utf-8"))
                if data.get("logged_in") and data.get("email"):
                    sanitized = dict(data)
                    sanitized["has_password"] = bool(data.get("password"))
                    sanitized.pop("password", None)
                    return sanitized
            except Exception:
                pass
        return {
            "logged_in": False,
            "email": "",
            "name": "",
            "avatar_url": "",
            "has_password": False,
            "is_verified_google": False,
            "logged_in_at": None,
        }

    def set_session(
        self,
        email: str,
        name: Optional[str] = None,
        password: Optional[str] = None,
        avatar_url: Optional[str] = None,
        is_verified_google: bool = False,
    ) -> Dict[str, Any]:
        """Stores the active Gmail login session (only requires email address)."""
        email_clean = (email or "").strip().lower()
        display_name = (name or "").strip() or email_clean.split("@")[0].capitalize()
        session_data = {
            "logged_in": True,
            "email": email_clean,
            "name": display_name,
            "password": (password or "").strip(),
            "is_verified_google": is_verified_google,
            "avatar_url": avatar_url or f"https://www.gravatar.com/avatar/{abs(hash(email_clean))}?d=mp",
            "logged_in_at": datetime.now(timezone.utc).isoformat(),
        }
        SESSION_FILE.write_text(json.dumps(session_data, indent=2), encoding="utf-8")

        sanitized = dict(session_data)
        sanitized["has_password"] = bool(session_data["password"])
        sanitized.pop("password", None)
        return sanitized

    def clear_session(self) -> None:
        """Clears the active session."""
        if SESSION_FILE.exists():
            SESSION_FILE.unlink(missing_ok=True)

    def verify_gmail_credentials(self, email: str, password: str) -> tuple[bool, str]:
        """Tests authentication directly against Google's live smtp.gmail.com:587 server."""
        clean_email = (email or "").strip().lower()
        clean_pass = (password or "").strip()
        if not clean_email or "@" not in clean_email:
            return False, "A valid Gmail email address is required."
        if not clean_pass:
            return False, "Password cannot be empty."

        try:
            with smtplib.SMTP("smtp.gmail.com", 587, timeout=10) as server:
                server.starttls()
                server.login(clean_email, clean_pass)
            return True, "Authenticated successfully with Google."
        except smtplib.SMTPAuthenticationError as sae:
            err_text = str(sae)
            if "BadCredentials" in err_text or "535" in err_text:
                return False, (
                    "Google Authentication Failed: Invalid password. "
                    "Google Accounts with 2-Step Verification require a 16-character App Password. "
                    "Generate one at: myaccount.google.com ➔ Security ➔ 2-Step Verification ➔ App passwords."
                )
            return False, f"Google authentication failed: {sae}"
        except Exception as exc:
            return False, f"Could not connect to Google SMTP server: {exc}"

    # =========================================================================
    # Outbound SMTP Configuration
    # =========================================================================
    def get_smtp_config(self) -> Dict[str, Any]:
        """Returns the current SMTP outbound sender configuration."""
        config = {
            "smtp_host": os.getenv("SMTP_HOST", "smtp.gmail.com"),
            "smtp_port": int(os.getenv("SMTP_PORT", 587)),
            "smtp_user": os.getenv("GMAIL_USER") or os.getenv("SMTP_USERNAME") or "",
            "is_configured": False,
        }
        if SMTP_CONFIG_FILE.exists():
            try:
                data = json.loads(SMTP_CONFIG_FILE.read_text(encoding="utf-8"))
                if data.get("smtp_host"):
                    config["smtp_host"] = data["smtp_host"]
                if data.get("smtp_port"):
                    config["smtp_port"] = int(data["smtp_port"])
                if data.get("smtp_user"):
                    config["smtp_user"] = data["smtp_user"]
                if data.get("smtp_pass"):
                    config["is_configured"] = True
            except Exception:
                pass
        if os.getenv("GMAIL_APP_PASSWORD") or os.getenv("SMTP_PASSWORD"):
            config["is_configured"] = True
        return config

    def save_smtp_config(
        self,
        smtp_user: str,
        smtp_pass: str,
        smtp_host: str = "smtp.gmail.com",
        smtp_port: int = 587,
    ) -> tuple[bool, str]:
        """Saves and verifies outbound SMTP sender credentials for live email delivery."""
        clean_user = (smtp_user or "").strip()
        clean_pass = (smtp_pass or "").strip()
        clean_host = (smtp_host or "").strip() or "smtp.gmail.com"
        port = int(smtp_port or 587)

        if not clean_user or not clean_pass:
            return False, "Sender Email and 16-character Google App Password are required."

        try:
            with smtplib.SMTP(clean_host, port, timeout=10) as server:
                server.starttls()
                server.login(clean_user, clean_pass)
        except smtplib.SMTPAuthenticationError as sae:
            return False, f"Google SMTP authentication failed (535 Bad Credentials). Use a 16-character Google App Password: {sae}"
        except Exception as exc:
            return False, f"Could not connect to SMTP server {clean_host}:{port}: {exc}"

        data = {
            "smtp_user": clean_user,
            "smtp_pass": clean_pass,
            "smtp_host": clean_host,
            "smtp_port": port,
            "updated_at": datetime.now(timezone.utc).isoformat(),
        }
        SMTP_CONFIG_FILE.write_text(json.dumps(data, indent=2), encoding="utf-8")
        return True, f"Outbound SMTP sender '{clean_user}' verified & active! All alerts will dispatch live to your Gmail inbox."

    # =========================================================================
    # Email Delivery Engine
    # =========================================================================
    def send_email(
        self,
        to_email: str,
        subject: str,
        html_body: str,
        plain_body: Optional[str] = None,
        event_type: str = "GENERIC_ALERT",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Dispatches an email alert to the user's Gmail ID.
        
        Attempts native SMTP if outbound sender credentials exist; always generates full HTML artifact and audit trail.
        """
        self._ensure_files()
        alert_id = f"eml-{uuid.uuid4().hex[:10]}"
        now = datetime.now(timezone.utc).isoformat()
        clean_to = (to_email or "").strip().lower()

        sess_raw: Dict[str, Any] = {}
        if SESSION_FILE.exists():
            try:
                sess_raw = json.loads(SESSION_FILE.read_text(encoding="utf-8"))
            except Exception:
                pass

        if not clean_to:
            clean_to = sess_raw.get("email", "")

        if not clean_to:
            return {
                "success": False,
                "error": "No recipient email address provided or logged in.",
                "alert_id": alert_id,
            }

        # Check SMTP configuration
        smtp_user = os.getenv("GMAIL_USER") or os.getenv("SMTP_USERNAME") or ""
        smtp_pass = os.getenv("GMAIL_APP_PASSWORD") or os.getenv("SMTP_PASSWORD") or ""
        smtp_host = os.getenv("SMTP_HOST") or "smtp.gmail.com"
        smtp_port = int(os.getenv("SMTP_PORT") or 587)

        if SMTP_CONFIG_FILE.exists():
            try:
                cfg = json.loads(SMTP_CONFIG_FILE.read_text(encoding="utf-8"))
                if cfg.get("smtp_user"):
                    smtp_user = cfg["smtp_user"]
                if cfg.get("smtp_pass"):
                    smtp_pass = cfg["smtp_pass"]
                if cfg.get("smtp_host"):
                    smtp_host = cfg["smtp_host"]
                if cfg.get("smtp_port"):
                    smtp_port = int(cfg["smtp_port"])
            except Exception:
                pass

        if not smtp_pass and sess_raw.get("password"):
            smtp_pass = sess_raw["password"]
            if not smtp_user:
                smtp_user = clean_to

        smtp_sent = False
        smtp_detail = ""

        if smtp_pass and smtp_user:
            try:
                msg = MIMEMultipart("alternative")
                msg["From"] = f"CloudOpt AI Alerts <{smtp_user}>"
                msg["To"] = clean_to
                msg["Subject"] = subject
                msg.attach(MIMEText(plain_body or subject, "plain"))
                msg.attach(MIMEText(html_body, "html"))

                with smtplib.SMTP(smtp_host, smtp_port, timeout=10) as server:
                    server.starttls()
                    server.login(smtp_user, smtp_pass)
                    server.send_message(msg)
                smtp_sent = True
                smtp_detail = f"Dispatched live via Google SMTP ({smtp_host}:{smtp_port}) directly to {clean_to} inbox."
            except Exception as exc:
                smtp_detail = f"Google SMTP delivery attempt failed ({exc}); recorded in audit log & HTML preview."
        else:
            smtp_detail = f"Recorded in local audit log & HTML preview. To receive directly in your Gmail inbox, configure Google App Password in Live In-Box Delivery settings."

        # Save HTML email artifact for immediate preview
        email_file = EMAILS_DIR / f"{alert_id}.html"
        email_file.write_text(html_body, encoding="utf-8")

        record = {
            "alert_id": alert_id,
            "timestamp": now,
            "to_email": clean_to,
            "subject": subject,
            "event_type": event_type,
            "delivery_status": "SENT_SMTP" if smtp_sent else "RECORDED_IN_AUDIT_LOG",
            "delivery_note": smtp_detail,
            "preview_url": f"/api/notifications/preview/{alert_id}",
            "metadata": metadata or {},
        }

        # Append to alert log
        try:
            records = json.loads(ALERTS_LOG.read_text(encoding="utf-8"))
        except Exception:
            records = []
        records.insert(0, record)
        records = records[:100]  # Keep last 100 alerts
        ALERTS_LOG.write_text(json.dumps(records, indent=2), encoding="utf-8")

        return {
            "success": True,
            "alert_id": alert_id,
            "to_email": clean_to,
            "subject": subject,
            "delivery_status": record["delivery_status"],
            "delivery_note": smtp_detail,
            "timestamp": now,
        }

    def notify_test_alert(self, to_email: Optional[str] = None) -> Dict[str, Any]:
        """Dispatches an interactive test alert to verify delivery to the signed-in Gmail."""
        target_email = to_email or self.get_session().get("email")
        if not target_email:
            return {"success": False, "error": "No recipient email address found"}

        now_str = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
        subject = f"[CloudOpt AI Alert] Live Verification Test Alert for {target_email}"
        html = f"""<!DOCTYPE html>
<html>
<head>
  <meta charset="utf-8">
  <title>{subject}</title>
  <style>
    body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; background: #0c0c0e; color: #f2f2f5; margin: 0; padding: 24px; }}
    .card {{ background: #151518; border: 1px solid #282830; border-radius: 8px; max-width: 580px; margin: 0 auto; padding: 28px; }}
    .header {{ border-bottom: 1px solid #282830; padding-bottom: 16px; margin-bottom: 20px; }}
    .title {{ font-size: 18px; font-weight: 700; color: #ffffff; }}
    .badge {{ display: inline-block; background: #22c55e; color: #0c0c0e; font-size: 11px; font-weight: 700; padding: 3px 8px; border-radius: 4px; text-transform: uppercase; }}
    .detail-row {{ display: flex; justify-content: space-between; padding: 8px 0; border-bottom: 1px solid #1e1e24; font-size: 13px; }}
    .detail-label {{ color: #8c8c9a; }}
    .detail-val {{ color: #ffffff; font-family: 'JetBrains Mono', Consolas, monospace; font-weight: 600; }}
  </style>
</head>
<body>
  <div class="card">
    <div class="header">
      <span class="badge">VERIFICATION TEST ALERT</span>
      <div class="title" style="margin-top: 10px;">CloudOpt AI Real-Time Alert Engine</div>
    </div>
    <p style="font-size: 13px; color: #a1a1aa; line-height: 1.5; margin-bottom: 18px;">
      This email verifies that your signed-in Gmail address is receiving real-time infrastructure alerts from CloudOpt AI.
    </p>
    <div class="detail-row">
      <span class="detail-label">Recipient Gmail</span>
      <span class="detail-val">{target_email}</span>
    </div>
    <div class="detail-row">
      <span class="detail-label">Dispatched At</span>
      <span class="detail-val">{now_str}</span>
    </div>
    <div class="detail-row">
      <span class="detail-label">Delivery Channel</span>
      <span class="detail-val">Native CloudOpt AI Dispatch Engine</span>
    </div>
  </div>
</body>
</html>"""
        return self.send_email(
            to_email=target_email,
            subject=subject,
            html_body=html,
            plain_body=f"CloudOpt AI Verification Test Alert delivered to {target_email} at {now_str}.",
            event_type="VERIFICATION_TEST",
            metadata={"test": True, "recipient": target_email},
        )

    # =========================================================================
    # Standard Event Notifiers
    # =========================================================================
    def notify_cloud_account_connected(
        self,
        provider: str,
        account_id: str,
        arn_or_identity: str,
        region: str,
        to_email: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Dispatches an alert when a real cloud account connects successfully."""
        prov_upper = (provider or "AWS").upper()
        target_email = to_email or self.get_session().get("email")
        if not target_email:
            return {"success": False, "error": "No logged in user"}

        subject = f"[CloudOpt AI] Cloud Infrastructure Connected: {prov_upper} ({account_id or 'Account'})"
        now_str = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")

        html = f"""<!DOCTYPE html>
<html>
<head>
  <meta charset="utf-8">
  <title>{subject}</title>
  <style>
    body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; background: #0c0c0e; color: #f2f2f5; margin: 0; padding: 24px; }}
    .card {{ background: #151518; border: 1px solid #282830; border-radius: 8px; max-width: 580px; margin: 0 auto; padding: 28px; }}
    .header {{ border-bottom: 1px solid #282830; padding-bottom: 16px; margin-bottom: 20px; }}
    .title {{ font-size: 18px; font-weight: 700; color: #ffffff; letter-spacing: -0.3px; }}
    .badge {{ display: inline-block; background: #ffffff; color: #0c0c0e; font-size: 11px; font-weight: 700; padding: 3px 8px; border-radius: 4px; text-transform: uppercase; }}
    .detail-row {{ display: flex; justify-content: space-between; padding: 8px 0; border-bottom: 1px solid #1e1e24; font-size: 13px; }}
    .detail-label {{ color: #8c8c9a; }}
    .detail-val {{ color: #ffffff; font-family: 'JetBrains Mono', Consolas, monospace; font-weight: 600; }}
    .footer {{ margin-top: 24px; padding-top: 14px; font-size: 11px; color: #5a5a68; border-top: 1px solid #282830; }}
  </style>
</head>
<body>
  <div class="card">
    <div class="header">
      <span class="badge">{prov_upper} CONNECTED</span>
      <div class="title" style="margin-top: 10px;">Cloud Infrastructure Authenticated</div>
    </div>
    <p style="font-size: 13px; color: #a1a1aa; line-height: 1.5; margin-bottom: 18px;">
      Your CloudOpt AI platform has successfully authenticated a live cloud provider. Real-time metrics collection and AI rightsizing surveillance are now active.
    </p>
    <div class="detail-row">
      <span class="detail-label">Provider</span>
      <span class="detail-val">{prov_upper}</span>
    </div>
    <div class="detail-row">
      <span class="detail-label">Account ID</span>
      <span class="detail-val">{account_id}</span>
    </div>
    <div class="detail-row">
      <span class="detail-label">Authenticated ARN / Identity</span>
      <span class="detail-val" style="font-size: 11px;">{arn_or_identity}</span>
    </div>
    <div class="detail-row">
      <span class="detail-label">Primary Region / Zone</span>
      <span class="detail-val">{region}</span>
    </div>
    <div class="detail-row">
      <span class="detail-label">Timestamp</span>
      <span class="detail-val">{now_str}</span>
    </div>
    <div class="footer">
      CloudOpt AI Autonomous Multi-Cloud FinOps & Resource Intelligence Engine • Zero Synthetic Data
    </div>
  </div>
</body>
</html>"""
        return self.send_email(
            to_email=target_email,
            subject=subject,
            html_body=html,
            plain_body=f"CloudOpt AI: {prov_upper} Account {account_id} successfully authenticated ({arn_or_identity}) in {region} at {now_str}.",
            event_type="CLOUD_ACCOUNT_CONNECTED",
            metadata={"provider": provider, "account_id": account_id, "arn": arn_or_identity, "region": region},
        )

    def notify_optimization_dispatched(
        self,
        resource_id: str,
        action: str,
        current_sku: str,
        target_sku: str,
        monthly_savings: float,
        environment: str = "production",
        risk_level: str = "low",
        action_id: Optional[str] = None,
        to_email: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Dispatches an email alert when an AI right-sizing action is queued/executed."""
        target_email = to_email or self.get_session().get("email")
        if not target_email:
            return {"success": False, "error": "No logged in user"}

        act_upper = (action or "scale_down").replace("_", " ").upper()
        env_upper = (environment or "production").upper()
        savings_str = f"${float(monthly_savings or 0):.2f}/mo"
        now_str = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
        subject = f"[CloudOpt AI Alert] Optimization Dispatched: {resource_id} ({current_sku} ➔ {target_sku})"

        html = f"""<!DOCTYPE html>
<html>
<head>
  <meta charset="utf-8">
  <title>{subject}</title>
  <style>
    body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; background: #0c0c0e; color: #f2f2f5; margin: 0; padding: 24px; }}
    .card {{ background: #151518; border: 1px solid #282830; border-radius: 8px; max-width: 580px; margin: 0 auto; padding: 28px; }}
    .header {{ border-bottom: 1px solid #282830; padding-bottom: 16px; margin-bottom: 20px; }}
    .title {{ font-size: 18px; font-weight: 700; color: #ffffff; letter-spacing: -0.3px; }}
    .badge {{ display: inline-block; background: #ffffff; color: #0c0c0e; font-size: 11px; font-weight: 700; padding: 3px 8px; border-radius: 4px; text-transform: uppercase; }}
    .detail-row {{ display: flex; justify-content: space-between; padding: 8px 0; border-bottom: 1px solid #1e1e24; font-size: 13px; }}
    .detail-label {{ color: #8c8c9a; }}
    .detail-val {{ color: #ffffff; font-family: 'JetBrains Mono', Consolas, monospace; font-weight: 600; }}
    .highlight-box {{ background: #1c1c22; border: 1px solid #2e2e38; border-radius: 6px; padding: 14px; margin: 16px 0; display: flex; justify-content: space-around; text-align: center; }}
    .footer {{ margin-top: 24px; padding-top: 14px; font-size: 11px; color: #5a5a68; border-top: 1px solid #282830; }}
  </style>
</head>
<body>
  <div class="card">
    <div class="header">
      <span class="badge">{act_upper} • {env_upper}</span>
      <div class="title" style="margin-top: 10px;">AI Right-Sizing Optimization Dispatched</div>
    </div>
    <p style="font-size: 13px; color: #a1a1aa; line-height: 1.5; margin-bottom: 14px;">
      An autonomous optimization job has been approved and dispatched to the native cloud queue. The post-optimization health surveillance engine is actively observing workload metrics.
    </p>
    <div class="highlight-box">
      <div>
        <div style="font-size: 10px; color: #8c8c9a; text-transform: uppercase;">Current SKU</div>
        <div style="font-size: 15px; font-weight: 700; color: #ffffff; font-family: monospace; margin-top: 4px;">{current_sku}</div>
      </div>
      <div style="font-size: 18px; color: #8c8c9a; align-self: center;">➔</div>
      <div>
        <div style="font-size: 10px; color: #8c8c9a; text-transform: uppercase;">Target SKU</div>
        <div style="font-size: 15px; font-weight: 700; color: #ffffff; font-family: monospace; margin-top: 4px;">{target_sku}</div>
      </div>
      <div>
        <div style="font-size: 10px; color: #8c8c9a; text-transform: uppercase;">Est. Savings</div>
        <div style="font-size: 15px; font-weight: 700; color: #ffffff; font-family: monospace; margin-top: 4px;">{savings_str}</div>
      </div>
    </div>
    <div class="detail-row">
      <span class="detail-label">Resource ID</span>
      <span class="detail-val">{resource_id}</span>
    </div>
    <div class="detail-row">
      <span class="detail-label">Environment Tier</span>
      <span class="detail-val">{env_upper}</span>
    </div>
    <div class="detail-row">
      <span class="detail-label">Risk Classification</span>
      <span class="detail-val">{risk_level.upper()}</span>
    </div>
    <div class="detail-row">
      <span class="detail-label">Action Identifier</span>
      <span class="detail-val">{action_id or 'opt-action'}</span>
    </div>
    <div class="detail-row">
      <span class="detail-label">Observation Window</span>
      <span class="detail-val">10 Minutes (Auto-Rollback if CPU > 85%)</span>
    </div>
    <div class="detail-row">
      <span class="detail-label">Dispatched At</span>
      <span class="detail-val">{now_str}</span>
    </div>
    <div class="footer">
      CloudOpt AI Autonomous Multi-Cloud FinOps & Resource Intelligence Engine • Zero Synthetic Data
    </div>
  </div>
</body>
</html>"""
        return self.send_email(
            to_email=target_email,
            subject=subject,
            html_body=html,
            plain_body=f"CloudOpt AI Alert: Optimization dispatched for {resource_id} ({current_sku} -> {target_sku}) with {savings_str} in {env_upper} at {now_str}.",
            event_type="OPTIMIZATION_DISPATCHED",
            metadata={
                "resource_id": resource_id,
                "action": action,
                "current_sku": current_sku,
                "target_sku": target_sku,
                "monthly_savings": monthly_savings,
                "environment": environment,
                "risk_level": risk_level,
                "action_id": action_id,
            },
        )

    def notify_rollback_alert(
        self,
        action_id: str,
        resource_id: str,
        status: str,
        reason: str,
        to_email: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Dispatches an email alert when a rollback is triggered or permanent commit is finalized."""
        target_email = to_email or self.get_session().get("email")
        if not target_email:
            return {"success": False, "error": "No logged in user"}

        stat_upper = (status or "ROLLED_BACK").upper()
        now_str = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
        subject = f"[CloudOpt AI Alert] Health Surveillance: {stat_upper} for {resource_id}"

        html = f"""<!DOCTYPE html>
<html>
<head>
  <meta charset="utf-8">
  <title>{subject}</title>
  <style>
    body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; background: #0c0c0e; color: #f2f2f5; margin: 0; padding: 24px; }}
    .card {{ background: #151518; border: 1px solid #282830; border-radius: 8px; max-width: 580px; margin: 0 auto; padding: 28px; }}
    .header {{ border-bottom: 1px solid #282830; padding-bottom: 16px; margin-bottom: 20px; }}
    .title {{ font-size: 18px; font-weight: 700; color: #ffffff; letter-spacing: -0.3px; }}
    .badge {{ display: inline-block; background: #ffffff; color: #0c0c0e; font-size: 11px; font-weight: 700; padding: 3px 8px; border-radius: 4px; text-transform: uppercase; }}
    .detail-row {{ display: flex; justify-content: space-between; padding: 8px 0; border-bottom: 1px solid #1e1e24; font-size: 13px; }}
    .detail-label {{ color: #8c8c9a; }}
    .detail-val {{ color: #ffffff; font-family: 'JetBrains Mono', Consolas, monospace; font-weight: 600; }}
    .footer {{ margin-top: 24px; padding-top: 14px; font-size: 11px; color: #5a5a68; border-top: 1px solid #282830; }}
  </style>
</head>
<body>
  <div class="card">
    <div class="header">
      <span class="badge">{stat_upper}</span>
      <div class="title" style="margin-top: 10px;">Post-Optimization Health Check Status</div>
    </div>
    <p style="font-size: 13px; color: #a1a1aa; line-height: 1.5; margin-bottom: 16px;">
      {reason}
    </p>
    <div class="detail-row">
      <span class="detail-label">Action ID</span>
      <span class="detail-val">{action_id}</span>
    </div>
    <div class="detail-row">
      <span class="detail-label">Resource ID</span>
      <span class="detail-val">{resource_id}</span>
    </div>
    <div class="detail-row">
      <span class="detail-label">Status</span>
      <span class="detail-val">{stat_upper}</span>
    </div>
    <div class="detail-row">
      <span class="detail-label">Timestamp</span>
      <span class="detail-val">{now_str}</span>
    </div>
    <div class="footer">
      CloudOpt AI Autonomous Multi-Cloud FinOps & Resource Intelligence Engine • Zero Synthetic Data
    </div>
  </div>
</body>
</html>"""
        return self.send_email(
            to_email=target_email,
            subject=subject,
            html_body=html,
            plain_body=f"CloudOpt AI Alert: {stat_upper} for {resource_id} ({action_id}). Reason: {reason} at {now_str}.",
            event_type=f"HEALTH_STATUS_{stat_upper}",
            metadata={"action_id": action_id, "resource_id": resource_id, "status": status, "reason": reason},
        )

    def get_history(self, limit: int = 50) -> List[Dict[str, Any]]:
        """Returns recent email alert history."""
        self._ensure_files()
        try:
            records = json.loads(ALERTS_LOG.read_text(encoding="utf-8"))
            return records[:limit]
        except Exception:
            return []


# Global singleton instance
notifier = GmailAlertService()
