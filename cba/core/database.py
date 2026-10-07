from __future__ import annotations

import json
import os
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Generator

from sqlalchemy import (
    Boolean,
    DateTime,
    Float,
    Integer,
    String,
    Text,
    create_engine,
    select,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, Session, mapped_column, sessionmaker


BASE_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = BASE_DIR / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)

DATABASE_URL = os.getenv(
    "CLOUDOPT_DATABASE_URL",
    f"sqlite:///{DATA_DIR / 'cloudopt.db'}",
)

engine = create_engine(
    DATABASE_URL,
    connect_args={"check_same_thread": False}
    if DATABASE_URL.startswith("sqlite")
    else {},
    pool_pre_ping=True,
)

SessionLocal = sessionmaker(
    bind=engine,
    autoflush=False,
    autocommit=False,
    expire_on_commit=False,
)


class Base(DeclarativeBase):
    pass


class CloudAccount(Base):
    __tablename__ = "cloud_accounts"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    provider: Mapped[str] = mapped_column(String(32), index=True)
    account_id: Mapped[str] = mapped_column(String(255), index=True)
    account_name: Mapped[str] = mapped_column(String(255), default="")
    region: Mapped[str] = mapped_column(String(128), default="")
    status: Mapped[str] = mapped_column(String(32), default="active")
    credential_ref: Mapped[str] = mapped_column(String(512), default="")
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
        onupdate=lambda: datetime.now(timezone.utc),
    )


class ResourceRecord(Base):
    __tablename__ = "resources"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    cloud_account_id: Mapped[int | None] = mapped_column(Integer, index=True)
    provider: Mapped[str] = mapped_column(String(32), index=True)
    resource_id: Mapped[str] = mapped_column(String(255), index=True)
    resource_name: Mapped[str] = mapped_column(String(255), default="")
    resource_type: Mapped[str] = mapped_column(String(128), default="")
    region: Mapped[str] = mapped_column(String(128), default="")
    status: Mapped[str] = mapped_column(String(64), default="running")
    cpu_utilization: Mapped[float] = mapped_column(Float, default=0.0)
    memory_utilization: Mapped[float] = mapped_column(Float, default=0.0)
    network_utilization: Mapped[float] = mapped_column(Float, default=0.0)
    monthly_cost: Mapped[float] = mapped_column(Float, default=0.0)
    metadata_json: Mapped[str] = mapped_column(Text, default="{}")
    observed_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
        index=True,
    )


class CostRecord(Base):
    __tablename__ = "cost_records"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    cloud_account_id: Mapped[int | None] = mapped_column(Integer, index=True)
    provider: Mapped[str] = mapped_column(String(32), index=True)
    service: Mapped[str] = mapped_column(String(255), default="")
    resource_id: Mapped[str] = mapped_column(String(255), default="", index=True)
    amount: Mapped[float] = mapped_column(Float, default=0.0)
    currency: Mapped[str] = mapped_column(String(16), default="USD")
    usage_quantity: Mapped[float] = mapped_column(Float, default=0.0)
    usage_unit: Mapped[str] = mapped_column(String(64), default="")
    recorded_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
        index=True,
    )
    metadata_json: Mapped[str] = mapped_column(Text, default="{}")


class DemandRecord(Base):
    __tablename__ = "demand_records"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    provider: Mapped[str] = mapped_column(String(32), default="simulation")
    metric: Mapped[str] = mapped_column(String(128), index=True)
    value: Mapped[float] = mapped_column(Float, default=0.0)
    unit: Mapped[str] = mapped_column(String(64), default="")
    source: Mapped[str] = mapped_column(String(128), default="manual")
    timestamp: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
        index=True,
    )
    metadata_json: Mapped[str] = mapped_column(Text, default="{}")


class ForecastRecord(Base):
    __tablename__ = "forecasts"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    forecast_type: Mapped[str] = mapped_column(String(64), index=True)
    provider: Mapped[str] = mapped_column(String(32), default="")
    target: Mapped[str] = mapped_column(String(255), default="")
    predicted_value: Mapped[float] = mapped_column(Float, default=0.0)
    lower_bound: Mapped[float | None] = mapped_column(Float, nullable=True)
    upper_bound: Mapped[float | None] = mapped_column(Float, nullable=True)
    confidence: Mapped[float] = mapped_column(Float, default=0.0)
    model_name: Mapped[str] = mapped_column(String(128), default="")
    model_version: Mapped[str] = mapped_column(String(64), default="")
    forecast_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
        index=True,
    )
    metadata_json: Mapped[str] = mapped_column(Text, default="{}")


class AnomalyRecord(Base):
    __tablename__ = "anomalies"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    provider: Mapped[str] = mapped_column(String(32), default="")
    resource_id: Mapped[str] = mapped_column(String(255), default="", index=True)
    metric: Mapped[str] = mapped_column(String(128), default="")
    score: Mapped[float] = mapped_column(Float, default=0.0)
    threshold: Mapped[float] = mapped_column(Float, default=0.0)
    is_anomaly: Mapped[bool] = mapped_column(Boolean, default=False)
    severity: Mapped[str] = mapped_column(String(32), default="low")
    explanation: Mapped[str] = mapped_column(Text, default="")
    detected_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
        index=True,
    )


class RecommendationRecord(Base):
    __tablename__ = "recommendations"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    provider: Mapped[str] = mapped_column(String(32), default="")
    resource_id: Mapped[str] = mapped_column(String(255), index=True)
    action: Mapped[str] = mapped_column(String(64))
    current_size: Mapped[str] = mapped_column(String(128), default="")
    recommended_size: Mapped[str] = mapped_column(String(128), default="")
    estimated_monthly_savings: Mapped[float] = mapped_column(Float, default=0.0)
    confidence: Mapped[float] = mapped_column(Float, default=0.0)
    risk_score: Mapped[float] = mapped_column(Float, default=0.0)
    risk_level: Mapped[str] = mapped_column(String(32), default="low")
    expected_utilization: Mapped[float] = mapped_column(Float, default=0.0)
    explanation: Mapped[str] = mapped_column(Text, default="")
    status: Mapped[str] = mapped_column(String(32), default="pending")
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
        index=True,
    )


class AutomationRecord(Base):
    __tablename__ = "automation_records"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    recommendation_id: Mapped[int | None] = mapped_column(Integer, nullable=True)
    provider: Mapped[str] = mapped_column(String(32), default="")
    resource_id: Mapped[str] = mapped_column(String(255), index=True)
    action: Mapped[str] = mapped_column(String(64))
    mode: Mapped[str] = mapped_column(String(32), default="dry_run")
    status: Mapped[str] = mapped_column(String(32), default="pending")
    health_status: Mapped[str] = mapped_column(String(32), default="unknown")
    rollback_available: Mapped[bool] = mapped_column(Boolean, default=False)
    rollback_state: Mapped[str] = mapped_column(Text, default="")
    message: Mapped[str] = mapped_column(Text, default="")
    executed_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
        index=True,
    )


class AuditRecord(Base):
    __tablename__ = "audit_records"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    actor: Mapped[str] = mapped_column(String(255), default="system")
    event_type: Mapped[str] = mapped_column(String(128), index=True)
    provider: Mapped[str] = mapped_column(String(32), default="")
    resource_id: Mapped[str] = mapped_column(String(255), default="", index=True)
    action: Mapped[str] = mapped_column(String(128), default="")
    status: Mapped[str] = mapped_column(String(32), default="success")
    details_json: Mapped[str] = mapped_column(Text, default="{}")
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
        index=True,
    )


def init_database() -> None:
    Base.metadata.create_all(bind=engine)


@contextmanager
def session_scope() -> Generator[Session, None, None]:
    session = SessionLocal()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


def get_session() -> Generator[Session, None, None]:
    session = SessionLocal()
    try:
        yield session
    finally:
        session.close()


def _json(value: Any) -> str:
    try:
        return json.dumps(value, default=str)
    except Exception:
        return "{}"


def create_cloud_account(
    provider: str,
    account_id: str,
    account_name: str = "",
    region: str = "",
    credential_ref: str = "",
) -> CloudAccount:
    with session_scope() as session:
        account = CloudAccount(
            provider=provider,
            account_id=account_id,
            account_name=account_name,
            region=region,
            credential_ref=credential_ref,
        )
        session.add(account)
        session.flush()
        session.refresh(account)
        return account


def get_cloud_accounts(provider: str | None = None) -> list[CloudAccount]:
    with SessionLocal() as session:
        stmt = select(CloudAccount).order_by(CloudAccount.created_at.desc())
        if provider:
            stmt = stmt.where(CloudAccount.provider == provider)
        return list(session.scalars(stmt).all())


def save_resource(data: dict[str, Any]) -> ResourceRecord:
    with session_scope() as session:
        record = ResourceRecord(
            cloud_account_id=data.get("cloud_account_id"),
            provider=str(data.get("provider", "")),
            resource_id=str(data.get("resource_id", "")),
            resource_name=str(data.get("resource_name", "")),
            resource_type=str(data.get("resource_type", "")),
            region=str(data.get("region", "")),
            status=str(data.get("status", "running")),
            cpu_utilization=float(data.get("cpu_utilization", 0.0)),
            memory_utilization=float(data.get("memory_utilization", 0.0)),
            network_utilization=float(data.get("network_utilization", 0.0)),
            monthly_cost=float(data.get("monthly_cost", 0.0)),
            metadata_json=_json(data.get("metadata", {})),
        )
        session.add(record)
        session.flush()
        session.refresh(record)
        return record


def save_cost(data: dict[str, Any]) -> CostRecord:
    with session_scope() as session:
        record = CostRecord(
            cloud_account_id=data.get("cloud_account_id"),
            provider=str(data.get("provider", "")),
            service=str(data.get("service", "")),
            resource_id=str(data.get("resource_id", "")),
            amount=float(data.get("amount", 0.0)),
            currency=str(data.get("currency", "USD")),
            usage_quantity=float(data.get("usage_quantity", 0.0)),
            usage_unit=str(data.get("usage_unit", "")),
            metadata_json=_json(data.get("metadata", {})),
        )
        session.add(record)
        session.flush()
        session.refresh(record)
        return record


def save_demand(
    metric: str,
    value: float,
    timestamp: datetime | None = None,
    provider: str = "simulation",
    unit: str = "",
    source: str = "manual",
    metadata: dict[str, Any] | None = None,
) -> DemandRecord:
    with session_scope() as session:
        record = DemandRecord(
            provider=provider,
            metric=metric,
            value=float(value),
            unit=unit,
            source=source,
            timestamp=timestamp or datetime.now(timezone.utc),
            metadata_json=_json(metadata or {}),
        )
        session.add(record)
        session.flush()
        session.refresh(record)
        return record


def get_demand_history(
    metric: str,
    provider: str | None = None,
    limit: int = 1000,
) -> list[DemandRecord]:
    with SessionLocal() as session:
        stmt = (
            select(DemandRecord)
            .where(DemandRecord.metric == metric)
            .order_by(DemandRecord.timestamp.desc())
            .limit(limit)
        )
        if provider:
            stmt = stmt.where(DemandRecord.provider == provider)
        return list(session.scalars(stmt).all())[::-1]


def save_forecast(data: dict[str, Any]) -> ForecastRecord:
    with session_scope() as session:
        record = ForecastRecord(
            forecast_type=str(data.get("forecast_type", "")),
            provider=str(data.get("provider", "")),
            target=str(data.get("target", "")),
            predicted_value=float(data.get("predicted_value", 0.0)),
            lower_bound=data.get("lower_bound"),
            upper_bound=data.get("upper_bound"),
            confidence=float(data.get("confidence", 0.0)),
            model_name=str(data.get("model_name", "")),
            model_version=str(data.get("model_version", "")),
            metadata_json=_json(data.get("metadata", {})),
        )
        session.add(record)
        session.flush()
        session.refresh(record)
        return record


def save_anomaly(data: dict[str, Any]) -> AnomalyRecord:
    with session_scope() as session:
        record = AnomalyRecord(
            provider=str(data.get("provider", "")),
            resource_id=str(data.get("resource_id", "")),
            metric=str(data.get("metric", "")),
            score=float(data.get("score", 0.0)),
            threshold=float(data.get("threshold", 0.0)),
            is_anomaly=bool(data.get("is_anomaly", False)),
            severity=str(data.get("severity", "low")),
            explanation=str(data.get("explanation", "")),
        )
        session.add(record)
        session.flush()
        session.refresh(record)
        return record


def save_recommendation(data: dict[str, Any]) -> RecommendationRecord:
    with session_scope() as session:
        record = RecommendationRecord(
            provider=str(data.get("provider", "")),
            resource_id=str(data.get("resource_id", "")),
            action=str(data.get("action", "")),
            current_size=str(data.get("current_size", "")),
            recommended_size=str(data.get("recommended_size", "")),
            estimated_monthly_savings=float(
                data.get("estimated_monthly_savings", 0.0)
            ),
            confidence=float(data.get("confidence", 0.0)),
            risk_score=float(data.get("risk_score", 0.0)),
            risk_level=str(data.get("risk_level", "low")),
            expected_utilization=float(data.get("expected_utilization", 0.0)),
            explanation=str(data.get("explanation", "")),
            status=str(data.get("status", "pending")),
        )
        session.add(record)
        session.flush()
        session.refresh(record)
        return record


def save_automation(data: dict[str, Any]) -> AutomationRecord:
    with session_scope() as session:
        record = AutomationRecord(
            recommendation_id=data.get("recommendation_id"),
            provider=str(data.get("provider", "")),
            resource_id=str(data.get("resource_id", "")),
            action=str(data.get("action", "")),
            mode=str(data.get("mode", "dry_run")),
            status=str(data.get("status", "pending")),
            health_status=str(data.get("health_status", "unknown")),
            rollback_available=bool(data.get("rollback_available", False)),
            rollback_state=_json(data.get("rollback_state", {})),
            message=str(data.get("message", "")),
        )
        session.add(record)
        session.flush()
        session.refresh(record)
        return record


def save_audit(
    event_type: str,
    actor: str = "system",
    provider: str = "",
    resource_id: str = "",
    action: str = "",
    status: str = "success",
    details: dict[str, Any] | None = None,
) -> AuditRecord:
    with session_scope() as session:
        record = AuditRecord(
            actor=actor,
            event_type=event_type,
            provider=provider,
            resource_id=resource_id,
            action=action,
            status=status,
            details_json=_json(details or {}),
        )
        session.add(record)
        session.flush()
        session.refresh(record)
        return record


def get_recent_audit(limit: int = 100) -> list[AuditRecord]:
    with SessionLocal() as session:
        stmt = (
            select(AuditRecord)
            .order_by(AuditRecord.created_at.desc())
            .limit(limit)
        )
        return list(session.scalars(stmt).all())


def get_recent_recommendations(limit: int = 100) -> list[RecommendationRecord]:
    with SessionLocal() as session:
        stmt = (
            select(RecommendationRecord)
            .order_by(RecommendationRecord.created_at.desc())
            .limit(limit)
        )
        return list(session.scalars(stmt).all())


def get_recent_automation(limit: int = 100) -> list[AutomationRecord]:
    with SessionLocal() as session:
        stmt = (
            select(AutomationRecord)
            .order_by(AutomationRecord.executed_at.desc())
            .limit(limit)
        )
        return list(session.scalars(stmt).all())


def database_health() -> dict[str, Any]:
    try:
        with SessionLocal() as session:
            session.execute(select(CloudAccount).limit(1))
        return {
            "status": "healthy",
            "database": DATABASE_URL.split("://", 1)[0],
        }
    except Exception as exc:
        return {
            "status": "unhealthy",
            "database": DATABASE_URL.split("://", 1)[0],
            "error": str(exc),
        }


init_database()
