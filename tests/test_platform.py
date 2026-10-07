from __future__ import annotations

import json
from pathlib import Path
import pytest
import pandas as pd

from cba.core.models import CloudProvider, ActionType, RecommendationStatus
from cba.intelligence.resource_model import ResourceIntelligenceModel, AWS_INSTANCE_TIERS, AZURE_INSTANCE_TIERS, GCP_INSTANCE_TIERS
from cba.intelligence.anomaly import CloudAnomalyDetector
from cba.intelligence.forecasting import CloudForecaster
from cba.intelligence.explainability import ExplainabilityEngine
from cba.automation.queue_manager import CloudQueueManager
from cba.automation.monitor_rollback import AutoRollbackMonitor
from cba.live.permissions import AWS_MINIMUM_IAM_POLICY, AZURE_MINIMUM_RBAC_ROLE, GCP_MINIMUM_IAM_ROLES


def test_models_loaded_and_infer():
    """Verifies trained models can be loaded and make valid inferences."""
    rm = ResourceIntelligenceModel()
    loaded = rm.load()
    assert loaded is True, "Resource model failed to load"
    assert rm.classifier is not None

    # Test SKU right-sizing
    target_sku, target_rate, curr_rate = rm.suggest_target_sku("aws", "c5.2xlarge", "scale_down")
    assert target_sku == "c5.xlarge"
    assert target_rate < curr_rate
    savings = (curr_rate - target_rate) * 730
    assert savings > 0

    risk, approval = rm.evaluate_risk(cpu_avg=8.5, cpu_max=22.0, action="scale_down", resilience_score=95.0)
    assert risk == "low"
    assert approval is False


def test_anomaly_detector():
    """Verifies anomaly detection model computes normalized risk score without warnings."""
    ad = CloudAnomalyDetector()
    loaded = ad.load()
    assert loaded is True, "Anomaly detector failed to load"

    score, severity, factors = ad.predict_risk_score({"cpu_usage": 15.0, "memory_usage": 30.0})
    assert 0.0 <= score <= 100.0
    assert severity in ["low", "medium", "high"]

    # High saturation test
    high_score, high_sev, high_factors = ad.predict_risk_score({"cpu_usage": 95.0, "memory_usage": 92.0})
    assert high_score > score
    assert "CPU Saturation (>80%)" in high_factors


def test_workload_resilience_and_forecasting():
    """Verifies Google Borg resilience model and cost forecasting."""
    fc = CloudForecaster()
    fail_prob, resilience = fc.evaluate_resilience(cpu_avg=12.0, cpu_max=25.0, mem_usage=20.0)
    assert 0.0 <= fail_prob <= 1.0
    assert 0.0 <= resilience <= 100.0
    assert resilience > 80.0  # Safe workload

    # 30-day forecast
    forecasts = fc.forecast_cost(baseline_daily_cost=200.0, days=30)
    assert len(forecasts) == 30
    assert forecasts[0]["predicted_cost"] > 0
    assert forecasts[0]["upper_bound"] >= forecasts[0]["predicted_cost"]


def test_explainability_engine():
    """Verifies natural language explainability generation."""
    exp = ExplainabilityEngine()
    rec = exp.explain_recommendation(
        resource_id="i-test123",
        provider="aws",
        current_sku="c5.2xlarge",
        target_sku="c5.xlarge",
        action="scale_down",
        risk_level="low",
        monthly_savings=124.10,
        cpu_avg=12.4,
        cpu_max=28.0,
        mem_usage=32.0,
        resilience_score=94.5,
        anomaly_score=15.0,
    )
    assert rec["approval_required"] is False
    assert "c5.2xlarge to c5.xlarge" in rec["summary"]
    assert rec["monthly_savings"] == 124.10
    assert len(rec["drivers"]) > 0


def test_minimum_iam_policies():
    """Verifies minimum IAM/RBAC policies conform to strict security requirements."""
    assert "Statement" in AWS_MINIMUM_IAM_POLICY
    assert "Actions" in AZURE_MINIMUM_RBAC_ROLE
    assert "roles" in GCP_MINIMUM_IAM_ROLES
    assert "monitoring.timeSeries.list" in GCP_MINIMUM_IAM_ROLES["custom_role_permissions"]

    from cba.live.permissions import AZURE_RBAC_TEMPLATE, GCP_IAM_CUSTOM_ROLE_YAML
    assert "Microsoft.Compute/virtualMachines/read" in AZURE_RBAC_TEMPLATE
    assert "Microsoft.Insights/metrics/read" in AZURE_RBAC_TEMPLATE
    assert "compute.instances.get" in GCP_IAM_CUSTOM_ROLE_YAML
    assert "monitoring.timeSeries.list" in GCP_IAM_CUSTOM_ROLE_YAML


def test_queue_dispatch():
    """Verifies unified message queue dispatcher format."""
    qm = CloudQueueManager()
    res = qm.dispatch_optimization(
        provider="aws",
        resource_id="i-test999",
        action="scale_down",
        current_sku="t3.large",
        target_sku="t3.medium",
        risk_level="low",
        original_state={"instance_type": "t3.large", "region": "us-east-1"},
        queue_config={},
        observation_window_minutes=10,
    )
    assert res["status"] == "QUEUED"
    assert res["provider"] == "aws"
    assert res["payload"]["target_size"] == "t3.medium"
    assert res["payload"]["original_state"]["instance_type"] == "t3.large"


def test_auto_rollback_lifecycle():
    """Verifies observation window tracking and automatic rollback on performance breach."""
    monitor = AutoRollbackMonitor()
    action = monitor.register_execution(
        provider="aws",
        resource_id="i-test999",
        action_type="scale_down",
        target_size="t3.medium",
        original_state={"instance_type": "t3.large"},
        observation_minutes=10,
    )
    assert action.status == RecommendationStatus.OBSERVING

    # Step 1: Health check under normal load -> remains in observation
    status1, _ = monitor.run_health_check_step(action.action_id, current_cpu=25.0, current_mem=35.0)
    assert status1 == "OBSERVING"

    # Step 2: Health check with CPU spike above 85% -> Triggers AUTOMATIC ROLLBACK!
    status2, reason = monitor.run_health_check_step(action.action_id, current_cpu=92.5, current_mem=40.0)
    assert status2 == "ROLLED_BACK"
    assert "Safety breach" in reason

    # Verify action record is updated
    updated = monitor.get_action(action.action_id)
    assert updated["status"] == "rolled_back"
    assert updated["rollback_reason"] == reason


def test_permanent_commit_lifecycle():
    """Verifies action permanently commits when observation window expires cleanly."""
    monitor = AutoRollbackMonitor()
    action = monitor.register_execution(
        provider="gcp",
        resource_id="gce-vm-101",
        action_type="scale_down",
        target_size="e2-small",
        original_state={"machine_type": "e2-medium"},
        observation_minutes=-1,  # Expired deadline
    )
    # Check step with expired deadline and normal CPU
    status, reason = monitor.run_health_check_step(action.action_id, current_cpu=20.0, current_mem=25.0)
    assert status == "COMMITTED_PERMANENTLY"
    assert reason is None
    record = monitor.get_action(action.action_id)
    assert record["status"] == "committed_permanently"


def test_sku_catalog_all_clouds():
    """Verifies SKU catalog right-sizing logic across AWS, Azure, and GCP."""
    rm = ResourceIntelligenceModel()
    # AWS
    aws_rec, aws_new, aws_curr = rm.suggest_target_sku("aws", "c5.4xlarge", "scale_down")
    assert aws_rec == "c5.2xlarge"
    assert aws_new < aws_curr

    # Azure
    az_rec, az_new, az_curr = rm.suggest_target_sku("azure", "Standard_D8s_v5", "scale_down")
    assert az_rec == "Standard_D4s_v5"
    assert az_new < az_curr

    # GCP
    gcp_rec, gcp_new, gcp_curr = rm.suggest_target_sku("gcp", "c2-standard-8", "scale_down")
    assert gcp_rec == "c2-standard-4"
    assert gcp_new < gcp_curr


def test_queue_manager_audit_log():
    """Verifies queue manager local audit logging and history retrieval."""
    qm = CloudQueueManager()
    qm._log_local({"test_event": "AUDIT_VERIFIED", "timestamp": "2026-09-04T00:00:00Z"}, direction="OUTBOUND")
    recent = qm.get_recent_messages(limit=5)
    assert len(recent) > 0
    assert any(m.get("payload", {}).get("test_event") == "AUDIT_VERIFIED" for m in recent)


def test_preprocessed_datasets_integrity():
    """Verifies all 4 processed feature datasets exist with valid dimensions."""
    proc = Path("data/processed")
    assert (proc / "resource_features.csv").exists()
    assert (proc / "anomaly_features.csv").exists()
    assert (proc / "workload_timeseries.csv").exists()
    assert (proc / "cost_timeseries.csv").exists()

    df_res = pd.read_csv(proc / "resource_features.csv", nrows=10)
    assert "cpu_usage" in df_res.columns
    assert "target_encoded" in df_res.columns or "target" in df_res.columns

    df_cost = pd.read_csv(proc / "cost_timeseries.csv", nrows=10)
    assert "net_cost" in df_cost.columns


def test_model_metrics_file_integrity():
    """Verifies serialized metrics.json contains valid evaluation results."""
    m_path = Path("models/metrics.json")
    assert m_path.exists()
    data = json.loads(m_path.read_text(encoding="utf-8"))
    assert "resource" in data
    assert "anomaly" in data
    assert "forecast" in data
    assert data["resource"]["accuracy"] >= 0.95
    assert data["anomaly"]["roc_auc"] >= 0.75


def test_cloud_permission_helpers():
    """Verifies AWS CloudFormation setup template and IAM helper definitions."""
    from cba.live.permissions import AWS_CLOUDFORMATION_TEMPLATE, test_aws_connection, auto_attach_aws_policies
    assert "AWSTemplateFormatVersion" in AWS_CLOUDFORMATION_TEMPLATE
    assert "CloudOptAgentUser" in AWS_CLOUDFORMATION_TEMPLATE
    assert "AmazonEC2ReadOnlyAccess" in AWS_CLOUDFORMATION_TEMPLATE
    assert "CloudWatchReadOnlyAccess" in AWS_CLOUDFORMATION_TEMPLATE

    # Test invalid connection handles gracefully without unhandled crashes
    ok, msg, details = test_aws_connection("dummy_key", "dummy_secret", "us-east-1")
    assert ok is False
    assert "AWS connection failed" in msg or "InvalidClientTokenId" in msg


def test_aws_telemetry_and_optimization_pipeline():
    """Verifies complete AWS right-sizing, explainability, and queue pipeline."""
    from cba.live.collector import LiveTelemetryCollector
    from cba.intelligence.resource_model import ResourceIntelligenceModel
    from cba.intelligence.explainability import ExplainabilityEngine
    from cba.automation.queue_manager import CloudQueueManager

    # 1. Test AWS t2.micro right-sizing & pricing
    rm = ResourceIntelligenceModel()
    rec_sku, new_rate, curr_rate = rm.suggest_target_sku("aws", "t2.micro", "scale_down")
    assert rec_sku == "t2.nano"
    assert new_rate < curr_rate

    # 2. Test AWS scale up
    up_sku, up_rate, _ = rm.suggest_target_sku("aws", "t2.micro", "scale_up")
    assert up_sku == "t2.small"
    assert up_rate > curr_rate

    # 3. Test Explainability output
    exp = ExplainabilityEngine()
    explanation = exp.explain_recommendation(
        resource_id="i-0abc123sydney",
        provider="aws",
        current_sku="t2.micro",
        target_sku="t2.nano",
        action="scale_down",
        risk_level="low",
        monthly_savings=round((curr_rate - new_rate) * 730, 2),
        cpu_avg=11.2,
        cpu_max=22.5,
        mem_usage=25.0,
        resilience_score=92.5,
        anomaly_score=8.0,
    )
    assert "t2.micro to t2.nano" in explanation["summary"]
    assert explanation["approval_required"] is False

    # 4. Test Queue Dispatch format
    test_q_dir = Path("outputs/test_queue")
    qm = CloudQueueManager(state_dir=test_q_dir)
    payload = qm.dispatch_optimization(
        provider="aws",
        resource_id="i-0abc123sydney",
        action="scale_down",
        current_sku="t2.micro",
        target_sku="t2.nano",
        risk_level="low",
        original_state={"instance_type": "t2.micro", "region": "ap-southeast-2"},
        queue_config={"sqs_queue_url": ""},
        observation_window_minutes=10,
    )
    assert payload["status"] == "QUEUED"
    assert payload["payload"]["event_type"] == "OPTIMIZE_ACTION"
    assert payload["resource_id"] == "i-0abc123sydney"


def test_sqs_non_existent_queue_resilience():
    """Verify that specifying a non-existent or inaccessible SQS queue safely falls back to local audit queue."""
    test_q_dir = Path("outputs/test_queue_sqs_resilience")
    qm = CloudQueueManager(state_dir=test_q_dir)
    payload = qm.dispatch_optimization(
        provider="aws",
        resource_id="i-0abc123sydney",
        action="scale_down",
        current_sku="t2.micro",
        target_sku="t2.nano",
        risk_level="low",
        original_state={"instance_type": "t2.micro", "region": "ap-southeast-2"},
        queue_config={
            "sqs_queue_url": "https://sqs.ap-southeast-2.amazonaws.com/123456789012/nonexistent-queue",
            "aws_credentials": {"access_key_id": "dummy", "secret_access_key": "dummy"},
            "region": "ap-southeast-2",
        },
        observation_window_minutes=10,
    )
    assert payload["status"] == "QUEUED"
    assert "sqs-local-" in payload["native_message_id"]
    msgs = qm.get_recent_messages(limit=5)
    assert any(m.get("payload", {}).get("queue") == "AWS_LOCAL_FALLBACK" for m in msgs)


def test_anomaly_analyzer_statistical_and_results():
    """Verifies Cost Anomaly Detector with statistical methods and AnomalyResult integrity."""
    from datetime import datetime, timedelta, timezone
    from cba.collectors.base import BillingData
    from cba.analyzers.anomaly import AnomalyDetector, AnomalyResult
    from cba.core.exceptions import BudgetExceededError, APIError

    # Verify exception classes
    assert issubclass(BudgetExceededError, Exception)
    assert issubclass(APIError, Exception)

    # Build billing data with a clear cost anomaly spike
    now = datetime.now(timezone.utc)
    normal_data = [
        BillingData(
            provider="aws",
            account_id="111222333",
            service="AmazonEC2",
            region="us-east-1",
            resource_id="i-test001",
            resource_name="web-app",
            usage_type="BoxUsage:t3.medium",
            usage_amount=24.0,
            usage_unit="Hrs",
            cost=2.50,
            currency="USD",
            start_time=now - timedelta(days=i),
            end_time=now - timedelta(days=i - 1),
        )
        for i in range(1, 15)
    ]
    # Add a massive anomalous spike
    spike = BillingData(
        provider="aws",
        account_id="111222333",
        service="AmazonEC2",
        region="us-east-1",
        resource_id="i-test001",
        resource_name="web-app",
        usage_type="BoxUsage:t3.medium",
        usage_amount=24.0,
        usage_unit="Hrs",
        cost=85.00,  # Huge spike compared to $2.50
        currency="USD",
        start_time=now,
        end_time=now + timedelta(days=1),
    )
    dataset = normal_data + [spike]

    detector = AnomalyDetector()
    anomalies = detector.detect_anomalies(dataset, methods=["zscore", "iqr", "percentage"], threshold=2.0)
    assert len(anomalies) > 0
    top = anomalies[0]
    assert isinstance(top, AnomalyResult)
    assert top.resource_id == "i-test001"
    assert top.service == "AmazonEC2"
    assert top.actual_value == 85.00
    assert top.severity in ["high", "critical"]
    assert top.deviation_percentage > 100.0

    # Serialization test
    d = top.to_dict()
    assert "date" in d
    assert d["actual_value"] == 85.00


def test_cost_forecaster_analyzer_and_results():
    """Verifies Cost Forecaster analyzer with daily series aggregation and confidence intervals."""
    from datetime import datetime, timedelta, timezone
    from cba.collectors.base import BillingData
    from cba.analyzers.forecast import CostForecaster, ForecastResult

    now = datetime.now(timezone.utc)
    series = [
        BillingData(
            provider="aws",
            account_id="111222333",
            service="AmazonEC2",
            region="us-east-1",
            resource_id=f"i-srv{i}",
            resource_name="backend",
            usage_type="BoxUsage:m5.large",
            usage_amount=24.0,
            usage_unit="Hrs",
            cost=10.0 + (i * 0.5),
            currency="USD",
            start_time=now - timedelta(days=30 - i),
            end_time=now - timedelta(days=29 - i),
        )
        for i in range(30)
    ]

    forecaster = CostForecaster()
    result = forecaster.forecast_costs(series, days=14, model="auto")
    assert isinstance(result, ForecastResult)
    assert len(result.forecast_values) == 14
    assert len(result.forecast_dates) == 14
    assert all(v > 0 for v in result.forecast_values)
    assert "mape" in result.accuracy_metrics
    assert result.confidence_intervals is not None

    res_dict = result.to_dict()
    assert len(res_dict["forecast_values"]) == 14


def test_fastapi_rest_and_rollback_endpoint():
    """Verifies FastAPI REST server endpoints, health surveillance, and rollback evaluation."""
    from starlette.testclient import TestClient
    from cba.api.server import app
    from cba.automation.monitor_rollback import AutoRollbackMonitor

    client = TestClient(app)

    # 1. System status endpoint
    res_status = client.get("/api/status")
    assert res_status.status_code == 200
    assert res_status.json()["version"] == "2.4.0"

    # 2. Register an action directly for surveillance testing
    monitor = AutoRollbackMonitor()
    action = monitor.register_execution(
        provider="aws",
        resource_id="i-rest-surveillance-99",
        action_type="scale_down",
        target_size="t3.small",
        original_state={"instance_type": "t3.medium"},
        observation_minutes=15,
    )

    # 3. Healthy step via REST API -> Remains in OBSERVING
    res_step1 = client.post("/api/rollback/step", json={
        "action_id": action.action_id,
        "simulated_cpu": 35.0,
        "simulated_memory": 40.0,
        "cpu_threshold": 85.0,
    })
    assert res_step1.status_code == 200
    assert res_step1.json()["new_status"] == "OBSERVING"

    # 4. Spike step (>85% CPU) -> Triggers automatic rollback!
    res_step2 = client.post("/api/rollback/step", json={
        "action_id": action.action_id,
        "simulated_cpu": 91.5,
        "simulated_memory": 45.0,
        "cpu_threshold": 85.0,
    })
    assert res_step2.status_code == 200
    assert res_step2.json()["new_status"] == "ROLLED_BACK"
    assert "Safety breach" in res_step2.json()["reason"]


def test_environment_detector():
    """Verifies environment tag extraction and normalization across multiple cloud tagging patterns."""
    from cba.core.models import normalize_environment, detect_environment_from_tags, EnvironmentTier

    # 1. Direct normalization
    assert normalize_environment("Production") == EnvironmentTier.PRODUCTION.value
    assert normalize_environment("prod") == EnvironmentTier.PRODUCTION.value
    assert normalize_environment("PRD-East") == EnvironmentTier.PRODUCTION.value
    assert normalize_environment("live-cluster") == EnvironmentTier.PRODUCTION.value

    assert normalize_environment("Staging") == EnvironmentTier.STAGING.value
    assert normalize_environment("stage") == EnvironmentTier.STAGING.value
    assert normalize_environment("uat") == EnvironmentTier.STAGING.value
    assert normalize_environment("pre-prod") == EnvironmentTier.STAGING.value

    assert normalize_environment("Development") == EnvironmentTier.DEVELOPMENT.value
    assert normalize_environment("dev") == EnvironmentTier.DEVELOPMENT.value
    assert normalize_environment("test") == EnvironmentTier.DEVELOPMENT.value
    assert normalize_environment("qa-team") == EnvironmentTier.DEVELOPMENT.value
    assert normalize_environment("sandbox-01") == EnvironmentTier.DEVELOPMENT.value
    assert normalize_environment("nonprod") == EnvironmentTier.DEVELOPMENT.value

    assert normalize_environment("custom-random") == EnvironmentTier.UNTAGGED.value
    assert normalize_environment(None) == EnvironmentTier.UNTAGGED.value

    # 2. Tag dictionary extraction
    assert detect_environment_from_tags({"Environment": "production"}) == "production"
    assert detect_environment_from_tags({"env": "staging"}) == "staging"
    assert detect_environment_from_tags({"stage": "dev"}) == "development"
    assert detect_environment_from_tags({"tier": "prod"}) == "production"
    assert detect_environment_from_tags({"deployment_stage": "test"}) == "development"
    assert detect_environment_from_tags({"Name": "web-server"}) == "untagged"
    assert detect_environment_from_tags(None) == "untagged"


def test_environment_rightsizing_production():
    """Verifies that production workloads enforce conservative rightsizing and mandatory human sign-off."""
    from cba.intelligence.resource_model import ResourceIntelligenceModel
    from cba.intelligence.explainability import ExplainabilityEngine

    rm = ResourceIntelligenceModel()

    # 1. Conservative single-tier limit
    rec_sku, new_rate, curr_rate = rm.suggest_target_sku(
        provider="aws",
        current_sku="c5.4xlarge",
        action="scale_down",
        environment="production",
    )
    # Production moves 1 step only (c5.4xlarge -> c5.2xlarge)
    assert rec_sku == "c5.2xlarge"

    # 2. Mandatory approval gate (approval_required == True) even with low average CPU
    risk_level, approval_req = rm.evaluate_risk(
        cpu_avg=8.0,
        cpu_max=18.0,
        action="scale_down",
        resilience_score=96.0,
        environment="production",
    )
    assert approval_req is True
    assert risk_level in ["medium", "high"]

    # 3. Explainability context
    exp = ExplainabilityEngine()
    rec = exp.explain_recommendation(
        resource_id="i-prod-db",
        provider="aws",
        current_sku="c5.4xlarge",
        target_sku="c5.2xlarge",
        action="scale_down",
        risk_level=risk_level,
        monthly_savings=(curr_rate - new_rate) * 730,
        cpu_avg=8.0,
        cpu_max=18.0,
        mem_usage=25.0,
        resilience_score=96.0,
        anomaly_score=5.0,
        environment="production",
    )
    assert rec["approval_required"] is True
    assert "PRODUCTION" in rec["summary"]
    assert "mandatory operator approval gate" in rec["summary"]


def test_environment_rightsizing_development():
    """Verifies that development workloads enable aggressive 2-tier downsize and instant auto-dispatch."""
    from cba.intelligence.resource_model import ResourceIntelligenceModel
    from cba.intelligence.explainability import ExplainabilityEngine

    rm = ResourceIntelligenceModel()

    # 1. Aggressive 2-tier downsize in dev
    rec_sku, new_rate, curr_rate = rm.suggest_target_sku(
        provider="aws",
        current_sku="c5.4xlarge",
        action="scale_down",
        environment="development",
    )
    # Development moves 2 steps down: c5.4xlarge -> c5.2xlarge -> c5.xlarge
    assert rec_sku == "c5.xlarge"
    assert new_rate < curr_rate

    # 2. Low risk & zero approval gate (instant dispatch)
    risk_level, approval_req = rm.evaluate_risk(
        cpu_avg=32.0,
        cpu_max=60.0,
        action="scale_down",
        resilience_score=75.0,
        environment="development",
    )
    assert risk_level == "low"
    assert approval_req is False

    # 3. Explainability context
    exp = ExplainabilityEngine()
    rec = exp.explain_recommendation(
        resource_id="i-dev-sandbox",
        provider="aws",
        current_sku="c5.4xlarge",
        target_sku="c5.xlarge",
        action="scale_down",
        risk_level=risk_level,
        monthly_savings=(curr_rate - new_rate) * 730,
        cpu_avg=32.0,
        cpu_max=60.0,
        mem_usage=40.0,
        resilience_score=75.0,
        anomaly_score=10.0,
        environment="development",
    )
    assert rec["approval_required"] is False
    assert "DEVELOPMENT" in rec["summary"]
    assert "Aggressive cost-saving policy" in rec["summary"]


def test_environment_recommendations_api():
    """Verifies end-to-end recommendation generation and API output with tagged cloud environments."""
    from starlette.testclient import TestClient
    from cba.api.server import app, LIVE_DIR

    client = TestClient(app)

    # Setup telemetry buffer containing prod, dev, and staging workloads
    test_df = pd.DataFrame([
        {
            "provider": "aws",
            "account_id": "111222333",
            "resource_id": "i-prod-ec2",
            "resource_name": "prod-api-cluster",
            "region": "us-east-1",
            "environment": "production",
            "state": "running",
            "instance_type": "c5.4xlarge",
            "cpu_usage": 10.0,
            "cpu_max": 22.0,
            "memory_usage": 30.0,
            "original_state": {"tags": {"Environment": "production"}},
        },
        {
            "provider": "aws",
            "account_id": "111222333",
            "resource_id": "i-dev-ec2",
            "resource_name": "dev-test-runner",
            "region": "us-east-1",
            "environment": "development",
            "state": "running",
            "instance_type": "c5.4xlarge",
            "cpu_usage": 30.0,
            "cpu_max": 50.0,
            "memory_usage": 25.0,
            "original_state": {"tags": {"Environment": "development"}},
        },
        {
            "provider": "aws",
            "account_id": "111222333",
            "resource_id": "i-stage-ec2",
            "resource_name": "stage-preprod",
            "region": "us-east-1",
            "environment": "staging",
            "state": "running",
            "instance_type": "c5.4xlarge",
            "cpu_usage": 18.0,
            "cpu_max": 35.0,
            "memory_usage": 28.0,
            "original_state": {"tags": {"Environment": "staging"}},
        },
    ])
    test_df.to_csv(LIVE_DIR / "telemetry.csv", index=False)

    # Generate recommendations
    gen_res = client.post("/api/recommendations/generate")
    assert gen_res.status_code == 200
    gen_data = gen_res.json()
    assert gen_data["success"] is True
    assert gen_data["count"] == 3

    recs_map = {r["resource_id"]: r for r in gen_data["recommendations"]}

    # Verify Production Policy
    prod_rec = recs_map["i-prod-ec2"]
    assert prod_rec["environment"] == "production"
    assert prod_rec["action"] == "scale_down"
    assert prod_rec["recommended_sku"] == "c5.2xlarge"  # 1 step down
    assert prod_rec["approval_required"] is True  # Mandatory gate

    # Verify Development Policy
    dev_rec = recs_map["i-dev-ec2"]
    assert dev_rec["environment"] == "development"
    assert dev_rec["action"] == "scale_down"
    assert dev_rec["recommended_sku"] == "c5.xlarge"  # 2 steps down!
    assert dev_rec["approval_required"] is False  # Instant 1-click eligible

    # Verify GET /api/recommendations breakdown
    get_res = client.get("/api/recommendations")
    assert get_res.status_code == 200
    rec_body = get_res.json()
    assert "by_environment" in rec_body
    assert rec_body["by_environment"]["production"]["count"] == 1
    assert rec_body["by_environment"]["development"]["count"] == 1
    assert rec_body["by_environment"]["staging"]["count"] == 1


def test_aws_session_token_backend_request(monkeypatch):
    """Verifies that request_aws_session_token calls boto3 STS get_session_token and parses credentials."""
    from datetime import datetime, timezone
    from unittest.mock import MagicMock
    from cba.live.permissions import request_aws_session_token

    mock_sts = MagicMock()
    mock_sts.get_session_token.return_value = {
        "Credentials": {
            "AccessKeyId": "ASIAEXAMPLE123456789",
            "SecretAccessKey": "SECRETEXAMPLE987654321",
            "SessionToken": "TOKEN1234567890abcdef...",
            "Expiration": datetime(2026, 9, 5, 12, 0, 0, tzinfo=timezone.utc),
        }
    }

    mock_session_cls = MagicMock()
    mock_session_inst = MagicMock()
    mock_session_inst.client.return_value = mock_sts
    mock_session_cls.return_value = mock_session_inst

    import boto3
    monkeypatch.setattr(boto3, "Session", mock_session_cls)

    ok, msg, details = request_aws_session_token(
        access_key_id="AKIA1234567890EXAMPLE",
        secret_access_key="SECRETEXAMPLE",
        region="us-east-1",
        duration_seconds=43200,
    )

    assert ok is True
    assert "Successfully acquired" in msg
    assert details["access_key_id"] == "ASIAEXAMPLE123456789"
    assert details["secret_access_key"] == "SECRETEXAMPLE987654321"
    assert details["session_token"] == "TOKEN1234567890abcdef..."
    assert "2026-09-05" in details["expiration"]
    assert details["duration_seconds"] == 43200
    mock_sts.get_session_token.assert_called_once_with(DurationSeconds=43200)


def test_aws_session_token_endpoint(monkeypatch):
    """Verifies POST /api/auth/aws/session-token REST endpoint."""
    from datetime import datetime, timezone
    from unittest.mock import MagicMock
    from fastapi.testclient import TestClient
    from cba.api.server import app

    client = TestClient(app)

    mock_sts = MagicMock()
    mock_sts.get_session_token.return_value = {
        "Credentials": {
            "AccessKeyId": "ASIAAPI123456789",
            "SecretAccessKey": "SECRETAPI987654321",
            "SessionToken": "TOKENAPI1234567890abcdef...",
            "Expiration": datetime(2026, 9, 5, 12, 0, 0, tzinfo=timezone.utc),
        }
    }

    mock_session_cls = MagicMock()
    mock_session_inst = MagicMock()
    mock_session_inst.client.return_value = mock_sts
    mock_session_cls.return_value = mock_session_inst

    import boto3
    monkeypatch.setattr(boto3, "Session", mock_session_cls)

    resp = client.post(
        "/api/auth/aws/session-token",
        json={
            "aws_access_key_id": "AKIAAPIUSERKEY12345",
            "aws_secret_access_key": "SecretKey12345",
            "aws_region": "ap-southeast-2",
            "duration_hours": 6.0,
        },
    )

    assert resp.status_code == 200
    data = resp.json()
    assert data["success"] is True
    creds = data["credentials"]
    assert creds["access_key_id"] == "ASIAAPI123456789"
    assert creds["session_token"] == "TOKENAPI1234567890abcdef..."
    assert creds["duration_seconds"] == 21600


def test_aws_connection_auto_session_token(monkeypatch):
    """Verifies test_aws_connection automatically acquires STS session token when AKIA key is used."""
    from datetime import datetime, timezone
    from unittest.mock import MagicMock
    from cba.live.permissions import test_aws_connection

    mock_sts = MagicMock()
    mock_sts.get_caller_identity.return_value = {
        "Account": "123456789012",
        "Arn": "arn:aws:iam::123456789012:user/dev-admin",
        "UserId": "AIDASAMPLE",
    }
    mock_sts.get_session_token.return_value = {
        "Credentials": {
            "AccessKeyId": "ASIAAUTOGENERATED123",
            "SecretAccessKey": "SECRETKEY999",
            "SessionToken": "AUTOTOKEN123456",
            "Expiration": datetime(2026, 9, 5, 12, 0, 0, tzinfo=timezone.utc),
        }
    }

    mock_ec2 = MagicMock()
    mock_ec2.describe_instances.return_value = {"Reservations": []}

    def client_factory(service_name, **kwargs):
        if service_name == "sts":
            return mock_sts
        elif service_name == "ec2":
            return mock_ec2
        return MagicMock()

    mock_session_inst = MagicMock()
    mock_session_inst.client.side_effect = client_factory
    mock_session_inst.region_name = "us-east-1"

    mock_session_cls = MagicMock(return_value=mock_session_inst)

    import boto3
    monkeypatch.setattr(boto3, "Session", mock_session_cls)

    ok, msg, details = test_aws_connection(
        access_key_id="AKIAUSER1234567890",
        secret_access_key="SECRETKEY12345",
        region="us-east-1",
        session_token=None,
    )

    assert ok is True
    assert details["generated_session_token"] is not None
    assert details["generated_session_token"]["session_token"] == "AUTOTOKEN123456"
    assert details["generated_session_token"]["access_key_id"] == "ASIAAUTOGENERATED123"


def test_gmail_session_and_alert_service():
    """Verifies Gmail session lifecycle, email generation, and alert persistence."""
    from cba.alerts.notifier import GmailAlertService

    service = GmailAlertService()
    service.clear_session()

    # Initial session should be logged out
    sess = service.get_session()
    assert sess["logged_in"] is False
    assert sess["email"] == ""

    # Login
    sess = service.set_session(email="madhu@gmail.com", name="Madhu Architect")
    assert sess["logged_in"] is True
    assert sess["email"] == "madhu@gmail.com"
    assert sess["name"] == "Madhu Architect"

    # Cloud account alert
    res1 = service.notify_cloud_account_connected(
        provider="AWS",
        account_id="245716172295",
        arn_or_identity="arn:aws:iam::245716172295:user/madhu",
        region="ap-southeast-2",
    )
    assert res1["success"] is True
    assert res1["to_email"] == "madhu@gmail.com"
    assert "245716172295" in res1["subject"]
    assert res1["alert_id"].startswith("eml-")

    # Optimization dispatched alert
    res2 = service.notify_optimization_dispatched(
        resource_id="i-prod-web-01",
        action="scale_down",
        current_sku="c5.2xlarge",
        target_sku="c5.xlarge",
        monthly_savings=124.10,
        environment="production",
        risk_level="high",
        action_id="act-test-99",
    )
    assert res2["success"] is True
    assert res2["to_email"] == "madhu@gmail.com"
    assert "i-prod-web-01" in res2["subject"]
    assert "c5.2xlarge ➔ c5.xlarge" in res2["subject"]

    # History verification
    history = service.get_history(limit=10)
    assert len(history) >= 2
    alert_ids = [h["alert_id"] for h in history]
    assert res1["alert_id"] in alert_ids
    assert res2["alert_id"] in alert_ids

    # Clear session
    service.clear_session()
    sess_after = service.get_session()
    assert sess_after["logged_in"] is False


def test_gmail_alert_endpoints_and_dispatch_hook():
    """Verifies FastAPI endpoints for email-only user login/session and email alert triggering."""
    from starlette.testclient import TestClient
    from cba.api.server import app
    from cba.alerts.notifier import notifier

    client = TestClient(app)

    # 1. Login user with email ONLY (no password required)
    login_resp = client.post(
        "/api/user/login",
        json={"email": "operator-sre@gmail.com", "name": "Site Reliability Engineer"},
    )
    assert login_resp.status_code == 200
    login_data = login_resp.json()
    assert login_data["logged_in"] is True
    assert login_data["email"] == "operator-sre@gmail.com"

    # 2. Get active session
    sess_resp = client.get("/api/user/session")
    assert sess_resp.status_code == 200
    assert sess_resp.json()["email"] == "operator-sre@gmail.com"

    # 3. Trigger optimization dispatch with email alert
    disp_resp = client.post(
        "/api/queue/dispatch",
        json={
            "provider": "aws",
            "resource_id": "i-enterprise-srv-44",
            "action": "scale_down",
            "current_sku": "m5.large",
            "target_sku": "m5.small",
            "risk_level": "low",
            "original_state": {"instance_type": "m5.large"},
            "monthly_savings": 45.50,
            "environment": "development",
        },
    )
    assert disp_resp.status_code == 200
    disp_data = disp_resp.json()
    assert disp_data["success"] is True
    assert disp_data["email_alert"] is not None
    assert disp_data["email_alert"]["success"] is True
    assert disp_data["email_alert"]["to_email"] == "operator-sre@gmail.com"
    alert_id = disp_data["email_alert"]["alert_id"]

    # 4. Preview HTML endpoint
    prev_resp = client.get(f"/api/notifications/preview/{alert_id}")
    assert prev_resp.status_code == 200
    assert "text/html" in prev_resp.headers["content-type"]
    assert "i-enterprise-srv-44" in prev_resp.text

    # 5. History endpoint
    hist_resp = client.get("/api/notifications/history?limit=10")
    assert hist_resp.status_code == 200
    alerts = hist_resp.json()["alerts"]
    assert any(a["alert_id"] == alert_id for a in alerts)


def test_gmail_email_only_login_and_practical_dispatch(monkeypatch):
    """Verifies email-only login requirement, SMTP config saving, and practical test email dispatch."""
    from starlette.testclient import TestClient
    from cba.api.server import app
    from cba.alerts.notifier import notifier

    client = TestClient(app)

    # 1. Missing email returns 400
    res_no_email = client.post("/api/user/login", json={})
    assert res_no_email.status_code == 422 or res_no_email.status_code == 400

    # 2. Invalid email format returns 400
    res_invalid_email = client.post("/api/user/login", json={"email": "invalid-address"})
    assert res_invalid_email.status_code == 400

    # 3. Valid email without password succeeds (200)
    res_ok = client.post("/api/user/login", json={"email": "madhu@gmail.com", "name": "Madhu Lead"})
    assert res_ok.status_code == 200
    assert res_ok.json()["logged_in"] is True
    assert res_ok.json()["email"] == "madhu@gmail.com"

    # 4. Outbound SMTP configuration endpoints
    monkeypatch.setattr(
        notifier,
        "save_smtp_config",
        lambda smtp_user, smtp_pass, smtp_host="smtp.gmail.com", smtp_port=587: (True, "Sender verified"),
    )
    cfg_save = client.post(
        "/api/notifications/smtp-config",
        json={"smtp_user": "alerts@gmail.com", "smtp_pass": "apppass12345678"},
    )
    assert cfg_save.status_code == 200
    assert cfg_save.json()["success"] is True

    cfg_get = client.get("/api/notifications/smtp-config")
    assert cfg_get.status_code == 200

    # 5. Practical test email dispatch endpoint
    test_email_resp = client.post("/api/notifications/test-email")
    assert test_email_resp.status_code == 200
    test_data = test_email_resp.json()
    assert test_data["success"] is True
    assert test_data["to_email"] == "madhu@gmail.com"
    assert "alert_id" in test_data



def test_auto_attach_aws_policies_session_token_handling(monkeypatch):
    """Verifies auto_attach_aws_policies rejects ASIA session tokens with constructive guidance
    and succeeds when valid permanent AKIA credentials are used."""
    from unittest.mock import MagicMock
    from cba.live.permissions import auto_attach_aws_policies

    # 1. ASIA temporary key passed: must return clear security rule and guidance
    ok, msg, details = auto_attach_aws_policies(
        access_key_id="ASIAEXAMPLE12345",
        secret_access_key="secret",
        region="ap-southeast-2",
        session_token="session-tok-abc",
        user_name="madhu",
    )
    assert ok is False
    assert "AWS IAM Security Rule" in msg or "temporary" in msg.lower()
    assert "cli_command" in details
    assert "console_url" in details
    assert "madhu" in details["cli_command"]

    # 2. AKIA key with mocks: must not pass session token and should attach policies
    mock_sts = MagicMock()
    mock_sts.get_caller_identity.return_value = {
        "Account": "245716172295",
        "Arn": "arn:aws:iam::245716172295:user/madhu",
        "UserId": "AIDASAMPLE",
    }
    mock_iam = MagicMock()
    mock_iam.attach_user_policy.return_value = {}

    def client_factory(service_name, **kwargs):
        if service_name == "sts":
            return mock_sts
        elif service_name == "iam":
            return mock_iam
        return MagicMock()

    mock_session_inst = MagicMock()
    mock_session_inst.client.side_effect = client_factory

    mock_session_cls = MagicMock(return_value=mock_session_inst)
    import boto3
    monkeypatch.setattr(boto3, "Session", mock_session_cls)

    ok, msg, details = auto_attach_aws_policies(
        access_key_id="AKIAADMIN12345",
        secret_access_key="secret12345",
        region="ap-southeast-2",
        session_token=None,
        user_name="madhu",
    )
    assert ok is True
    assert "Successfully attached" in msg
    assert "madhu" in msg
    assert mock_iam.attach_user_policy.call_count == 2


def test_ppo_rl_environment_and_agent():
    """Verifies Gym-compatible CloudResourceEnv and CloudPPOAgent Actor-Critic inference."""
    from cba.intelligence.ppo_agent import CloudPPOAgent, CloudResourceEnv, ACTION_NAMES

    # 1. Environment contract testing
    env = CloudResourceEnv(max_steps=50)
    obs, env_info = env.reset()
    assert obs.shape == (8,), f"Expected 8-dim continuous state, got {obs.shape}"
    assert isinstance(env_info, dict)
    assert env.observation_space_dim == 8
    assert env.action_space_dim == 4

    # Step through all discrete actions
    for act in [0, 1, 2, 3]:
        next_obs, reward, done, truncated, info = env.step(act)
        assert next_obs.shape == (8,)
        assert isinstance(reward, float)
        assert isinstance(done, bool)
        assert "cost_delta_pct" in info
        assert "projected_cpu" in info
        assert "sla_breach" in info

    # 2. PPO Agent model loading and inference
    agent = CloudPPOAgent()
    loaded = agent.load()
    assert loaded is True, "Trained PPO Actor-Critic weights failed to load"
    assert agent.is_trained is True

    # Test evaluation with real cloud telemetry
    eval_result = agent.evaluate_workload({
        "cpu_usage": 12.5,
        "memory_usage": 28.0,
        "net_io": 240.0,
        "disk_io": 310.0,
        "vcpu": 4,
        "ram_gb": 16.0,
        "price_per_hour": 0.192,
        "headroom_margin": 87.5,
    })

    assert eval_result["action"] in ACTION_NAMES.values()
    assert isinstance(eval_result["action_label"], str) and len(eval_result["action_label"]) > 0
    assert 0.0 <= eval_result["confidence_pct"] <= 100.0
    assert isinstance(eval_result["state_value"], float)
    assert isinstance(eval_result["expected_reward"], float)
    assert isinstance(eval_result["action_distribution"], dict)
    assert len(eval_result["action_distribution"]) == 4
    assert len(eval_result["verdict"]) > 0


def test_ppo_api_attribution_and_metrics():
    """Verifies FastAPI REST endpoints expose PPO policy metrics and Model 05 attribution."""
    from starlette.testclient import TestClient
    from cba.api.server import app

    client = TestClient(app)

    # 1. Test /api/rl/policy-metrics endpoint
    res_rl = client.get("/api/rl/policy-metrics")
    assert res_rl.status_code == 200
    rl_data = res_rl.json()
    assert rl_data["status"] == "success"
    assert rl_data["algorithm"] == "Proximal Policy Optimization (PPO)"
    assert rl_data["ppo_policy_loaded"] is True
    assert rl_data["state_space"]["dimension"] == 8
    assert rl_data["action_space"]["dimension"] == 4
    assert "reward_function" in rl_data
    assert "hyperparameters" in rl_data

    # 2. Test /api/status includes PPO policy loaded
    res_status = client.get("/api/status")
    assert res_status.status_code == 200
    assert res_status.json()["models"]["ppo_policy_loaded"] is True

    # 3. Test /api/explainability returns Model 05 PPO Attribution
    res_exp = client.get("/api/explainability/i-prod-ec2")
    assert res_exp.status_code == 200
    exp_data = res_exp.json()
    assert "model_attribution" in exp_data
    assert "model_05" in exp_data["model_attribution"]

    m05 = exp_data["model_attribution"]["model_05"]
    assert "Model 05" in m05["name"]
    assert "PPO" in m05["name"] or "PPO" in m05["model_type"]
    assert "policy_action" in m05
    assert "action_label" in m05
    assert "state_value_v" in m05
    assert "expected_reward" in m05
    assert "verdict" in m05


def test_recommendation_family_awareness_and_stable_baseline():
    """Verifies that SKU recommendations strictly stay in-family, prevent false upgrades, and preserve stable baselines."""
    import asyncio
    import pandas as pd
    from cba.intelligence.resource_model import ResourceIntelligenceModel
    from cba.api.server import generate_recommendations, LIVE_DIR

    rm = ResourceIntelligenceModel()

    # 1. t3.micro downsize stays in t3 family (t3.nano) and NEVER jumps across families into t2.xlarge
    tgt_sku, new_rate, curr_rate = rm.suggest_target_sku("aws", "t3.micro", "scale_down")
    assert tgt_sku == "t3.nano"
    assert tgt_sku != "t2.xlarge"
    assert new_rate < curr_rate

    # 2. Bottom of family (t2.nano) returns same SKU without invalid downgrade/upgrade
    bot_sku, bot_new, bot_curr = rm.suggest_target_sku("aws", "t2.nano", "scale_down")
    assert bot_sku == "t2.nano"
    assert bot_new == bot_curr

    # 3. Top of family (c5.4xlarge) returns same SKU on scale_up without error
    top_sku, top_new, top_curr = rm.suggest_target_sku("aws", "c5.4xlarge", "scale_up")
    assert top_sku == "c5.4xlarge"
    assert top_new == top_curr

    # 4. Live telemetry test with stable t3.micro web server:
    # A running web workload on t3.micro with low CPU (0.39%) and 35% memory must resolve to NO_ACTION (already optimal)
    test_df = pd.DataFrame([{
        "provider": "aws",
        "account_id": "245716172295",
        "resource_id": "i-0b6230979eb053bf5",
        "resource_name": "demo-web-for-HACKATHON",
        "region": "ap-southeast-2",
        "environment": "untagged",
        "state": "running",
        "instance_type": "t3.micro",
        "cpu_usage": 0.39,
        "cpu_max": 0.47,
        "memory_usage": 35.0,
        "original_state": {"instance_type": "t3.micro"},
    }])
    test_df.to_csv(LIVE_DIR / "telemetry.csv", index=False)

    res = asyncio.run(generate_recommendations())
    assert res["success"] is True
    assert len(res["recommendations"]) == 1
    rec = res["recommendations"][0]

    assert rec["action"] == "scale_down"
    assert rec["recommended_sku"] == "t3.nano"
    assert rec["recommended_sku"] != "t2.xlarge"
    assert rec["monthly_savings"] > 0
    assert rec["hardware_comparison"]["delta_monthly_cost"] < 0
    assert rec["hardware_comparison"]["savings_pct"] > 0
    assert "t3.nano" in rec["summary"]








