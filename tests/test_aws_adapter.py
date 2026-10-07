"""Tests for AWSAutomationAdapter and live AWS optimization execution."""
from __future__ import annotations

from unittest.mock import MagicMock, patch
import pytest
from botocore.exceptions import ClientError

from cba.automation.aws_adapter import AWSAutomationAdapter
from cba.core.models import ActionType, OptimizationRecommendation, RecommendationStatus
from cba.automation.actions import AutomationEngine
from cba.automation.monitor_rollback import AutoRollbackMonitor


def test_aws_adapter_initialization_with_credentials():
    """Verifies that AWSAutomationAdapter initializes boto3 session with explicit credentials and region."""
    with patch("boto3.Session") as mock_session_cls:
        mock_session = MagicMock()
        mock_session_cls.return_value = mock_session

        creds = {
            "access_key_id": "AKIA_TEST_KEY",
            "secret_access_key": "SECRET_TEST_KEY",
            "session_token": "TOKEN_TEST",
        }
        adapter = AWSAutomationAdapter(credentials=creds, region="ap-southeast-2", dry_run=True)

        mock_session_cls.assert_called_once_with(
            region_name="ap-southeast-2",
            aws_access_key_id="AKIA_TEST_KEY",
            aws_secret_access_key="SECRET_TEST_KEY",
            aws_session_token="TOKEN_TEST",
        )
        assert adapter.region == "ap-southeast-2"
        assert adapter.dry_run is True


def test_aws_adapter_test_connection_success():
    """Verifies caller identity and region test on AWS connection."""
    with patch("boto3.Session") as mock_session_cls:
        mock_session = MagicMock()
        mock_session_cls.return_value = mock_session
        mock_sts = MagicMock()
        mock_ec2 = MagicMock()
        mock_session.client.side_effect = lambda svc, **kw: mock_sts if svc == "sts" else mock_ec2

        mock_sts.get_caller_identity.return_value = {
            "Account": "123456789012",
            "Arn": "arn:aws:iam::123456789012:user/cloudopt-operator",
            "UserId": "AIDA_TEST",
        }

        adapter = AWSAutomationAdapter(region="us-east-1")
        result = adapter.test_connection()

        assert result["success"] is True
        assert result["account_id"] == "123456789012"
        assert "arn:aws:iam::123456789012:user/cloudopt-operator" in result["arn"]


def test_aws_adapter_get_instance_details():
    """Verifies instance inspection extracts state, type, architecture, and attached EBS volumes."""
    with patch("boto3.Session") as mock_session_cls:
        mock_session = MagicMock()
        mock_session_cls.return_value = mock_session
        mock_ec2 = MagicMock()
        mock_session.client.side_effect = lambda svc, **kw: mock_ec2

        mock_ec2.describe_instances.return_value = {
            "Reservations": [
                {
                    "Instances": [
                        {
                            "InstanceId": "i-0123456789abcdef0",
                            "InstanceType": "c5.2xlarge",
                            "State": {"Name": "running"},
                            "Architecture": "x86_64",
                            "Tags": [{"Key": "Name", "Value": "prod-api-cluster"}, {"Key": "Env", "Value": "production"}],
                            "BlockDeviceMappings": [
                                {
                                    "DeviceName": "/dev/xvda",
                                    "Ebs": {"VolumeId": "vol-0987654321fedcba0", "Status": "attached"},
                                }
                            ],
                        }
                    ]
                }
            ]
        }

        adapter = AWSAutomationAdapter(region="us-east-1")
        details = adapter.get_instance_details("i-0123456789abcdef0")

        assert details["found"] is True
        assert details["instance_type"] == "c5.2xlarge"
        assert details["state"] == "running"
        assert details["name"] == "prod-api-cluster"
        assert len(details["volumes"]) == 1
        assert details["volumes"][0]["volume_id"] == "vol-0987654321fedcba0"


def test_aws_adapter_dry_run_validation():
    """Verifies DryRunOperation ClientError is correctly recognized as permission validation success."""
    with patch("boto3.Session") as mock_session_cls:
        mock_session = MagicMock()
        mock_session_cls.return_value = mock_session
        mock_ec2 = MagicMock()
        mock_session.client.side_effect = lambda svc, **kw: mock_ec2

        # Mock describe_instances
        mock_ec2.describe_instances.return_value = {
            "Reservations": [
                {
                    "Instances": [
                        {
                            "InstanceId": "i-test001",
                            "InstanceType": "t3.xlarge",
                            "State": {"Name": "running"},
                            "Tags": [],
                        }
                    ]
                }
            ]
        }

        # Mock DryRunOperation exception
        mock_ec2.modify_instance_attribute.side_effect = ClientError(
            error_response={"Error": {"Code": "DryRunOperation", "Message": "Request would have succeeded, but DryRun flag is set."}},
            operation_name="ModifyInstanceAttribute",
        )

        adapter = AWSAutomationAdapter(region="us-east-1")
        result = adapter.resize_instance("i-test001", target_sku="t3.medium", dry_run=True)

        assert result["success"] is True
        assert result["dry_run"] is True
        assert "DRY-RUN CONFIRMED" in result["message"]
        assert result["previous_sku"] == "t3.xlarge"
        assert result["target_sku"] == "t3.medium"


def test_aws_adapter_live_resize_lifecycle():
    """Verifies complete real AWS lifecycle: Describe -> Stop -> Waiter -> ModifyAttribute -> Start -> Waiter -> Describe."""
    with patch("boto3.Session") as mock_session_cls:
        mock_session = MagicMock()
        mock_session_cls.return_value = mock_session
        mock_ec2 = MagicMock()
        mock_session.client.side_effect = lambda svc, **kw: mock_ec2

        # Mock initial state (running, c5.2xlarge) then post state (running, c5.xlarge)
        initial_desc = {
            "Reservations": [
                {"Instances": [{"InstanceId": "i-live999", "InstanceType": "c5.2xlarge", "State": {"Name": "running"}, "Tags": []}]}
            ]
        }
        post_desc = {
            "Reservations": [
                {"Instances": [{"InstanceId": "i-live999", "InstanceType": "c5.xlarge", "State": {"Name": "running"}, "Tags": []}]}
            ]
        }
        mock_ec2.describe_instances.side_effect = [initial_desc, post_desc]

        mock_waiter_stopped = MagicMock()
        mock_waiter_running = MagicMock()
        mock_ec2.get_waiter.side_effect = lambda w_name: mock_waiter_stopped if w_name == "instance_stopped" else mock_waiter_running

        adapter = AWSAutomationAdapter(region="us-east-1", dry_run=False)
        result = adapter.resize_instance("i-live999", target_sku="c5.xlarge", restart_after=True)

        assert result["success"] is True
        assert result["previous_sku"] == "c5.2xlarge"
        assert result["new_sku"] == "c5.xlarge"
        assert result["state"] == "running"

        # Verify exact call order
        mock_ec2.stop_instances.assert_called_once_with(InstanceIds=["i-live999"])
        mock_waiter_stopped.wait.assert_called_once()
        mock_ec2.modify_instance_attribute.assert_called_once_with(
            InstanceId="i-live999",
            InstanceType={"Value": "c5.xlarge"},
        )
        mock_ec2.start_instances.assert_called_once_with(InstanceIds=["i-live999"])
        mock_waiter_running.wait.assert_called_once()


def test_aws_adapter_storage_optimization():
    """Verifies EBS volume gp2 to gp3 modification."""
    with patch("boto3.Session") as mock_session_cls:
        mock_session = MagicMock()
        mock_session_cls.return_value = mock_session
        mock_ec2 = MagicMock()
        mock_session.client.side_effect = lambda svc, **kw: mock_ec2

        mock_ec2.modify_volume.return_value = {
            "VolumeModification": {
                "VolumeId": "vol-gp2test",
                "ModificationState": "modifying",
                "TargetVolumeType": "gp3",
            }
        }

        adapter = AWSAutomationAdapter(region="us-east-1")
        res = adapter.modify_ebs_volume(volume_id="vol-gp2test", volume_type="gp3")

        assert res["success"] is True
        assert res["volume_id"] == "vol-gp2test"
        assert res["target_type"] == "gp3"
        mock_ec2.modify_volume.assert_called_once_with(VolumeId="vol-gp2test", VolumeType="gp3", DryRun=False)


def test_aws_adapter_rollback_instance():
    """Verifies rollback reversion back to original SKU."""
    with patch("boto3.Session") as mock_session_cls:
        mock_session = MagicMock()
        mock_session_cls.return_value = mock_session
        mock_ec2 = MagicMock()
        mock_session.client.side_effect = lambda svc, **kw: mock_ec2

        initial_desc = {
            "Reservations": [
                {"Instances": [{"InstanceId": "i-rollback123", "InstanceType": "t3.medium", "State": {"Name": "running"}, "Tags": []}]}
            ]
        }
        post_desc = {
            "Reservations": [
                {"Instances": [{"InstanceId": "i-rollback123", "InstanceType": "t3.xlarge", "State": {"Name": "running"}, "Tags": []}]}
            ]
        }
        mock_ec2.describe_instances.side_effect = [initial_desc, post_desc]
        mock_ec2.get_waiter.return_value = MagicMock()

        adapter = AWSAutomationAdapter(region="us-east-1")
        result = adapter.rollback_instance("i-rollback123", original_sku="t3.xlarge")

        assert result["success"] is True
        assert "Safety Rollback" in result["message"]
        assert result["new_sku"] == "t3.xlarge"


def test_aws_adapter_cloudwatch_live_metrics():
    """Verifies CloudWatch get_metric_data extraction."""
    with patch("boto3.Session") as mock_session_cls:
        mock_session = MagicMock()
        mock_session_cls.return_value = mock_session
        mock_cw = MagicMock()
        mock_session.client.side_effect = lambda svc, **kw: mock_cw

        mock_cw.get_metric_data.return_value = {
            "MetricDataResults": [
                {"Id": "cpu_avg", "Values": [14.2, 16.8]},
                {"Id": "cpu_max", "Values": [28.5]},
                {"Id": "net_in", "Values": [102400.0]},
                {"Id": "net_out", "Values": [51200.0]},
            ]
        }

        adapter = AWSAutomationAdapter(region="us-east-1")
        res = adapter.get_live_metrics("i-metricstest")

        assert res["success"] is True
        assert res["metrics"]["cpu_avg"] == 15.5
        assert res["metrics"]["cpu_max"] == 28.5
        assert res["metrics"]["network_in"] == 102400.0
        assert res["metrics"]["network_out"] == 51200.0


def test_automation_engine_with_aws_adapter():
    """Verifies that AutomationEngine invokes AWSAutomationAdapter for scale_down and storage actions."""
    with patch("cba.automation.aws_adapter.AWSAutomationAdapter.resize_instance") as mock_resize:
        mock_resize.return_value = {
            "success": True,
            "instance_id": "i-auto999",
            "previous_sku": "m5.xlarge",
            "new_sku": "m5.large",
            "message": "AWS instance i-auto999 resized to m5.large.",
        }

        from cba.core.models import RiskLevel, CloudProvider

        engine = AutomationEngine(dry_run=False, require_approval=False)
        rec = OptimizationRecommendation(
            recommendation_id="rec-auto-1",
            resource_id="i-auto999",
            provider=CloudProvider.AWS,
            action=ActionType.SCALE_DOWN,
            risk_level=RiskLevel.LOW,
            confidence=0.95,
            estimated_monthly_savings=85.0,
            current_utilization=12.0,
            expected_utilization=24.0,
            current_size="m5.xlarge",
            recommended_size="m5.large",
            rationale="Downsize idle AWS EC2 instance",
            status=RecommendationStatus.APPROVED,
            metadata={"target_sku": "m5.large"},
        )

        execution = engine.execute(
            recommendation=rec,
            resource={"resource_id": "i-auto999", "instance_type": "m5.xlarge", "region": "us-east-1"},
            force=True,
        )

        assert execution.success is True
        assert execution.status == RecommendationStatus.EXECUTED
        assert "resized to m5.large" in execution.message


def test_api_action_execute_endpoint():
    """Verifies that FastAPI /api/actions/execute directly invokes the AWS adapter and returns real mutation status."""
    from starlette.testclient import TestClient
    from cba.api.server import app

    with patch("cba.automation.aws_adapter.AWSAutomationAdapter.resize_instance") as mock_resize:
        mock_resize.return_value = {
            "success": True,
            "instance_id": "i-direct999",
            "previous_sku": "c5.2xlarge",
            "new_sku": "c5.xlarge",
            "message": "AWS EC2 instance i-direct999 successfully resized to c5.xlarge.",
        }

        client = TestClient(app)
        res = client.post(
            "/api/actions/execute",
            json={
                "provider": "aws",
                "resource_id": "i-direct999",
                "action": "scale_down",
                "current_sku": "c5.2xlarge",
                "target_sku": "c5.xlarge",
                "cloud_config": {
                    "region": "us-east-1",
                    "aws_credentials": {"access_key_id": "AKIA_EXEC", "secret_access_key": "SK_EXEC"},
                },
                "dry_run": False,
            },
        )
        assert res.status_code == 200
        data = res.json()
        assert data["success"] is True
        assert data["target_sku"] == "c5.xlarge"
        assert "successfully resized to c5.xlarge" in data["message"]



def test_autorollback_monitor_aws_mutation_and_persisted_config():
    """Verifies AutoRollbackMonitor uses AWSAutomationAdapter and persists cloud credentials for rollback."""
    with patch("cba.automation.aws_adapter.AWSAutomationAdapter.resize_instance") as mock_resize:
        mock_resize.return_value = {
            "success": True,
            "instance_id": "i-roll999",
            "previous_sku": "t3.medium",
            "new_sku": "t3.large",
            "message": "AWS EC2 instance i-roll999 successfully resized to t3.large.",
        }

        monitor = AutoRollbackMonitor()
        cloud_cfg = {
            "aws_credentials": {"access_key_id": "AKIA_ROLL", "secret_access_key": "SK_ROLL"},
            "region": "us-east-1",
        }
        action = monitor.register_execution(
            provider="aws",
            resource_id="i-roll999",
            action_type="scale_down",
            target_size="t3.medium",
            original_state={"instance_type": "t3.large"},
            observation_minutes=10,
            cloud_config=cloud_cfg,
        )

        # Confirm persisted cloud_config in metadata
        act_rec = monitor.get_action(action.action_id)
        assert act_rec["metadata"]["cloud_config"]["aws_credentials"]["access_key_id"] == "AKIA_ROLL"

        # Trigger rollback without re-passing cloud_config -> it should pull from metadata
        rb_res = monitor.rollback_action(action.action_id, reason="Operator test rollback")
        assert rb_res["status"] == "rolled_back"
        assert "successfully resized to t3.large" in rb_res["metadata"]["rollback_note"]


def test_aws_adapter_list_instances():
    """Verifies that list_instances discovers EC2 instances and parses tags, specs, and attached volumes."""
    with patch("boto3.Session") as mock_session_cls:
        mock_session = MagicMock()
        mock_session_cls.return_value = mock_session
        mock_ec2 = MagicMock()
        mock_session.client.side_effect = lambda svc, **kw: mock_ec2

        mock_paginator = MagicMock()
        mock_ec2.get_paginator.return_value = mock_paginator
        mock_paginator.paginate.return_value = [
            {
                "Reservations": [
                    {
                        "Instances": [
                            {
                                "InstanceId": "i-list01",
                                "InstanceType": "t3.medium",
                                "State": {"Name": "running"},
                                "Architecture": "x86_64",
                                "Tags": [{"Key": "Name", "Value": "web-prod-01"}, {"Key": "Environment", "Value": "production"}],
                                "BlockDeviceMappings": [
                                    {"DeviceName": "/dev/sda1", "Ebs": {"VolumeId": "vol-01", "Status": "attached"}}
                                ],
                            }
                        ]
                    }
                ]
            }
        ]

        adapter = AWSAutomationAdapter(region="us-east-1")
        instances = adapter.list_instances()

        assert len(instances) == 1
        inst = instances[0]
        assert inst["instance_id"] == "i-list01"
        assert inst["name"] == "web-prod-01"
        assert inst["instance_type"] == "t3.medium"
        assert inst["environment"] == "production"
        assert len(inst["volumes"]) == 1
        assert inst["volumes"][0]["volume_id"] == "vol-01"


def test_aws_adapter_fetch_realtime_metrics():
    """Verifies that fetch_realtime_metrics batches CloudWatch queries for instances."""
    with patch("boto3.Session") as mock_session_cls:
        mock_session = MagicMock()
        mock_session_cls.return_value = mock_session
        mock_cw = MagicMock()
        mock_session.client.side_effect = lambda svc, **kw: mock_cw

        mock_cw.get_metric_data.return_value = {
            "MetricDataResults": [
                {"Id": "cpu_avg_0", "Values": [12.4]},
                {"Id": "cpu_max_0", "Values": [24.8]},
                {"Id": "net_in_0", "Values": [50000.0]},
                {"Id": "net_out_0", "Values": [25000.0]},
            ]
        }

        adapter = AWSAutomationAdapter(region="us-east-1")
        metrics_map = adapter.fetch_realtime_metrics(instance_ids=["i-metric01"], lookback_minutes=15)

        assert "i-metric01" in metrics_map
        m = metrics_map["i-metric01"]
        assert m["cpu_avg"] == 12.4
        assert m["cpu_max"] == 24.8
        assert m["network_in"] == 50000.0
        assert m["network_out"] == 25000.0


def test_aws_adapter_stop_and_start_dry_run():
    """Verifies that stop_instance and start_instance recognize DryRunOperation as valid permission confirmation."""
    with patch("boto3.Session") as mock_session_cls:
        mock_session = MagicMock()
        mock_session_cls.return_value = mock_session
        mock_ec2 = MagicMock()
        mock_session.client.side_effect = lambda svc, **kw: mock_ec2

        dry_run_err = ClientError(
            error_response={"Error": {"Code": "DryRunOperation", "Message": "Dry-run successful."}},
            operation_name="StopInstances",
        )
        mock_ec2.stop_instances.side_effect = dry_run_err
        mock_ec2.start_instances.side_effect = dry_run_err

        adapter = AWSAutomationAdapter(region="us-east-1", dry_run=True)
        stop_res = adapter.stop_instance("i-drytest", dry_run=True)
        assert stop_res["success"] is True
        assert stop_res["dry_run"] is True
        assert "DRY-RUN CONFIRMED" in stop_res["message"]

        start_res = adapter.start_instance("i-drytest", dry_run=True)
        assert start_res["success"] is True
        assert start_res["dry_run"] is True
        assert "DRY-RUN CONFIRMED" in start_res["message"]


def test_aws_adapter_modify_ebs_volume_dry_run():
    """Verifies that modify_ebs_volume recognizes DryRunOperation as permission validation success."""
    with patch("boto3.Session") as mock_session_cls:
        mock_session = MagicMock()
        mock_session_cls.return_value = mock_session
        mock_ec2 = MagicMock()
        mock_session.client.side_effect = lambda svc, **kw: mock_ec2

        mock_ec2.modify_volume.side_effect = ClientError(
            error_response={"Error": {"Code": "DryRunOperation", "Message": "Dry-run successful."}},
            operation_name="ModifyVolume",
        )

        adapter = AWSAutomationAdapter(region="us-east-1", dry_run=True)
        res = adapter.modify_ebs_volume(volume_id="vol-dryebs", volume_type="gp3", dry_run=True)

        assert res["success"] is True
        assert res["dry_run"] is True
        assert res["modification_state"] == "validated"
        assert "DRY-RUN CONFIRMED" in res["message"]


def test_aws_adapter_auto_optimize_flow():
    """Verifies the autonomous auto_optimize method end-to-end with right-sizing and storage optimization."""
    with patch("boto3.Session") as mock_session_cls:
        mock_session = MagicMock()
        mock_session_cls.return_value = mock_session
        mock_ec2 = MagicMock()
        mock_cw = MagicMock()
        mock_sts = MagicMock()
        mock_session.client.side_effect = lambda svc, **kw: (
            mock_sts if svc == "sts" else (mock_cw if svc == "cloudwatch" else mock_ec2)
        )

        mock_sts.get_caller_identity.return_value = {"Account": "123456789012"}

        adapter = AWSAutomationAdapter(region="us-east-1", dry_run=False)

        # Mock inventory with 1 under-utilized instance (t3.xlarge, CPU 8.5%) and 1 gp2 volume
        mock_inventory = [
            {
                "provider": "aws",
                "resource_id": "i-idle01",
                "resource_name": "worker-idle-01",
                "region": "us-east-1",
                "environment": "production",
                "state": "running",
                "instance_type": "t3.xlarge",
                "architecture": "x86_64",
                "cpu_usage": 8.5,
                "cpu_max": 18.0,
                "volumes": [{"volume_id": "vol-gp2-01", "device_name": "/dev/xvda"}],
                "tags": {"Name": "worker-idle-01"},
            }
        ]

        with patch.object(adapter, "fetch_realtime_inventory", return_value=mock_inventory), \
             patch.object(adapter, "resize_instance") as mock_resize, \
             patch.object(adapter, "modify_ebs_volume") as mock_ebs:

            mock_resize.return_value = {
                "success": True,
                "instance_id": "i-idle01",
                "previous_sku": "t3.xlarge",
                "new_sku": "t3.large",
                "message": "Resized i-idle01 from t3.xlarge to t3.large",
            }
            mock_ebs.return_value = {
                "success": True,
                "volume_id": "vol-gp2-01",
                "message": "Volume optimized to gp3",
            }

            result = adapter.auto_optimize(dry_run=False, cpu_idle_threshold=15.0, optimize_storage=True)

            assert result["success"] is True
            assert result["scanned_count"] == 1
            assert result["optimized_count"] == 1
            assert result["total_monthly_savings"] > 0
            assert len(result["actions"]) == 2  # 1 resize + 1 ebs optimize
            assert result["actions"][0]["action"] == "scale_down"
            assert result["actions"][1]["action"] == "storage_optimization"


def test_automation_engine_live_aws_rollback():
    """Verifies that AutomationEngine.rollback invokes AWSAutomationAdapter.rollback_instance for AWS resources."""
    with patch("cba.automation.aws_adapter.AWSAutomationAdapter.rollback_instance") as mock_rollback:
        mock_rollback.return_value = {
            "success": True,
            "instance_id": "i-engine-rb",
            "previous_sku": "t3.medium",
            "new_sku": "t3.xlarge",
            "message": "Safety Rollback: EC2 instance i-engine-rb restored to pre-optimization SKU t3.xlarge.",
        }

        engine = AutomationEngine(dry_run=False)
        rollback_id = "rb-aws-test-01"
        engine.rollback_store[rollback_id] = {
            "resource_id": "i-engine-rb",
            "provider": "aws",
            "original_state": {"instance_type": "t3.xlarge", "region": "us-east-1"},
        }

        mock_execution = MagicMock()
        mock_execution.rollback_id = rollback_id

        result = engine.rollback(mock_execution, resource={"resource_id": "i-engine-rb", "region": "us-east-1"})

        assert result["success"] is True
        assert result["mode"] == "live"
        assert result["resource_id"] == "i-engine-rb"
        assert "restored to pre-optimization SKU t3.xlarge" in result["message"]
        mock_rollback.assert_called_once_with(
            resource_id="i-engine-rb",
            original_sku="t3.xlarge",
            dry_run=False,
        )


def test_api_aws_auto_optimize_endpoint():
    """Verifies that FastAPI /api/aws/auto-optimize invokes the adapter auto_optimize method."""
    from starlette.testclient import TestClient
    from cba.api.server import app

    with patch("cba.automation.aws_adapter.AWSAutomationAdapter.auto_optimize") as mock_opt:
        mock_opt.return_value = {
            "success": True,
            "region": "us-east-1",
            "dry_run": True,
            "scanned_count": 3,
            "optimized_count": 2,
            "skipped_count": 1,
            "total_monthly_savings": 145.50,
            "actions": [
                {
                    "resource_id": "i-001",
                    "action": "scale_down",
                    "current_sku": "m5.2xlarge",
                    "target_sku": "m5.xlarge",
                    "success": True,
                    "estimated_monthly_savings": 140.50,
                    "dry_run": True,
                }
            ],
            "duration_seconds": 1.25,
            "timestamp": "2026-09-05T00:00:00Z",
        }

        client = TestClient(app)
        res = client.post(
            "/api/aws/auto-optimize",
            json={
                "region": "us-east-1",
                "dry_run": True,
                "cpu_idle_threshold": 15.0,
            },
        )
        assert res.status_code == 200
        data = res.json()
        assert data["success"] is True
        assert data["optimized_count"] == 2
        assert data["total_monthly_savings"] == 145.50


def test_detect_resize_intent_from_tags():
    """Verifies that detect_resize_intent_from_tags parses various EC2 tag formats correctly."""
    from cba.core.models import detect_resize_intent_from_tags

    # 1. Explicit SKU in ResizeRecommendation tag
    intent = detect_resize_intent_from_tags({"ResizeRecommendation": "t3.nano"})
    assert intent["has_resize_intent"] is True
    assert intent["target_sku"] == "t3.nano"
    assert intent["action"] == "scale_down"
    assert intent["matched_key"] == "ResizeRecommendation"

    # 2. RightSize tag with prefix
    intent2 = detect_resize_intent_from_tags({"RightSize": "scale_down:t3.micro"})
    assert intent2["has_resize_intent"] is True
    assert intent2["target_sku"] == "t3.micro"
    assert intent2["action"] == "scale_down"

    # 3. TargetSKU tag
    intent3 = detect_resize_intent_from_tags({"TargetSKU": "t3.small", "Name": "test-vm"})
    assert intent3["has_resize_intent"] is True
    assert intent3["target_sku"] == "t3.small"

    # 4. AutoOptimize tag boolean
    intent4 = detect_resize_intent_from_tags({"AutoOptimize": "true"})
    assert intent4["has_resize_intent"] is True
    assert intent4["target_sku"] is None

    # 5. CloudOpt:Recommendation tag
    intent5 = detect_resize_intent_from_tags({"CloudOpt:Recommendation": "downsize to t3.nano"})
    assert intent5["has_resize_intent"] is True
    assert intent5["target_sku"] == "t3.nano"

    # 6. Standard tags without resize intent
    intent6 = detect_resize_intent_from_tags({"Name": "web-prod-01", "Environment": "production"})
    assert intent6["has_resize_intent"] is False
    assert intent6["target_sku"] is None


def test_sanitize_aws_credential():
    """Verifies that sanitize_aws_credential cleans quotes, carriage returns, and whitespace."""
    from cba.live.permissions import sanitize_aws_credential

    assert sanitize_aws_credential(None) == ""
    assert sanitize_aws_credential('  "AKIA12345678" \r\n ') == "AKIA12345678"
    assert sanitize_aws_credential("'ASIA987654321'\n") == "ASIA987654321"
    assert sanitize_aws_credential('IQoJb3JpZ2luX2VjE...\r\n') == "IQoJb3JpZ2luX2VjE..."


def test_test_aws_connection_with_sqs_verification():
    """Verifies test_aws_connection authenticates STS with regional client and verifies SQS queue attributes."""
    from cba.live.permissions import test_aws_connection

    with patch("boto3.Session") as mock_session_cls:
        mock_session = MagicMock()
        mock_session_cls.return_value = mock_session
        mock_sts = MagicMock()
        mock_ec2 = MagicMock()
        mock_sqs = MagicMock()

        def mock_client(service_name, **kwargs):
            if service_name == "sts":
                return mock_sts
            elif service_name == "sqs":
                return mock_sqs
            return mock_ec2

        mock_session.client.side_effect = mock_client

        mock_sts.get_caller_identity.return_value = {
            "Account": "123456789012",
            "Arn": "arn:aws:iam::123456789012:user/admin",
            "UserId": "AIDA12345678",
        }
        mock_ec2.describe_instances.return_value = {
            "Reservations": [
                {"Instances": [{"InstanceId": "i-test01", "State": {"Name": "running"}}]}
            ]
        }
        mock_sqs.get_queue_attributes.return_value = {
            "Attributes": {
                "ApproximateNumberOfMessages": "5",
                "ApproximateNumberOfMessagesNotVisible": "0",
                "QueueArn": "arn:aws:sqs:ap-south-1:123456789012:cloudopt-queue",
            }
        }

        ok, msg, details = test_aws_connection(
            aws_access_key_id='"AKIA12345678"',
            aws_secret_access_key="secretKey123",
            aws_session_token="sessionToken123\r\n",
            region="ap-south-1",
            sqs_queue_url="https://sqs.ap-south-1.amazonaws.com/123456789012/cloudopt-queue",
        )

        assert ok is True
        assert details["account_id"] == "123456789012"
        assert details["instances_discovered"] == 1
        assert details["sqs_verified"] is True
        sqs_v = details["sqs_details"]
        assert sqs_v["status"] == "connected"
        assert sqs_v["approximate_number_of_messages"] == 5
        assert sqs_v["queue_arn"] == "arn:aws:sqs:ap-south-1:123456789012:cloudopt-queue"


def test_aws_adapter_tag_instance():
    """Verifies that AWSAutomationAdapter.tag_instance correctly applies tags via EC2 API."""
    with patch("boto3.Session") as mock_session_cls:
        mock_session = MagicMock()
        mock_session_cls.return_value = mock_session
        mock_ec2 = MagicMock()
        mock_sts = MagicMock()
        mock_session.client.side_effect = lambda svc, **kw: (
            mock_sts if svc == "sts" else mock_ec2
        )
        mock_sts.get_caller_identity.return_value = {"Account": "123456789012"}

        adapter = AWSAutomationAdapter(region="ap-south-1", dry_run=False)

        # 1. Dry run
        dry_res = adapter.tag_instance(
            instance_id="i-testtag01",
            tags={"CloudOpt:Recommendation": "t3.nano", "CloudOpt:Action": "scale_down"},
            dry_run=True,
        )
        assert dry_res["success"] is True
        assert dry_res["dry_run"] is True
        assert "DRY-RUN" in dry_res["message"]

        # 2. Live execution
        live_res = adapter.tag_instance(
            instance_id="i-testtag01",
            tags={"CloudOpt:Recommendation": "t3.nano", "CloudOpt:Action": "scale_down"},
            dry_run=False,
        )
        assert live_res["success"] is True
        assert live_res["dry_run"] is False
        assert mock_ec2.create_tags.call_count == 2
        call_kwargs = mock_ec2.create_tags.call_args[1]
        assert call_kwargs["Resources"] == ["i-testtag01"]
        tag_keys = [t["Key"] for t in call_kwargs["Tags"]]
        assert "CloudOpt:Recommendation" in tag_keys
        assert "CloudOpt:Action" in tag_keys


def test_api_aws_tag_recommendation_endpoint():
    """Verifies the /api/aws/tag-recommendation FastAPI endpoint."""
    from starlette.testclient import TestClient
    from cba.api.server import app

    with patch("cba.automation.aws_adapter.AWSAutomationAdapter.tag_instance") as mock_tag:
        mock_tag.return_value = {
            "success": True,
            "instance_id": "i-api-tag01",
            "tags": {"CloudOpt:Recommendation": "t3.nano", "CloudOpt:Action": "scale_down"},
            "dry_run": False,
            "message": "Successfully tagged instance i-api-tag01 on AWS.",
        }

        client = TestClient(app)
        res = client.post(
            "/api/aws/tag-recommendation",
            json={
                "instance_id": "i-api-tag01",
                "target_sku": "t3.nano",
                "action": "scale_down",
                "region": "ap-south-1",
                "dry_run": False,
            },
        )
        assert res.status_code == 200
        data = res.json()
        assert data["success"] is True
        assert data["instance_id"] == "i-api-tag01"
        assert "CloudOpt:Recommendation" in data["tags"]


def test_aws_adapter_auto_optimize_unauthorized_fails_truthfully():
    """Verifies that auto_optimize does not falsify success if IAM returns UnauthorizedOperation."""
    with patch("boto3.Session") as mock_session_cls:
        mock_session = MagicMock()
        mock_session_cls.return_value = mock_session
        mock_ec2 = MagicMock()
        mock_cw = MagicMock()
        mock_sts = MagicMock()
        mock_session.client.side_effect = lambda svc, **kw: (
            mock_sts if svc == "sts" else (mock_cw if svc == "cloudwatch" else mock_ec2)
        )
        mock_sts.get_caller_identity.return_value = {"Account": "123456789012"}

        adapter = AWSAutomationAdapter(region="ap-southeast-2", dry_run=False)

        mock_inventory = [
            {
                "provider": "aws",
                "resource_id": "i-unauth01",
                "resource_name": "worker-unauth-01",
                "region": "ap-southeast-2",
                "environment": "production",
                "state": "running",
                "instance_type": "t3.micro",
                "cpu_usage": 0.4,
                "cpu_max": 0.5,
                "volumes": [],
                "tags": {},
            }
        ]

        with patch.object(adapter, "fetch_realtime_inventory", return_value=mock_inventory), \
             patch.object(adapter, "resize_instance") as mock_resize:

            mock_resize.return_value = {
                "success": False,
                "instance_id": "i-unauth01",
                "previous_sku": "t3.micro",
                "target_sku": "t3.nano",
                "error": "ClientError: (UnauthorizedOperation) You are not authorized to perform this operation.",
                "message": "AWS live instance resize failed: ClientError: (UnauthorizedOperation)",
            }

            result = adapter.auto_optimize(
                instance_ids=["i-unauth01"],
                dry_run=False,
                target_sku_map={"i-unauth01": "t3.nano"},
                optimize_storage=False,
            )

            # Must NOT mask UnauthorizedOperation as success!
            assert result["success"] is False
            assert result["optimized_count"] == 0
            assert result["skipped_count"] == 1
            assert len(result["actions"]) == 1
            assert result["actions"][0]["success"] is False
            assert "UnauthorizedOperation" in result["actions"][0]["message"]


def test_aws_adapter_sync_local_caches_post_resize():
    """Verifies that _sync_local_caches_post_resize correctly updates telemetry and recommendation CSV files."""
    import tempfile
    import os
    from pathlib import Path
    import pandas as pd

    with tempfile.TemporaryDirectory() as td:
        old_cwd = os.getcwd()
        try:
            os.chdir(td)
            live_dir = Path(td) / "outputs" / "live"
            live_dir.mkdir(parents=True, exist_ok=True)

            t_csv = live_dir / "telemetry.csv"
            t_csv.write_text(
                "provider,account_id,resource_id,resource_name,region,environment,state,instance_type,cpu_usage,cpu_max,memory_usage,original_state\n"
                "aws,245716172295,i-sync01,demo-instance,ap-southeast-2,untagged,running,t3.micro,0.39,0.47,35.0,{'instance_type': 't3.micro'}\n",
                encoding="utf-8",
            )

            r_csv = live_dir / "recommendations.csv"
            r_csv.write_text(
                "resource_id,provider,environment,current_sku,recommended_sku,action,monthly_savings,summary\n"
                "i-sync01,aws,untagged,t3.micro,t3.nano,scale_down,3.80,Downsizing i-sync01 from t3.micro to t3.nano\n",
                encoding="utf-8",
            )

            adapter = AWSAutomationAdapter(region="ap-southeast-2")
            adapter._sync_local_caches_post_resize("i-sync01", "t3.micro", "t3.nano")

            tdf = pd.read_csv(t_csv)
            assert tdf.iloc[0]["instance_type"] == "t3.nano"

            rdf = pd.read_csv(r_csv)
            assert rdf.iloc[0]["current_sku"] == "t3.nano"
            assert rdf.iloc[0]["action"] == "no_action"
            assert float(rdf.iloc[0]["monthly_savings"]) == 0.0
        finally:
            os.chdir(old_cwd)



