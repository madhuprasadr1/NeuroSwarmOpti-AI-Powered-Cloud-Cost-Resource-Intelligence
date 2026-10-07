from __future__ import annotations

import json
from typing import Any, Dict, Optional, Tuple


# ==============================================================================
# MINIMUM IAM & RBAC POLICY DOCUMENTS
# ==============================================================================

AWS_MINIMUM_IAM_POLICY = {
    "Version": "2012-10-17",
    "Statement": [
        {
            "Sid": "CloudOptMetricsAndDiscovery",
            "Effect": "Allow",
            "Action": [
                "ec2:DescribeInstances",
                "ec2:DescribeInstanceStatus",
                "cloudwatch:GetMetricData",
                "cloudwatch:GetMetricStatistics",
                "cloudwatch:ListMetrics",
                "sts:GetCallerIdentity",
            ],
            "Resource": "*",
        },
        {
            "Sid": "CloudOptSQSQueueOperations",
            "Effect": "Allow",
            "Action": [
                "sqs:SendMessage",
                "sqs:ReceiveMessage",
                "sqs:DeleteMessage",
                "sqs:GetQueueAttributes",
                "sqs:GetQueueUrl",
            ],
            "Resource": "arn:aws:sqs:*:*:cloudopt-*",
        },
        {
            "Sid": "CloudOptAutoOptimizationAndRollback",
            "Effect": "Allow",
            "Action": [
                "ec2:ModifyInstanceAttribute",
                "ec2:StartInstances",
                "ec2:StopInstances",
            ],
            "Resource": "arn:aws:ec2:*:*:instance/*",
        },
    ],
}

AZURE_MINIMUM_RBAC_ROLE = {
    "Name": "CloudOpt AI Minimum Role",
    "IsCustom": True,
    "Description": "Minimum permissions for CloudOpt AI real-time metrics, VM right-sizing, queue dispatch, and rollback.",
    "Actions": [
        "Microsoft.Compute/virtualMachines/read",
        "Microsoft.Compute/virtualMachines/write",
        "Microsoft.Compute/virtualMachines/start/action",
        "Microsoft.Compute/virtualMachines/deallocate/action",
        "Microsoft.Insights/metrics/read",
        "Microsoft.Resources/subscriptions/resourceGroups/read",
    ],
    "DataActions": [
        "Microsoft.Storage/storageAccounts/queueServices/queues/messages/read",
        "Microsoft.Storage/storageAccounts/queueServices/queues/messages/write",
        "Microsoft.Storage/storageAccounts/queueServices/queues/messages/process/action",
        "Microsoft.Storage/storageAccounts/queueServices/queues/messages/delete",
    ],
    "AssignableScopes": ["/subscriptions/<SUBSCRIPTION_ID>"],
}

GCP_MINIMUM_IAM_ROLES = {
    "description": "Minimum GCP IAM Roles required for CloudOpt AI real-time monitoring and auto-optimization.",
    "roles": [
        {
            "role": "roles/monitoring.viewer",
            "purpose": "Read Compute Engine CPU and network metrics from Google Cloud Monitoring API.",
        },
        {
            "role": "roles/compute.instanceAdmin.v1",
            "purpose": "Modify GCE machine types (setMachineType), start/stop instances during right-sizing.",
            "note": "Can be scoped to specific instances or service accounts.",
        },
        {
            "role": "roles/pubsub.publisher",
            "purpose": "Publish optimization commands and real-time alert events to Cloud Pub/Sub.",
        },
        {
            "role": "roles/pubsub.subscriber",
            "purpose": "Consume optimization queue messages in the automation worker.",
        },
    ],
    "custom_role_permissions": [
        "monitoring.timeSeries.list",
        "compute.instances.get",
        "compute.instances.list",
        "compute.instances.setMachineType",
        "compute.instances.start",
        "compute.instances.stop",
        "pubsub.topics.publish",
        "pubsub.subscriptions.consume",
    ],
}


# ==============================================================================
# LIVE CONNECTION TESTERS
# ==============================================================================

AWS_CLOUDFORMATION_TEMPLATE = """AWSTemplateFormatVersion: '2010-09-09'
Description: 'CloudOpt AI — Automated Least-Privilege IAM User & Access Keys'
Resources:
  CloudOptAgentUser:
    Type: 'AWS::IAM::User'
    Properties:
      UserName: 'cloudopt-agent'
      ManagedPolicyArns:
        - 'arn:aws:iam::aws:policy/AmazonEC2ReadOnlyAccess'
        - 'arn:aws:iam::aws:policy/CloudWatchReadOnlyAccess'
  CloudOptAccessKey:
    Type: 'AWS::IAM::AccessKey'
    Properties:
      UserName: !Ref CloudOptAgentUser
Outputs:
  CloudOptAccessKeyId:
    Description: 'AWS Access Key ID for CloudOpt AI'
    Value: !Ref CloudOptAccessKey
  CloudOptSecretAccessKey:
    Description: 'AWS Secret Access Key for CloudOpt AI'
    Value: !GetAtt CloudOptAccessKey.SecretAccessKey
"""

AZURE_RBAC_TEMPLATE = json.dumps(AZURE_MINIMUM_RBAC_ROLE, indent=2)

GCP_IAM_CUSTOM_ROLE_YAML = """title: 'CloudOpt AI Least-Privilege Custom Role'
description: 'Minimum permissions for CloudOpt AI real-time monitoring and autonomous compute optimization.'
stage: 'GA'
includedPermissions:
  - 'monitoring.timeSeries.list'
  - 'compute.instances.get'
  - 'compute.instances.list'
  - 'compute.instances.setMachineType'
  - 'compute.instances.start'
  - 'compute.instances.stop'
  - 'pubsub.topics.publish'
  - 'pubsub.subscriptions.consume'
"""


def auto_attach_aws_policies(
    access_key_id: str,
    secret_access_key: str,
    region: str = "us-east-1",
    session_token: Optional[str] = None,
    user_name: Optional[str] = None,
) -> Tuple[bool, str, Dict[str, Any]]:
    """Attempts to auto-attach AmazonEC2ReadOnlyAccess & CloudWatchReadOnlyAccess via IAM API.
    
    Handles temporary session tokens gracefully: AWS IAM APIs forbid GetSessionToken credentials,
    so this attempts permanent ambient credentials if available, or returns exact CLI commands.
    """
    import os
    import boto3
    from botocore.exceptions import ClientError

    ak = (access_key_id or "").strip()
    sk = (secret_access_key or "").strip()
    st = (session_token or "").strip() or None
    reg = (region or "").strip() or "us-east-1"
    target = user_name

    # If key starts with ASIA (temporary STS credential), AWS strictly forbids calling IAM APIs with GetSessionToken creds.
    # Check if ambient permanent AKIA credentials exist in environment.
    if ak.startswith("ASIA"):
        env_ak = os.getenv("AWS_ACCESS_KEY_ID", "").strip()
        env_sk = os.getenv("AWS_SECRET_ACCESS_KEY", "").strip()
        if env_ak.startswith("AKIA") and env_sk:
            ak = env_ak
            sk = env_sk
            st = None

    cli_cmd = f"aws iam attach-user-policy --user-name {target or 'user'} --policy-arn arn:aws:iam::aws:policy/AmazonEC2ReadOnlyAccess"
    console_url = f"https://console.aws.amazon.com/iam/home#/users/{target}" if target else "https://console.aws.amazon.com/iam/home#/users"
    details: Dict[str, Any] = {
        "target_user": target or "",
        "cli_command": cli_cmd,
        "console_url": console_url,
    }

    # If still using ASIA temporary session token, AWS IAM will reject it
    if ak.startswith("ASIA"):
        return False, (
            "AWS IAM Security Rule: AWS IAM APIs cannot be executed using temporary STS session tokens (ASIA...). "
            f"Please attach AmazonEC2ReadOnlyAccess using the AWS Console link or run the AWS CLI command below as an administrator."
        ), details

    try:
        session_kwargs: Dict[str, Any] = {
            "aws_access_key_id": ak,
            "aws_secret_access_key": sk,
            "region_name": reg,
        }
        # Only pass session_token if key is not permanent AKIA
        if st and not ak.startswith("AKIA"):
            session_kwargs["aws_session_token"] = st

        session = boto3.Session(**session_kwargs)
        sts = session.client("sts")
        identity = sts.get_caller_identity()
        arn = identity.get("Arn", "")

        if not target:
            if ":user/" in arn:
                target = arn.split(":user/")[-1]
            else:
                return False, f"Auto-attach requires an IAM User. Active identity is: {arn}", details

        details["target_user"] = target
        details["cli_command"] = f"aws iam attach-user-policy --user-name {target} --policy-arn arn:aws:iam::aws:policy/AmazonEC2ReadOnlyAccess"
        details["console_url"] = f"https://console.aws.amazon.com/iam/home#/users/{target}"

        iam = session.client("iam")
        policies = [
            "arn:aws:iam::aws:policy/AmazonEC2ReadOnlyAccess",
            "arn:aws:iam::aws:policy/CloudWatchReadOnlyAccess",
        ]
        attached = []
        for pol in policies:
            iam.attach_user_policy(UserName=target, PolicyArn=pol)
            attached.append(pol.split("/")[-1])

        return True, f"Successfully attached {', '.join(attached)} to IAM user '{target}'.", details

    except ClientError as ce:
        err_code = ce.response.get("Error", {}).get("Code", "")
        err_msg = ce.response.get("Error", {}).get("Message", "")
        if err_code in ["InvalidClientTokenId", "InvalidToken"]:
            return False, (
                "AWS IAM Security Policy: AWS IAM APIs cannot be executed using temporary STS session credentials. "
                f"Please attach AmazonEC2ReadOnlyAccess in the AWS Console or use the AWS CLI command below."
            ), details
        elif err_code in ["AccessDenied", "UnauthorizedOperation"]:
            return False, (
                f"IAM User '{target or 'current'}' does not have the 'iam:AttachUserPolicy' privilege to attach policies to itself. "
                f"Please attach AmazonEC2ReadOnlyAccess in the AWS Console or ask your AWS account administrator to run the command below."
            ), details
        return False, f"Auto-attach failed ({err_code}): {err_msg}", details

    except Exception as exc:
        return False, f"Auto-attach failed: {str(exc)}", details


def request_aws_session_token(
    access_key_id: Optional[str] = None,
    secret_access_key: Optional[str] = None,
    region: str = "ap-southeast-2",
    duration_seconds: int = 43200,
) -> Tuple[bool, str, Dict[str, Any]]:
    """Requests an AWS STS Session Token by running AWS STS commands in the backend.
    
    Can use provided access_key_id & secret_access_key, or fall back to system
    AWS environment variables / AWS credentials file.
    Returns: (success, message, details_dict)
      details_dict contains:
        - access_key_id (ASIA...)
        - secret_access_key
        - session_token
        - expiration (ISO string)
        - duration_seconds
    """
    try:
        import os
        import boto3
        
        session_kwargs: Dict[str, Any] = {"region_name": region.strip() or "ap-southeast-2"}
        ak = (access_key_id or "").strip() or os.getenv("AWS_ACCESS_KEY_ID", "").strip()
        sk = (secret_access_key or "").strip() or os.getenv("AWS_SECRET_ACCESS_KEY", "").strip()

        # AWS STS Security Rule:
        # GetSessionToken requires permanent IAM user credentials (AKIA...).
        # If an ASIA key was passed (e.g. from an expired temporary session),
        # fall back to ambient permanent credentials (env vars or ~/.aws/credentials).
        if ak.startswith("ASIA"):
            env_ak = os.getenv("AWS_ACCESS_KEY_ID", "").strip()
            env_sk = os.getenv("AWS_SECRET_ACCESS_KEY", "").strip()
            if env_ak.startswith("AKIA") and env_sk:
                ak = env_ak
                sk = env_sk
            else:
                ak = ""
                sk = ""
        
        if ak:
            session_kwargs["aws_access_key_id"] = ak
        if sk:
            session_kwargs["aws_secret_access_key"] = sk
            
        session = boto3.Session(**session_kwargs)
        sts = session.client("sts")
        
        # Duration between 15 mins (900s) and 36 hrs (129600s)
        dur = max(900, min(129600, int(duration_seconds)))
        resp = sts.get_session_token(DurationSeconds=dur)
        creds = resp.get("Credentials", {})
        
        exp = creds.get("Expiration")
        exp_iso = exp.isoformat() if hasattr(exp, "isoformat") else str(exp)
        
        return True, "Successfully acquired AWS STS Session Token via backend command.", {
            "access_key_id": creds.get("AccessKeyId"),
            "secret_access_key": creds.get("SecretAccessKey"),
            "session_token": creds.get("SessionToken"),
            "expiration": exp_iso,
            "duration_seconds": dur,
        }
    except Exception as exc:
        return False, f"AWS STS Session Token command failed: {str(exc)}", {}


def sanitize_aws_credential(val: Optional[str]) -> str:
    """Cleans AWS credential strings by stripping whitespace, linebreaks, and enclosing quotes."""
    if not val:
        return ""
    return str(val).strip().strip("\"'").replace("\r", "").replace("\n", "").strip()


def test_aws_connection(
    access_key_id: Optional[str] = None,
    secret_access_key: Optional[str] = None,
    region: str = "us-east-1",
    session_token: Optional[str] = None,
    sqs_queue_url: Optional[str] = None,
    **kwargs: Any,
) -> Tuple[bool, str, Dict[str, Any]]:
    """Tests real AWS authentication using STS GetCallerIdentity, checks EC2 DescribeInstances,
    and validates SQS Queue connectivity if URL is provided.
    """
    try:
        import re
        import boto3
        from botocore.exceptions import ClientError

        raw_ak = access_key_id or kwargs.get("aws_access_key_id", "")
        raw_sk = secret_access_key or kwargs.get("aws_secret_access_key", "")
        raw_st = session_token or kwargs.get("aws_session_token")
        raw_reg = region or kwargs.get("aws_region", "us-east-1")
        raw_sqs = sqs_queue_url or kwargs.get("sqs_url")

        ak = sanitize_aws_credential(raw_ak)
        sk = sanitize_aws_credential(raw_sk)
        st = sanitize_aws_credential(raw_st) or None
        reg = sanitize_aws_credential(raw_reg) or "us-east-1"
        sqs_url = sanitize_aws_credential(raw_sqs) or None

        # Check if SQS URL implies a specific AWS region
        detected_sqs_region = None
        if sqs_url:
            m = re.search(r"sqs[.-]([a-z0-9-]+)\.amazonaws\.com", sqs_url, re.IGNORECASE)
            if m:
                detected_sqs_region = m.group(1).lower()

        # AWS STS Security Rule:
        # - AKIA keys are permanent IAM User credentials.
        # - ASIA keys are temporary session credentials and require an aws_session_token.
        if ak.startswith("ASIA") and not st:
            return False, (
                f"AWS STS Security Requirement: The Access Key ID '{ak[:8]}...' starts with 'ASIA', "
                f"which indicates temporary session credentials. A valid AWS Session Token is mandatory. "
                f"Please paste your session token into the AWS Session Token field."
            ), {}

        session_kwargs: Dict[str, Any] = {
            "aws_access_key_id": ak,
            "aws_secret_access_key": sk,
            "region_name": reg,
        }
        if st:
            session_kwargs["aws_session_token"] = st

        session = boto3.Session(**session_kwargs)
        # Always configure STS client with region to support regional STS endpoints
        sts = session.client("sts", region_name=reg)

        # Step 1: Validate STS Authentication (Gold Standard of AWS Identity)
        try:
            identity = sts.get_caller_identity()
        except ClientError as ce:
            err_code = ce.response.get("Error", {}).get("Code", "")
            err_msg = ce.response.get("Error", {}).get("Message", "")
            if err_code == "InvalidClientTokenId":
                if ak.startswith("AKIA"):
                    return False, (
                        f"AWS Authentication failed (InvalidClientTokenId): The Access Key ID '{ak[:8]}...' "
                        f"does not exist in AWS IAM, has been deleted, or was typed incorrectly. "
                        f"Please verify your credentials in the AWS IAM Console."
                    ), {}
                elif ak.startswith("ASIA"):
                    # Attempt automatic seamless auto-recovery using ambient permanent credentials
                    try:
                        st_ok, _, st_details = request_aws_session_token(
                            region=reg,
                            duration_seconds=43200,
                        )
                        if st_ok and st_details.get("session_token"):
                            retry_ok, retry_msg, retry_details = test_aws_connection(
                                access_key_id=st_details["access_key_id"],
                                secret_access_key=st_details["secret_access_key"],
                                region=reg,
                                session_token=st_details["session_token"],
                                sqs_queue_url=sqs_url,
                            )
                            if retry_ok:
                                retry_details["generated_session_token"] = st_details
                                return True, f"Expired STS token automatically renewed: {retry_msg}", retry_details
                    except Exception:
                        pass

                    return False, (
                        f"AWS Authentication failed (InvalidClientTokenId): The temporary STS session token "
                        f"for key '{ak[:8]}...' is invalid or has expired. "
                        f"Please verify or renew your AWS Session Token."
                    ), {}
                else:
                    return False, f"AWS Authentication failed (InvalidClientTokenId): {err_msg}", {}
            raise ce

        account_id = identity.get("Account", "")
        arn = identity.get("Arn", "")
        user_id = identity.get("UserId", "")
        user_name = arn.split(":user/")[-1] if ":user/" in arn else arn.split("/")[-1]

        # Step 1.5: Auto-fetch session token if IAM user key (AKIA...) used
        generated_token = None
        if ak.startswith("AKIA") and not st:
            try:
                st_ok, _, st_details = request_aws_session_token(
                    access_key_id=ak,
                    secret_access_key=sk,
                    region=reg,
                    duration_seconds=43200,
                )
                if st_ok:
                    generated_token = st_details
            except Exception:
                pass

        # Step 2: Check EC2 Read Permission & Active Instances
        ec2_authorized = True
        ec2_instances_count = 0
        ec2 = session.client("ec2", region_name=reg)
        try:
            desc_res = ec2.describe_instances(MaxResults=10)
            for res in desc_res.get("Reservations", []):
                ec2_instances_count += len(res.get("Instances", []))
        except ClientError as ce:
            code = ce.response.get("Error", {}).get("Code", "")
            if code in ("UnauthorizedOperation", "AccessDenied"):
                ec2_authorized = False
            else:
                raise ce

        # Step 3: Validate SQS Queue if provided
        sqs_verified = False
        sqs_details: Dict[str, Any] = {}
        if sqs_url:
            sqs_reg = detected_sqs_region or reg
            try:
                sqs = session.client("sqs", region_name=sqs_reg)
                resolved_url = sqs_url
                if not resolved_url.startswith("http://") and not resolved_url.startswith("https://"):
                    try:
                        q_res = sqs.get_queue_url(QueueName=resolved_url)
                        resolved_url = q_res.get("QueueUrl", resolved_url)
                    except Exception:
                        pass
                attrs = sqs.get_queue_attributes(
                    QueueUrl=resolved_url,
                    AttributeNames=["ApproximateNumberOfMessages", "QueueArn"],
                ).get("Attributes", {})
                sqs_verified = True
                sqs_details = {
                    "status": "connected",
                    "queue_url": resolved_url,
                    "queue_arn": attrs.get("QueueArn", ""),
                    "pending_messages": int(attrs.get("ApproximateNumberOfMessages", 0)),
                    "approximate_number_of_messages": int(attrs.get("ApproximateNumberOfMessages", 0)),
                    "region": sqs_reg,
                    "verified": True,
                }
            except Exception as sqs_err:
                sqs_details = {
                    "status": "failed",
                    "queue_url": sqs_url,
                    "verified": False,
                    "error": str(sqs_err),
                    "region": sqs_reg,
                }

        cli_fix = (
            f"aws iam attach-user-policy --user-name {user_name} --policy-arn arn:aws:iam::aws:policy/AdministratorAccess"
        )
        console_url = f"https://console.aws.amazon.com/iam/home#/users/{user_name}?section=permissions"

        details = {
            "account_id": account_id,
            "arn": arn,
            "user_id": user_id,
            "user_name": user_name,
            "region": reg,
            "sts_authenticated": True,
            "ec2_authorized": ec2_authorized,
            "instances_discovered": ec2_instances_count,
            "status": "fully_authorized" if ec2_authorized else "sts_authenticated_ec2_pending",
            "cli_command": cli_fix,
            "console_url": console_url,
            "generated_session_token": generated_token,
            "has_session_token": bool(st or generated_token),
            "sqs_verified": sqs_verified,
            "sqs_details": sqs_details,
            "detected_sqs_region": detected_sqs_region,
        }

        if ec2_authorized:
            sqs_tag = " & SQS Queue Verified" if sqs_verified else ""
            msg = f"Connected & Authenticated with AWS Account {account_id} as '{arn}'. EC2 Authorized ({ec2_instances_count} instances discovered in {reg}){sqs_tag}."
        else:
            msg = f"Connected & Authenticated with AWS Account {account_id} as '{user_name}'. IAM policy setup recommended."

        return True, msg, details
    except Exception as exc:
        return False, f"AWS connection failed: {exc}", {}


def test_azure_connection(
    tenant_id: str,
    client_id: str,
    client_secret: str,
    subscription_id: str,
) -> Tuple[bool, str, Dict[str, Any]]:
    """Tests real Azure Service Principal authentication against Compute Management API."""
    try:
        from azure.identity import ClientSecretCredential
        from azure.mgmt.compute import ComputeManagementClient

        credential = ClientSecretCredential(
            tenant_id=tenant_id.strip(),
            client_id=client_id.strip(),
            client_secret=client_secret.strip(),
        )
        compute = ComputeManagementClient(credential, subscription_id.strip())
        # Query 1 page to test authorization
        vms = list(compute.virtual_machines.list_all())
        return True, f"Successfully connected to Azure subscription. Found {len(vms)} VMs.", {
            "subscription_id": subscription_id.strip(),
            "vm_count": len(vms),
        }
    except Exception as exc:
        return False, f"Azure connection failed: {exc}", {}


def test_gcp_connection(
    project_id: str,
    service_account_json: str,
) -> Tuple[bool, str, Dict[str, Any]]:
    """Tests real GCP Service Account key authentication against Compute Engine API."""
    try:
        from google.oauth2 import service_account
        from google.cloud import compute_v1

        info = json.loads(service_account_json)
        credentials = service_account.Credentials.from_service_account_info(info)
        instances_client = compute_v1.InstancesClient(credentials=credentials)
        # Test aggregated list
        aggregated = instances_client.aggregated_list(project=project_id.strip())
        count = sum(1 for _, scoped in aggregated if scoped.instances)
        return True, f"Successfully connected to GCP project '{project_id}'.", {
            "project_id": project_id.strip(),
            "service_account_email": info.get("client_email"),
        }
    except Exception as exc:
        return False, f"GCP connection failed: {exc}", {}
