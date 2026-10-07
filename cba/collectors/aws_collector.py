import time
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import boto3
from botocore.exceptions import BotoCoreError, ClientError

from .base import BaseCollector, BillingData, ResourceData
from ..core.exceptions import (
    AuthenticationError,
    AuthorizationError,
    CollectorError,
)


class AWSCollector(BaseCollector):
    def __init__(
        self,
        config: Any,
        credentials: Optional[Dict[str, str]] = None,
    ):
        super().__init__(
            config,
            credentials,
        )

        self.session = None
        self.ce_client = None
        self.ec2_client = None
        self.rds_client = None
        self.lambda_client = None
        self.s3_client = None
        self.cloudwatch_client = None
        self.sts_client = None

    def authenticate(self) -> None:
        try:
            access_key_id = self.credentials.get(
                "access_key_id"
            )
            secret_access_key = self.credentials.get(
                "secret_access_key"
            )
            session_token = self.credentials.get(
                "session_token"
            )

            if not access_key_id or not secret_access_key:
                raise AuthenticationError(
                    "AWS access key ID and secret access key are required."
                )

            session_kwargs = {
                "aws_access_key_id": access_key_id,
                "aws_secret_access_key": secret_access_key,
                "region_name": "us-east-1",
            }

            if session_token:
                session_kwargs[
                    "aws_session_token"
                ] = session_token

            self.session = boto3.Session(
                **session_kwargs
            )

            self.ce_client = self.session.client(
                "ce",
                region_name="us-east-1",
            )
            self.ec2_client = self.session.client(
                "ec2"
            )
            self.rds_client = self.session.client(
                "rds"
            )
            self.lambda_client = self.session.client(
                "lambda"
            )
            self.s3_client = self.session.client(
                "s3"
            )
            self.cloudwatch_client = self.session.client(
                "cloudwatch"
            )
            self.sts_client = self.session.client(
                "sts"
            )

            self.sts_client.get_caller_identity()

            self.authenticated = True
            self.last_authentication_time = (
                datetime.now(timezone.utc)
            )

        except ClientError as exc:
            error_code = (
                exc.response.get("Error", {})
                .get("Code", "")
            )

            if error_code in {
                "AccessDenied",
                "AccessDeniedException",
                "UnauthorizedOperation",
                "Unauthorized",
            }:
                raise AuthorizationError(
                    f"AWS authorization failed: {exc}"
                ) from exc

            raise AuthenticationError(
                f"AWS authentication failed: {exc}"
            ) from exc

        except BotoCoreError as exc:
            raise AuthenticationError(
                f"AWS connection failed: {exc}"
            ) from exc

        except (
            AuthenticationError,
            AuthorizationError,
        ):
            raise

        except Exception as exc:
            raise AuthenticationError(
                f"AWS authentication failed: {exc}"
            ) from exc

    def collect_billing_data(
        self,
        start_date: datetime,
        end_date: datetime,
    ) -> List[BillingData]:
        self.validate_date_range(
            start_date,
            end_date,
        )
        self.ensure_authenticated()

        try:
            start = self._normalize_datetime(
                start_date
            )
            end = self._normalize_datetime(
                end_date
            )

            response = self.ce_client.get_cost_and_usage(
                TimePeriod={
                    "Start": start.strftime(
                        "%Y-%m-%d"
                    ),
                    "End": end.strftime(
                        "%Y-%m-%d"
                    ),
                },
                Granularity="DAILY",
                Metrics=[
                    "UnblendedCost",
                    "UsageQuantity",
                ],
                GroupBy=[
                    {
                        "Type": "DIMENSION",
                        "Key": "SERVICE",
                    },
                    {
                        "Type": "DIMENSION",
                        "Key": "REGION",
                    },
                    {
                        "Type": "DIMENSION",
                        "Key": "USAGE_TYPE",
                    },
                ],
            )

            billing_data = []

            for result in response.get(
                "ResultsByTime",
                [],
            ):
                period_start = self._parse_date(
                    result["TimePeriod"]["Start"]
                )

                period_end = self._parse_date(
                    result["TimePeriod"]["End"]
                )

                for group in result.get(
                    "Groups",
                    [],
                ):
                    keys = group.get(
                        "Keys",
                        [],
                    )

                    metrics = group.get(
                        "Metrics",
                        {},
                    )

                    service = self._extract_group_value(
                        keys,
                        0,
                    )

                    region = self._extract_group_value(
                        keys,
                        1,
                    )

                    usage_type = (
                        self._extract_group_value(
                            keys,
                            2,
                        )
                    )

                    cost_metric = metrics.get(
                        "UnblendedCost",
                        {},
                    )

                    usage_metric = metrics.get(
                        "UsageQuantity",
                        {},
                    )

                    cost = self._safe_float(
                        cost_metric.get(
                            "Amount",
                            0.0,
                        )
                    )

                    usage_amount = self._safe_float(
                        usage_metric.get(
                            "Amount",
                            0.0,
                        )
                    )

                    currency = str(
                        cost_metric.get(
                            "Unit",
                            "USD",
                        )
                    )

                    resource_id = (
                        "service:"
                        + service
                    )

                    billing_data.append(
                        BillingData(
                            provider="aws",
                            account_id=(
                                self.config.aws.account_id
                                or "unknown"
                            ),
                            service=service,
                            region=region,
                            resource_id=resource_id,
                            resource_name=service,
                            usage_type=usage_type,
                            usage_amount=usage_amount,
                            usage_unit=str(
                                usage_metric.get(
                                    "Unit",
                                    "",
                                )
                            ),
                            cost=max(
                                0.0,
                                cost,
                            ),
                            currency=currency,
                            start_time=period_start,
                            end_time=period_end,
                            tags={},
                            environment=None,
                            cost_center=None,
                        )
                    )

            return billing_data

        except ClientError as exc:
            raise CollectorError(
                f"Failed to collect AWS billing data: {exc}"
            ) from exc

        except BotoCoreError as exc:
            raise CollectorError(
                f"AWS billing service unavailable: {exc}"
            ) from exc

        except Exception as exc:
            if isinstance(
                exc,
                CollectorError,
            ):
                raise

            raise CollectorError(
                f"Failed to collect AWS billing data: {exc}"
            ) from exc

    def collect_resource_data(
        self,
    ) -> List[ResourceData]:
        self.ensure_authenticated()

        resources = []

        try:
            regions = self._get_target_regions()

            for region in regions:
                self.ec2_client = self.session.client(
                    "ec2",
                    region_name=region,
                )

                resources.extend(
                    self._collect_ec2_instances(
                        region
                    )
                )

                self.rds_client = self.session.client(
                    "rds",
                    region_name=region,
                )

                resources.extend(
                    self._collect_rds_instances(
                        region
                    )
                )

                self.lambda_client = (
                    self.session.client(
                        "lambda",
                        region_name=region,
                    )
                )

                resources.extend(
                    self._collect_lambda_functions(
                        region
                    )
                )

            resources.extend(
                self._collect_s3_buckets()
            )

            return resources

        except ClientError as exc:
            raise CollectorError(
                f"Failed to collect AWS resource data: {exc}"
            ) from exc

        except BotoCoreError as exc:
            raise CollectorError(
                f"AWS resource service unavailable: {exc}"
            ) from exc

        except Exception as exc:
            if isinstance(
                exc,
                CollectorError,
            ):
                raise

            raise CollectorError(
                f"Failed to collect AWS resource data: {exc}"
            ) from exc

    def get_cost_breakdown(
        self,
        start_date: datetime,
        end_date: datetime,
        group_by: str = "service",
    ) -> Dict[str, float]:
        self.validate_date_range(
            start_date,
            end_date,
        )
        self.ensure_authenticated()

        dimension_map = {
            "service": "SERVICE",
            "region": "REGION",
            "usage_type": "USAGE_TYPE",
            "account_id": "LINKED_ACCOUNT",
            "operation": "OPERATION",
            "platform": "PLATFORM",
            "instance_type": "INSTANCE_TYPE",
            "purchase_option": "PURCHASE_TYPE",
        }

        normalized_group = str(
            group_by
        ).strip().lower()

        if normalized_group not in dimension_map:
            raise CollectorError(
                f"Unsupported AWS cost grouping: "
                f"{group_by}"
            )

        try:
            start = self._normalize_datetime(
                start_date
            )
            end = self._normalize_datetime(
                end_date
            )

            response = self.ce_client.get_cost_and_usage(
                TimePeriod={
                    "Start": start.strftime(
                        "%Y-%m-%d"
                    ),
                    "End": end.strftime(
                        "%Y-%m-%d"
                    ),
                },
                Granularity="DAILY",
                Metrics=[
                    "UnblendedCost"
                ],
                GroupBy=[
                    {
                        "Type": "DIMENSION",
                        "Key": dimension_map[
                            normalized_group
                        ],
                    }
                ],
            )

            cost_breakdown: Dict[
                str,
                float,
            ] = {}

            for result in response.get(
                "ResultsByTime",
                [],
            ):
                for group in result.get(
                    "Groups",
                    [],
                ):
                    keys = group.get(
                        "Keys",
                        [],
                    )

                    key = (
                        keys[0]
                        if keys
                        else "Unknown"
                    )

                    amount = self._safe_float(
                        group.get(
                            "Metrics",
                            {},
                        )
                        .get(
                            "UnblendedCost",
                            {},
                        )
                        .get(
                            "Amount",
                            0.0,
                        )
                    )

                    cost_breakdown[key] = (
                        cost_breakdown.get(
                            key,
                            0.0,
                        )
                        + amount
                    )

            return dict(
                sorted(
                    cost_breakdown.items(),
                    key=lambda item: item[1],
                    reverse=True,
                )
            )

        except ClientError as exc:
            raise CollectorError(
                f"Failed to get AWS cost breakdown: {exc}"
            ) from exc

        except Exception as exc:
            raise CollectorError(
                f"Failed to get AWS cost breakdown: {exc}"
            ) from exc

    def get_account_identity(
        self,
    ) -> Dict[str, str]:
        self.ensure_authenticated()

        try:
            response = (
                self.sts_client.get_caller_identity()
            )

            return {
                "account_id": str(
                    response.get(
                        "Account",
                        "",
                    )
                ),
                "user_id": str(
                    response.get(
                        "UserId",
                        "",
                    )
                ),
                "arn": str(
                    response.get(
                        "Arn",
                        "",
                    )
                ),
            }

        except Exception as exc:
            raise CollectorError(
                f"Failed to retrieve AWS account identity: {exc}"
            ) from exc

    def _get_target_regions(self) -> List[str]:
        configured = getattr(
            self.config.aws,
            "regions",
            None,
        )

        if configured:
            return sorted(
                {
                    str(region).strip()
                    for region in configured
                    if str(region).strip()
                }
            )

        try:
            ec2 = self.session.client(
                "ec2",
                region_name="us-east-1",
            )

            response = (
                ec2.describe_regions(
                    AllRegions=False
                )
            )

            regions = [
                region["RegionName"]
                for region in response.get(
                    "Regions",
                    [],
                )
                if region.get("RegionName")
            ]

            return sorted(
                set(regions)
            )

        except Exception:
            return ["us-east-1"]

    def _collect_ec2_instances(
        self,
        region: str,
    ) -> List[ResourceData]:
        resources = []

        paginator = (
            self.ec2_client.get_paginator(
                "describe_instances"
            )
        )

        try:
            for page in paginator.paginate():
                for reservation in page.get(
                    "Reservations",
                    [],
                ):
                    for instance in reservation.get(
                        "Instances",
                        [],
                    ):
                        instance_id = instance.get(
                            "InstanceId"
                        )

                        if not instance_id:
                            continue

                        tags = self.standardize_tags(
                            {
                                tag.get(
                                    "Key"
                                ): tag.get(
                                    "Value"
                                )
                                for tag in instance.get(
                                    "Tags",
                                    [],
                                )
                                if tag.get(
                                    "Key"
                                )
                            }
                        )

                        state = (
                            instance.get(
                                "State",
                                {},
                            ).get(
                                "Name",
                                "unknown",
                            )
                        )

                        launch_time = (
                            instance.get(
                                "LaunchTime"
                            )
                        )

                        if not launch_time:
                            launch_time = (
                                datetime.now(
                                    timezone.utc
                                )
                            )

                        resources.append(
                            ResourceData(
                                provider="aws",
                                account_id=(
                                    self.config.aws.account_id
                                    or "unknown"
                                ),
                                resource_id=instance_id,
                                resource_name=(
                                    tags.get(
                                        "Name",
                                        instance_id,
                                    )
                                ),
                                resource_type="EC2 Instance",
                                region=region,
                                state=state,
                                creation_time=(
                                    launch_time
                                ),
                                tags=tags,
                                environment=(
                                    self.extract_environment_from_tags(
                                        tags,
                                        self.config.aws.environment_tag,
                                    )
                                ),
                                cost_center=(
                                    self.extract_cost_center_from_tags(
                                        tags,
                                        self.config.aws.cost_center_tag,
                                    )
                                ),
                                is_idle=(
                                    state
                                    == "stopped"
                                ),
                                last_used_time=(
                                    self._get_ec2_last_used(
                                        instance_id
                                    )
                                ),
                            )
                        )

            return resources

        except ClientError as exc:
            raise CollectorError(
                f"Failed to collect EC2 instances "
                f"in {region}: {exc}"
            ) from exc

    def _collect_rds_instances(
        self,
        region: str,
    ) -> List[ResourceData]:
        resources = []

        paginator = (
            self.rds_client.get_paginator(
                "describe_db_instances"
            )
        )

        try:
            for page in paginator.paginate():
                for db_instance in page.get(
                    "DBInstances",
                    [],
                ):
                    identifier = db_instance.get(
                        "DBInstanceIdentifier"
                    )

                    if not identifier:
                        continue

                    arn = db_instance.get(
                        "DBInstanceArn"
                    )

                    tags = (
                        self._get_rds_tags(
                            arn
                        )
                        if arn
                        else {}
                    )

                    creation_time = (
                        db_instance.get(
                            "InstanceCreateTime"
                        )
                    )

                    if not creation_time:
                        creation_time = (
                            datetime.now(
                                timezone.utc
                            )
                        )

                    status = str(
                        db_instance.get(
                            "DBInstanceStatus",
                            "unknown",
                        )
                    )

                    resources.append(
                        ResourceData(
                            provider="aws",
                            account_id=(
                                self.config.aws.account_id
                                or "unknown"
                            ),
                            resource_id=identifier,
                            resource_name=identifier,
                            resource_type="RDS Instance",
                            region=region,
                            state=status,
                            creation_time=(
                                creation_time
                            ),
                            tags=tags,
                            environment=(
                                self.extract_environment_from_tags(
                                    tags,
                                    self.config.aws.environment_tag,
                                )
                            ),
                            cost_center=(
                                self.extract_cost_center_from_tags(
                                    tags,
                                    self.config.aws.cost_center_tag,
                                )
                            ),
                            is_idle=(
                                status == "stopped"
                            ),
                            last_used_time=None,
                        )
                    )

            return resources

        except ClientError as exc:
            raise CollectorError(
                f"Failed to collect RDS instances "
                f"in {region}: {exc}"
            ) from exc

    def _collect_lambda_functions(
        self,
        region: str,
    ) -> List[ResourceData]:
        resources = []

        paginator = (
            self.lambda_client.get_paginator(
                "list_functions"
            )
        )

        try:
            for page in paginator.paginate():
                for function in page.get(
                    "Functions",
                    [],
                ):
                    arn = function.get(
                        "FunctionArn"
                    )

                    name = function.get(
                        "FunctionName"
                    )

                    if not arn or not name:
                        continue

                    tags = self._get_lambda_tags(
                        arn
                    )

                    last_modified = (
                        function.get(
                            "LastModified"
                        )
                    )

                    creation_time = (
                        self._parse_lambda_timestamp(
                            last_modified
                        )
                    )

                    last_used_time = (
                        self._get_lambda_last_used(
                            arn
                        )
                    )

                    resources.append(
                        ResourceData(
                            provider="aws",
                            account_id=(
                                self.config.aws.account_id
                                or "unknown"
                            ),
                            resource_id=arn,
                            resource_name=name,
                            resource_type="Lambda Function",
                            region=region,
                            state="Active",
                            creation_time=(
                                creation_time
                            ),
                            tags=tags,
                            environment=(
                                self.extract_environment_from_tags(
                                    tags,
                                    self.config.aws.environment_tag,
                                )
                            ),
                            cost_center=(
                                self.extract_cost_center_from_tags(
                                    tags,
                                    self.config.aws.cost_center_tag,
                                )
                            ),
                            is_idle=(
                                last_used_time is not None
                                and (
                                    datetime.now(
                                        timezone.utc
                                    )
                                    - last_used_time
                                ).days
                                >= 30
                            ),
                            last_used_time=(
                                last_used_time
                            ),
                        )
                    )

            return resources

        except ClientError as exc:
            raise CollectorError(
                f"Failed to collect Lambda functions "
                f"in {region}: {exc}"
            ) from exc

    def _collect_s3_buckets(
        self,
    ) -> List[ResourceData]:
        resources = []

        try:
            response = (
                self.s3_client.list_buckets()
            )

            for bucket in response.get(
                "Buckets",
                [],
            ):
                name = bucket.get(
                    "Name"
                )

                if not name:
                    continue

                tags = self._get_s3_tags(
                    name
                )

                region = self._get_s3_region(
                    name
                )

                creation_time = bucket.get(
                    "CreationDate"
                )

                if not creation_time:
                    creation_time = (
                        datetime.now(
                            timezone.utc
                        )
                    )

                last_used_time = (
                    self._get_s3_last_used(
                        name
                    )
                )

                resources.append(
                    ResourceData(
                        provider="aws",
                        account_id=(
                            self.config.aws.account_id
                            or "unknown"
                        ),
                        resource_id=name,
                        resource_name=name,
                        resource_type="S3 Bucket",
                        region=region,
                        state="Active",
                        creation_time=(
                            creation_time
                        ),
                        tags=tags,
                        environment=(
                            self.extract_environment_from_tags(
                                tags,
                                self.config.aws.environment_tag,
                            )
                        ),
                        cost_center=(
                            self.extract_cost_center_from_tags(
                                tags,
                                self.config.aws.cost_center_tag,
                            )
                        ),
                        is_idle=(
                            last_used_time is not None
                            and (
                                datetime.now(
                                    timezone.utc
                                )
                                - last_used_time
                            ).days
                            >= 90
                        ),
                        last_used_time=(
                            last_used_time
                        ),
                    )
                )

            return resources

        except ClientError as exc:
            raise CollectorError(
                f"Failed to collect S3 buckets: {exc}"
            ) from exc

    def _get_resource_tags(
        self,
        resource_id: str,
        service: str,
    ) -> Dict[str, str]:
        try:
            if (
                service == "Amazon Elastic Compute Cloud - Compute"
                or service == "Amazon EC2"
            ) and resource_id.startswith("i-"):
                return self._get_ec2_tags(
                    resource_id
                )

            if (
                service == "Amazon Relational Database Service"
                or service == "Amazon RDS"
            ):
                if resource_id.startswith("db-"):
                    return {}

            if (
                service == "AWS Lambda"
                and resource_id.startswith("arn:")
            ):
                return self._get_lambda_tags(
                    resource_id
                )

            if service == "Amazon Simple Storage Service":
                return self._get_s3_tags(
                    resource_id
                )

        except Exception:
            return {}

        return {}

    def _get_ec2_tags(
        self,
        instance_id: str,
    ) -> Dict[str, str]:
        try:
            response = (
                self.ec2_client.describe_tags(
                    Filters=[
                        {
                            "Name": "resource-id",
                            "Values": [
                                instance_id
                            ],
                        }
                    ]
                )
            )

            return self.standardize_tags(
                {
                    tag.get("Key"): tag.get(
                        "Value"
                    )
                    for tag in response.get(
                        "Tags",
                        [],
                    )
                    if tag.get("Key")
                }
            )

        except Exception:
            return {}

    def _get_rds_tags(
        self,
        resource_arn: str,
    ) -> Dict[str, str]:
        try:
            response = (
                self.rds_client.list_tags_for_resource(
                    ResourceName=resource_arn
                )
            )

            return self.standardize_tags(
                {
                    tag.get("Key"): tag.get(
                        "Value"
                    )
                    for tag in response.get(
                        "TagList",
                        [],
                    )
                    if tag.get("Key")
                }
            )

        except Exception:
            return {}

    def _get_lambda_tags(
        self,
        function_arn: str,
    ) -> Dict[str, str]:
        try:
            response = (
                self.lambda_client.list_tags(
                    Resource=function_arn
                )
            )

            return self.standardize_tags(
                response.get(
                    "Tags",
                    {},
                )
            )

        except Exception:
            return {}

    def _get_s3_tags(
        self,
        bucket_name: str,
    ) -> Dict[str, str]:
        try:
            response = (
                self.s3_client.get_bucket_tagging(
                    Bucket=bucket_name
                )
            )

            return self.standardize_tags(
                {
                    tag.get("Key"): tag.get(
                        "Value"
                    )
                    for tag in response.get(
                        "TagSet",
                        [],
                    )
                    if tag.get("Key")
                }
            )

        except ClientError as exc:
            error_code = (
                exc.response.get(
                    "Error",
                    {},
                ).get(
                    "Code",
                    "",
                )
            )

            if error_code in {
                "NoSuchTagSet",
                "AccessDenied",
                "AccessDeniedException",
            }:
                return {}

            return {}

        except Exception:
            return {}

    def _get_s3_region(
        self,
        bucket_name: str,
    ) -> str:
        try:
            response = (
                self.s3_client.get_bucket_location(
                    Bucket=bucket_name
                )
            )

            location = response.get(
                "LocationConstraint"
            )

            if not location:
                return "us-east-1"

            if location == "EU":
                return "eu-west-1"

            return str(location)

        except Exception:
            return "unknown"

    def _get_ec2_last_used(
        self,
        instance_id: str,
    ) -> Optional[datetime]:
        if not self.cloudwatch_client:
            return None

        try:
            end = datetime.now(
                timezone.utc
            )

            start = (
                end
                - __import__(
                    "datetime"
                ).timedelta(days=30)
            )

            response = (
                self.cloudwatch_client.get_metric_statistics(
                    Namespace="AWS/EC2",
                    MetricName="CPUUtilization",
                    Dimensions=[
                        {
                            "Name": "InstanceId",
                            "Value": instance_id,
                        }
                    ],
                    StartTime=start,
                    EndTime=end,
                    Period=86400,
                    Statistics=[
                        "Average"
                    ],
                )
            )

            datapoints = response.get(
                "Datapoints",
                [],
            )

            active_points = [
                point
                for point in datapoints
                if self._safe_float(
                    point.get(
                        "Average",
                        0.0,
                    )
                )
                > 1.0
            ]

            if not active_points:
                return None

            latest = max(
                active_points,
                key=lambda item: item.get(
                    "Timestamp",
                    start,
                ),
            )

            timestamp = latest.get(
                "Timestamp"
            )

            return (
                self._normalize_datetime(
                    timestamp
                )
                if timestamp
                else None
            )

        except Exception:
            return None

    def _get_lambda_last_used(
        self,
        function_arn: str,
    ) -> Optional[datetime]:
        if not self.cloudwatch_client:
            return None

        try:
            function_name = (
                function_arn.split(":")[-1]
            )

            end = datetime.now(
                timezone.utc
            )

            start = (
                end
                - __import__(
                    "datetime"
                ).timedelta(days=30)
            )

            response = (
                self.cloudwatch_client.get_metric_statistics(
                    Namespace="AWS/Lambda",
                    MetricName="Invocations",
                    Dimensions=[
                        {
                            "Name": "FunctionName",
                            "Value": function_name,
                        }
                    ],
                    StartTime=start,
                    EndTime=end,
                    Period=86400,
                    Statistics=[
                        "Sum"
                    ],
                )
            )

            datapoints = response.get(
                "Datapoints",
                [],
            )

            active_points = [
                point
                for point in datapoints
                if self._safe_float(
                    point.get(
                        "Sum",
                        0.0,
                    )
                )
                > 0
            ]

            if not active_points:
                return None

            latest = max(
                active_points,
                key=lambda item: item.get(
                    "Timestamp",
                    start,
                ),
            )

            timestamp = latest.get(
                "Timestamp"
            )

            return (
                self._normalize_datetime(
                    timestamp
                )
                if timestamp
                else None
            )

        except Exception:
            return None

    def _get_s3_last_used(
        self,
        bucket_name: str,
    ) -> Optional[datetime]:
        return None

    def _is_ec2_idle(
        self,
        instance: Dict[str, Any],
    ) -> bool:
        state = (
            instance.get(
                "State",
                {},
            ).get(
                "Name",
                "",
            )
        )

        return state in {
            "stopped",
            "stopping",
        }

    def _is_rds_idle(
        self,
        db_instance: Dict[str, Any],
    ) -> bool:
        status = str(
            db_instance.get(
                "DBInstanceStatus",
                "",
            )
        ).lower()

        return status == "stopped"

    def _is_lambda_idle(
        self,
        function: Dict[str, Any],
    ) -> bool:
        arn = function.get(
            "FunctionArn"
        )

        if not arn:
            return False

        last_used = (
            self._get_lambda_last_used(
                arn
            )
        )

        if last_used is None:
            return False

        return (
            datetime.now(
                timezone.utc
            )
            - last_used
        ).days >= 30

    def _is_s3_idle(
        self,
        bucket_name: str,
    ) -> bool:
        last_used = (
            self._get_s3_last_used(
                bucket_name
            )
        )

        if last_used is None:
            return False

        return (
            datetime.now(
                timezone.utc
            )
            - last_used
        ).days >= 90

    def _extract_dimension(
        self,
        keys: List[str],
        dimension: str,
    ) -> str:
        prefix = f"{dimension}$"

        for key in keys:
            if str(key).startswith(prefix):
                return str(key).split(
                    "$",
                    1,
                )[1]

        return "Unknown"

    def _extract_group_value(
        self,
        keys: List[str],
        index: int,
    ) -> str:
        if index >= len(keys):
            return "Unknown"

        value = str(
            keys[index]
        ).strip()

        return value or "Unknown"

    def _parse_date(
        self,
        value: str,
    ) -> datetime:
        parsed = datetime.strptime(
            value,
            "%Y-%m-%d",
        )

        return parsed.replace(
            tzinfo=timezone.utc
        )

    def _parse_lambda_timestamp(
        self,
        value: Optional[str],
    ) -> datetime:
        if not value:
            return datetime.now(
                timezone.utc
            )

        normalized = value.strip()

        if normalized.endswith(
            "+0000"
        ):
            normalized = (
                normalized[:-5]
                + "+00:00"
            )

        try:
            parsed = datetime.fromisoformat(
                normalized
            )

            return self._normalize_datetime(
                parsed
            )

        except ValueError:
            return datetime.now(
                timezone.utc
            )

    def _normalize_datetime(
        self,
        value: datetime,
    ) -> datetime:
        if value is None:
            return datetime.now(
                timezone.utc
            )

        if value.tzinfo is None:
            return value.replace(
                tzinfo=timezone.utc
            )

        return value.astimezone(
            timezone.utc
        )

    def _safe_float(
        self,
        value: Any,
        default: float = 0.0,
    ) -> float:
        try:
            result = float(
                value
            )

            if result != result:
                return default

            return result

        except (
            TypeError,
            ValueError,
        ):
            return default

    def _retry(
        self,
        function: Any,
        attempts: int = 3,
    ) -> Any:
        last_error = None

        for attempt in range(
            max(1, attempts)
        ):
            try:
                return function()

            except (
                ClientError,
                BotoCoreError,
            ) as exc:
                last_error = exc

                if attempt == attempts - 1:
                    break

                time.sleep(
                    2 ** attempt
                )

        if last_error:
            raise last_error

        return None