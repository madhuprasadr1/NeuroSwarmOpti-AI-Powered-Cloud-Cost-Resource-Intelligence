from __future__ import annotations

import ast
import json
import re
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder


class CloudDataPreprocessor:
    _CPU_REGEX = re.compile(r"['\"]?(?:cpus?|cpu_usage|0)['\"]?\s*:\s*([0-9\.eE\-]+)")
    _MEM_REGEX = re.compile(r"['\"]?(?:memory|mem|memory_usage|0)['\"]?\s*:\s*([0-9\.eE\-]+)")

    def __init__(self, data_root: str | Path = "data/raw"):
        self.data_root = Path(data_root)
        self.encoders: Dict[str, LabelEncoder] = {}

    def _resolve(self, *parts: str) -> Path:
        path = self.data_root.joinpath(*parts)
        if path.exists():
            return path
        raise FileNotFoundError(str(path))

    def _read_csv(
        self,
        path: str | Path,
        usecols: Optional[Iterable[str]] = None,
        parse_dates: Optional[Iterable[str]] = None,
        dtype: Optional[Dict[str, Any]] = None,
        nrows: Optional[int] = None,
    ) -> pd.DataFrame:
        path = Path(path)
        kwargs: Dict[str, Any] = {"low_memory": False}
        if usecols is not None:
            kwargs["usecols"] = list(usecols)
        if parse_dates is not None:
            kwargs["parse_dates"] = list(parse_dates)
        if dtype is not None:
            kwargs["dtype"] = dtype
        if nrows is not None:
            kwargs["nrows"] = nrows
        return pd.read_csv(path, **kwargs)

    @staticmethod
    def _normalise_columns(df: pd.DataFrame) -> pd.DataFrame:
        result = df.copy()
        result.columns = [
            str(column).strip().lower().replace(" ", "_").replace("-", "_")
            for column in result.columns
        ]
        return result

    @staticmethod
    def _ensure_columns(df: pd.DataFrame, columns: Iterable[str], default: Any = 0) -> pd.DataFrame:
        result = df.copy()
        for column in columns:
            if column not in result.columns:
                result[column] = default
        return result

    def _clean_numeric(
        self,
        df: pd.DataFrame,
        columns: Iterable[str],
        fill_strategy: str = "median",
    ) -> pd.DataFrame:
        result = df.copy()
        for column in columns:
            if column not in result.columns:
                continue
            result[column] = pd.to_numeric(result[column], errors="coerce")
            if result[column].isna().all():
                result[column] = 0.0
            elif fill_strategy == "zero":
                result[column] = result[column].fillna(0.0)
            else:
                med = result[column].median()
                result[column] = result[column].fillna(med if pd.notna(med) else 0.0)
        return result

    def _clean_categories(
        self,
        df: pd.DataFrame,
        columns: Iterable[str],
    ) -> pd.DataFrame:
        result = df.copy()
        for column in columns:
            if column not in result.columns:
                continue
            result[column] = result[column].astype("string").fillna("unknown").str.strip()
        return result

    def _encode(
        self,
        df: pd.DataFrame,
        columns: Iterable[str],
        prefix: str = "",
    ) -> pd.DataFrame:
        result = df.copy()
        for column in columns:
            if column not in result.columns:
                continue
            key = f"{prefix}{column}"
            if key not in self.encoders:
                encoder = LabelEncoder()
                values = result[column].astype(str)
                encoder.fit(values)
                self.encoders[key] = encoder
            encoder = self.encoders[key]
            values = result[column].astype(str)
            mapping = {c: i for i, c in enumerate(encoder.classes_)}
            encoded = values.map(mapping).fillna(-1)
            result[f"{column}_encoded"] = encoded.astype(np.int32)
        return result

    def _extract_cpu(self, value: Any) -> float:
        if value is None or (isinstance(value, float) and np.isnan(value)):
            return 0.0
        if isinstance(value, (int, float)):
            return float(value)
        s = str(value)
        m = self._CPU_REGEX.search(s)
        if m:
            try:
                return float(m.group(1))
            except ValueError:
                pass
        return 0.0

    def _extract_memory(self, value: Any) -> float:
        if value is None or (isinstance(value, float) and np.isnan(value)):
            return 0.0
        if isinstance(value, (int, float)):
            return float(value)
        s = str(value)
        m = self._MEM_REGEX.search(s)
        if m:
            try:
                return float(m.group(1))
            except ValueError:
                pass
        return 0.0

    # 1. Multi-Cloud Dataset (with 50,000-instance physics-informed SKU augmentation)
    def load_multi_cloud(
        self,
        path: str | Path | None = None,
        augment: bool = True,
        target_samples: int = 50000,
    ) -> pd.DataFrame:
        if path is None:
            path = self._resolve("multi_cloud", "Cloud_Dataset.csv")
        df = self._read_csv(path, parse_dates=["timestamp"])
        df = self._normalise_columns(df)
        df = self._ensure_columns(df, [
            "cpu_usage", "memory_usage", "net_io", "disk_io", "vcpu", "ram_gb",
            "price_per_hour", "latency_ms", "throughput", "cost", "utilization",
            "cloud_provider", "region", "vm_type", "target",
        ])
        if "resource_id" not in df.columns:
            df["resource_id"] = [f"resource-{index + 1}" for index in range(len(df))]
        if (df["utilization"] == 0).all() and "cpu_usage" in df.columns:
            df["utilization"] = df["cpu_usage"]

        # High-Fidelity Multi-Cloud SKU Augmentation
        if augment and len(df) < target_samples:
            n_aug = target_samples - len(df)
            np.random.seed(42)
            skus = [
                ('AWS', 't3.nano', 2, 0.5, 0.0052),
                ('AWS', 't3.micro', 2, 1.0, 0.0104),
                ('AWS', 't3.small', 2, 2.0, 0.0208),
                ('AWS', 't3.medium', 2, 4.0, 0.0416),
                ('AWS', 't3.large', 2, 8.0, 0.0832),
                ('AWS', 't3.xlarge', 4, 16.0, 0.1664),
                ('AWS', 'c5.large', 2, 4.0, 0.085),
                ('AWS', 'c5.xlarge', 4, 8.0, 0.17),
                ('AWS', 'c5.2xlarge', 8, 16.0, 0.34),
                ('AWS', 'm5.large', 2, 8.0, 0.096),
                ('AWS', 'm5.xlarge', 4, 16.0, 0.192),
                ('AWS', 'r5.large', 2, 16.0, 0.126),
                ('AWS', 'r5.xlarge', 4, 32.0, 0.252),
                ('Azure', 'Standard_B1s', 1, 1.0, 0.0104),
                ('Azure', 'Standard_B2s', 2, 4.0, 0.0416),
                ('Azure', 'Standard_D2s_v5', 2, 8.0, 0.096),
                ('Azure', 'Standard_D4s_v5', 4, 16.0, 0.192),
                ('Azure', 'Standard_D8s_v5', 8, 32.0, 0.384),
                ('Azure', 'Standard_E2s_v5', 2, 16.0, 0.126),
                ('Azure', 'Standard_E4s_v5', 4, 32.0, 0.252),
                ('Azure', 'Standard_F2s_v2', 2, 4.0, 0.085),
                ('Azure', 'Standard_F4s_v2', 4, 8.0, 0.169),
                ('GCP', 'e2-micro', 2, 1.0, 0.0084),
                ('GCP', 'e2-small', 2, 2.0, 0.0168),
                ('GCP', 'e2-medium', 2, 4.0, 0.0336),
                ('GCP', 'e2-standard-2', 2, 8.0, 0.067),
                ('GCP', 'e2-standard-4', 4, 16.0, 0.134),
                ('GCP', 'n2-standard-2', 2, 8.0, 0.097),
                ('GCP', 'n2-standard-4', 4, 16.0, 0.194),
                ('GCP', 'c2-standard-4', 4, 16.0, 0.208),
            ]
            sku_idx = np.random.choice(len(skus), size=n_aug)
            chosen = [skus[i] for i in sku_idx]
            providers = [c[0] for c in chosen]
            vm_types = [c[1] for c in chosen]
            vcpus = np.array([c[2] for c in chosen])
            rams = np.array([c[3] for c in chosen])
            base_prices = np.array([c[4] for c in chosen])

            regions_aws = ['us-east-1', 'us-east-2', 'us-west-2', 'eu-west-1', 'ap-south-1']
            regions_az = ['eastus', 'westeurope', 'centralus', 'southeastasia']
            regions_gcp = ['us-central1', 'europe-west1', 'asia-south1', 'us-east4']

            regions = []
            for p in providers:
                if p == 'AWS':
                    regions.append(np.random.choice(regions_aws))
                elif p == 'Azure':
                    regions.append(np.random.choice(regions_az))
                else:
                    regions.append(np.random.choice(regions_gcp))

            cpu = np.clip(np.random.beta(2, 5, size=n_aug) * 100 + np.random.choice([0, 18, 52], size=n_aug, p=[0.4, 0.35, 0.25]), 1.0, 99.0)
            mem = np.clip(cpu * 0.72 + np.random.normal(14, 10, size=n_aug), 5.0, 98.0)
            net_io = np.clip(cpu * 8.0 + np.random.exponential(160, size=n_aug), 10.0, 5000.0)
            disk_io = np.clip(mem * 6.5 + np.random.exponential(180, size=n_aug), 20.0, 8000.0)
            latency = np.clip(100.0 + (cpu / 100.0) ** 2 * 280.0 + np.random.normal(0, 15, size=n_aug), 15.0, 900.0)
            throughput = np.clip((1000.0 - latency) * 4.0 + net_io * 0.4, 50.0, 15000.0)

            targets = []
            for c, m in zip(cpu, mem):
                if c < 20.0 and m < 45.0:
                    targets.append('scale_down')
                elif c > 75.0 or m > 82.0:
                    targets.append('scale_up')
                else:
                    targets.append('no_action')

            base_time = pd.Timestamp("2024-01-01")
            time_offsets = pd.to_timedelta(np.random.randint(0, 365 * 86400, size=n_aug), unit="s")
            timestamps = base_time + time_offsets

            df_aug = pd.DataFrame({
                'timestamp': timestamps,
                'resource_id': [f'res-aug-{i:06d}' for i in range(n_aug)],
                'cloud_provider': providers,
                'region': regions,
                'vm_type': vm_types,
                'vcpu': vcpus,
                'ram_gb': rams,
                'price_per_hour': np.round(base_prices, 4),
                'cpu_usage': np.round(cpu, 2),
                'memory_usage': np.round(mem, 2),
                'net_io': np.round(net_io, 2),
                'disk_io': np.round(disk_io, 2),
                'latency_ms': np.round(latency, 2),
                'throughput': np.round(throughput, 2),
                'cost': np.round(base_prices * (cpu / 100.0 * 0.5 + 0.5), 4),
                'utilization': np.round(cpu * 0.6 + mem * 0.4, 2),
                'target': targets,
            })
            df = pd.concat([df, df_aug], ignore_index=True)

        numeric_columns = [
            "cpu_usage", "memory_usage", "net_io", "disk_io", "vcpu", "ram_gb",
            "price_per_hour", "latency_ms", "throughput", "cost", "utilization",
        ]
        categorical_columns = ["cloud_provider", "region", "vm_type", "target"]
        df = self._clean_numeric(df, numeric_columns)
        df = self._clean_categories(df, categorical_columns)

        if "timestamp" in df.columns:
            df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce").ffill().bfill()

        df = self._encode(df, ["cloud_provider", "region", "vm_type"], prefix="resource_")

        target_mapping = {"no_action": 0, "scale_down": 1, "scale_up": 2}
        if "target" in df.columns:
            df["target_encoded"] = (
                df["target"].astype(str).str.lower().map(target_mapping).fillna(0).astype(np.int8)
            )

        df["cpu_ratio"] = df["cpu_usage"] / 100.0
        df["memory_ratio"] = df["memory_usage"] / 100.0
        df["utilization_ratio"] = df["utilization"] / 100.0
        df["estimated_hourly_cost"] = df["price_per_hour"].clip(lower=0)
        df["cpu_memory_pressure"] = df["cpu_ratio"] * 0.55 + df["memory_ratio"] * 0.45
        df["performance_index"] = df["throughput"].clip(lower=0) / (df["latency_ms"].clip(lower=1))

        return df.sort_values("timestamp").reset_index(drop=True)

    # 2. Cloud Anomaly Dataset
    def load_anomaly(self, path: str | Path | None = None, max_rows: Optional[int] = None) -> pd.DataFrame:
        if path is None:
            path = self._resolve("anomaly", "Cloud_Anomaly_Dataset.csv")
        df = self._normalise_columns(self._read_csv(path, nrows=max_rows))
        columns = [
            "vm_id", "timestamp", "cpu_usage", "memory_usage", "network_traffic",
            "power_consumption", "num_executed_instructions", "execution_time",
            "energy_efficiency", "task_type", "task_priority", "task_status", "anomaly_status",
        ]
        df = self._ensure_columns(df, columns, "unknown")

        numeric_columns = [
            "cpu_usage", "memory_usage", "network_traffic", "power_consumption",
            "num_executed_instructions", "execution_time", "energy_efficiency", "task_priority",
        ]
        categorical_columns = ["vm_id", "task_type", "task_status", "anomaly_status"]
        df = self._clean_numeric(df, numeric_columns)
        df = self._clean_categories(df, categorical_columns)

        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce").ffill().bfill()
        df["cpu_ratio"] = df["cpu_usage"] / 100.0
        df["memory_ratio"] = df["memory_usage"] / 100.0
        df["resource_pressure"] = df["cpu_ratio"] * 0.5 + df["memory_ratio"] * 0.5
        df["execution_efficiency"] = (
            df["num_executed_instructions"].clip(lower=0) / df["execution_time"].clip(lower=0.001)
        )
        df["network_power_ratio"] = (
            df["network_traffic"].clip(lower=0) / df["power_consumption"].clip(lower=0.001)
        )

        anomaly_mapping = {"normal": 0, "anomaly": 1, "1": 1, "0": 0}
        df["anomaly_target"] = (
            df["anomaly_status"].astype(str).str.lower().map(anomaly_mapping).fillna(0).astype(np.int8)
        )
        df = self._encode(df, ["task_type", "task_status"], prefix="anomaly_")
        return df.sort_values(["vm_id", "timestamp"]).reset_index(drop=True)

    # 3. Google Borg Traces Dataset (Full 1.32M-line ingestion)
    def load_google_borg(self, path: str | Path | None = None, max_rows: Optional[int] = None) -> pd.DataFrame:
        if path is None:
            path = self._resolve("google_borg", "borg_traces_data.csv")
        df = self._normalise_columns(self._read_csv(path, nrows=max_rows))
        columns = [
            "time", "collection_id", "scheduling_class", "priority", "instance_index",
            "machine_id", "resource_request", "collection_type", "instance_events_type",
            "average_usage", "maximum_usage", "assigned_memory", "page_cache_memory",
            "cluster", "event", "failed",
        ]
        df = self._ensure_columns(df, columns, "unknown")

        raw_time = pd.to_numeric(df["time"], errors="coerce").fillna(0.0)
        clipped_time = raw_time.clip(lower=0.0, upper=35.0 * 86400 * 1_000_000)
        df["timestamp"] = pd.Timestamp("2024-01-01") + pd.to_timedelta(clipped_time, unit="us")

        df["cpu_requested"] = df["resource_request"].map(self._extract_cpu)
        df["cpu_average"] = df["average_usage"].map(self._extract_cpu)
        df["cpu_maximum"] = df["maximum_usage"].map(self._extract_cpu)
        df["memory_requested"] = df["resource_request"].map(self._extract_memory)
        df["assigned_memory"] = pd.to_numeric(df["assigned_memory"], errors="coerce").fillna(0.0)
        df["page_cache_memory"] = pd.to_numeric(df["page_cache_memory"], errors="coerce").fillna(0.0)
        df["priority"] = pd.to_numeric(df["priority"], errors="coerce").fillna(0.0)
        df["scheduling_class"] = pd.to_numeric(df["scheduling_class"], errors="coerce").fillna(0.0)
        df["instance_index"] = pd.to_numeric(df["instance_index"], errors="coerce").fillna(0.0)
        df["failed"] = (pd.to_numeric(df["failed"], errors="coerce").fillna(0) > 0).astype(np.int8)

        df["active_signal"] = (~df["event"].astype(str).isin(["FINISH", "FAIL", "LOST"])).astype(np.int8)
        df["memory_pressure"] = df["assigned_memory"] / df["memory_requested"].clip(lower=0.001)
        df["cpu_pressure"] = df["cpu_average"] / df["cpu_requested"].clip(lower=0.001)
        df["resource_intensity"] = df["cpu_average"].clip(lower=0) + df["assigned_memory"].clip(lower=0)

        df = df.drop(columns=["resource_request", "average_usage", "maximum_usage", "time"], errors="ignore")
        df = df.dropna(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)
        return df

    def aggregate_workload(self, df: pd.DataFrame, frequency: str = "5min") -> pd.DataFrame:
        required = {"timestamp", "cpu_average", "cpu_maximum", "cpu_requested", "assigned_memory"}
        working = df.copy()
        working["timestamp"] = pd.to_datetime(working["timestamp"], errors="coerce")
        working = working.dropna(subset=["timestamp"])

        aggregations = {
            "cpu_average": "mean",
            "cpu_maximum": "max",
            "cpu_requested": "mean",
            "memory_requested": "mean",
            "assigned_memory": "mean",
            "active_signal": "sum",
            "failed": "sum",
            "priority": "mean",
        }
        available = {k: v for k, v in aggregations.items() if k in working.columns}
        result = working.set_index("timestamp").resample(frequency).agg(available).reset_index()
        result = result.rename(columns={"active_signal": "active_jobs", "failed": "failed_jobs"})

        if "cpu_requested" in result.columns and "cpu_average" in result.columns:
            result["cpu_pressure"] = result["cpu_average"] / result["cpu_requested"].clip(lower=0.001)
        if "assigned_memory" in result.columns and "memory_requested" in result.columns:
            result["memory_pressure"] = result["assigned_memory"] / result["memory_requested"].clip(lower=0.001)
        if "failed_jobs" in result.columns and "active_jobs" in result.columns:
            result["failure_rate"] = result["failed_jobs"] / result["active_jobs"].clip(lower=1)
            result["demand_index"] = result["active_jobs"].rank(pct=True)

        num_cols = result.select_dtypes(include=[np.number]).columns
        result[num_cols] = result[num_cols].replace([np.inf, -np.inf], np.nan).fillna(0)
        return result

    # 4. Cloud Budget Dataset
    def load_cloud_budget(self, path: str | Path | None = None) -> pd.DataFrame:
        if path is None:
            path = self._resolve("cloud_budget", "cloud_budget_2023_dataset.csv")
        df = self._normalise_columns(self._read_csv(path))
        numeric_columns = [
            "usage_quantity", "list_cost", "savings_plan_coverage_pct",
            "reserved_instance_coverage_pct", "discount_rate_pct", "discount_amount",
            "net_cost", "on_demand_cost", "reserved_savings", "savings_plan_savings",
            "spot_savings", "amortized_cost", "forecast_monthly_cost", "budget_amount",
            "budget_utilization_pct", "cost_variance_7d_pct", "cost_variance_30d_pct",
            "anomaly_score", "is_anomaly",
        ]
        categorical_columns = [
            "cloud_provider", "account_id", "project_id", "environment", "business_unit",
            "department", "cost_center", "region", "service", "resource_type",
            "usage_unit", "budget_status", "currency",
        ]
        df = self._ensure_columns(df, numeric_columns, 0.0)
        df = self._ensure_columns(df, categorical_columns, "unknown")
        df = self._ensure_columns(df, ["date"], pd.NaT)
        df = self._clean_numeric(df, numeric_columns)
        df = self._clean_categories(df, categorical_columns)

        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        df["daily_cost"] = df["net_cost"].clip(lower=0)
        df["budget_amount"] = df["budget_amount"].clip(lower=0)
        df["cost_efficiency"] = df["net_cost"].clip(lower=0) / df["usage_quantity"].clip(lower=0.001)
        df["discount_savings_total"] = (
            df["discount_amount"].clip(lower=0)
            + df["reserved_savings"].clip(lower=0)
            + df["savings_plan_savings"].clip(lower=0)
            + df["spot_savings"].clip(lower=0)
        )
        df["effective_discount_pct"] = (
            df["discount_savings_total"] / df["on_demand_cost"].clip(lower=0.001) * 100
        )
        df["budget_remaining"] = (df["budget_amount"] - df["net_cost"]).clip(lower=0)
        df["budget_overrun"] = (df["net_cost"] - df["budget_amount"]).clip(lower=0)
        df["budget_pressure"] = (df["budget_utilization_pct"] / 100.0).clip(lower=0)

        df = self._encode(df, [
            "cloud_provider", "environment", "business_unit", "department",
            "region", "service", "resource_type", "budget_status",
        ], prefix="cost_")
        return df.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)

    def aggregate_cost(self, df: pd.DataFrame, frequency: str = "D") -> pd.DataFrame:
        working = df.copy()
        working["date"] = pd.to_datetime(working["date"], errors="coerce")
        working = working.dropna(subset=["date"])

        numeric_aggregations = {
            "usage_quantity": "sum", "list_cost": "sum", "net_cost": "sum",
            "on_demand_cost": "sum", "amortized_cost": "sum", "discount_amount": "sum",
            "reserved_savings": "sum", "savings_plan_savings": "sum", "spot_savings": "sum",
            "budget_amount": "sum", "budget_utilization_pct": "mean",
            "cost_variance_7d_pct": "mean", "cost_variance_30d_pct": "mean",
            "anomaly_score": "mean", "is_anomaly": "max",
        }
        available = {k: v for k, v in numeric_aggregations.items() if k in working.columns}
        result = working.set_index("date").resample(frequency).agg(available).reset_index()

        result["cost_change_pct"] = (
            result["net_cost"].pct_change().replace([np.inf, -np.inf], np.nan).fillna(0) * 100
        )
        result["rolling_7d_cost"] = result["net_cost"].rolling(7, min_periods=1).mean()
        result["rolling_30d_cost"] = result["net_cost"].rolling(30, min_periods=1).mean()
        result["budget_remaining"] = (result["budget_amount"] - result["net_cost"]).clip(lower=0)
        result["budget_overrun"] = (result["net_cost"] - result["budget_amount"]).clip(lower=0)
        result["cost_anomaly_signal"] = result["anomaly_score"].abs()

        num_cols = result.select_dtypes(include=[np.number]).columns
        result[num_cols] = result[num_cols].replace([np.inf, -np.inf], np.nan).fillna(0)
        return result

    # Feature Generators
    def create_resource_features(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
        feature_columns = [
            "cpu_usage", "memory_usage", "net_io", "disk_io", "vcpu", "ram_gb",
            "price_per_hour", "latency_ms", "throughput", "utilization",
            "cpu_ratio", "memory_ratio", "utilization_ratio",
            "cpu_memory_pressure", "performance_index",
            "resource_cloud_provider_encoded", "resource_region_encoded", "resource_vm_type_encoded",
        ]
        available = [c for c in feature_columns if c in df.columns]
        if len(available) < 5:
            num = df.select_dtypes(include=[np.number]).columns
            available = [c for c in num if c not in ["target_encoded", "cost"]]
        X = df[available].replace([np.inf, -np.inf], np.nan).fillna(0)
        if "target_encoded" in df.columns:
            y = df["target_encoded"].astype(np.int8)
        else:
            y = pd.Series(np.zeros(len(df), dtype=np.int8))
        return X, y

    def create_anomaly_features(self, df: pd.DataFrame) -> pd.DataFrame:
        feature_columns = [
            "cpu_usage", "memory_usage", "network_traffic", "power_consumption",
            "num_executed_instructions", "execution_time", "energy_efficiency",
            "task_priority", "cpu_ratio", "memory_ratio", "resource_pressure",
            "execution_efficiency", "network_power_ratio",
            "anomaly_task_type_encoded", "anomaly_task_status_encoded",
        ]
        available = [c for c in feature_columns if c in df.columns]
        if not available:
            num = df.select_dtypes(include=[np.number]).columns
            available = [c for c in num if c not in ["anomaly_target", "anomaly_status"]]
        return df[available].replace([np.inf, -np.inf], np.nan).fillna(0)

    def create_workload_features(self, df: pd.DataFrame) -> pd.DataFrame:
        feature_columns = [
            "cpu_average", "cpu_maximum", "cpu_requested", "memory_requested",
            "assigned_memory", "active_jobs", "failed_jobs", "priority",
            "cpu_pressure", "memory_pressure", "failure_rate", "demand_index",
        ]
        available = [c for c in feature_columns if c in df.columns]
        if not available:
            available = list(df.select_dtypes(include=[np.number]).columns)
        return df[available].replace([np.inf, -np.inf], np.nan).fillna(0)

    def create_cost_features(self, df: pd.DataFrame) -> pd.DataFrame:
        feature_columns = [
            "usage_quantity", "list_cost", "net_cost", "on_demand_cost", "amortized_cost",
            "forecast_monthly_cost", "budget_amount", "budget_utilization_pct",
            "cost_variance_7d_pct", "cost_variance_30d_pct", "anomaly_score",
            "cost_efficiency", "discount_savings_total", "effective_discount_pct",
            "budget_remaining", "budget_overrun", "budget_pressure",
            "cost_cloud_provider_encoded", "cost_environment_encoded",
            "cost_business_unit_encoded", "cost_department_encoded",
            "cost_region_encoded", "cost_service_encoded",
            "cost_resource_type_encoded", "cost_budget_status_encoded",
        ]
        available = [c for c in feature_columns if c in df.columns]
        if not available:
            available = list(df.select_dtypes(include=[np.number]).columns)
        return df[available].replace([np.inf, -np.inf], np.nan).fillna(0)

    def save_processed(self, df: pd.DataFrame, filename: str) -> Path:
        output_dir = self.data_root.parent / "processed"
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / filename
        df.to_csv(output_path, index=False)
        return output_path

    def prepare_all(self, save: bool = True) -> Dict[str, pd.DataFrame]:
        resource = self.load_multi_cloud(augment=True, target_samples=50000)
        anomaly = self.load_anomaly()
        borg = self.load_google_borg(max_rows=None)
        budget = self.load_cloud_budget()

        workload = self.aggregate_workload(borg, frequency="5min")
        cost = self.aggregate_cost(budget)

        result = {
            "resource": resource,
            "anomaly": anomaly,
            "borg": borg,
            "workload": workload,
            "budget": budget,
            "cost": cost,
        }

        if save:
            self.save_processed(resource, "resource_features.csv")
            self.save_processed(anomaly, "anomaly_features.csv")
            self.save_processed(workload, "workload_timeseries.csv")
            self.save_processed(cost, "cost_timeseries.csv")
            self.save_processed(budget, "budget_granular.csv")

        return result


def prepare_datasets(
    data_root: str | Path = "data/raw",
    save: bool = True,
) -> Dict[str, pd.DataFrame]:
    processor = CloudDataPreprocessor(data_root)
    return processor.prepare_all(save=save)


__all__ = [
    "CloudDataPreprocessor",
    "prepare_datasets",
]
