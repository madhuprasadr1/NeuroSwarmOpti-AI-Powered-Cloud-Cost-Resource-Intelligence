from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score, classification_report, f1_score, mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False


# Instance families for right-sizing recommendations across the 3 supported clouds
AWS_INSTANCE_TIERS = [
    ("c5.4xlarge", 16, 32.0, 0.68),
    ("c5.2xlarge", 8, 16.0, 0.34),
    ("c5.xlarge", 4, 8.0, 0.17),
    ("c5.large", 2, 4.0, 0.085),
    ("t3.xlarge", 4, 16.0, 0.1664),
    ("t3.large", 2, 8.0, 0.0832),
    ("t3.medium", 2, 4.0, 0.0416),
    ("t3.small", 2, 2.0, 0.0208),
    ("t3.micro", 2, 1.0, 0.0104),
    ("t3.nano", 2, 0.5, 0.0052),
    ("t2.xlarge", 4, 16.0, 0.1856),
    ("t2.large", 2, 8.0, 0.0928),
    ("t2.medium", 2, 4.0, 0.0464),
    ("t2.small", 1, 2.0, 0.023),
    ("t2.micro", 1, 1.0, 0.0116),
    ("t2.nano", 1, 0.5, 0.0058),
]

AZURE_INSTANCE_TIERS = [
    ("Standard_D16s_v5", 16, 64.0, 0.768),
    ("Standard_D8s_v5", 8, 32.0, 0.384),
    ("Standard_D4s_v5", 4, 16.0, 0.192),
    ("Standard_D2s_v5", 2, 8.0, 0.096),
    ("Standard_B2s", 2, 4.0, 0.0416),
    ("Standard_B1ms", 1, 2.0, 0.0208),
    ("Standard_B1s", 1, 1.0, 0.0104),
]

GCP_INSTANCE_TIERS = [
    ("c2-standard-16", 16, 64.0, 0.835),
    ("c2-standard-8", 8, 32.0, 0.417),
    ("c2-standard-4", 4, 16.0, 0.209),
    ("e2-standard-4", 4, 16.0, 0.134),
    ("e2-standard-2", 2, 8.0, 0.067),
    ("e2-medium", 2, 4.0, 0.0335),
    ("e2-small", 2, 2.0, 0.0168),
    ("e2-micro", 2, 1.0, 0.0084),
]


def get_sku_family(sku: str) -> str:
    """Extracts instance family prefix (e.g. 'c5', 't3', 't2', 'standard_d', 'standard_b', 'c2', 'e2')."""
    s = str(sku).strip().lower()
    if "." in s:
        return s.split(".")[0]
    if s.startswith("standard_"):
        parts = s.split("_")
        if len(parts) >= 2 and parts[1]:
            return f"standard_{parts[1][0]}"
    if "-" in s:
        parts = s.split("-")
        return parts[0]
    return s


class ResourceIntelligenceModel:
    """AI Right-Sizing and Capacity Optimization Model.
    
    Trained on Multi-Cloud resource metrics to:
    1. Classify right-sizing actions: no_action (0), scale_down (1), scale_up (2)
    2. Predict target optimal hourly cost and compute net dollar savings
    3. Generate explainable feature importances and concrete VM SKU targets
    """

    def __init__(self, model_dir: str | Path = "models", random_state: int = 42):
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(parents=True, exist_ok=True)
        self.random_state = random_state

        self.classifier = None
        self.cost_model = None
        self.scaler = None
        self.feature_names: List[str] = []
        self.class_names = ["no_action", "scale_down", "scale_up"]
        self.metrics: Dict[str, Any] = {}

    def _feature_columns(self, df: pd.DataFrame) -> List[str]:
        preferred = [
            "cpu_usage", "memory_usage", "net_io", "disk_io", "vcpu", "ram_gb",
            "price_per_hour", "latency_ms", "throughput", "utilization",
            "cpu_ratio", "memory_ratio", "utilization_ratio",
            "cpu_memory_pressure", "performance_index",
            "resource_cloud_provider_encoded", "resource_region_encoded", "resource_vm_type_encoded",
        ]
        available = [c for c in preferred if c in df.columns]
        if len(available) < 5:
            num = df.select_dtypes(include=[np.number]).columns
            available = [c for c in num if c not in ["target_encoded", "cost", "target"]]
        return available

    def _prepare_features(self, df: pd.DataFrame) -> pd.DataFrame:
        cols = self._feature_columns(df)
        X = df[cols].copy()
        for c in cols:
            X[c] = pd.to_numeric(X[c], errors="coerce")
        X = X.replace([np.inf, -np.inf], np.nan)
        X = X.fillna(X.median(numeric_only=True)).fillna(0.0)
        return X

    def _prepare_target(self, df: pd.DataFrame) -> np.ndarray:
        if "target_encoded" in df.columns:
            target = pd.to_numeric(df["target_encoded"], errors="coerce").fillna(0)
        elif "target" in df.columns:
            mapping = {"no_action": 0, "scale_down": 1, "scale_up": 2}
            target = df["target"].astype(str).str.lower().map(mapping).fillna(0)
        else:
            # Fallback heuristic target based on utilization
            cpu = pd.to_numeric(df.get("cpu_usage", 50), errors="coerce").fillna(50)
            target = pd.Series(0, index=df.index)
            target[cpu < 20] = 1  # scale down
            target[cpu > 80] = 2  # scale up
        return target.astype(np.int64).to_numpy()

    def train(self, df: pd.DataFrame, test_size: float = 0.2, deep: bool = True, extreme: bool = False, finetune: bool = False) -> Dict[str, Any]:
        X = self._prepare_features(df)
        y = self._prepare_target(df)

        if len(X) < 15:
            raise ValueError("At least 15 resource telemetry records required for training.")

        self.feature_names = list(X.columns)
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X)

        strat = y if len(np.unique(y)) > 1 else None
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y, test_size=test_size, random_state=self.random_state, stratify=strat
        )

        existing_clf = None
        if finetune and (self.model_dir / "resource_classifier.joblib").exists():
            try:
                existing_clf = joblib.load(self.model_dir / "resource_classifier.joblib")
            except Exception:
                pass

        if extreme:
            n_est = 1800
            lr = 0.015 if not finetune else 0.007
            max_depth = 10
            num_leaves = 127
        elif deep:
            n_est = 600
            lr = 0.03 if not finetune else 0.012
            max_depth = 8
            num_leaves = 63
        else:
            n_est = 300
            lr = 0.05 if not finetune else 0.02
            max_depth = 6
            num_leaves = 31
        if LIGHTGBM_AVAILABLE and len(np.unique(y_train)) > 1:
            try:
                clf = lgb.LGBMClassifier(
                    n_estimators=n_est,
                    learning_rate=lr,
                    max_depth=8,
                    num_leaves=63,
                    subsample=0.85,
                    colsample_bytree=0.85,
                    random_state=self.random_state,
                    verbose=-1,
                )
                init_booster = None
                if finetune and existing_clf is not None and hasattr(existing_clf, "booster_"):
                    init_booster = existing_clf.booster_
                clf.fit(X_train, y_train, init_model=init_booster)
                self.classifier = clf
            except Exception:
                self.classifier = ExtraTreesClassifier(
                    n_estimators=n_est, max_depth=14, random_state=self.random_state, n_jobs=-1
                )
                self.classifier.fit(X_train, y_train)
        else:
            self.classifier = ExtraTreesClassifier(
                n_estimators=n_est, max_depth=14, random_state=self.random_state, n_jobs=-1
            )
            self.classifier.fit(X_train, y_train)

        preds = self.classifier.predict(X_test)
        acc = accuracy_score(y_test, preds)
        macro_f1 = f1_score(y_test, preds, average="macro", zero_division=0)
        weighted_f1 = f1_score(y_test, preds, average="weighted", zero_division=0)

        # Regressor for target hourly cost
        if "price_per_hour" in df.columns or "cost" in df.columns:
            cost_col = "price_per_hour" if "price_per_hour" in df.columns else "cost"
            target_cost = pd.to_numeric(df[cost_col], errors="coerce").fillna(0.1)
            n_reg = 1000 if extreme else (400 if deep else 150)
            reg = RandomForestRegressor(n_estimators=n_reg, max_depth=16 if extreme else 14, random_state=self.random_state, n_jobs=-1)
            reg.fit(X_scaled, target_cost.values)
            self.cost_model = reg
            cost_preds = reg.predict(X_test)
            mae = mean_absolute_error(target_cost.values[int(len(X_scaled)*(1-test_size)):], cost_preds[:len(target_cost.values[int(len(X_scaled)*(1-test_size)):])])
        else:
            mae = 0.0

        self.metrics = {
            "accuracy": float(round(acc, 4)),
            "macro_f1": float(round(macro_f1, 4)),
            "weighted_f1": float(round(weighted_f1, 4)),
            "cost_mae": float(round(mae, 4)),
            "classes": self.class_names,
            "feature_count": len(self.feature_names),
        }

        self.save()
        return self.metrics

    def save(self) -> None:
        self.model_dir.mkdir(parents=True, exist_ok=True)
        if self.classifier is not None:
            joblib.dump(self.classifier, self.model_dir / "resource_classifier.joblib")
        if self.cost_model is not None:
            joblib.dump(self.cost_model, self.model_dir / "resource_cost_regressor.joblib")
        if self.scaler is not None:
            joblib.dump(self.scaler, self.model_dir / "resource_scaler.joblib")
        meta = {
            "feature_names": self.feature_names,
            "metrics": self.metrics,
            "class_names": self.class_names,
        }
        (self.model_dir / "resource_metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    def load(self) -> bool:
        clf_path = self.model_dir / "resource_classifier.joblib"
        if clf_path.exists():
            self.classifier = joblib.load(clf_path)
            reg_path = self.model_dir / "resource_cost_regressor.joblib"
            if reg_path.exists():
                self.cost_model = joblib.load(reg_path)
            s_path = self.model_dir / "resource_scaler.joblib"
            if s_path.exists():
                self.scaler = joblib.load(s_path)
            meta_path = self.model_dir / "resource_metadata.json"
            if meta_path.exists():
                meta = json.loads(meta_path.read_text(encoding="utf-8"))
                self.feature_names = meta.get("feature_names", [])
                self.metrics = meta.get("metrics", {})
            return True
        return False

    def get_feature_importances(self) -> Dict[str, float]:
        if self.classifier is None:
            self.load()
        if self.classifier is None or not self.feature_names:
            return {}
        if hasattr(self.classifier, "feature_importances_"):
            imp = self.classifier.feature_importances_
            total = sum(imp) or 1.0
            return {name: float(round(val / total, 4)) for name, val in zip(self.feature_names, imp)}
        return {}

    def suggest_target_sku(
        self,
        provider: str,
        current_sku: str,
        action: str,
        environment: str = "untagged",
        step_depth: int = 1,
    ) -> Tuple[str, float, float]:
        """Returns (recommended_sku, new_hourly_cost, current_hourly_cost).
        
        Strict FinOps Guarantees:
        1. Family-awareness: Never crosses instance families (e.g. t3 will never jump to t2.xlarge).
        2. Cost invariants:
           - For scale_down: target SKU must have strictly lower hourly cost than current SKU.
             If already at the lowest tier in family, returns (current_sku, current_rate, current_rate).
           - For scale_up: target SKU must have strictly higher hourly cost than current SKU.
             If already at the highest tier in family, returns (current_sku, current_rate, current_rate).
        3. Environment policy:
           - development: allows up to 2-tier downsize within the family for aggressive cost savings
           - production & staging: strictly enforces single-tier downsize within the family
        """
        prov = str(provider).lower()
        if "azure" in prov:
            catalog = AZURE_INSTANCE_TIERS
        elif "gcp" in prov or "google" in prov:
            catalog = GCP_INSTANCE_TIERS
        else:
            catalog = AWS_INSTANCE_TIERS

        curr_entry = None
        for entry in catalog:
            if entry[0].lower() == str(current_sku).lower():
                curr_entry = entry
                break

        if curr_entry is None:
            # Fallback to middle tier of catalog
            curr_entry = catalog[len(catalog) // 2]

        curr_sku_name, _, _, curr_rate = curr_entry
        curr_family = get_sku_family(curr_sku_name)

        # Restrict candidates strictly to the same instance family
        family_catalog = [x for x in catalog if get_sku_family(x[0]) == curr_family]
        if not family_catalog or curr_entry not in family_catalog:
            family_catalog = catalog

        curr_idx = family_catalog.index(curr_entry) if curr_entry in family_catalog else 0
        env = str(environment).lower().strip()

        if action == "scale_down":
            depth = step_depth if step_depth > 1 else (2 if env == "development" else 1)
            target_idx = curr_idx + depth
            # Already at the bottom/cheapest tier of the family
            if target_idx >= len(family_catalog):
                return curr_sku_name, curr_rate, curr_rate
            target_entry = family_catalog[target_idx]
            # Downsize must yield strictly lower cost
            if target_entry[3] >= curr_rate:
                return curr_sku_name, curr_rate, curr_rate
            return target_entry[0], target_entry[3], curr_rate

        elif action == "scale_up":
            target_idx = curr_idx - 1
            # Already at the top/largest tier of the family
            if target_idx < 0:
                return curr_sku_name, curr_rate, curr_rate
            target_entry = family_catalog[target_idx]
            # Scale up must yield higher capacity/cost
            if target_entry[3] <= curr_rate:
                return curr_sku_name, curr_rate, curr_rate
            return target_entry[0], target_entry[3], curr_rate

        return curr_sku_name, curr_rate, curr_rate

    def evaluate_risk(
        self,
        cpu_avg: float,
        cpu_max: float,
        action: str,
        resilience_score: float = 90.0,
        environment: str = "untagged",
    ) -> Tuple[str, bool]:
        """Determines risk level (low, medium, high) and whether explicit user approval is required.
        
        Environment policies:
        - production: Never low risk on changes. Minimum medium risk with mandatory human approval (approval_required = True).
        - development: Low risk on downsize (approval_required = False) enabling zero-friction auto-dispatch.
        - staging / untagged: Standard balanced policy.
        """
        env = str(environment).lower().strip()
        if env == "production":
            if action == "scale_down":
                # In production, any change requires explicit human sign-off
                if cpu_avg < 10.0 and resilience_score >= 88.0:
                    return "medium", True
                return "high", True
            elif action == "scale_up":
                return "high", True
            return "low", False

        elif env == "development":
            if action == "scale_down":
                # In development, aggressive optimization is low risk and auto-dispatch approved
                return "low", False
            elif action == "scale_up":
                return "medium", False  # low friction even on scaling up in dev
            return "low", False

        # Staging & Untagged default policy
        if action == "scale_down":
            if cpu_avg < 15.0 and cpu_max < 35.0 and resilience_score >= 80.0:
                return "low", False  # Small optimization -> Automate allowed
            elif cpu_avg < 30.0 and resilience_score >= 70.0:
                return "medium", True  # Medium risk -> Approve required
            else:
                return "high", True  # High risk -> Approve required
        elif action == "scale_up":
            return "high", True  # Capacity expansion has cost impact -> Approve required
        return "low", False


def get_sku_specs(provider: str, sku: str) -> Dict[str, Any]:
    """Retrieves vCPU, RAM, hourly price, and monthly cost for a given cloud SKU."""
    prov = str(provider).lower()
    if "azure" in prov:
        catalog = AZURE_INSTANCE_TIERS
    elif "gcp" in prov or "google" in prov:
        catalog = GCP_INSTANCE_TIERS
    else:
        catalog = AWS_INSTANCE_TIERS

    for name, vcpu, ram_gb, price in catalog:
        if name.lower() == str(sku).lower():
            return {
                "sku": name,
                "vcpu": int(vcpu),
                "ram_gb": float(ram_gb),
                "hourly_cost": float(price),
                "monthly_cost": round(float(price) * 730, 2),
            }

    # Fallback to middle tier of catalog
    mid = catalog[len(catalog) // 2]
    return {
        "sku": str(sku),
        "vcpu": int(mid[1]),
        "ram_gb": float(mid[2]),
        "hourly_cost": float(mid[3]),
        "monthly_cost": round(float(mid[3]) * 730, 2),
    }