from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score
from sklearn.preprocessing import StandardScaler

try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False


class CloudAnomalyDetector:
    """Infrastructure Anomaly and Operational Risk Detection Engine.
    
    Trained on Cloud Anomaly dataset (277,570 records) to:
    1. Detect multi-variate telemetry outliers via Isolation Forest.
    2. Classify operational anomalies and predict risk score (0 to 100).
    3. Identify contributing anomaly metrics (power, cpu, execution time, instructions).
    """

    def __init__(
        self,
        model_dir: str | Path = "models",
        contamination: float = 0.05,
        random_state: int = 42,
    ):
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(parents=True, exist_ok=True)
        self.contamination = contamination
        self.random_state = random_state

        self.iso_forest: Optional[IsolationForest] = None
        self.supervised_model: Optional[Any] = None
        self.scaler: Optional[StandardScaler] = None
        self.feature_names: List[str] = []
        self.threshold: float = 0.5
        self.metrics: Dict[str, Any] = {}

    def _select_features(self, df: pd.DataFrame) -> pd.DataFrame:
        preferred = [
            "cpu_usage", "memory_usage", "network_traffic", "power_consumption",
            "num_executed_instructions", "execution_time", "energy_efficiency",
            "task_priority", "cpu_ratio", "memory_ratio", "resource_pressure",
            "execution_efficiency", "network_power_ratio",
            "anomaly_task_type_encoded", "anomaly_task_status_encoded",
            "task_type_encoded", "task_status_encoded",
        ]
        available = [c for c in preferred if c in df.columns]
        if not available:
            num = df.select_dtypes(include=[np.number]).columns
            available = [c for c in num if c not in ["anomaly_target", "anomaly_status", "target"]]
        X = df[available].replace([np.inf, -np.inf], np.nan)
        X = X.fillna(X.median(numeric_only=True)).fillna(0.0)
        return X

    def train(self, df: pd.DataFrame, validation_ratio: float = 0.2, deep: bool = True, extreme: bool = False, finetune: bool = False) -> Dict[str, Any]:
        """Trains dual-engine anomaly detector with deep gradient boosting and isolation trees."""
        X = self._select_features(df)
        self.feature_names = list(X.columns)

        y = None
        if "anomaly_target" in df.columns:
            y = pd.to_numeric(df["anomaly_target"], errors="coerce").fillna(0).astype(int).values

        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X)

        split = int(len(X) * (1.0 - validation_ratio))
        X_train, X_val = X_scaled[:split], X_scaled[split:]

        existing_sup = None
        if finetune and (self.model_dir / "anomaly_supervised_model.joblib").exists():
            try:
                existing_sup = joblib.load(self.model_dir / "anomaly_supervised_model.joblib")
            except Exception:
                pass

        # 1. Deep Unsupervised Isolation Forest
        n_iso = 800 if extreme else (500 if deep else 200)
        self.iso_forest = IsolationForest(
            n_estimators=n_iso,
            contamination=self.contamination,
            max_samples="auto",
            n_jobs=-1,
            random_state=self.random_state,
        )
        self.iso_forest.fit(X_train)

        # 2. Deep Supervised LightGBM with balanced class weights
        if y is not None and LIGHTGBM_AVAILABLE and len(np.unique(y[:split])) > 1:
            n_lgb = 1600 if extreme else (800 if deep else 300)
            lr = (0.015 if extreme else (0.03 if deep else 0.05)) * (0.6 if finetune else 1.0)
            leaves = 127 if (extreme or deep) else 63
            depth = 10 if extreme else (9 if deep else 7)
            try:
                clf = lgb.LGBMClassifier(
                    n_estimators=n_lgb,
                    learning_rate=lr,
                    max_depth=depth,
                    num_leaves=leaves,
                    class_weight="balanced",
                    subsample=0.9,
                    colsample_bytree=0.9,
                    min_child_samples=25,
                    random_state=self.random_state,
                    verbose=-1,
                )
                init_booster = None
                if finetune and existing_sup is not None and hasattr(existing_sup, "booster_"):
                    init_booster = existing_sup.booster_
                clf.fit(X_train, y[:split], init_model=init_booster)
                self.supervised_model = clf
            except Exception:
                self.supervised_model = None

        # Evaluation
        raw_scores = -self.iso_forest.decision_function(X_val)
        norm_scores = (raw_scores - raw_scores.min()) / (raw_scores.max() - raw_scores.min() + 1e-9)

        if y is not None:
            y_val = y[split:]
            if self.supervised_model is not None:
                sup_probs = self.supervised_model.predict_proba(X_val)[:, 1]
                ensemble_scores = 0.3 * norm_scores + 0.7 * sup_probs
            else:
                ensemble_scores = norm_scores

            preds = (ensemble_scores >= 0.5).astype(int)
            prec = precision_score(y_val, preds, zero_division=0)
            rec = recall_score(y_val, preds, zero_division=0)
            f1 = f1_score(y_val, preds, zero_division=0)
            try:
                auc = roc_auc_score(y_val, ensemble_scores)
            except Exception:
                auc = 0.5

            self.metrics = {
                "precision": float(round(prec, 4)),
                "recall": float(round(rec, 4)),
                "f1": float(round(f1, 4)),
                "roc_auc": float(round(auc, 4)),
                "training_samples": len(X_train),
                "validation_samples": len(X_val),
            }
        else:
            self.metrics = {
                "training_samples": len(X_train),
                "validation_samples": len(X_val),
            }

        self.save()
        return self.metrics

    def predict_risk_score(self, telemetry_dict: Dict[str, Any]) -> Tuple[float, str, List[str]]:
        """Given a resource's telemetry metrics, predicts (risk_score_0_to_100, severity, factors)."""
        if self.iso_forest is None:
            self.load()

        if self.iso_forest is None or not self.feature_names:
            cpu = float(telemetry_dict.get("cpu_usage", 20.0) or 20.0)
            mem = float(telemetry_dict.get("memory_usage", 30.0) or 30.0)
            score = min(100.0, max(0.0, (cpu * 0.6 + mem * 0.4)))
            sev = "high" if score > 80 else ("medium" if score > 50 else "low")
            return float(round(score, 1)), sev, ["CPU Utilization" if cpu > 75 else "Stable Operations"]

        # Build feature DataFrame matching trained feature names to prevent StandardScaler warnings
        row_dict = {}
        for name in self.feature_names:
            val = telemetry_dict.get(name)
            if val is None:
                if "cpu_ratio" in name:
                    val = float(telemetry_dict.get("cpu_usage", 20.0)) / 100.0
                elif "memory_ratio" in name:
                    val = float(telemetry_dict.get("memory_usage", 30.0)) / 100.0
                elif "resource_pressure" in name:
                    val = (float(telemetry_dict.get("cpu_usage", 20.0)) + float(telemetry_dict.get("memory_usage", 30.0))) / 200.0
                else:
                    val = 0.0
            row_dict[name] = float(val)

        feat_df = pd.DataFrame([row_dict], columns=self.feature_names)
        if self.scaler is not None:
            feat_arr = self.scaler.transform(feat_df)
        else:
            feat_arr = feat_df.values

        raw = -self.iso_forest.decision_function(feat_arr)[0]
        # Normalization to [0, 100]
        iso_score = float(np.clip((raw + 0.3) / 0.6 * 100.0, 0.0, 100.0))

        if self.supervised_model is not None:
            sup_prob = float(self.supervised_model.predict_proba(feat_arr)[0, 1]) * 100.0
            risk_score = 0.4 * iso_score + 0.6 * sup_prob
        else:
            risk_score = iso_score

        risk_score = float(np.clip(risk_score, 0.0, 100.0))
        if risk_score >= 70.0:
            sev = "high"
        elif risk_score >= 35.0:
            sev = "medium"
        else:
            sev = "low"

        factors = []
        if float(telemetry_dict.get("cpu_usage", 0.0)) > 80.0:
            factors.append("CPU Saturation (>80%)")
        if float(telemetry_dict.get("memory_usage", 0.0)) > 85.0:
            factors.append("Memory Exhaustion (>85%)")
        if not factors:
            factors.append("Nominal Execution Profile")

        return float(round(risk_score, 1)), sev, factors

    def save(self) -> None:
        self.model_dir.mkdir(parents=True, exist_ok=True)
        if self.iso_forest is not None:
            joblib.dump(self.iso_forest, self.model_dir / "anomaly_isolation_forest.joblib")
        if self.supervised_model is not None:
            joblib.dump(self.supervised_model, self.model_dir / "anomaly_supervised_model.joblib")
        if self.scaler is not None:
            joblib.dump(self.scaler, self.model_dir / "anomaly_scaler.joblib")
        meta = {
            "feature_names": self.feature_names,
            "metrics": self.metrics,
            "contamination": self.contamination,
        }
        (self.model_dir / "anomaly_metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    def load(self) -> bool:
        iso_p = self.model_dir / "anomaly_isolation_forest.joblib"
        if iso_p.exists():
            self.iso_forest = joblib.load(iso_p)
            sup_p = self.model_dir / "anomaly_supervised_model.joblib"
            if sup_p.exists():
                self.supervised_model = joblib.load(sup_p)
            s_p = self.model_dir / "anomaly_scaler.joblib"
            if s_p.exists():
                self.scaler = joblib.load(s_p)
            meta_p = self.model_dir / "anomaly_metadata.json"
            if meta_p.exists():
                meta = json.loads(meta_p.read_text(encoding="utf-8"))
                self.feature_names = meta.get("feature_names", [])
                self.metrics = meta.get("metrics", {})
            return True
        return False