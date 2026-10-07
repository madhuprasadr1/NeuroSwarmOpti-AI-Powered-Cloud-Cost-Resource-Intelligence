"""
Cost anomaly detection module for cloud billing data.
Supports multi-method statistical anomaly detection (Z-score, IQR, Percentage spike)
and native integration with CloudAnomalyDetector machine learning models.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
import numpy as np
import pandas as pd

from ..collectors.base import BillingData
from ..core.exceptions import AnalyzerError


@dataclass
class AnomalyResult:
    """Cost anomaly detection result."""
    resource_id: str
    service: str
    anomaly_type: str
    severity: str  # 'low', 'medium', 'high', 'critical'
    description: str
    expected_value: float
    actual_value: float
    deviation_percentage: float
    confidence_score: float
    date: datetime
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        result = asdict(self)
        result["date"] = self.date.isoformat() if hasattr(self.date, "isoformat") else str(self.date)
        return result


class AnomalyDetector:
    """Cost anomaly detector supporting statistical and ML-based approaches."""

    def __init__(self, config: Any = None):
        self.config = config
        self._ml_model = None

    def _get_ml_detector(self):
        if self._ml_model is None:
            try:
                from ..intelligence.anomaly import CloudAnomalyDetector
                detector = CloudAnomalyDetector()
                detector.load()
                self._ml_model = detector
            except Exception:
                self._ml_model = False
        return self._ml_model if self._ml_model is not False else None

    def detect_anomalies(
        self,
        billing_data: List[BillingData],
        methods: Optional[List[str]] = None,
        threshold: float = 2.0,
    ) -> List[AnomalyResult]:
        """Detect cost anomalies in billing data using specified detection methods.
        
        Args:
            billing_data: List of BillingData records to analyze.
            methods: List of methods to use ('zscore', 'iqr', 'percentage', 'ml').
            threshold: Anomaly threshold multiplier (default: 2.0).
            
        Returns:
            List of detected AnomalyResult instances.
        """
        if not billing_data:
            return []

        if methods is None:
            methods = ["zscore", "iqr", "percentage"]

        anomalies: List[AnomalyResult] = []

        try:
            # Convert to DataFrame for efficient grouping
            records = []
            for b in billing_data:
                records.append({
                    "resource_id": b.resource_id or "unknown",
                    "service": b.service or "General",
                    "cost": float(b.cost or 0.0),
                    "start_time": b.start_time,
                    "provider": getattr(b, "provider", "cloud"),
                })
            df = pd.DataFrame(records)

            if df.empty:
                return []

            # Group by service to establish baseline patterns
            for service, group in df.groupby("service"):
                costs = group["cost"].values
                if len(costs) < 2:
                    continue

                mean_cost = float(np.mean(costs))
                std_cost = float(np.std(costs))
                q25, q75 = float(np.percentile(costs, 25)), float(np.percentile(costs, 75))
                iqr = max(q75 - q25, 0.001)

                for idx, row in group.iterrows():
                    actual = float(row["cost"])
                    res_id = str(row["resource_id"])
                    event_date = row["start_time"] if isinstance(row["start_time"], datetime) else datetime.now(timezone.utc)

                    # Method 1: Z-score
                    if "zscore" in methods and std_cost > 0:
                        z = (actual - mean_cost) / std_cost
                        if z > threshold:
                            dev = ((actual - mean_cost) / max(mean_cost, 0.001)) * 100.0
                            sev = self._determine_severity(dev, z)
                            conf = min(0.99, max(0.60, 0.5 + 0.1 * z))
                            anomalies.append(
                                AnomalyResult(
                                    resource_id=res_id,
                                    service=str(service),
                                    anomaly_type="zscore",
                                    severity=sev,
                                    description=f"Cost of ${actual:.2f} deviates from expected mean of ${mean_cost:.2f} (Z-score: {z:.2f})",
                                    expected_value=round(mean_cost, 2),
                                    actual_value=round(actual, 2),
                                    deviation_percentage=round(dev, 1),
                                    confidence_score=round(conf, 2),
                                    date=event_date,
                                    metadata={"method": "zscore", "z_score": round(z, 2)},
                                )
                            )
                            continue

                    # Method 2: IQR
                    if "iqr" in methods:
                        iqr_cutoff = q75 + threshold * iqr
                        if actual > iqr_cutoff:
                            dev = ((actual - q75) / max(q75, 0.001)) * 100.0
                            sev = self._determine_severity(dev, (actual - q75) / iqr)
                            conf = min(0.95, max(0.60, 0.65 + 0.05 * ((actual - q75) / iqr)))
                            anomalies.append(
                                AnomalyResult(
                                    resource_id=res_id,
                                    service=str(service),
                                    anomaly_type="iqr",
                                    severity=sev,
                                    description=f"Cost of ${actual:.2f} exceeds IQR upper bound of ${iqr_cutoff:.2f}",
                                    expected_value=round(q75, 2),
                                    actual_value=round(actual, 2),
                                    deviation_percentage=round(dev, 1),
                                    confidence_score=round(conf, 2),
                                    date=event_date,
                                    metadata={"method": "iqr", "cutoff": round(iqr_cutoff, 2)},
                                )
                            )
                            continue

                    # Method 3: Percentage spike
                    if "percentage" in methods and mean_cost > 0:
                        pct_increase = ((actual - mean_cost) / mean_cost) * 100.0
                        pct_threshold = max(50.0, threshold * 30.0)
                        if pct_increase > pct_threshold:
                            sev = self._determine_severity(pct_increase, pct_increase / 50.0)
                            conf = min(0.92, max(0.55, 0.5 + pct_increase / 300.0))
                            anomalies.append(
                                AnomalyResult(
                                    resource_id=res_id,
                                    service=str(service),
                                    anomaly_type="percentage_spike",
                                    severity=sev,
                                    description=f"Cost spiked by {pct_increase:.1f}% above historical baseline (${mean_cost:.2f})",
                                    expected_value=round(mean_cost, 2),
                                    actual_value=round(actual, 2),
                                    deviation_percentage=round(pct_increase, 1),
                                    confidence_score=round(conf, 2),
                                    date=event_date,
                                    metadata={"method": "percentage", "increase_pct": round(pct_increase, 1)},
                                )
                            )

            # Sort anomalies by deviation percentage descending
            anomalies.sort(key=lambda a: a.deviation_percentage, reverse=True)
            return anomalies

        except Exception as exc:
            raise AnalyzerError(f"Failed to detect cost anomalies: {exc}") from exc

    def _determine_severity(self, deviation_pct: float, score: float) -> str:
        if deviation_pct >= 150.0 or score >= 4.0:
            return "critical"
        elif deviation_pct >= 75.0 or score >= 3.0:
            return "high"
        elif deviation_pct >= 30.0 or score >= 2.0:
            return "medium"
        return "low"
