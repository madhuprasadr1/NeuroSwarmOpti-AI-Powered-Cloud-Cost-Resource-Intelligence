from __future__ import annotations

from dataclasses import asdict
from datetime import datetime, timezone
from typing import Any, Dict, Iterable, List, Optional

import numpy as np
import pandas as pd

from cba.core.models import (
    ActionType,
    AnomalyResult,
    CloudProvider,
    OptimizationRecommendation,
    RecommendationStatus,
    ResourceCandidate,
    RiskLevel,
    WorkloadForecast,
)


class CloudOptimizer:
    def __init__(
        self,
        min_confidence: float = 0.60,
        max_risk_score: float = 0.70,
        min_monthly_savings: float = 1.0,
        scale_down_utilization: float = 0.35,
        scale_up_utilization: float = 0.80,
    ):
        self.min_confidence = float(min_confidence)
        self.max_risk_score = float(max_risk_score)
        self.min_monthly_savings = float(min_monthly_savings)
        self.scale_down_utilization = float(scale_down_utilization)
        self.scale_up_utilization = float(scale_up_utilization)

    def _value(self, value: Any, default: float = 0.0) -> float:
        try:
            if value is None:
                return default
            result = float(value)
            if not np.isfinite(result):
                return default
            return result
        except (TypeError, ValueError):
            return default

    def _clamp(self, value: float, low: float = 0.0, high: float = 1.0) -> float:
        return max(low, min(high, float(value)))

    def _get(self, obj: Any, key: str, default: Any = None) -> Any:
        if obj is None:
            return default

        if isinstance(obj, dict):
            return obj.get(key, default)

        return getattr(obj, key, default)

    def _resource_id(self, resource: Any, index: int) -> str:
        value = self._get(resource, "resource_id")
        if value:
            return str(value)

        value = self._get(resource, "id")
        if value:
            return str(value)

        value = self._get(resource, "vm_id")
        if value:
            return str(value)

        value = self._get(resource, "name")
        if value:
            return str(value)

        return f"resource-{index + 1}"

    def _provider(self, resource: Any) -> CloudProvider:
        provider = self._get(resource, "cloud_provider", CloudProvider.AWS)

        if isinstance(provider, CloudProvider):
            return provider

        normalized = str(provider).strip().lower()

        mapping = {
            "aws": CloudProvider.AWS,
            "amazon": CloudProvider.AWS,
            "azure": CloudProvider.AZURE,
            "gcp": CloudProvider.GCP,
            "google": CloudProvider.GCP,
        }

        return mapping.get(normalized, CloudProvider.AWS)

    def _monthly_cost(self, resource: Any) -> float:
        monthly = self._get(resource, "monthly_cost")

        if monthly is not None:
            return max(0.0, self._value(monthly))

        hourly = self._get(resource, "price_per_hour")

        if hourly is not None:
            return max(0.0, self._value(hourly) * 730.0)

        cost = self._get(resource, "cost")

        if cost is not None:
            value = self._value(cost)
            if value <= 100:
                return max(0.0, value * 730.0)
            return max(0.0, value)

        return 0.0

    def _utilization(self, resource: Any) -> float:
        utilization = self._get(resource, "utilization")

        if utilization is None:
            cpu = self._value(self._get(resource, "cpu_usage"))
            memory = self._value(self._get(resource, "memory_usage"))

            if cpu > 1:
                cpu /= 100.0

            if memory > 1:
                memory /= 100.0

            utilization = np.mean([cpu, memory])

        utilization = self._value(utilization)

        if utilization > 1:
            utilization /= 100.0

        return self._clamp(utilization)

    def _pressure(self, resource: Any) -> float:
        pressure = self._get(resource, "resource_pressure")

        if pressure is not None:
            return self._clamp(self._value(pressure))

        cpu = self._value(self._get(resource, "cpu_usage"))
        memory = self._value(self._get(resource, "memory_usage"))

        if cpu > 1:
            cpu /= 100.0

        if memory > 1:
            memory /= 100.0

        return self._clamp(max(cpu, memory))

    def _model_prediction(self, resource: Any) -> Dict[str, Any]:
        prediction = self._get(resource, "prediction")

        if isinstance(prediction, dict):
            return prediction

        prediction = self._get(resource, "model_prediction")

        if isinstance(prediction, dict):
            return prediction

        return {}

    def _action_from_prediction(
        self,
        resource: Any,
        prediction: Dict[str, Any],
    ) -> ActionType:
        action = prediction.get("action")

        if action:
            normalized = str(action).strip().lower()

            mapping = {
                "no_action": ActionType.NO_ACTION,
                "scale_down": ActionType.SCALE_DOWN,
                "scale_up": ActionType.SCALE_UP,
                "right_sizing": ActionType.RIGHTSIZING,
                "rightsizing": ActionType.RIGHTSIZING,
                "stop": ActionType.STOP,
                "start": ActionType.START,
                "delete": ActionType.DELETE,
                "storage_optimization": ActionType.STORAGE_OPTIMIZATION,
                "reserved_capacity": ActionType.RESERVED_CAPACITY,
                "spot": ActionType.SPOT,
                "schedule": ActionType.SCHEDULE,
            }

            if normalized in mapping:
                return mapping[normalized]

        utilization = self._utilization(resource)
        pressure = self._pressure(resource)

        if pressure >= self.scale_up_utilization:
            return ActionType.SCALE_UP

        if utilization <= self.scale_down_utilization:
            return ActionType.SCALE_DOWN

        return ActionType.RIGHTSIZING

    def _confidence(
        self,
        resource: Any,
        prediction: Dict[str, Any],
        anomaly: Optional[Any],
        forecast: Optional[Any],
    ) -> float:
        confidence = self._value(prediction.get("confidence"), 0.0)

        if confidence <= 0:
            probabilities = prediction.get("probabilities")

            if isinstance(probabilities, dict) and probabilities:
                confidence = max(
                    self._value(value)
                    for value in probabilities.values()
                )

        if confidence <= 0:
            confidence = 0.55

        utilization = self._utilization(resource)

        stability = 1.0 - abs(utilization - 0.5) * 0.35

        anomaly_penalty = 0.0

        if anomaly is not None:
            anomaly_flag = bool(self._get(anomaly, "is_anomaly", False))
            anomaly_score = self._value(
                self._get(anomaly, "anomaly_score"),
                0.0,
            )

            if anomaly_flag:
                anomaly_penalty = 0.25
            else:
                anomaly_penalty = anomaly_score * 0.10

        forecast_confidence = 1.0

        if forecast is not None:
            forecast_confidence = self._value(
                self._get(forecast, "confidence"),
                0.85,
            )

            if forecast_confidence <= 0:
                forecast_confidence = 0.85

        confidence *= stability
        confidence *= self._clamp(forecast_confidence)
        confidence -= anomaly_penalty

        return self._clamp(confidence, 0.0, 1.0)

    def _risk_score(
        self,
        resource: Any,
        action: ActionType,
        anomaly: Optional[Any],
        forecast: Optional[Any],
    ) -> float:
        utilization = self._utilization(resource)
        pressure = self._pressure(resource)

        risk = 0.10

        if action == ActionType.SCALE_DOWN:
            if pressure > 0.65:
                risk += 0.35

            if utilization > 0.50:
                risk += 0.25

        elif action == ActionType.SCALE_UP:
            if pressure < 0.55:
                risk += 0.20

        elif action in {
            ActionType.STOP,
            ActionType.DELETE,
        }:
            risk += 0.35

        elif action == ActionType.RIGHTSIZING:
            risk += 0.10

        if anomaly is not None:
            if bool(self._get(anomaly, "is_anomaly", False)):
                risk += 0.30

            risk += self._value(
                self._get(anomaly, "anomaly_score"),
                0.0,
            ) * 0.10

        if forecast is not None:
            forecast_confidence = self._value(
                self._get(forecast, "confidence"),
                0.85,
            )

            risk += (1.0 - self._clamp(forecast_confidence)) * 0.20

        return self._clamp(risk)

    def _savings_factor(
        self,
        action: ActionType,
        utilization: float,
        pressure: float,
        prediction: Dict[str, Any],
    ) -> float:
        model_savings = self._value(
            prediction.get("savings_factor"),
            -1.0,
        )

        if model_savings >= 0:
            return self._clamp(model_savings, 0.0, 0.90)

        optimization_score = self._value(
            prediction.get("optimization_score"),
            -1.0,
        )

        if optimization_score >= 0:
            return self._clamp(optimization_score * 0.60, 0.0, 0.90)

        if action == ActionType.SCALE_DOWN:
            return self._clamp(
                (self.scale_down_utilization - utilization + 0.20) * 0.80,
                0.0,
                0.70,
            )

        if action == ActionType.RIGHTSIZING:
            return self._clamp(
                (1.0 - utilization) * 0.35,
                0.0,
                0.45,
            )

        if action == ActionType.STORAGE_OPTIMIZATION:
            return self._clamp(
                (1.0 - utilization) * 0.25,
                0.0,
                0.35,
            )

        return 0.0

    def _expected_utilization(
        self,
        resource: Any,
        action: ActionType,
        forecast: Optional[Any],
    ) -> float:
        utilization = self._utilization(resource)

        forecast_utilization = None

        if forecast is not None:
            forecast_utilization = self._get(
                forecast,
                "predicted_utilization",
            )

            if forecast_utilization is None:
                forecast_utilization = self._get(
                    forecast,
                    "predicted_cpu",
                )

        if forecast_utilization is not None:
            forecast_utilization = self._value(forecast_utilization)

            if forecast_utilization > 1:
                forecast_utilization /= 100.0

            utilization = max(utilization, forecast_utilization)

        if action == ActionType.SCALE_DOWN:
            utilization += 0.08

        elif action == ActionType.SCALE_UP:
            utilization -= 0.10

        elif action == ActionType.RIGHTSIZING:
            utilization += 0.05

        return self._clamp(utilization)

    def _risk_level(self, score: float) -> RiskLevel:
        if score >= 0.80:
            return RiskLevel.CRITICAL

        if score >= 0.60:
            return RiskLevel.HIGH

        if score >= 0.35:
            return RiskLevel.MEDIUM

        return RiskLevel.LOW

    def _rationale(
        self,
        resource: Any,
        action: ActionType,
        confidence: float,
        risk_score: float,
        savings: float,
        forecast: Optional[Any],
        anomaly: Optional[Any],
    ) -> str:
        utilization = self._utilization(resource)
        pressure = self._pressure(resource)

        reasons = []

        reasons.append(
            f"current utilization is {utilization * 100:.1f}%"
        )

        if pressure != utilization:
            reasons.append(
                f"resource pressure is {pressure * 100:.1f}%"
            )

        if forecast is not None:
            predicted = self._get(
                forecast,
                "predicted_utilization",
            )

            if predicted is None:
                predicted = self._get(
                    forecast,
                    "predicted_cpu",
                )

            if predicted is not None:
                predicted = self._value(predicted)

                if predicted > 1:
                    predicted /= 100.0

                reasons.append(
                    f"forecast utilization is {predicted * 100:.1f}%"
                )

        if anomaly is not None and bool(
            self._get(anomaly, "is_anomaly", False)
        ):
            reasons.append("anomaly detection indicates elevated operational risk")

        reasons.append(f"model confidence is {confidence * 100:.1f}%")
        reasons.append(f"estimated monthly savings are {savings:.2f}")

        return (
            f"{action.value} recommended because "
            + ", ".join(reasons)
            + f"; calculated risk is {risk_score * 100:.1f}%."
        )

    def _safe_action(
        self,
        action: ActionType,
        confidence: float,
        risk_score: float,
        savings: float,
    ) -> ActionType:
        if action == ActionType.NO_ACTION:
            return action

        if confidence < self.min_confidence:
            return ActionType.NO_ACTION

        if risk_score > self.max_risk_score:
            return ActionType.NO_ACTION

        if savings < self.min_monthly_savings:
            return ActionType.NO_ACTION

        return action

    def optimize_resource(
        self,
        resource: Any,
        prediction: Optional[Dict[str, Any]] = None,
        anomaly: Optional[Any] = None,
        forecast: Optional[Any] = None,
        budget_context: Optional[Dict[str, Any]] = None,
    ) -> OptimizationRecommendation:
        prediction = prediction or self._model_prediction(resource)

        resource_id = self._resource_id(resource, 0)
        provider = self._provider(resource)
        monthly_cost = self._monthly_cost(resource)
        utilization = self._utilization(resource)
        pressure = self._pressure(resource)

        proposed_action = self._action_from_prediction(
            resource,
            prediction,
        )

        confidence = self._confidence(
            resource,
            prediction,
            anomaly,
            forecast,
        )

        risk_score = self._risk_score(
            resource,
            proposed_action,
            anomaly,
            forecast,
        )

        factor = self._savings_factor(
            proposed_action,
            utilization,
            pressure,
            prediction,
        )

        savings = max(0.0, monthly_cost * factor)

        budget_priority = 0.0

        if budget_context:
            budget_utilization = self._value(
                budget_context.get("budget_utilization_pct"),
                0.0,
            )

            if budget_utilization > 1:
                budget_utilization /= 100.0

            budget_priority = self._clamp(budget_utilization)

            if budget_priority >= 0.90:
                savings *= 1.15

        action = self._safe_action(
            proposed_action,
            confidence,
            risk_score,
            savings,
        )

        if action == ActionType.NO_ACTION:
            savings = 0.0
            status = RecommendationStatus.REJECTED
        else:
            status = RecommendationStatus.PENDING

        expected_utilization = self._expected_utilization(
            resource,
            action,
            forecast,
        )

        rationale = self._rationale(
            resource,
            action,
            confidence,
            risk_score,
            savings,
            forecast,
            anomaly,
        )

        target = self._get(resource, "target")
        if target is None:
            target = action.value

        current_size = self._get(resource, "vm_type")
        recommended_size = prediction.get("recommended_vm_type")

        if recommended_size is None:
            recommended_size = prediction.get("recommended_size")

        if recommended_size is None:
            recommended_size = current_size

        metadata = {
            "provider": provider.value,
            "current_utilization": utilization,
            "resource_pressure": pressure,
            "monthly_cost": monthly_cost,
            "budget_priority": budget_priority,
            "model_prediction": prediction,
            "target": target,
            "generated_at": datetime.now(timezone.utc).isoformat(),
        }

        return OptimizationRecommendation(
            recommendation_id=f"opt-{resource_id}-{datetime.now(timezone.utc).strftime('%Y%m%d%H%M%S%f')}",
            resource_id=resource_id,
            provider=provider,
            action=action,
            risk_level=self._risk_level(risk_score),
            confidence=confidence,
            estimated_monthly_savings=savings,
            current_utilization=utilization,
            expected_utilization=expected_utilization,
            current_size=current_size,
            recommended_size=recommended_size,
            rationale=rationale,
            status=status,
            metadata=metadata,
        )

    def optimize(
        self,
        resources: Iterable[Any],
        predictions: Optional[Dict[str, Dict[str, Any]]] = None,
        anomalies: Optional[Dict[str, Any]] = None,
        forecasts: Optional[Dict[str, Any]] = None,
        budget_context: Optional[Dict[str, Any]] = None,
    ) -> List[OptimizationRecommendation]:
        predictions = predictions or {}
        anomalies = anomalies or {}
        forecasts = forecasts or {}

        recommendations = []

        for index, resource in enumerate(resources):
            resource_id = self._resource_id(resource, index)

            prediction = predictions.get(
                resource_id,
                self._model_prediction(resource),
            )

            anomaly = anomalies.get(resource_id)
            forecast = forecasts.get(resource_id)

            recommendation = self.optimize_resource(
                resource=resource,
                prediction=prediction,
                anomaly=anomaly,
                forecast=forecast,
                budget_context=budget_context,
            )

            recommendations.append(recommendation)

        recommendations.sort(
            key=lambda item: (
                item.estimated_monthly_savings,
                item.confidence,
            ),
            reverse=True,
        )

        return recommendations

    def rank(
        self,
        recommendations: Iterable[OptimizationRecommendation],
    ) -> List[OptimizationRecommendation]:
        return sorted(
            recommendations,
            key=lambda item: (
                item.estimated_monthly_savings
                * item.confidence
                * (1.0 - self._risk_value(item.risk_level))
            ),
            reverse=True,
        )

    def _risk_value(self, level: RiskLevel) -> float:
        mapping = {
            RiskLevel.LOW: 0.15,
            RiskLevel.MEDIUM: 0.40,
            RiskLevel.HIGH: 0.70,
            RiskLevel.CRITICAL: 1.00,
        }

        return mapping.get(level, 0.50)

    def summary(
        self,
        recommendations: Iterable[OptimizationRecommendation],
    ) -> Dict[str, Any]:
        items = list(recommendations)

        total_savings = sum(
            max(0.0, self._value(item.estimated_monthly_savings))
            for item in items
        )

        action_counts: Dict[str, int] = {}

        for item in items:
            key = item.action.value
            action_counts[key] = action_counts.get(key, 0) + 1

        risk_counts: Dict[str, int] = {}

        for item in items:
            key = item.risk_level.value
            risk_counts[key] = risk_counts.get(key, 0) + 1

        approved = [
            item
            for item in items
            if item.status == RecommendationStatus.PENDING
        ]

        return {
            "total_resources": len(items),
            "optimization_candidates": len(approved),
            "estimated_monthly_savings": round(total_savings, 2),
            "estimated_annual_savings": round(total_savings * 12.0, 2),
            "action_counts": action_counts,
            "risk_counts": risk_counts,
            "average_confidence": round(
                float(
                    np.mean(
                        [
                            item.confidence
                            for item in items
                        ]
                    )
                )
                if items
                else 0.0,
                4,
            ),
        }

    def to_dataframe(
        self,
        recommendations: Iterable[OptimizationRecommendation],
    ) -> pd.DataFrame:
        rows = []

        for recommendation in recommendations:
            row = asdict(recommendation)

            provider = row.get("provider")
            action = row.get("action")
            risk = row.get("risk_level")
            status = row.get("status")

            if hasattr(provider, "value"):
                row["provider"] = provider.value

            if hasattr(action, "value"):
                row["action"] = action.value

            if hasattr(risk, "value"):
                row["risk_level"] = risk.value

            if hasattr(status, "value"):
                row["status"] = status.value

            rows.append(row)

        return pd.DataFrame(rows)

    def convert_candidates(
        self,
        recommendations: Iterable[OptimizationRecommendation],
    ) -> List[ResourceCandidate]:
        candidates = []

        for recommendation in recommendations:
            candidates.append(
                ResourceCandidate(
                    resource_id=recommendation.resource_id,
                    provider=recommendation.provider,
                    resource_type=self._resource_type(
                        recommendation.action
                    ),
                    current_cost=(
                        recommendation.estimated_monthly_savings
                        / max(
                            0.01,
                            self._estimated_savings_ratio(
                                recommendation
                            ),
                        )
                    ),
                    current_utilization=recommendation.current_utilization,
                    recommended_action=recommendation.action,
                    confidence=recommendation.confidence,
                    risk_score=self._risk_value(
                        recommendation.risk_level
                    ),
                    estimated_savings=(
                        recommendation.estimated_monthly_savings
                    ),
                )
            )

        return candidates

    def _estimated_savings_ratio(
        self,
        recommendation: OptimizationRecommendation,
    ) -> float:
        current = self._value(
            recommendation.current_utilization
        )

        if recommendation.action == ActionType.SCALE_DOWN:
            return self._clamp(
                (1.0 - current) * 0.60,
                0.05,
                0.70,
            )

        if recommendation.action == ActionType.RIGHTSIZING:
            return self._clamp(
                (1.0 - current) * 0.30,
                0.03,
                0.45,
            )

        return 0.10

    def _resource_type(self, action: ActionType):
        from cba.core.models import ResourceType

        if action == ActionType.STORAGE_OPTIMIZATION:
            return ResourceType.STORAGE

        if action in {
            ActionType.SCALE_UP,
            ActionType.SCALE_DOWN,
            ActionType.RIGHTSIZING,
            ActionType.STOP,
            ActionType.START,
        }:
            return ResourceType.COMPUTE

        return ResourceType.OTHER


Optimizer = CloudOptimizer
