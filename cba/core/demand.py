from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler

try:
    import joblib
except ImportError:
    joblib = None


@dataclass
class DemandModelMetrics:
    mae: float
    rmse: float
    r2: float
    samples: int
    model_name: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "mae": float(self.mae),
            "rmse": float(self.rmse),
            "r2": float(self.r2),
            "samples": int(self.samples),
            "model_name": self.model_name,
        }


@dataclass
class DemandForecastResult:
    metric: str
    predictions: list[float]
    timestamps: list[datetime]
    confidence: float
    model_name: str
    model_version: str
    lower_bounds: list[float]
    upper_bounds: list[float]
    metrics: dict[str, float]

    @property
    def peak_demand(self) -> float:
        return max(self.predictions) if self.predictions else 0.0

    @property
    def average_demand(self) -> float:
        if not self.predictions:
            return 0.0
        return float(np.mean(self.predictions))

    @property
    def growth_rate(self) -> float:
        if len(self.predictions) < 2 or self.predictions[0] == 0:
            return 0.0
        return (
            (self.predictions[-1] - self.predictions[0])
            / abs(self.predictions[0])
        ) * 100.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "metric": self.metric,
            "predictions": [float(x) for x in self.predictions],
            "timestamps": [x.isoformat() for x in self.timestamps],
            "confidence": float(self.confidence),
            "model_name": self.model_name,
            "model_version": self.model_version,
            "lower_bounds": [float(x) for x in self.lower_bounds],
            "upper_bounds": [float(x) for x in self.upper_bounds],
            "metrics": {
                key: float(value)
                for key, value in self.metrics.items()
            },
            "peak_demand": float(self.peak_demand),
            "average_demand": float(self.average_demand),
            "growth_rate": float(self.growth_rate),
        }


class BusinessDemandForecaster:
    def __init__(
        self,
        model_dir: str | Path = "models",
        random_state: int = 42,
    ) -> None:
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(parents=True, exist_ok=True)
        self.random_state = random_state
        self.model: Any = None
        self.scaler = StandardScaler()
        self.metric_name = "demand"
        self.model_name = "HistGradientBoostingDemandForecaster"
        self.model_version = "1.0"
        self.feature_columns = [
            "hour",
            "day_of_week",
            "day_of_month",
            "month",
            "is_weekend",
            "is_business_hour",
            "lag_1",
            "lag_2",
            "lag_3",
            "lag_6",
            "lag_12",
            "lag_24",
            "rolling_mean_3",
            "rolling_mean_6",
            "rolling_mean_12",
            "rolling_std_6",
            "rolling_std_12",
            "trend_6",
            "trend_12",
        ]
        self.metrics: dict[str, float] = {}
        self.residual_std = 0.0

    def _prepare_dataframe(
        self,
        data: pd.DataFrame,
        timestamp_column: str = "timestamp",
        value_column: str = "value",
    ) -> pd.DataFrame:
        if data is None or data.empty:
            raise ValueError("Demand dataset is empty")

        frame = data.copy()

        if timestamp_column not in frame.columns:
            candidates = [
                "datetime",
                "date",
                "time",
                "recorded_at",
                "observed_at",
            ]
            found = next(
                (column for column in candidates if column in frame.columns),
                None,
            )
            if found is None:
                raise ValueError("No timestamp column found")
            timestamp_column = found

        if value_column not in frame.columns:
            candidates = [
                "demand",
                "requests",
                "request_rate",
                "active_users",
                "business_demand",
                "users",
                "value",
            ]
            found = next(
                (column for column in candidates if column in frame.columns),
                None,
            )
            if found is None:
                raise ValueError("No demand value column found")
            value_column = found

        frame[timestamp_column] = pd.to_datetime(
            frame[timestamp_column],
            errors="coerce",
        )

        frame[value_column] = pd.to_numeric(
            frame[value_column],
            errors="coerce",
        )

        frame = frame.dropna(
            subset=[timestamp_column, value_column]
        ).copy()

        frame = frame.sort_values(timestamp_column)
        frame = frame.drop_duplicates(
            subset=[timestamp_column],
            keep="last",
        )

        frame = frame.rename(
            columns={
                timestamp_column: "timestamp",
                value_column: "value",
            }
        )

        frame["value"] = frame["value"].clip(lower=0.0)

        return frame[["timestamp", "value"]].reset_index(drop=True)

    def _feature_engineering(
        self,
        data: pd.DataFrame,
    ) -> pd.DataFrame:
        frame = data.copy()

        timestamp = frame["timestamp"]

        frame["hour"] = timestamp.dt.hour
        frame["day_of_week"] = timestamp.dt.dayofweek
        frame["day_of_month"] = timestamp.dt.day
        frame["month"] = timestamp.dt.month
        frame["is_weekend"] = (
            timestamp.dt.dayofweek >= 5
        ).astype(int)

        frame["is_business_hour"] = (
            (timestamp.dt.hour >= 9)
            & (timestamp.dt.hour < 18)
            & (timestamp.dt.dayofweek < 5)
        ).astype(int)

        for lag in [1, 2, 3, 6, 12, 24]:
            frame[f"lag_{lag}"] = frame["value"].shift(lag)

        for window in [3, 6, 12]:
            frame[f"rolling_mean_{window}"] = (
                frame["value"]
                .shift(1)
                .rolling(window)
                .mean()
            )

        for window in [6, 12]:
            frame[f"rolling_std_{window}"] = (
                frame["value"]
                .shift(1)
                .rolling(window)
                .std()
            )

        frame["trend_6"] = (
            frame["value"].shift(1)
            - frame["value"].shift(6)
        )

        frame["trend_12"] = (
            frame["value"].shift(1)
            - frame["value"].shift(12)
        )

        return frame

    def _build_training_set(
        self,
        data: pd.DataFrame,
    ) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
        frame = self._feature_engineering(data)

        frame = frame.dropna(
            subset=self.feature_columns + ["value"]
        ).reset_index(drop=True)

        if len(frame) < 30:
            raise ValueError(
                "At least 30 valid demand observations are required"
            )

        x = frame[self.feature_columns].astype(float)
        y = frame["value"].astype(float)

        return x, y, frame

    def _create_model(self) -> Any:
        return HistGradientBoostingRegressor(
            learning_rate=0.05,
            max_iter=300,
            max_leaf_nodes=31,
            l2_regularization=1.0,
            random_state=self.random_state,
        )

    def train(
        self,
        data: pd.DataFrame,
        timestamp_column: str = "timestamp",
        value_column: str = "value",
        test_size: float = 0.2,
    ) -> DemandModelMetrics:
        frame = self._prepare_dataframe(
            data,
            timestamp_column=timestamp_column,
            value_column=value_column,
        )

        self.metric_name = value_column

        x, y, _ = self._build_training_set(frame)

        split_index = int(len(x) * (1.0 - test_size))

        split_index = max(
            1,
            min(split_index, len(x) - 1),
        )

        x_train = x.iloc[:split_index]
        x_test = x.iloc[split_index:]

        y_train = y.iloc[:split_index]
        y_test = y.iloc[split_index:]

        self.model = self._create_model()
        self.model.fit(x_train, y_train)

        predictions = self.model.predict(x_test)

        mae = mean_absolute_error(
            y_test,
            predictions,
        )

        rmse = float(
            np.sqrt(
                mean_squared_error(
                    y_test,
                    predictions,
                )
            )
        )

        r2 = r2_score(
            y_test,
            predictions,
        )

        residuals = (
            np.asarray(y_test)
            - np.asarray(predictions)
        )

        self.residual_std = float(
            np.std(residuals)
        )

        self.metrics = {
            "mae": float(mae),
            "rmse": float(rmse),
            "r2": float(r2),
            "samples": float(len(x)),
        }

        self.save()

        return DemandModelMetrics(
            mae=float(mae),
            rmse=rmse,
            r2=float(r2),
            samples=len(x),
            model_name=self.model_name,
        )

    def _fallback_model(
        self,
        data: pd.DataFrame,
    ) -> None:
        frame = self._prepare_dataframe(data)

        x, y, _ = self._build_training_set(frame)

        model = RandomForestRegressor(
            n_estimators=200,
            max_depth=12,
            min_samples_leaf=2,
            random_state=self.random_state,
            n_jobs=-1,
        )

        model.fit(x, y)

        self.model = model
        self.model_name = "RandomForestDemandForecaster"

    def _predict_one(
        self,
        history: pd.DataFrame,
    ) -> float:
        features = self._feature_engineering(history)

        row = features.iloc[-1:]

        if row[self.feature_columns].isnull().any().any():
            recent = history["value"].tail(6)
            return float(
                recent.mean()
                if not recent.empty
                else 0.0
            )

        prediction = self.model.predict(
            row[self.feature_columns]
        )[0]

        return max(0.0, float(prediction))

    def forecast(
        self,
        data: pd.DataFrame,
        horizon: int = 24,
        frequency: str = "h",
        timestamp_column: str = "timestamp",
        value_column: str = "value",
    ) -> DemandForecastResult:
        if horizon <= 0:
            raise ValueError("Forecast horizon must be positive")

        frame = self._prepare_dataframe(
            data,
            timestamp_column=timestamp_column,
            value_column=value_column,
        )

        self.metric_name = value_column

        if self.model is None:
            try:
                self.load()
            except Exception:
                self.train(
                    frame,
                    timestamp_column="timestamp",
                    value_column="value",
                )

        history = frame.copy()

        if len(history) < 24:
            raise ValueError(
                "At least 24 observations are required for forecasting"
            )

        last_timestamp = history["timestamp"].iloc[-1]

        if frequency in {"h", "H"}:
            delta = pd.Timedelta(hours=1)
        elif frequency in {"d", "D"}:
            delta = pd.Timedelta(days=1)
        elif frequency in {"m", "min"}:
            delta = pd.Timedelta(minutes=1)
        else:
            delta = pd.Timedelta(hours=1)

        predictions: list[float] = []
        timestamps: list[datetime] = []

        for step in range(1, horizon + 1):
            timestamp = last_timestamp + (
                delta * step
            )

            predicted = self._predict_one(history)

            predictions.append(predicted)
            timestamps.append(
                timestamp.to_pydatetime()
            )

            history = pd.concat(
                [
                    history,
                    pd.DataFrame(
                        {
                            "timestamp": [timestamp],
                            "value": [predicted],
                        }
                    ),
                ],
                ignore_index=True,
            )

        confidence = self._confidence_score()

        uncertainty = max(
            self.residual_std,
            np.std(predictions) * 0.1
            if len(predictions) > 1
            else 0.0,
        )

        lower_bounds = [
            max(
                0.0,
                value - (1.96 * uncertainty),
            )
            for value in predictions
        ]

        upper_bounds = [
            value + (1.96 * uncertainty)
            for value in predictions
        ]

        return DemandForecastResult(
            metric=self.metric_name,
            predictions=predictions,
            timestamps=timestamps,
            confidence=confidence,
            model_name=self.model_name,
            model_version=self.model_version,
            lower_bounds=lower_bounds,
            upper_bounds=upper_bounds,
            metrics=self.metrics.copy(),
        )

    def _confidence_score(self) -> float:
        if not self.metrics:
            return 0.5

        r2 = float(self.metrics.get("r2", 0.0))

        if np.isnan(r2):
            r2 = 0.0

        r2_component = max(
            0.0,
            min(1.0, (r2 + 1.0) / 2.0),
        )

        return float(
            max(
                0.0,
                min(
                    1.0,
                    0.5 + (0.5 * r2_component),
                ),
            )
        )

    def feature_importance(self) -> dict[str, float]:
        if self.model is None:
            return {}

        if hasattr(self.model, "feature_importances_"):
            values = self.model.feature_importances_
        else:
            return {}

        values = np.asarray(values, dtype=float)

        total = float(values.sum())

        if total <= 0:
            return {
                feature: 0.0
                for feature in self.feature_columns
            }

        return {
            feature: float(value / total)
            for feature, value in zip(
                self.feature_columns,
                values,
            )
        }

    def demand_capacity_signal(
        self,
        forecast: DemandForecastResult,
        current_capacity: float,
        safety_margin: float = 0.20,
    ) -> dict[str, Any]:
        current_capacity = max(
            0.0,
            float(current_capacity),
        )

        safety_margin = max(
            0.0,
            min(1.0, float(safety_margin)),
        )

        required_capacity = (
            forecast.peak_demand
            * (1.0 + safety_margin)
        )

        if current_capacity <= 0:
            utilization = 0.0
        else:
            utilization = (
                forecast.peak_demand
                / current_capacity
            ) * 100.0

        if current_capacity <= 0:
            action = "scale_up"
        elif required_capacity > current_capacity * 1.10:
            action = "scale_up"
        elif required_capacity < current_capacity * 0.65:
            action = "scale_down"
        else:
            action = "maintain"

        return {
            "current_capacity": current_capacity,
            "forecast_peak_demand": forecast.peak_demand,
            "required_capacity": required_capacity,
            "projected_peak_utilization": utilization,
            "recommended_action": action,
            "confidence": forecast.confidence,
            "risk": self._capacity_risk(
                utilization
            ),
        }

    def _capacity_risk(
        self,
        projected_utilization: float,
    ) -> str:
        if projected_utilization >= 95:
            return "critical"

        if projected_utilization >= 85:
            return "high"

        if projected_utilization >= 70:
            return "medium"

        return "low"

    def generate_business_features(
        self,
        data: pd.DataFrame,
    ) -> pd.DataFrame:
        frame = self._prepare_dataframe(data)

        result = frame.copy()

        result["business_hour"] = (
            (result["timestamp"].dt.hour >= 9)
            & (result["timestamp"].dt.hour < 18)
            & (result["timestamp"].dt.dayofweek < 5)
        ).astype(int)

        result["weekend"] = (
            result["timestamp"].dt.dayofweek >= 5
        ).astype(int)

        result["hour_sin"] = np.sin(
            2 * np.pi
            * result["timestamp"].dt.hour
            / 24.0
        )

        result["hour_cos"] = np.cos(
            2 * np.pi
            * result["timestamp"].dt.hour
            / 24.0
        )

        result["weekly_sin"] = np.sin(
            2 * np.pi
            * result["timestamp"].dt.dayofweek
            / 7.0
        )

        result["weekly_cos"] = np.cos(
            2 * np.pi
            * result["timestamp"].dt.dayofweek
            / 7.0
        )

        return result

    def save(self) -> Path:
        if joblib is None:
            raise RuntimeError(
                "joblib is required to save the demand model"
            )

        path = (
            self.model_dir
            / "business_demand_forecaster.joblib"
        )

        payload = {
            "model": self.model,
            "feature_columns": self.feature_columns,
            "metric_name": self.metric_name,
            "model_name": self.model_name,
            "model_version": self.model_version,
            "metrics": self.metrics,
            "residual_std": self.residual_std,
        }

        joblib.dump(payload, path)

        return path

    def load(self) -> Path:
        if joblib is None:
            raise RuntimeError(
                "joblib is required to load the demand model"
            )

        path = (
            self.model_dir
            / "business_demand_forecaster.joblib"
        )

        if not path.exists():
            raise FileNotFoundError(
                f"Demand model not found: {path}"
            )

        payload = joblib.load(path)

        self.model = payload["model"]
        self.feature_columns = payload.get(
            "feature_columns",
            self.feature_columns,
        )
        self.metric_name = payload.get(
            "metric_name",
            "demand",
        )
        self.model_name = payload.get(
            "model_name",
            self.model_name,
        )
        self.model_version = payload.get(
            "model_version",
            "1.0",
        )
        self.metrics = payload.get(
            "metrics",
            {},
        )
        self.residual_std = float(
            payload.get(
                "residual_std",
                0.0,
            )
        )

        return path

    def train_from_csv(
        self,
        path: str | Path,
        timestamp_column: str = "timestamp",
        value_column: str = "value",
    ) -> DemandModelMetrics:
        frame = pd.read_csv(path)

        return self.train(
            frame,
            timestamp_column=timestamp_column,
            value_column=value_column,
        )

    def forecast_from_csv(
        self,
        path: str | Path,
        horizon: int = 24,
        timestamp_column: str = "timestamp",
        value_column: str = "value",
    ) -> DemandForecastResult:
        frame = pd.read_csv(path)

        return self.forecast(
            frame,
            horizon=horizon,
            timestamp_column=timestamp_column,
            value_column=value_column,
        )


def generate_demo_demand(
    periods: int = 720,
    frequency: str = "h",
    seed: int = 42,
) -> pd.DataFrame:
    if periods < 30:
        periods = 30

    rng = np.random.default_rng(seed)

    timestamps = pd.date_range(
        end=pd.Timestamp.now().floor("h"),
        periods=periods,
        freq=frequency,
    )

    values: list[float] = []

    for index, timestamp in enumerate(timestamps):
        hour = timestamp.hour
        weekday = timestamp.dayofweek

        daily = (
            120
            + 70
            * np.sin(
                (2 * np.pi * (hour - 7)) / 24
            )
        )

        weekly = (
            35
            if weekday < 5
            else -25
        )

        business_peak = (
            80
            if 9 <= hour <= 17 and weekday < 5
            else 0
        )

        trend = index * 0.035

        noise = rng.normal(
            0,
            12,
        )

        value = max(
            5.0,
            daily
            + weekly
            + business_peak
            + trend
            + noise,
        )

        values.append(value)

    return pd.DataFrame(
        {
            "timestamp": timestamps,
            "value": values,
        }
    )


def forecast_business_demand(
    data: pd.DataFrame,
    horizon: int = 24,
    timestamp_column: str = "timestamp",
    value_column: str = "value",
) -> DemandForecastResult:
    forecaster = BusinessDemandForecaster()

    return forecaster.forecast(
        data,
        horizon=horizon,
        timestamp_column=timestamp_column,
        value_column=value_column,
    )


def train_business_demand_model(
    data: pd.DataFrame,
    timestamp_column: str = "timestamp",
    value_column: str = "value",
) -> DemandModelMetrics:
    forecaster = BusinessDemandForecaster()

    return forecaster.train(
        data,
        timestamp_column=timestamp_column,
        value_column=value_column,
    )


def build_business_demand_dataset(
    periods: int = 720,
    seed: int = 42,
) -> pd.DataFrame:
    return generate_demo_demand(
        periods=periods,
        seed=seed,
    )