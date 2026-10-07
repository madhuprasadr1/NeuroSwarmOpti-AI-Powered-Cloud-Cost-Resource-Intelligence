from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd
import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from cba.automation.actions import AutomationEngine
from cba.intelligence.anomaly import CloudAnomalyDetector
from cba.intelligence.explainability import ExplainabilityEngine
from cba.intelligence.optimizer import CloudOptimizer
from cba.intelligence.resource_model import ResourceIntelligenceModel


app = typer.Typer(
    name="optimize",
    help="Analyze cloud resources and generate AI optimization recommendations.",
)

console = Console()

ROOT = Path(__file__).resolve().parents[3]


def serialize(value: Any) -> Any:
    if value is None:
        return None

    if isinstance(value, dict):
        return {str(k): serialize(v) for k, v in value.items()}

    if isinstance(value, (list, tuple)):
        return [serialize(v) for v in value]

    if hasattr(value, "to_dict"):
        try:
            return serialize(value.to_dict())
        except Exception:
            pass

    if hasattr(value, "__dict__"):
        return serialize(vars(value))

    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            pass

    return value


def load_data(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")

    if path.suffix.lower() != ".csv":
        raise ValueError("Only CSV datasets are currently supported.")

    frame = pd.read_csv(path)

    if frame.empty:
        raise ValueError("The supplied dataset is empty.")

    return frame


def normalize_columns(frame: pd.DataFrame) -> pd.DataFrame:
    df = frame.copy()

    aliases = {
        "vm_id": "resource_id",
        "instance_id": "resource_id",
        "resource": "resource_id",
        "id": "resource_id",
        "provider": "cloud_provider",
        "cloud": "cloud_provider",
        "cpu": "cpu_usage",
        "cpu_utilization": "cpu_usage",
        "memory": "memory_usage",
        "memory_utilization": "memory_usage",
        "net_io": "network_traffic",
        "network": "network_traffic",
        "latency": "latency_ms",
        "cost": "monthly_cost",
        "monthly_spend": "monthly_cost",
        "price": "price_per_hour",
    }

    for source, target in aliases.items():
        if target not in df.columns and source in df.columns:
            df[target] = df[source]

    if "resource_id" not in df.columns:
        df["resource_id"] = [
            f"resource-{index + 1:05d}"
            for index in range(len(df))
        ]

    if "cloud_provider" not in df.columns:
        df["cloud_provider"] = "SIMULATION"

    numeric_defaults = {
        "cpu_usage": 50.0,
        "memory_usage": 50.0,
        "network_traffic": 50.0,
        "latency_ms": 80.0,
        "price_per_hour": 0.40,
        "monthly_cost": 288.0,
    }

    for column, default in numeric_defaults.items():
        if column not in df.columns:
            df[column] = default

        df[column] = pd.to_numeric(
            df[column],
            errors="coerce",
        ).fillna(default)

    if (
        "monthly_cost" not in frame.columns
        and "price_per_hour" in df.columns
    ):
        df["monthly_cost"] = (
            df["price_per_hour"] * 24 * 30
        )

    df["cloud_provider"] = (
        df["cloud_provider"]
        .astype(str)
        .str.upper()
    )

    return df


def heuristic_fallback(row: dict[str, Any]) -> dict[str, Any]:
    cpu = float(row.get("cpu_usage", 50) or 50)
    memory = float(row.get("memory_usage", 50) or 50)
    latency = float(row.get("latency_ms", 80) or 80)
    monthly_cost = float(row.get("monthly_cost", 0) or 0)

    if cpu < 20 and memory < 45:
        action = "SCALE_DOWN"
        risk = "LOW"
        savings = monthly_cost * 0.30
        confidence = 0.88
    elif cpu > 85 or memory > 90:
        action = "SCALE_UP"
        risk = "HIGH"
        savings = 0.0
        confidence = 0.90
    elif latency > 180:
        action = "INVESTIGATE"
        risk = "MEDIUM"
        savings = 0.0
        confidence = 0.72
    else:
        action = "NO_ACTION"
        risk = "LOW"
        savings = 0.0
        confidence = 0.80

    return {
        "resource_id": row.get("resource_id"),
        "provider": row.get("cloud_provider"),
        "action": action,
        "risk": risk,
        "confidence": confidence,
        "estimated_savings": savings,
    }


def build_recommendation(
    optimizer: CloudOptimizer,
    row: dict[str, Any],
) -> dict[str, Any]:
    try:
        result = optimizer.optimize(row)
        data = serialize(result)

        if isinstance(data, dict):
            return data

    except Exception:
        pass

    return heuristic_fallback(row)


def add_model_signals(
    frame: pd.DataFrame,
    resource_model: ResourceIntelligenceModel,
    anomaly_detector: CloudAnomalyDetector,
) -> pd.DataFrame:
    df = frame.copy()

    try:
        prediction = resource_model.predict(df)

        if isinstance(prediction, pd.DataFrame):
            for column in prediction.columns:
                values = prediction[column]

                if len(values) == len(df):
                    df[f"model_{column}"] = values.to_numpy()

        elif hasattr(prediction, "__len__"):
            values = list(prediction)

            if len(values) == len(df):
                df["model_action"] = values

    except Exception:
        pass

    try:
        anomaly = anomaly_detector.predict(df)

        if isinstance(anomaly, pd.DataFrame):
            for column in anomaly.columns:
                if len(anomaly[column]) == len(df):
                    df[f"anomaly_{column}"] = anomaly[column].to_numpy()

        elif isinstance(anomaly, tuple):
            if len(anomaly) >= 1:
                scores = list(anomaly[0])

                if len(scores) == len(df):
                    df["anomaly_score"] = scores

            if len(anomaly) >= 2:
                flags = list(anomaly[1])

                if len(flags) == len(df):
                    df["is_anomaly"] = flags

    except Exception:
        pass

    return df


@app.command("run")
def run(
    dataset: Path = typer.Argument(
        ...,
        help="CSV dataset containing cloud resource telemetry.",
    ),
    limit: int = typer.Option(
        500,
        min=1,
        max=100000,
        help="Maximum number of resources to analyze.",
    ),
    provider: str | None = typer.Option(
        None,
        help="Optional cloud provider filter.",
    ),
    execute: bool = typer.Option(
        False,
        help="Execute approved safe actions after analysis.",
    ),
    approve: bool = typer.Option(
        False,
        help="Approve selected automation actions.",
    ),
    output: Path | None = typer.Option(
        None,
        help="Optional JSON output file.",
    ),
) -> None:
    console.print(
        Panel(
            "AI-powered resource optimization",
            title="CloudOpt AI",
        )
    )

    try:
        frame = load_data(dataset)
        frame = normalize_columns(frame)

    except Exception as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(1)

    if provider:
        provider_value = provider.upper()
        frame = frame[
            frame["cloud_provider"].astype(str).str.upper()
            == provider_value
        ]

    frame = frame.head(limit).copy()

    if frame.empty:
        console.print(
            "[yellow]No resources matched the supplied filters.[/yellow]"
        )
        raise typer.Exit(0)

    resource_model = ResourceIntelligenceModel()
    anomaly_detector = CloudAnomalyDetector()
    optimizer = CloudOptimizer()
    explainability = ExplainabilityEngine()

    frame = add_model_signals(
        frame,
        resource_model,
        anomaly_detector,
    )

    recommendations: list[dict[str, Any]] = []

    for row in frame.to_dict("records"):
        recommendation = build_recommendation(
            optimizer,
            row,
        )

        if "resource_id" not in recommendation:
            recommendation["resource_id"] = row.get("resource_id")

        if "provider" not in recommendation:
            recommendation["provider"] = row.get("cloud_provider")

        recommendations.append(recommendation)

    actionable = [
        item
        for item in recommendations
        if str(
            item.get(
                "action",
                item.get("action_type", "NO_ACTION"),
            )
        ).upper()
        not in {
            "NO_ACTION",
            "NONE",
            "HOLD",
        }
    ]

    table = Table(title="AI Optimization Recommendations")

    table.add_column("Resource")
    table.add_column("Provider")
    table.add_column("Action")
    table.add_column("Confidence")
    table.add_column("Risk")
    table.add_column("Savings")

    for item in actionable[:50]:
        resource = str(
            item.get(
                "resource_id",
                item.get("resource", "unknown"),
            )
        )

        cloud = str(
            item.get(
                "provider",
                item.get("cloud_provider", "unknown"),
            )
        )

        action = str(
            item.get(
                "action",
                item.get("action_type", "unknown"),
            )
        )

        confidence = item.get(
            "confidence",
            item.get("confidence_score", 0),
        )

        risk = str(
            item.get(
                "risk",
                item.get("risk_level", "UNKNOWN"),
            )
        )

        savings = item.get(
            "estimated_savings",
            item.get("monthly_savings", 0),
        )

        try:
            confidence_text = f"{float(confidence) * 100:.1f}%"
        except Exception:
            confidence_text = str(confidence)

        try:
            savings_text = f"${float(savings):,.2f}"
        except Exception:
            savings_text = str(savings)

        table.add_row(
            resource,
            cloud,
            action,
            confidence_text,
            risk,
            savings_text,
        )

    console.print(table)

    total_savings = 0.0

    for item in actionable:
        try:
            total_savings += float(
                item.get(
                    "estimated_savings",
                    item.get("monthly_savings", 0),
                )
                or 0
            )
        except Exception:
            pass

    console.print(
        Panel(
            f"Resources analyzed: {len(frame):,}\n"
            f"Actionable recommendations: {len(actionable):,}\n"
            f"Estimated monthly savings: ${total_savings:,.2f}",
            title="Optimization Summary",
        )
    )

    if actionable:
        selected = actionable[:10]

        explanation_rows = []

        for item in selected:
            try:
                resource_id = item.get("resource_id")

                source_row = frame[
                    frame["resource_id"].astype(str)
                    == str(resource_id)
                ]

                if not source_row.empty:
                    row_data = source_row.iloc[0].to_dict()

                    if hasattr(
                        explainability,
                        "explain_optimization",
                    ):
                        explanation = explainability.explain_optimization(
                            row_data,
                            item,
                        )
                        explanation_rows.append(
                            serialize(explanation)
                        )
            except Exception:
                continue

        if explanation_rows:
            console.print(
                Panel(
                    json.dumps(
                        explanation_rows[:5],
                        indent=2,
                        default=str,
                    ),
                    title="Explainable AI",
                )
            )

    execution_results = []

    if execute:
        if not approve:
            console.print(
                "[yellow]Execution requested but approval was not supplied. "
                "No actions were executed.[/yellow]"
            )
        else:
            automation = AutomationEngine(
                dry_run=True,
            )

            for recommendation in actionable:
                try:
                    result = automation.execute(
                        recommendation,
                        approved=True,
                    )

                    execution_results.append(
                        serialize(result)
                    )

                except Exception as exc:
                    execution_results.append(
                        {
                            "resource_id": recommendation.get(
                                "resource_id"
                            ),
                            "success": False,
                            "status": "FAILED",
                            "message": str(exc),
                        }
                    )

            console.print(
                Panel(
                    json.dumps(
                        execution_results,
                        indent=2,
                        default=str,
                    ),
                    title="Automation Results",
                )
            )

    result = {
        "dataset": str(dataset),
        "resources_analyzed": len(frame),
        "provider_filter": provider,
        "recommendations": recommendations,
        "actionable_recommendations": actionable,
        "estimated_monthly_savings": total_savings,
        "execution_results": execution_results,
    }

    if output:
        output.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        output.write_text(
            json.dumps(
                result,
                indent=2,
                default=str,
            ),
            encoding="utf-8",
        )

        console.print(
            f"[green]Saved optimization report:[/green] {output}"
        )


@app.command("rank")
def rank(
    dataset: Path = typer.Argument(
        ...,
        help="CSV dataset to rank.",
    ),
    limit: int = typer.Option(
        20,
        min=1,
        max=1000,
        help="Number of top opportunities.",
    ),
) -> None:
    try:
        frame = normalize_columns(
            load_data(dataset)
        )
    except Exception as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(1)

    frame["underutilization_score"] = (
        (100 - frame["cpu_usage"]).clip(0, 100) * 0.55
        + (100 - frame["memory_usage"]).clip(0, 100) * 0.45
    )

    frame["optimization_score"] = (
        frame["underutilization_score"] * 0.60
        + frame["monthly_cost"].rank(pct=True) * 100 * 0.40
    )

    ranked = frame.sort_values(
        "optimization_score",
        ascending=False,
    ).head(limit)

    table = Table(
        title="Highest-Value Optimization Opportunities"
    )

    table.add_column("Resource")
    table.add_column("Provider")
    table.add_column("CPU")
    table.add_column("Memory")
    table.add_column("Monthly Cost")
    table.add_column("Score")

    for row in ranked.to_dict("records"):
        table.add_row(
            str(row["resource_id"]),
            str(row["cloud_provider"]),
            f"{float(row['cpu_usage']):.1f}%",
            f"{float(row['memory_usage']):.1f}%",
            f"${float(row['monthly_cost']):,.2f}",
            f"{float(row['optimization_score']):.1f}",
        )

    console.print(table)


@app.command("explain")
def explain(
    dataset: Path = typer.Argument(
        ...,
        help="CSV dataset to explain.",
    ),
    resource_id: str = typer.Argument(
        ...,
        help="Resource identifier.",
    ),
) -> None:
    try:
        frame = normalize_columns(
            load_data(dataset)
        )
    except Exception as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(1)

    selected = frame[
        frame["resource_id"].astype(str)
        == str(resource_id)
    ]

    if selected.empty:
        console.print(
            f"[red]Resource not found: {resource_id}[/red]"
        )
        raise typer.Exit(1)

    row = selected.iloc[0].to_dict()

    optimizer = CloudOptimizer()
    explainability = ExplainabilityEngine()

    recommendation = build_recommendation(
        optimizer,
        row,
    )

    explanation = None

    try:
        if hasattr(
            explainability,
            "explain_optimization",
        ):
            explanation = explainability.explain_optimization(
                row,
                recommendation,
            )
    except Exception:
        explanation = None

    console.print(
        Panel(
            json.dumps(
                {
                    "resource": row,
                    "recommendation": serialize(
                        recommendation
                    ),
                    "explanation": serialize(
                        explanation
                    ),
                },
                indent=2,
                default=str,
            ),
            title=f"Explainability: {resource_id}",
        )
    )


@app.command("dry-run")
def dry_run(
    dataset: Path = typer.Argument(
        ...,
        help="CSV dataset.",
    ),
    limit: int = typer.Option(
        10,
        min=1,
        max=100,
        help="Maximum actions to preview.",
    ),
) -> None:
    try:
        frame = normalize_columns(
            load_data(dataset)
        )
    except Exception as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(1)

    optimizer = CloudOptimizer()
    automation = AutomationEngine(
        dry_run=True,
    )

    results = []

    for row in frame.head(limit).to_dict("records"):
        recommendation = build_recommendation(
            optimizer,
            row,
        )

        action = str(
            recommendation.get(
                "action",
                recommendation.get(
                    "action_type",
                    "NO_ACTION",
                ),
            )
        ).upper()

        if action in {
            "NO_ACTION",
            "NONE",
            "HOLD",
        }:
            continue

        try:
            result = automation.execute(
                recommendation,
                approved=True,
            )
            results.append(
                serialize(result)
            )
        except Exception as exc:
            results.append(
                {
                    "resource_id": row.get(
                        "resource_id"
                    ),
                    "success": False,
                    "status": "FAILED",
                    "message": str(exc),
                }
            )

    console.print(
        Panel(
            json.dumps(
                results,
                indent=2,
                default=str,
            ),
            title="Automation Dry Run",
        )
    )


if __name__ == "__main__":
    app()
