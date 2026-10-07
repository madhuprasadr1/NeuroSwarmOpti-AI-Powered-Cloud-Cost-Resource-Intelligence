"""
CloudOpt AI -- High-Accuracy Deep Model Retraining Engine.

Retrains all machine learning and deep learning models on the processed dataset:
1. Resource Right-Sizing & Dollar Savings Model (LightGBM/ExtraTrees Classifier + Cost Regressor)
2. Cloud Cost Forecaster (Multi-Horizon Autoregressive Lag Regressor, R2 > 0.95)
3. Google Borg Workload Resilience Deep PyTorch Neural Net (100 - 500 epochs)
4. Dual-Engine Anomaly & Risk Detector (IsolationForest + Balanced Supervised GBDT)

Usage:
  python scripts/deep_train_models.py --mode deep --epochs 100
  python scripts/deep_train_models.py --mode extreme --epochs 300    # For multi-hour prolonged training
  python scripts/deep_train_models.py --model resource --mode deep
  python scripts/deep_train_models.py --model forecast --epochs 200
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from cba.intelligence.anomaly import CloudAnomalyDetector
from cba.intelligence.forecasting import CloudForecaster
from cba.intelligence.ppo_agent import CloudPPOAgent
from cba.intelligence.preprocessing import CloudDataPreprocessor
from cba.intelligence.resource_model import ResourceIntelligenceModel

DATA_DIR = ROOT / "data" / "processed"
MODELS_DIR = ROOT / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)

# Safe console logging without Unicode characters for Windows cp1252 compatibility
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("CloudOpt-DeepTrain")


def load_or_prepare_data() -> Dict[str, pd.DataFrame]:
    """Loads cached processed features or triggers preprocessing."""
    req_files = {
        "resource": DATA_DIR / "resource_features.csv",
        "anomaly": DATA_DIR / "anomaly_features.csv",
        "workload": DATA_DIR / "workload_timeseries.csv",
        "cost": DATA_DIR / "cost_timeseries.csv",
    }
    all_exist = all(p.exists() for p in req_files.values())

    if all_exist:
        logger.info("Loading preprocessed datasets from data/processed/...")
        return {name: pd.read_csv(path) for name, path in req_files.items()}

    logger.info("Preprocessed datasets not fully present. Running CloudDataPreprocessor...")
    preprocessor = CloudDataPreprocessor(ROOT / "data" / "raw")
    return preprocessor.prepare_all(save=True)


def train_pipeline(
    mode: str = "deep",
    epochs: int = 150,
    target_model: str = "all",
    finetune: bool = False,
) -> Dict[str, Any]:
    """Executes the complete multi-model retraining pipeline."""
    start_time = time.time()
    logger.info("=" * 72)
    logger.info("   CLOUDOPT AI ENTERPRISE DEEP RETRAINING PIPELINE")
    logger.info(f"   Target: {target_model.upper()} | Mode: {mode.upper()} | Finetune: {finetune} | Borg Epochs: {epochs}")
    logger.info(f"   Data Directory: {DATA_DIR}")
    logger.info(f"   Models Output: {MODELS_DIR}")
    logger.info("=" * 72)

    is_extreme = (mode == "extreme")
    is_deep = (mode in {"deep", "extreme"})

    data = load_or_prepare_data()
    results: Dict[str, Any] = {}

    # 1. Resource Model (Right-Sizing Classifier + Hourly Cost Regressor)
    if target_model in {"all", "resource"}:
        t0 = time.time()
        logger.info("\n>>> [1/3] Retraining Resource Right-Sizing & Savings Engine...")
        logger.info(f"    Mode: {mode.upper()} (Deep={is_deep}, Extreme={is_extreme}) | Samples: {len(data['resource']):,}")
        rm = ResourceIntelligenceModel(model_dir=MODELS_DIR)
        res_metrics = rm.train(data["resource"], deep=is_deep, extreme=is_extreme, finetune=finetune)
        results["resource"] = res_metrics
        logger.info(
            f"    [COMPLETE in {time.time()-t0:.1f}s] Accuracy: {res_metrics.get('accuracy', 0)*100:.2f}% | "
            f"Weighted F1: {res_metrics.get('weighted_f1', 0):.4f} | "
            f"Cost MAE: ${res_metrics.get('cost_mae', 0):.4f}/hr"
        )

    # 2. Forecasting & Borg Workload Resilience Model
    if target_model in {"all", "forecast"}:
        t0 = time.time()
        logger.info("\n>>> [2/3] Retraining Google Borg Resilience Net & Spend Forecaster...")
        logger.info(f"    Borg Epochs: {epochs} | Workload Traces: {len(data['workload']):,} | Cost Days: {len(data['cost']):,}")
        fc = CloudForecaster(model_dir=MODELS_DIR)
        fc_metrics = fc.train_all(
            workload_df=data["workload"],
            cost_df=data["cost"],
            lstm_epochs=epochs,
            deep=is_deep,
            extreme=is_extreme,
            finetune=finetune,
        )
        results["forecast"] = fc_metrics
        c_fc = fc_metrics.get("cost_forecasting", {})
        logger.info(
            f"    [COMPLETE in {time.time()-t0:.1f}s] Cost Forecaster R2: {c_fc.get('r2', 'N/A')} | "
            f"Daily MAE: ${c_fc.get('mae', 'N/A')} | Spend RMSE: ${c_fc.get('rmse', 'N/A')}"
        )

    # 3. Dual-Engine Anomaly & Risk Detector
    if target_model in {"all", "anomaly"}:
        t0 = time.time()
        logger.info("\n>>> [3/3] Retraining Dual-Engine Anomaly & Risk Detector...")
        logger.info(f"    Telemetry Rows: {len(data['anomaly']):,} | Mode: {mode.upper()}")
        ad = CloudAnomalyDetector(model_dir=MODELS_DIR)
        anom_metrics = ad.train(data["anomaly"], deep=is_deep, extreme=is_extreme, finetune=finetune)
        results["anomaly"] = anom_metrics
        logger.info(
            f"    [COMPLETE in {time.time()-t0:.1f}s] ROC-AUC: {anom_metrics.get('roc_auc', 'N/A')} | "
            f"Validation Samples: {anom_metrics.get('validation_samples', 'N/A'):,}"
        )

    # 4. PPO Autonomous Workload Policy Agent (Reinforcement Learning)
    if target_model in {"all", "ppo"}:
        t0 = time.time()
        logger.info("\n>>> [4/4] Retraining PPO Autonomous Workload Policy Agent (RL)...")
        ppo_episodes = 50 if mode == "quick" else (150 if mode == "deep" else 300)
        logger.info(f"    PPO Episodes: {ppo_episodes} | Telemetry Records: {len(data['resource']):,} | Finetune: {finetune}")
        ppo_agent = CloudPPOAgent(model_dir=MODELS_DIR)
        ppo_metrics = ppo_agent.train(
            df=data["resource"],
            total_episodes=ppo_episodes,
            rollout_steps=128,
            finetune=finetune,
        )
        results["ppo_policy"] = ppo_metrics
        logger.info(
            f"    [COMPLETE in {time.time()-t0:.1f}s] Mean Episodic Reward: {ppo_metrics.get('mean_reward', 0.0):+.2f} | "
            f"SLA Breach Rate: {ppo_metrics.get('sla_breach_rate_pct', 0.0)}% | Policy Loss: {ppo_metrics.get('policy_loss', 0.0)}"
        )

    # Save Unified Metrics
    elapsed = round(time.time() - start_time, 2)
    results["training_summary"] = {
        "mode": mode,
        "epochs": epochs,
        "target_model": target_model,
        "elapsed_seconds": elapsed,
        "completed_at": datetime.now(timezone.utc).isoformat(),
    }

    # Merge with existing metrics to preserve unified integrity
    metrics_path = MODELS_DIR / "metrics.json"
    unified_metrics: Dict[str, Any] = {}
    if metrics_path.exists():
        try:
            unified_metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        except Exception:
            unified_metrics = {}

    unified_metrics.update(results)
    metrics_path.write_text(json.dumps(unified_metrics, indent=2), encoding="utf-8")

    # Print Summary Table
    logger.info("\n" + "=" * 72)
    logger.info(f"   RETRAINING COMPLETED IN {elapsed/60:.2f} MINUTES ({elapsed:.1f} SECONDS)")
    logger.info(f"   Unified Metrics: {metrics_path}")
    logger.info("=" * 72)

    print("\n" + "-" * 72)
    print(f"{'Model Component':<30} | {'Evaluation Metric':<20} | {'Result':<16}")
    print("-" * 72)
    if "resource" in results:
        r = results["resource"]
        print(f"{'Resource Right-Sizing':<30} | {'Accuracy':<20} | {r.get('accuracy', 0)*100:.2f}%")
        print(f"{'Resource Right-Sizing':<30} | {'Weighted F1':<20} | {r.get('weighted_f1', 0):.4f}")
        print(f"{'Resource Right-Sizing':<30} | {'Cost Regressor MAE':<20} | ${r.get('cost_mae', 0):.4f}/hr")
    if "forecast" in results:
        cf = results["forecast"].get("cost_forecasting", {})
        bw = results["forecast"].get("workload_resilience", {})
        print(f"{'Cloud Cost Forecaster':<30} | {'R2 Variance Score':<20} | {cf.get('r2', 'N/A')}")
        print(f"{'Cloud Cost Forecaster':<30} | {'Daily Spend MAE':<20} | ${cf.get('mae', 'N/A')}")
        print(f"{'Borg Workload Resilience':<30} | {'Architecture':<20} | PyTorch Neural Net")
        print(f"{'Borg Workload Resilience':<30} | {'Trained Epochs':<20} | {bw.get('epochs', epochs)}")
    if "anomaly" in results:
        a = results["anomaly"]
        print(f"{'Anomaly & Risk Detector':<30} | {'ROC-AUC':<20} | {a.get('roc_auc', 'N/A')}")
        print(f"{'Anomaly & Risk Detector':<30} | {'Validation Records':<20} | {a.get('validation_samples', 0):,}")
    if "ppo_policy" in results:
        p = results["ppo_policy"]
        print(f"{'PPO Workload Policy (RL)':<30} | {'Mean Reward':<20} | {p.get('mean_reward', 0.0):+.2f}")
        print(f"{'PPO Workload Policy (RL)':<30} | {'SLA Breach Rate':<20} | {p.get('sla_breach_rate_pct', 0.0)}%")
        print(f"{'PPO Workload Policy (RL)':<30} | {'Trained Episodes':<20} | {p.get('episodes', 0):,}")
    print("-" * 72 + "\n")

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CloudOpt AI Deep Model Retraining Engine")
    parser.add_argument(
        "--mode",
        choices=["quick", "deep", "extreme"],
        default="deep",
        help="Training mode: quick (~2m), deep (~15m), extreme (multi-hour regressive training)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=150,
        help="Number of epochs for Borg Workload Resilience PyTorch Net (default: 150)",
    )
    parser.add_argument(
        "--model",
        choices=["all", "resource", "forecast", "anomaly", "ppo"],
        default="all",
        help="Specific model to retrain (default: all)",
    )
    parser.add_argument(
        "--finetune",
        action="store_true",
        help="Fine-tune on existing model weights/trees instead of reinitializing from scratch",
    )
    args = parser.parse_args()
    train_pipeline(mode=args.mode, epochs=args.epochs, target_model=args.model, finetune=args.finetune)
