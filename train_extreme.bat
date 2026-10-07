@echo off
echo ========================================================
echo   CloudOpt AI -- Extreme Multi-Hour Model Retraining
echo ========================================================
python scripts/deep_train_models.py --mode extreme --epochs 300 %*
