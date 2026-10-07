@echo off
echo ========================================================
echo   CloudOpt AI -- Deep Model Retraining Engine
echo ========================================================
python scripts/deep_train_models.py --mode deep --epochs 100 %*
