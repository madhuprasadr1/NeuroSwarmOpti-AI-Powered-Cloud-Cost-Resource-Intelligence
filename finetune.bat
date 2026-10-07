@echo off
echo ========================================================
echo   CloudOpt AI -- Model Fine-Tuning Engine (Warm-Start)
echo ========================================================
python scripts/deep_train_models.py --finetune --mode deep --epochs 50 %*
