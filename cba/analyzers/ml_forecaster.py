"""
ML budget and cost forecaster analyzer module.
Aliases and integrates CloudForecaster from intelligence module.
"""
from __future__ import annotations

from ..intelligence.forecasting import CloudForecaster

# Alias for backward compatibility
BudgetForecaster = CloudForecaster

__all__ = ["BudgetForecaster", "CloudForecaster"]
