"""
Cost optimizer analyzer module.
Aliases and integrates CloudOptimizer from intelligence module.
"""
from __future__ import annotations

from ..intelligence.optimizer import CloudOptimizer

# Alias for backward compatibility
CostOptimizer = CloudOptimizer

__all__ = ["CostOptimizer", "CloudOptimizer"]
