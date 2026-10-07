"""
Cloud Billing Automation Tool

A DevOps-centric cloud cost governance and automation project built for 
Cloud Engineers and DevOps Engineers to gain visibility, control, and 
automation over cloud billing.
"""

__version__ = "0.6.0"
__author__ = "H A R S H H A A"
__email__ = "contact@example.com"

# The package root stays lightweight. Live-provider collectors and credential
# managers are optional integrations and should not make the local CSV/ML
# product emit warnings or fail at import time.
__all__ = ["__version__", "__author__", "__email__"]
