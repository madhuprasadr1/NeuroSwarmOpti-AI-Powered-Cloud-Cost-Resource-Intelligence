"""
Automation package for CloudOpt AI.
Provides action execution, rollback monitoring, and message queue management.
"""
from __future__ import annotations

from .actions import AutomationEngine
from .aws_adapter import AWSAutomationAdapter
from .monitor_rollback import AutoRollbackMonitor
from .queue_manager import CloudQueueManager

__all__ = [
    "AutomationEngine",
    "AWSAutomationAdapter",
    "AutoRollbackMonitor",
    "CloudQueueManager",
]

