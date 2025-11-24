"""
Legal Guarddog: AI Chatbot Legal Compliance Layer

Extends Asimov Box for legal compliance testing of chatbots.

Author: Alexandra Bodrova
Course: AI & Law Final Project
Institution: Princeton University
"""

__version__ = "1.0.0"
__author__ = "Alexandra Bodrova"

from .core.legal_guarddog_core import (
    LegalGuarddog,
    Config,
    create_default_config,
    EvaluationResult,
    AttackAttempt,
    ConversationTurn,
    AgentRole,
    ActionType
)

from .policies.legal_policy_engine import (
    LegalPolicyEngine,
    PolicyRule,
    PolicyCheckResult,
    PolicyViolation,
    RiskCategory,
    ViolationSeverity
)

__all__ = [
    # Core
    "LegalGuarddog",
    "Config",
    "create_default_config",
    "EvaluationResult",
    "AttackAttempt",
    "ConversationTurn",
    "AgentRole",
    "ActionType",

    # Policy Engine
    "LegalPolicyEngine",
    "PolicyRule",
    "PolicyCheckResult",
    "PolicyViolation",
    "RiskCategory",
    "ViolationSeverity",
]
