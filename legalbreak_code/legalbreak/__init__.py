"""
LegalBreak: Law-Aware Adversarial Testing for LLM Legal Compliance

A systematic framework for measuring LLM vulnerabilities to legal compliance
violations across dual-use content, copyright infringement, and defamation.
"""

__version__ = "0.1.0"
__author__ = "Alexandra Bodrova"
__email__ = ""

from legalbreak.core.legal_guarddog_core import LegalGuarddog
from legalbreak.policies.legal_policy_engine import (
    LegalPolicyEngine,
    PolicyRule,
    RiskCategory,
    ViolationSeverity
)

__all__ = [
    "LegalGuarddog",
    "LegalPolicyEngine",
    "PolicyRule",
    "RiskCategory",
    "ViolationSeverity",
]
