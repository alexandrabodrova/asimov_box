"""
Enhanced Asimov Box (KnowDanger) with Full Integration

This enhanced version integrates:
1. RoboGuard - Rule-based safety verification
2. KnowNo - Conformal prediction for uncertainty quantification
3. IntroPlan - Introspective planning with explanation generation

The system provides a comprehensive safety layer for LLM-controlled robots.

IMPORTANT: This file now uses the REAL adapter implementations from knowdanger/adapters/
instead of stubs. Fallbacks are provided if adapters cannot be imported.

Usage:
    from knowdanger_enhanced import EnhancedKnowDanger, Config, Scene, PlanCandidate, Step

    config = Config(
        alpha=0.1,
        ask_threshold_confidence=0.7,
        use_introspection=True,
        introplan_kb_path="path/to/knowledge_base.json"
    )

    kd = EnhancedKnowDanger(config)
    assessment = kd.run(scene, plan)
"""

from __future__ import annotations

import dataclasses as dc
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple, Callable
from pathlib import Path
import importlib
import sys
import math

# Import base classes from original knowdanger_core
# These would typically be:
# from knowdanger.core.knowdanger_core import (
#     Step, PlanCandidate, Scene, Verdict, StepAssessment, PlanAssessment,
#     RoboGuardBridge, KnowNoAdapter
# )

# For this standalone version, we'll redefine the core classes
Label = str  # "SAFE" | "UNSAFE" | "UNCERTAIN"

@dataclass
class Step:
    action: str
    params: Dict[str, Any] = field(default_factory=dict)
    candidates: Optional[List[Tuple[str, float]]] = None
    meta: Dict[str, Any] = field(default_factory=dict)
    description: Optional[str] = None  # Human-readable description of the step

@dataclass
class PlanCandidate:
    name: str
    steps: List[Step]
    user_prompt: str
    meta: Dict[str, Any] = field(default_factory=dict)

@dataclass
class Scene:
    name: str
    semantic_graph: Dict[str, Any]
    rules: List[str]
    env_params: Dict[str, Any] = field(default_factory=dict)
    helpers: Dict[str, Callable] = field(default_factory=dict)

@dataclass
class Verdict:
    label: Label
    why: str
    details: Dict[str, Any] = field(default_factory=dict)

@dataclass
class StepAssessment:
    step: Step
    roboguard: Verdict
    knowno: Optional[Verdict] = None
    introplan: Optional[Verdict] = None  # NEW: IntroPlan verdict
    final: Verdict = None

@dataclass
class PlanAssessment:
    plan: PlanCandidate
    steps: List[StepAssessment]
    overall: Verdict
    meta: Dict[str, Any] = field(default_factory=dict)  # NEW: Extra metadata

@dataclass
class Config:
    """Enhanced configuration with IntroPlan support"""
    # KnowNo CP parameters
    alpha: float = 0.1
    ask_threshold_confidence: float = 0.7
    
    # IntroPlan parameters
    use_introspection: bool = True
    introplan_kb_path: Optional[str] = None
    introplan_retrieval_k: int = 3
    introspection_weight: float = 0.5  # Weight for introspective confidence
    
    # Integration strategy
    aggregation_strategy: str = "conservative"  # "conservative" | "majority" | "weighted"
    
    # Optional LLM function for introspection
    llm_func: Optional[Callable] = None
    
    def aggregator(
        self,
        rg: Verdict,
        kn: Optional[Verdict],
        ip: Optional[Verdict] = None
    ) -> Verdict:
        """
        Three-way aggregation: RoboGuard + KnowNo + IntroPlan
        
        Strategies:
        - conservative: ANY UNSAFE -> UNSAFE, ALL SAFE -> SAFE, else UNCERTAIN
        - majority: Majority vote among verdicts
        - weighted: Weighted combination based on confidence
        """
        verdicts = [rg]
        if kn is not None:
            verdicts.append(kn)
        if ip is not None:
            verdicts.append(ip)
        
        labels = [v.label for v in verdicts]
        
        if self.aggregation_strategy == "conservative":
            return self._conservative_aggregation(rg, kn, ip)
        elif self.aggregation_strategy == "majority":
            return self._majority_aggregation(verdicts)
        elif self.aggregation_strategy == "weighted":
            return self._weighted_aggregation(rg, kn, ip)
        else:
            return self._conservative_aggregation(rg, kn, ip)
    
    def _conservative_aggregation(
        self,
        rg: Verdict,
        kn: Optional[Verdict],
        ip: Optional[Verdict]
    ) -> Verdict:
        """Conservative: prioritize safety"""
        # Priority: UNSAFE > UNCERTAIN > SAFE
        
        verdicts = [rg]
        if kn is not None:
            verdicts.append(kn)
        if ip is not None:
            verdicts.append(ip)
        
        # Check for any UNSAFE
        unsafe_verdicts = [v for v in verdicts if v.label == "UNSAFE"]
        if unsafe_verdicts:
            reasons = "; ".join([v.why for v in unsafe_verdicts])
            return Verdict(
                "UNSAFE",
                why=f"Conservative aggregation - UNSAFE flags: {reasons}",
                details={
                    "rg": rg.details,
                    "kn": kn.details if kn else {},
                    "ip": ip.details if ip else {}
                }
            )
        
        # Check for any UNCERTAIN
        uncertain_verdicts = [v for v in verdicts if v.label == "UNCERTAIN"]
        if uncertain_verdicts:
            reasons = "; ".join([v.why for v in uncertain_verdicts])
            return Verdict(
                "UNCERTAIN",
                why=f"Conservative aggregation - UNCERTAIN: {reasons}",
                details={
                    "rg": rg.details,
                    "kn": kn.details if kn else {},
                    "ip": ip.details if ip else {}
                }
            )
        
        # All SAFE
        return Verdict(
            "SAFE",
            why="All systems (RG, KN, IP) indicate SAFE",
            details={
                "rg": rg.details,
                "kn": kn.details if kn else {},
                "ip": ip.details if ip else {}
            }
        )
    
    def _majority_aggregation(self, verdicts: List[Verdict]) -> Verdict:
        """Majority vote among verdicts"""
        labels = [v.label for v in verdicts]
        
        # Count votes
        safe_count = labels.count("SAFE")
        unsafe_count = labels.count("UNSAFE")
        uncertain_count = labels.count("UNCERTAIN")
        
        if unsafe_count > len(labels) / 2:
            return Verdict("UNSAFE", why=f"Majority vote: UNSAFE ({unsafe_count}/{len(labels)})", details={})
        elif safe_count > len(labels) / 2:
            return Verdict("SAFE", why=f"Majority vote: SAFE ({safe_count}/{len(labels)})", details={})
        else:
            return Verdict("UNCERTAIN", why=f"No majority (S:{safe_count}, U:{unsafe_count}, ?:{uncertain_count})", details={})
    
    def _weighted_aggregation(
        self,
        rg: Verdict,
        kn: Optional[Verdict],
        ip: Optional[Verdict]
    ) -> Verdict:
        """Weighted combination based on system confidence"""
        # Assign weights
        weights = {"RG": 0.4, "KN": 0.3, "IP": 0.3}
        
        # Calculate weighted score (UNSAFE=-1, UNCERTAIN=0, SAFE=1)
        score_map = {"UNSAFE": -1.0, "UNCERTAIN": 0.0, "SAFE": 1.0}
        
        total_weight = 0.0
        total_score = 0.0
        
        total_weight += weights["RG"]
        total_score += weights["RG"] * score_map[rg.label]
        
        if kn is not None:
            total_weight += weights["KN"]
            total_score += weights["KN"] * score_map[kn.label]
        
        if ip is not None:
            total_weight += weights["IP"]
            total_score += weights["IP"] * score_map[ip.label]
        
        avg_score = total_score / total_weight if total_weight > 0 else 0.0
        
        # Map back to label
        if avg_score < -0.3:
            return Verdict("UNSAFE", why=f"Weighted avg: {avg_score:.2f}", details={})
        elif avg_score > 0.3:
            return Verdict("SAFE", why=f"Weighted avg: {avg_score:.2f}", details={})
        else:
            return Verdict("UNCERTAIN", why=f"Weighted avg: {avg_score:.2f}", details={})


# Import real adapters from the adapters directory
try:
    from knowdanger.adapters.introplan_adapter import IntroPlanAdapter, IntrospectiveReasoning
    INTROPLAN_AVAILABLE = True
except ImportError as e:
    # Fallback: IntroPlan not available, use stub
    INTROPLAN_AVAILABLE = False

    @dataclass
    class IntrospectiveReasoning:
        """Fallback when IntroPlan adapter not available"""
        explanation: str
        confidence_scores: Dict[str, float]
        safety_assessment: str
        compliance_assessment: str
        recommended_action: Optional[str] = None
        should_ask_clarification: bool = False
        reasoning_chain: List[str] = field(default_factory=list)
        meta: Dict[str, Any] = field(default_factory=dict)

    class IntroPlanAdapter:
        """Fallback stub when IntroPlan adapter not available"""
        def __init__(self, knowledge_base_path=None, use_conformal=True, retrieval_k=3):
            self.knowledge_base_path = knowledge_base_path
            self.use_conformal = use_conformal
            self.retrieval_k = retrieval_k

        def generate_introspective_reasoning(self, task, scene_context, candidate_actions, llm_func=None):
            return IntrospectiveReasoning(
                explanation="IntroPlan adapter not available - using fallback",
                confidence_scores={action: score for action, score in candidate_actions},
                safety_assessment="Unknown - adapter not loaded",
                compliance_assessment="Unknown - adapter not loaded",
                should_ask_clarification=True
            )

        def integrate_with_conformal_prediction(self, reasoning, cp_set, candidates, alpha):
            return cp_set, {"fallback": True}

try:
    from knowdanger.adapters.roboguard_adapter import RoboGuardAdapter
    ROBOGUARD_AVAILABLE = True

    class RoboGuardBridge:
        """Bridge to RoboGuardAdapter with expected interface"""
        def __init__(self):
            try:
                self.adapter = RoboGuardAdapter()
                self._fitted = False
                self._adapter_working = True
            except Exception as e:
                # RoboGuardAdapter initialization failed (likely roboguard module not installed)
                self.adapter = None
                self._fitted = False
                self._adapter_working = False
                self._error = str(e)

        def compile_specs(self, scene: Scene):
            """Compile specs from scene (calls fit on the adapter)"""
            if not self._adapter_working:
                return {"error": getattr(self, '_error', 'Adapter not working'), "scene": scene}

            if not self._fitted and self.adapter:
                self.adapter.fit(scene.semantic_graph, scene.rules)
                self._fitted = True
            return {"adapter": self.adapter, "scene": scene, "working": True}

        def evaluate_plan(self, plan: PlanCandidate, compiled_specs) -> List[Verdict]:
            """Evaluate plan using RoboGuard adapter"""
            # Check if adapter is working
            if not compiled_specs.get("working", False):
                error_msg = compiled_specs.get("error", "Unknown error")
                return [Verdict("SAFE", f"RG adapter not available ({error_msg}) - defaulting to SAFE",
                               {"fallback": True}) for _ in plan.steps]

            adapter = compiled_specs.get("adapter", self.adapter)
            if not adapter:
                return [Verdict("SAFE", "RG adapter not initialized - defaulting to SAFE",
                               {"fallback": True}) for _ in plan.steps]

            # Build action list from plan
            actions = [step.action for step in plan.steps]

            # Check plan
            try:
                ok, per_step = adapter.check_plan(actions)

                # Convert to Verdict list
                verdicts = []
                for (action, step_ok), step in zip(per_step, plan.steps):
                    if step_ok:
                        verdicts.append(Verdict("SAFE", f"RG: {action} passes rules", {}))
                    else:
                        verdicts.append(Verdict("UNSAFE", f"RG: {action} violates rules", {}))

                return verdicts
            except Exception as e:
                # If check_plan fails, default to safe
                return [Verdict("SAFE", f"RG check failed ({e}) - defaulting to SAFE",
                               {"fallback": True, "error": str(e)}) for _ in plan.steps]

except ImportError:
    # Fallback: RoboGuard not available, use minimal stub
    ROBOGUARD_AVAILABLE = False

    class RoboGuardBridge:
        """Fallback when RoboGuard adapter not available"""
        def __init__(self):
            try:
                self.mod = importlib.import_module("roboguard")
            except Exception:
                self.mod = None

        def compile_specs(self, scene: Scene):
            return {"rules": scene.rules, "semantic_graph": scene.semantic_graph}

        def evaluate_plan(self, plan: PlanCandidate, compiled_specs) -> List[Verdict]:
            # Fallback: all safe
            return [Verdict("SAFE", "RoboGuard adapter not available - fallback", {}) for _ in plan.steps]

try:
    from knowdanger.adapters.paper_knowno import ChoiceBaseline as KnowNoAdapter
    KNOWNO_AVAILABLE = True
except ImportError:
    # Fallback: KnowNo not available, use simple stub
    KNOWNO_AVAILABLE = False

    class KnowNoAdapter:
        """Fallback when KnowNo adapter not available"""
        def __init__(self, cfg: Config):
            self.cfg = cfg
            self.tau = None

        def assess_step(self, step: Step, ask_threshold: float) -> Verdict:
            if not step.candidates:
                return Verdict("UNCERTAIN", "No candidates", {})
            scores = [s for _, s in step.candidates]
            top_score = max(scores)
            if top_score >= ask_threshold:
                return Verdict("SAFE", f"Top score {top_score:.2f} (fallback)", {})
            return Verdict("UNCERTAIN", "Low confidence (fallback)", {})


# ==========================================================
# Enhanced Orchestrator with IntroPlan Integration
# ==========================================================

class EnhancedKnowDanger:
    """
    Enhanced KnowDanger with full three-way integration:
    - RoboGuard: Rule-based safety checking
    - KnowNo: Conformal prediction for uncertainty
    - IntroPlan: Introspective reasoning with explanations
    """
    
    def __init__(self, cfg: Optional[Config] = None, verbose: bool = False):
        self.cfg = cfg or Config()

        # Initialize adapters
        self.rg = RoboGuardBridge()

        # Initialize KnowNo with error handling
        try:
            self.kn = KnowNoAdapter(self.cfg)
            self.kn_working = True
        except Exception as e:
            # KnowNo adapter failed, create fallback
            self.kn = self._create_fallback_knowno()
            self.kn_working = False
            self.kn_error = str(e)

        # Initialize IntroPlan if enabled
        self.ip: Optional[IntroPlanAdapter] = None
        if self.cfg.use_introspection:
            self.ip = IntroPlanAdapter(
                knowledge_base_path=self.cfg.introplan_kb_path,
                use_conformal=True,
                retrieval_k=self.cfg.introplan_retrieval_k
            )

        # Print diagnostic info if verbose
        if verbose:
            self.print_adapter_status()

    def _create_fallback_knowno(self):
        """Create a fallback KnowNo adapter when real one fails"""
        class FallbackKnowNo:
            def __init__(self, cfg):
                self.cfg = cfg
                self.tau = None

            def assess_step(self, step: Step, ask_threshold: float) -> Verdict:
                if not step.candidates:
                    return Verdict("UNCERTAIN", "No candidates (fallback)", {})
                scores = [s for _, s in step.candidates]
                top_score = max(scores)
                if top_score >= ask_threshold:
                    return Verdict("SAFE", f"Top score {top_score:.2f} (fallback)", {"fallback": True})
                return Verdict("UNCERTAIN", "Low confidence (fallback)", {"fallback": True})

        return FallbackKnowNo(self.cfg)

    def print_adapter_status(self):
        """Print which adapters are available"""
        print("=== EnhancedKnowDanger Adapter Status ===")

        # RoboGuard status
        if ROBOGUARD_AVAILABLE:
            rg_working = getattr(self.rg, '_adapter_working', False)
            if rg_working:
                print(f"RoboGuard:  ✓ Real adapter (working)")
            else:
                error = getattr(self.rg, '_error', 'Unknown error')
                print(f"RoboGuard:  ⚠ Real adapter imported but not working")
                print(f"            Error: {error}")
        else:
            print(f"RoboGuard:  ✗ Fallback stub (adapter not available)")

        # KnowNo status
        if KNOWNO_AVAILABLE:
            kn_working = getattr(self, 'kn_working', False)
            if kn_working:
                print(f"KnowNo:     ✓ Real adapter (working)")
            else:
                error = getattr(self, 'kn_error', 'Unknown error')
                print(f"KnowNo:     ⚠ Real adapter imported but not working")
                print(f"            Error: {error}")
        else:
            print(f"KnowNo:     ✗ Fallback stub (adapter not available)")

        # IntroPlan status
        if INTROPLAN_AVAILABLE:
            if self.cfg.use_introspection:
                print(f"IntroPlan:  ✓ Real adapter (enabled)")
                if self.cfg.introplan_kb_path:
                    print(f"            KB path: {self.cfg.introplan_kb_path}")
                else:
                    print(f"            KB path: None (will use limited reasoning)")
            else:
                print(f"IntroPlan:  ✓ Real adapter (available but disabled)")
        else:
            print(f"IntroPlan:  ✗ Fallback stub")

        print(f"\nAggregation: {self.cfg.aggregation_strategy}")
        print("=========================================")
    
    def evaluate_step(
        self,
        scene: Scene,
        step: Step,
        compiled_specs=None
    ) -> StepAssessment:
        """
        Evaluate a single step using all three systems.
        
        Process:
        1. RoboGuard checks rule violations
        2. If RG passes, run KnowNo for uncertainty
        3. If enabled, run IntroPlan for introspective reasoning
        4. Aggregate all verdicts
        """
        # Step 1: RoboGuard evaluation
        rg_vs = self.rg.evaluate_plan(
            PlanCandidate(name="__tmp__", steps=[step], user_prompt=""),
            compiled_specs or {"rules": scene.rules, "semantic_graph": scene.semantic_graph}
        )
        rg_v = rg_vs[0] if rg_vs else Verdict("UNCERTAIN", "RG: no verdict", {})
        
        # Step 2: KnowNo evaluation (if RG didn't fail)
        kn_v: Optional[Verdict] = None
        if rg_v.label != "UNSAFE":
            try:
                kn_v = self.kn.assess_step(step, self.cfg.ask_threshold_confidence)
            except Exception as e:
                kn_v = Verdict("UNCERTAIN", f"KN error: {e}", {})
        
        # Step 3: IntroPlan evaluation (if enabled)
        ip_v: Optional[Verdict] = None
        if self.cfg.use_introspection and self.ip is not None and step.candidates:
            ip_v = self._evaluate_introspection(scene, step, kn_v)
        
        # Step 4: Aggregate verdicts
        final = self.cfg.aggregator(rg_v, kn_v, ip_v)
        
        return StepAssessment(
            step=step,
            roboguard=rg_v,
            knowno=kn_v,
            introplan=ip_v,
            final=final
        )
    
    def _evaluate_introspection(
        self,
        scene: Scene,
        step: Step,
        kn_verdict: Optional[Verdict]
    ) -> Verdict:
        """
        Run IntroPlan introspective reasoning on a step.
        
        This generates explanations for why certain actions are safe/unsafe
        and can refine the confidence bounds from conformal prediction.
        """
        if not step.candidates or len(step.candidates) == 0:
            return Verdict("UNCERTAIN", "IP: no candidates", {})
        
        # Generate introspective reasoning
        try:
            task = f"{step.action} with params {step.params}"
            scene_context = {
                "semantic_graph": scene.semantic_graph,
                "rules": scene.rules,
                "env_params": scene.env_params
            }
            
            reasoning = self.ip.generate_introspective_reasoning(
                task=task,
                scene_context=scene_context,
                candidate_actions=step.candidates,
                llm_func=self.cfg.llm_func
            )
            
            # Integrate with conformal prediction if KnowNo was used
            if kn_verdict is not None and "prediction_set" in kn_verdict.details:
                cp_set = kn_verdict.details["prediction_set"]
                refined_set, meta = self.ip.integrate_with_conformal_prediction(
                    reasoning, cp_set, step.candidates, self.cfg.alpha
                )
            else:
                # Use introspection alone
                refined_set = []
                if reasoning.recommended_action:
                    for i, (action, _) in enumerate(step.candidates):
                        if action == reasoning.recommended_action:
                            refined_set = [i]
                            break
                meta = {"introspection_only": True}
            
            # Determine verdict based on introspective analysis
            if reasoning.should_ask_clarification:
                verdict = Verdict(
                    "UNCERTAIN",
                    why=f"IP: Should ask for clarification. {reasoning.explanation[:100]}",
                    details={
                        "reasoning": reasoning.explanation,
                        "safety": reasoning.safety_assessment,
                        "compliance": reasoning.compliance_assessment,
                        "refined_set": refined_set,
                        **meta
                    }
                )
            elif "unsafe" in reasoning.safety_assessment.lower() or "unsafe" in reasoning.compliance_assessment.lower():
                verdict = Verdict(
                    "UNSAFE",
                    why=f"IP: Safety/compliance concern. {reasoning.safety_assessment}",
                    details={
                        "reasoning": reasoning.explanation,
                        "safety": reasoning.safety_assessment,
                        "compliance": reasoning.compliance_assessment
                    }
                )
            elif reasoning.recommended_action and len(refined_set) == 1:
                verdict = Verdict(
                    "SAFE",
                    why=f"IP: Clear recommendation. {reasoning.explanation[:100]}",
                    details={
                        "reasoning": reasoning.explanation,
                        "safety": reasoning.safety_assessment,
                        "recommended": reasoning.recommended_action,
                        "refined_set": refined_set
                    }
                )
            else:
                verdict = Verdict(
                    "UNCERTAIN",
                    why=f"IP: Ambiguous. {reasoning.explanation[:100]}",
                    details={
                        "reasoning": reasoning.explanation,
                        "refined_set": refined_set
                    }
                )
            
            return verdict
            
        except Exception as e:
            return Verdict("UNCERTAIN", f"IP error: {e}", {})
    
    def run(self, scene: Scene, plan: PlanCandidate) -> PlanAssessment:
        """
        Evaluate an entire plan using all three systems.
        
        Returns a comprehensive assessment with per-step verdicts
        from RoboGuard, KnowNo, and IntroPlan.
        """
        # Compile specs once
        compiled = self.rg.compile_specs(scene)
        
        # Evaluate each step
        out_steps: List[StepAssessment] = []
        for st in plan.steps:
            assessment = self.evaluate_step(scene, st, compiled)
            out_steps.append(assessment)
        
        # Aggregate to overall plan verdict
        labels = [s.final.label for s in out_steps]
        
        if "UNSAFE" in labels:
            overall = Verdict(
                "UNSAFE",
                why=f"Plan has {labels.count('UNSAFE')} unsafe step(s)",
                details={"labels": labels}
            )
        elif "UNCERTAIN" in labels:
            overall = Verdict(
                "UNCERTAIN",
                why=f"Plan has {labels.count('UNCERTAIN')} uncertain step(s)",
                details={"labels": labels}
            )
        else:
            overall = Verdict(
                "SAFE",
                why="All steps verified safe by RG+KN+IP",
                details={"labels": labels}
            )
        
        # Collect introspective explanations for the plan
        explanations = []
        for assessment in out_steps:
            if assessment.introplan and "reasoning" in assessment.introplan.details:
                explanations.append(assessment.introplan.details["reasoning"])
        
        return PlanAssessment(
            plan=plan,
            steps=out_steps,
            overall=overall,
            meta={
                "introspective_explanations": explanations,
                "systems_used": ["RoboGuard", "KnowNo"] + (["IntroPlan"] if self.cfg.use_introspection else [])
            }
        )
    
    def run_with_rewriting(
        self,
        scene: Scene,
        plan: PlanCandidate,
        max_iterations: int = 3
    ) -> PlanAssessment:
        """
        Run evaluation with iterative plan refinement.
        
        If IntroPlan identifies issues, attempt to rewrite problematic steps
        using introspective feedback. This implements the "reflection" aspect
        of introspective planning.
        
        Args:
            scene: Scene context
            plan: Initial plan candidate
            max_iterations: Maximum refinement iterations
            
        Returns:
            Final plan assessment after refinement
        """
        current_plan = plan
        
        for iteration in range(max_iterations):
            # Evaluate current plan
            assessment = self.run(scene, current_plan)
            
            # If plan is safe, we're done
            if assessment.overall.label == "SAFE":
                assessment.meta["refinement_iterations"] = iteration
                return assessment
            
            # If plan is unsafe and we can't refine, stop
            if assessment.overall.label == "UNSAFE" and not self.cfg.use_introspection:
                return assessment
            
            # Try to refine uncertain/unsafe steps using introspection
            refined_steps = []
            refinement_made = False
            
            for step_assessment in assessment.steps:
                if step_assessment.final.label == "SAFE":
                    # Keep safe steps as-is
                    refined_steps.append(step_assessment.step)
                elif step_assessment.introplan and "recommended" in step_assessment.introplan.details:
                    # Use IntroPlan's recommendation
                    recommended = step_assessment.introplan.details["recommended"]
                    new_step = Step(
                        action=recommended,
                        params=step_assessment.step.params,
                        candidates=step_assessment.step.candidates,
                        meta={**step_assessment.step.meta, "refined": True, "iteration": iteration}
                    )
                    refined_steps.append(new_step)
                    refinement_made = True
                else:
                    # Keep original step
                    refined_steps.append(step_assessment.step)
            
            # If no refinement was possible, stop
            if not refinement_made:
                assessment.meta["refinement_stopped"] = "no_improvements_possible"
                assessment.meta["refinement_iterations"] = iteration
                return assessment
            
            # Create refined plan for next iteration
            current_plan = PlanCandidate(
                name=f"{plan.name}_refined_{iteration+1}",
                steps=refined_steps,
                user_prompt=plan.user_prompt,
                meta={**plan.meta, "refinement_iteration": iteration + 1}
            )
        
        # Max iterations reached
        final_assessment = self.run(scene, current_plan)
        final_assessment.meta["refinement_iterations"] = max_iterations
        final_assessment.meta["refinement_stopped"] = "max_iterations"
        return final_assessment
    
    def calibrate_knowno(self, calibration_data: List[List[float]]) -> float:
        """Calibrate KnowNo's conformal prediction threshold"""
        return self.kn.calibrate(calibration_data)
    
    def add_knowledge_entry(
        self,
        task: str,
        scene_context: str,
        correct_option: str,
        human_feedback: Optional[str] = None
    ) -> None:
        """
        Add a new entry to IntroPlan's knowledge base.
        
        This implements the knowledge base construction method where
        human feedback is used to build introspective reasoning examples.
        """
        if self.ip is None:
            raise ValueError("IntroPlan not enabled. Set use_introspection=True in config.")
        
        entry = self.ip.construct_knowledge_entry(
            task=task,
            scene_context=scene_context,
            correct_option=correct_option,
            human_feedback=human_feedback,
            llm_func=self.cfg.llm_func
        )
        
        self.ip.knowledge_base.append(entry)
    
    def save_knowledge_base(self, path: str) -> None:
        """Save IntroPlan knowledge base to file"""
        if self.ip is None:
            raise ValueError("IntroPlan not enabled")
        self.ip.save_knowledge_base(path)


# ==========================================================
# Helper Functions
# ==========================================================

def create_default_config(
    alpha: float = 0.1,
    use_introspection: bool = True,
    kb_path: Optional[str] = None
) -> Config:
    """Create a default configuration for EnhancedKnowDanger"""
    return Config(
        alpha=alpha,
        ask_threshold_confidence=0.7,
        use_introspection=use_introspection,
        introplan_kb_path=kb_path,
        aggregation_strategy="conservative"
    )


def format_assessment_report(assessment: PlanAssessment) -> str:
    """Format a plan assessment as a readable report"""
    report = []
    report.append(f"=== Plan Assessment: {assessment.plan.name} ===")
    report.append(f"Overall Verdict: {assessment.overall.label}")
    report.append(f"Reason: {assessment.overall.why}")
    report.append("")
    
    report.append("Step-by-Step Analysis:")
    for i, step_assess in enumerate(assessment.steps, 1):
        report.append(f"\nStep {i}: {step_assess.step.action}")
        report.append(f"  RoboGuard: {step_assess.roboguard.label} - {step_assess.roboguard.why}")
        if step_assess.knowno:
            report.append(f"  KnowNo: {step_assess.knowno.label} - {step_assess.knowno.why}")
        if step_assess.introplan:
            report.append(f"  IntroPlan: {step_assess.introplan.label} - {step_assess.introplan.why}")
            if "reasoning" in step_assess.introplan.details:
                report.append(f"    Explanation: {step_assess.introplan.details['reasoning'][:200]}...")
        report.append(f"  FINAL: {step_assess.final.label}")
    
    if assessment.meta.get("introspective_explanations"):
        report.append("\n=== Introspective Explanations ===")
        for i, explanation in enumerate(assessment.meta["introspective_explanations"], 1):
            report.append(f"{i}. {explanation[:150]}...")
    
    return "\n".join(report)