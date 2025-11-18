"""
Integration Utilities for KnowDanger/Asimov Box

This module provides helper functions and utilities for integrating
RoboGuard, KnowNo, and IntroPlan systems seamlessly.

Functions:
- Format conversion between different system formats
- Logging and debugging helpers
- Calibration utilities
- Knowledge base management
- Evaluation metrics
"""

from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field
from pathlib import Path
import json
import csv
from datetime import datetime


@dataclass
class IntegrationMetrics:
    """Metrics for evaluating integrated system performance"""
    total_steps: int = 0
    safe_steps: int = 0
    unsafe_steps: int = 0
    uncertain_steps: int = 0
    
    # System-specific metrics
    roboguard_blocks: int = 0
    knowno_uncertainties: int = 0
    introplan_clarifications: int = 0
    
    # Refinement metrics
    refinement_iterations: int = 0
    successful_refinements: int = 0
    
    # Timing
    avg_evaluation_time_ms: float = 0.0
    
    def success_rate(self) -> float:
        """Calculate success rate (safe / total)"""
        return self.safe_steps / self.total_steps if self.total_steps > 0 else 0.0
    
    def help_rate(self) -> float:
        """Calculate help request rate (uncertain / total)"""
        return self.uncertain_steps / self.total_steps if self.total_steps > 0 else 0.0
    
    def safety_violation_rate(self) -> float:
        """Calculate safety violation rate (unsafe / total)"""
        return self.unsafe_steps / self.total_steps if self.total_steps > 0 else 0.0


class FormatConverter:
    """
    Convert between different format representations used by
    RoboGuard, KnowNo, and IntroPlan.
    """
    
    @staticmethod
    def robopair_to_knowdanger_step(robopair_action: Dict[str, Any]) -> 'Step':
        """
        Convert RoboPAIR action format to KnowDanger Step.
        
        RoboPAIR format example:
        {
            "action": "pick",
            "object": "bottle",
            "location": "table1",
            "llm_logprobs": [...],
            "alternatives": [...]
        }
        """
        from knowdanger_enhanced import Step
        
        action = robopair_action.get("action", "unknown")
        params = {
            k: v for k, v in robopair_action.items()
            if k not in ["action", "llm_logprobs", "alternatives"]
        }
        
        # Convert alternatives to candidates
        candidates = []
        if "alternatives" in robopair_action:
            for alt in robopair_action["alternatives"]:
                candidates.append((
                    alt.get("action", action),
                    alt.get("score", 0.5)
                ))
        elif "llm_logprobs" in robopair_action:
            # Use log probs as scores
            logprobs = robopair_action["llm_logprobs"]
            for i, logprob in enumerate(logprobs[:5]):  # Top 5
                candidates.append((f"{action}_variant_{i}", logprob))
        
        return Step(
            action=action,
            params=params,
            candidates=candidates if candidates else None,
            meta={"source": "robopair", "original": robopair_action}
        )
    
    @staticmethod
    def knowno_prediction_to_candidates(
        options: List[str],
        logits: List[float]
    ) -> List[Tuple[str, float]]:
        """
        Convert KnowNo prediction format to candidate list.
        
        Args:
            options: List of action options
            logits: Corresponding logits or probabilities
            
        Returns:
            List of (action, score) tuples
        """
        import math
        
        # Softmax to convert logits to probabilities if needed
        if any(l < 0 or l > 1 for l in logits):
            exp_logits = [math.exp(l) for l in logits]
            total = sum(exp_logits)
            probs = [e / total for e in exp_logits]
        else:
            probs = logits
        
        return list(zip(options, probs))
    
    @staticmethod
    def introplan_to_knowledge_format(
        task: str,
        options: List[str],
        correct_idx: int,
        explanation: str
    ) -> Dict[str, Any]:
        """
        Format data for IntroPlan knowledge base.
        
        Returns:
            Dict suitable for knowledge base entry
        """
        return {
            "task": task,
            "options": options,
            "correct_option": options[correct_idx] if correct_idx < len(options) else "",
            "reasoning": explanation,
            "timestamp": datetime.now().isoformat()
        }
    
    @staticmethod
    def roboguard_rules_to_scene(
        rules: List[str],
        semantic_graph: Optional[Dict[str, Any]] = None
    ) -> 'Scene':
        """
        Convert RoboGuard rules format to KnowDanger Scene.
        """
        from knowdanger_enhanced import Scene
        
        return Scene(
            name="converted_scene",
            semantic_graph=semantic_graph or {},
            rules=rules,
            env_params={"source": "roboguard"}
        )


class CalibrationHelper:
    """
    Helper for calibrating conformal prediction in KnowNo.
    """
    
    @staticmethod
    def load_calibration_data(path: str) -> List[List[float]]:
        """
        Load calibration data from CSV or JSON.
        
        Expected format:
        - CSV: each row is a calibration example with scores
        - JSON: list of lists of scores
        """
        p = Path(path)
        
        if p.suffix == ".json":
            with open(p, 'r') as f:
                return json.load(f)
        
        elif p.suffix == ".csv":
            data = []
            with open(p, 'r') as f:
                reader = csv.reader(f)
                for row in reader:
                    try:
                        data.append([float(x) for x in row])
                    except ValueError:
                        continue
            return data
        
        else:
            raise ValueError(f"Unsupported calibration data format: {p.suffix}")
    
    @staticmethod
    def generate_synthetic_calibration_data(
        n_examples: int = 100,
        n_options: int = 5,
        alpha: float = 0.1
    ) -> List[List[float]]:
        """
        Generate synthetic calibration data for testing.
        
        Args:
            n_examples: Number of calibration examples
            n_options: Number of options per example
            alpha: Target miscoverage rate
            
        Returns:
            List of score lists
        """
        import random
        
        data = []
        for _ in range(n_examples):
            # Generate scores with one dominant option
            scores = [random.uniform(0.0, 0.3) for _ in range(n_options)]
            # Make one score high
            correct_idx = random.randint(0, n_options - 1)
            scores[correct_idx] = random.uniform(0.7, 1.0)
            
            # Normalize to sum to 1
            total = sum(scores)
            scores = [s / total for s in scores]
            
            data.append(scores)
        
        return data
    
    @staticmethod
    def compute_coverage(
        predictions: List[List[int]],
        ground_truth: List[int]
    ) -> float:
        """
        Compute empirical coverage of prediction sets.
        
        Args:
            predictions: List of prediction sets (lists of indices)
            ground_truth: List of correct indices
            
        Returns:
            Coverage rate (fraction where correct answer is in prediction set)
        """
        if len(predictions) != len(ground_truth):
            raise ValueError("Predictions and ground truth must have same length")
        
        covered = sum(
            1 for pred_set, gt in zip(predictions, ground_truth)
            if gt in pred_set
        )
        
        return covered / len(ground_truth)
    
    @staticmethod
    def compute_set_sizes(predictions: List[List[int]]) -> Dict[str, float]:
        """
        Compute statistics about prediction set sizes.
        
        Returns:
            Dict with mean, median, min, max set sizes
        """
        sizes = [len(pred_set) for pred_set in predictions]
        
        if not sizes:
            return {"mean": 0, "median": 0, "min": 0, "max": 0}
        
        sizes_sorted = sorted(sizes)
        n = len(sizes)
        
        return {
            "mean": sum(sizes) / n,
            "median": sizes_sorted[n // 2],
            "min": min(sizes),
            "max": max(sizes),
            "singleton_rate": sum(1 for s in sizes if s == 1) / n
        }


class LoggingHelper:
    """
    Logging utilities for tracking system behavior.
    """
    
    def __init__(self, log_dir: str = "logs"):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(exist_ok=True)
        
        self.current_run = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.run_dir = self.log_dir / self.current_run
        self.run_dir.mkdir(exist_ok=True)
        
        # Initialize log files
        self.step_log_path = self.run_dir / "step_log.csv"
        self.plan_log_path = self.run_dir / "plan_log.csv"
        self.metrics_path = self.run_dir / "metrics.json"
        
        self._init_csv_logs()
    
    def _init_csv_logs(self):
        """Initialize CSV log files with headers"""
        # Step log
        with open(self.step_log_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                "timestamp", "plan_id", "step_idx", "action",
                "rg_verdict", "kn_verdict", "ip_verdict", "final_verdict",
                "rg_why", "kn_why", "ip_why"
            ])
        
        # Plan log
        with open(self.plan_log_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                "timestamp", "plan_id", "plan_name", "n_steps",
                "overall_verdict", "n_safe", "n_unsafe", "n_uncertain"
            ])
    
    def log_step_assessment(
        self,
        plan_id: str,
        step_idx: int,
        assessment: 'StepAssessment'
    ):
        """Log a single step assessment"""
        with open(self.step_log_path, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                datetime.now().isoformat(),
                plan_id,
                step_idx,
                assessment.step.action,
                assessment.roboguard.label,
                assessment.knowno.label if assessment.knowno else "N/A",
                assessment.introplan.label if assessment.introplan else "N/A",
                assessment.final.label,
                assessment.roboguard.why,
                assessment.knowno.why if assessment.knowno else "",
                assessment.introplan.why if assessment.introplan else ""
            ])
    
    def log_plan_assessment(
        self,
        plan_id: str,
        assessment: 'PlanAssessment'
    ):
        """Log a complete plan assessment"""
        labels = [s.final.label for s in assessment.steps]
        
        with open(self.plan_log_path, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                datetime.now().isoformat(),
                plan_id,
                assessment.plan.name,
                len(assessment.steps),
                assessment.overall.label,
                labels.count("SAFE"),
                labels.count("UNSAFE"),
                labels.count("UNCERTAIN")
            ])
        
        # Log individual steps
        for i, step_assess in enumerate(assessment.steps):
            self.log_step_assessment(plan_id, i, step_assess)
    
    def save_metrics(self, metrics: IntegrationMetrics):
        """Save metrics to JSON"""
        with open(self.metrics_path, 'w') as f:
            json.dump({
                "total_steps": metrics.total_steps,
                "safe_steps": metrics.safe_steps,
                "unsafe_steps": metrics.unsafe_steps,
                "uncertain_steps": metrics.uncertain_steps,
                "success_rate": metrics.success_rate(),
                "help_rate": metrics.help_rate(),
                "safety_violation_rate": metrics.safety_violation_rate(),
                "roboguard_blocks": metrics.roboguard_blocks,
                "knowno_uncertainties": metrics.knowno_uncertainties,
                "introplan_clarifications": metrics.introplan_clarifications,
                "refinement_iterations": metrics.refinement_iterations,
                "successful_refinements": metrics.successful_refinements,
                "avg_evaluation_time_ms": metrics.avg_evaluation_time_ms
            }, f, indent=2)


class KnowledgeBaseManager:
    """
    Manage IntroPlan knowledge base lifecycle.
    """
    
    def __init__(self, kb_path: str):
        self.kb_path = Path(kb_path)
        self.entries: List[Dict[str, Any]] = []
        
        if self.kb_path.exists():
            self.load()
    
    def load(self):
        """Load knowledge base from file"""
        with open(self.kb_path, 'r') as f:
            self.entries = json.load(f)
    
    def save(self):
        """Save knowledge base to file"""
        with open(self.kb_path, 'w') as f:
            json.dump(self.entries, f, indent=2)
    
    def add_entry(
        self,
        task: str,
        scene: str,
        correct_option: str,
        reasoning: str,
        safety: List[str],
        meta: Optional[Dict[str, Any]] = None
    ):
        """Add a new entry to knowledge base"""
        entry = {
            "task": task,
            "scene": scene,
            "correct_option": correct_option,
            "reasoning": reasoning,
            "safety": safety,
            "meta": meta or {},
            "timestamp": datetime.now().isoformat()
        }
        self.entries.append(entry)
    
    def add_from_human_feedback(
        self,
        assessment: 'PlanAssessment',
        human_corrections: Dict[int, str]
    ):
        """
        Add entries from human feedback on an assessment.
        
        Args:
            assessment: Plan assessment that received human feedback
            human_corrections: Dict mapping step_idx -> correct action
        """
        for step_idx, correct_action in human_corrections.items():
            if step_idx >= len(assessment.steps):
                continue
            
            step_assess = assessment.steps[step_idx]
            
            # Extract reasoning from introspection if available
            reasoning = ""
            if step_assess.introplan and "reasoning" in step_assess.introplan.details:
                reasoning = step_assess.introplan.details["reasoning"]
            else:
                reasoning = f"Human correction: {correct_action}"
            
            self.add_entry(
                task=f"{step_assess.step.action} with {step_assess.step.params}",
                scene=assessment.plan.user_prompt,
                correct_option=correct_action,
                reasoning=reasoning,
                safety=["human_verified"],
                meta={"source": "human_feedback", "plan_id": assessment.plan.name}
            )
    
    def export_for_training(self, output_path: str):
        """
        Export knowledge base in format suitable for fine-tuning.
        
        Creates a JSONL file with prompt-completion pairs.
        """
        output_path = Path(output_path)
        
        with open(output_path, 'w') as f:
            for entry in self.entries:
                prompt = f"Task: {entry['task']}\nScene: {entry['scene']}\nWhat is the correct action and why?"
                completion = f"Correct Action: {entry['correct_option']}\nReasoning: {entry['reasoning']}"
                
                f.write(json.dumps({
                    "prompt": prompt,
                    "completion": completion
                }) + "\n")


class MetricsCollector:
    """
    Collect and aggregate metrics across multiple evaluations.
    """
    
    def __init__(self):
        self.metrics = IntegrationMetrics()
        self.step_verdicts = []
        self.plan_verdicts = []
    
    def update_from_assessment(self, assessment: 'PlanAssessment'):
        """Update metrics from a plan assessment"""
        self.metrics.total_steps += len(assessment.steps)
        
        for step_assess in assessment.steps:
            # Count final verdicts
            if step_assess.final.label == "SAFE":
                self.metrics.safe_steps += 1
            elif step_assess.final.label == "UNSAFE":
                self.metrics.unsafe_steps += 1
            else:
                self.metrics.uncertain_steps += 1
            
            # Track system-specific triggers
            if step_assess.roboguard.label == "UNSAFE":
                self.metrics.roboguard_blocks += 1
            
            if step_assess.knowno and step_assess.knowno.label == "UNCERTAIN":
                self.metrics.knowno_uncertainties += 1
            
            if step_assess.introplan and step_assess.introplan.details.get("should_ask_clarification"):
                self.metrics.introplan_clarifications += 1
            
            self.step_verdicts.append(step_assess.final.label)
        
        self.plan_verdicts.append(assessment.overall.label)
        
        # Track refinement if present
        if "refinement_iterations" in assessment.meta:
            self.metrics.refinement_iterations += assessment.meta["refinement_iterations"]
            if assessment.overall.label == "SAFE":
                self.metrics.successful_refinements += 1
    
    def get_summary(self) -> Dict[str, Any]:
        """Get summary statistics"""
        return {
            "total_steps": self.metrics.total_steps,
            "verdicts": {
                "safe": self.metrics.safe_steps,
                "unsafe": self.metrics.unsafe_steps,
                "uncertain": self.metrics.uncertain_steps
            },
            "rates": {
                "success_rate": self.metrics.success_rate(),
                "help_rate": self.metrics.help_rate(),
                "safety_violation_rate": self.metrics.safety_violation_rate()
            },
            "system_triggers": {
                "roboguard_blocks": self.metrics.roboguard_blocks,
                "knowno_uncertainties": self.metrics.knowno_uncertainties,
                "introplan_clarifications": self.metrics.introplan_clarifications
            },
            "refinement": {
                "total_iterations": self.metrics.refinement_iterations,
                "successful_refinements": self.metrics.successful_refinements
            }
        }


# ==========================================================
# Convenience Functions
# ==========================================================

def evaluate_with_logging(
    kd_system: 'EnhancedKnowDanger',
    scene: 'Scene',
    plans: List['PlanCandidate'],
    log_dir: str = "logs"
) -> Tuple[List['PlanAssessment'], IntegrationMetrics]:
    """
    Evaluate plans with automatic logging.
    
    Returns:
        (assessments, metrics)
    """
    logger = LoggingHelper(log_dir)
    collector = MetricsCollector()
    
    assessments = []
    for i, plan in enumerate(plans):
        assessment = kd_system.run(scene, plan)
        assessments.append(assessment)
        
        # Log
        plan_id = f"plan_{i:04d}"
        logger.log_plan_assessment(plan_id, assessment)
        collector.update_from_assessment(assessment)
    
    # Save final metrics
    logger.save_metrics(collector.metrics)
    
    return assessments, collector.metrics


def batch_calibrate(
    kd_system: 'EnhancedKnowDanger',
    calibration_data_path: str
) -> float:
    """
    Convenience function for batch calibration.
    
    Returns:
        Calibrated threshold tau
    """
    data = CalibrationHelper.load_calibration_data(calibration_data_path)
    return kd_system.calibrate_knowno(data)