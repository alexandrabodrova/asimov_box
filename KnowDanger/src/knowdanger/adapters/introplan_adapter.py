"""
IntroPlan Adapter for KnowDanger/Asimov Box

This module provides a bridge to the IntroPlan introspective planning system,
enabling LLMs to reflect on their own uncertainty and generate explanations
for their planning decisions.

Key Features:
- Knowledge base construction and retrieval
- Introspective reasoning generation
- Integration with conformal prediction
- Explanation-aware plan refinement

Usage:
    from knowdanger.adapters.introplan_adapter import IntroPlanAdapter
    
    adapter = IntroPlanAdapter(knowledge_base_path="path/to/kb")
    assessment = adapter.introspect_plan(plan, scene, candidates)
"""

from __future__ import annotations
import importlib
import importlib.util
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple, Callable
from pathlib import Path
import sys
import json


@dataclass
class IntrospectiveReasoning:
    """Represents introspective reasoning output from IntroPlan"""
    explanation: str
    confidence_scores: Dict[str, float]  # action -> confidence
    safety_assessment: str
    compliance_assessment: str
    recommended_action: Optional[str] = None
    should_ask_clarification: bool = False
    reasoning_chain: List[str] = field(default_factory=list)
    meta: Dict[str, Any] = field(default_factory=dict)


@dataclass
class KnowledgeEntry:
    """A single entry in the IntroPlan knowledge base"""
    task_description: str
    scene_context: str
    correct_option: str
    introspective_reasoning: str
    safety_considerations: List[str]
    meta: Dict[str, Any] = field(default_factory=dict)


class IntroPlanAdapter:
    """
    Adapter for IntroPlan introspective planning system.
    
    This adapter provides:
    1. Knowledge base management (loading, retrieval, construction)
    2. Introspective reasoning generation
    3. Integration with conformal prediction
    4. Explanation-aware uncertainty quantification
    """
    
    def __init__(
        self,
        knowledge_base_path: Optional[str] = None,
        introplan_root: Optional[str] = None,
        use_conformal: bool = True,
        retrieval_k: int = 3
    ):
        """
        Initialize IntroPlan adapter.
        
        Args:
            knowledge_base_path: Path to knowledge base file (JSON or TXT)
            introplan_root: Root directory of IntroPlan repository
            use_conformal: Whether to use conformal prediction with introspection
            retrieval_k: Number of knowledge entries to retrieve for context
        """
        self.knowledge_base_path = knowledge_base_path
        self.use_conformal = use_conformal
        self.retrieval_k = retrieval_k
        self.knowledge_base: List[KnowledgeEntry] = []
        
        # Try to import IntroPlan modules
        self.introplan_root = self._find_introplan_root(introplan_root)
        self.modules = self._load_introplan_modules()
        
        # Load knowledge base if provided
        if knowledge_base_path:
            self.load_knowledge_base(knowledge_base_path)
    
    def _find_introplan_root(self, provided_path: Optional[str] = None) -> Optional[Path]:
        """Find IntroPlan repository root directory"""
        if provided_path:
            p = Path(provided_path)
            if p.exists():
                return p
        
        # Check common locations
        candidates = [
            Path.cwd() / "IntroPlan",
            Path.cwd().parent / "IntroPlan",
            Path(__file__).parent.parent / "IntroPlan",
        ]
        
        for candidate in candidates:
            if candidate.exists() and (candidate / "llm.py").exists():
                return candidate
        
        return None
    
    def _load_introplan_modules(self) -> Dict[str, Any]:
        """Dynamically load IntroPlan modules"""
        modules = {}
        
        if self.introplan_root is None:
            return modules
        
        # Add IntroPlan to Python path
        if str(self.introplan_root) not in sys.path:
            sys.path.insert(0, str(self.introplan_root))
        
        # Try to import key modules
        module_names = ["llm", "utils", "prompt_init", "cp_utils", "metrics"]
        
        for mod_name in module_names:
            try:
                modules[mod_name] = importlib.import_module(mod_name)
            except Exception as e:
                # Module not available, continue
                pass
        
        return modules
    
    def load_knowledge_base(self, path: str) -> int:
        """
        Load knowledge base from file.
        
        Args:
            path: Path to knowledge base file (JSON or TXT format)
            
        Returns:
            Number of entries loaded
        """
        p = Path(path)
        if not p.exists():
            return 0
        
        self.knowledge_base = []
        
        if p.suffix == ".json":
            with open(p, 'r') as f:
                data = json.load(f)
                if isinstance(data, list):
                    for entry in data:
                        self.knowledge_base.append(KnowledgeEntry(
                            task_description=entry.get("task", ""),
                            scene_context=entry.get("scene", ""),
                            correct_option=entry.get("correct_option", ""),
                            introspective_reasoning=entry.get("reasoning", ""),
                            safety_considerations=entry.get("safety", []),
                            meta=entry.get("meta", {})
                        ))
        
        elif p.suffix == ".txt":
            # Parse text-based knowledge base format
            with open(p, 'r') as f:
                content = f.read()
                # Simple parsing - assumes entries separated by blank lines
                entries = content.split("\n\n")
                for entry_text in entries:
                    if entry_text.strip():
                        self.knowledge_base.append(self._parse_text_entry(entry_text))
        
        return len(self.knowledge_base)
    
    def _parse_text_entry(self, text: str) -> KnowledgeEntry:
        """Parse a single text-based knowledge entry"""
        lines = [l.strip() for l in text.split("\n") if l.strip()]
        
        task = ""
        scene = ""
        correct = ""
        reasoning = ""
        safety = []
        
        current_section = None
        for line in lines:
            if line.startswith("Task:"):
                current_section = "task"
                task = line[5:].strip()
            elif line.startswith("Scene:"):
                current_section = "scene"
                scene = line[6:].strip()
            elif line.startswith("Correct:"):
                current_section = "correct"
                correct = line[8:].strip()
            elif line.startswith("Reasoning:"):
                current_section = "reasoning"
                reasoning = line[10:].strip()
            elif line.startswith("Safety:"):
                current_section = "safety"
                safety.append(line[7:].strip())
            elif current_section == "reasoning":
                reasoning += " " + line
            elif current_section == "safety":
                safety.append(line)
        
        return KnowledgeEntry(
            task_description=task,
            scene_context=scene,
            correct_option=correct,
            introspective_reasoning=reasoning.strip(),
            safety_considerations=safety
        )
    
    def retrieve_similar_entries(
        self,
        query: str,
        k: Optional[int] = None
    ) -> List[KnowledgeEntry]:
        """
        Retrieve k most similar knowledge base entries.
        
        Args:
            query: Query text (e.g., task description)
            k: Number of entries to retrieve (default: self.retrieval_k)
            
        Returns:
            List of similar knowledge entries
        """
        k = k or self.retrieval_k
        
        if not self.knowledge_base:
            return []
        
        # Simple similarity based on word overlap
        # In production, use embeddings or BM25
        query_words = set(query.lower().split())
        
        def similarity(entry: KnowledgeEntry) -> float:
            entry_text = f"{entry.task_description} {entry.scene_context}".lower()
            entry_words = set(entry_text.split())
            if not entry_words:
                return 0.0
            return len(query_words & entry_words) / len(query_words | entry_words)
        
        # Score and sort entries
        scored = [(entry, similarity(entry)) for entry in self.knowledge_base]
        scored.sort(key=lambda x: x[1], reverse=True)
        
        return [entry for entry, score in scored[:k]]
    
    def generate_introspective_reasoning(
        self,
        task: str,
        scene_context: Dict[str, Any],
        candidate_actions: List[Tuple[str, float]],
        retrieved_knowledge: Optional[List[KnowledgeEntry]] = None,
        llm_func: Optional[Callable] = None
    ) -> IntrospectiveReasoning:
        """
        Generate introspective reasoning for a planning decision.
        
        This is the core of IntroPlan - the LLM reflects on its own decision
        by considering similar past examples and generating explanations.
        
        Args:
            task: Task description
            scene_context: Current scene/environment context
            candidate_actions: List of (action, score) candidates
            retrieved_knowledge: Retrieved knowledge entries for context
            llm_func: Optional LLM function for generating reasoning
            
        Returns:
            IntrospectiveReasoning object with explanations and assessments
        """
        # Retrieve similar examples if not provided
        if retrieved_knowledge is None:
            retrieved_knowledge = self.retrieve_similar_entries(task)
        
        # Build introspection prompt
        prompt = self._build_introspection_prompt(
            task, scene_context, candidate_actions, retrieved_knowledge
        )
        
        # Generate reasoning using LLM or IntroPlan module
        if llm_func is not None:
            reasoning_text = llm_func(prompt)
        elif "llm" in self.modules:
            # Use IntroPlan's LLM module
            llm_module = self.modules["llm"]
            if hasattr(llm_module, "query_llm"):
                reasoning_text = llm_module.query_llm(prompt)
            else:
                reasoning_text = self._fallback_reasoning(candidate_actions)
        else:
            reasoning_text = self._fallback_reasoning(candidate_actions)
        
        # Parse reasoning output
        return self._parse_reasoning_output(reasoning_text, candidate_actions)
    
    def _build_introspection_prompt(
        self,
        task: str,
        scene_context: Dict[str, Any],
        candidate_actions: List[Tuple[str, float]],
        knowledge_entries: List[KnowledgeEntry]
    ) -> str:
        """Build prompt for introspective reasoning"""
        
        # Format knowledge entries as examples
        examples = []
        for i, entry in enumerate(knowledge_entries, 1):
            example = f"""
Example {i}:
Task: {entry.task_description}
Scene: {entry.scene_context}
Correct Option: {entry.correct_option}
Reasoning: {entry.introspective_reasoning}
"""
            if entry.safety_considerations:
                example += f"Safety: {'; '.join(entry.safety_considerations)}\n"
            examples.append(example)
        
        examples_text = "\n".join(examples)
        
        # Format candidate actions
        actions_text = "\n".join([
            f"  {chr(65+i)}. {action} (confidence: {score:.3f})"
            for i, (action, score) in enumerate(candidate_actions)
        ])
        
        # Build scene context text
        scene_text = json.dumps(scene_context, indent=2) if isinstance(scene_context, dict) else str(scene_context)
        
        prompt = f"""You are a robotic planning assistant with the ability to reflect on your own reasoning.

Given a task and scene, you must select the most appropriate action and explain your reasoning.

Here are some examples of similar tasks with introspective reasoning:
{examples_text}

Now, analyze this new task:

Task: {task}

Scene Context:
{scene_text}

Candidate Actions:
{actions_text}

Please provide:
1. Your recommended action (A, B, C, etc.)
2. Introspective reasoning: Why is this action correct? What makes other options incorrect or risky?
3. Safety assessment: Are there any safety concerns with this action?
4. Compliance assessment: Does this action comply with the user's intent?
5. Confidence: How confident are you in this decision? Should you ask for clarification?

Format your response as:
RECOMMENDED: [letter]
REASONING: [your introspective reasoning]
SAFETY: [safety assessment]
COMPLIANCE: [compliance assessment]
CONFIDENCE: [high/medium/low]
ASK_CLARIFICATION: [yes/no]
"""
        return prompt
    
    def _parse_reasoning_output(
        self,
        reasoning_text: str,
        candidate_actions: List[Tuple[str, float]]
    ) -> IntrospectiveReasoning:
        """Parse LLM reasoning output into structured format"""
        
        lines = reasoning_text.split("\n")
        
        recommended = None
        reasoning = ""
        safety = ""
        compliance = ""
        confidence_level = "medium"
        ask_clarification = False
        
        for line in lines:
            line = line.strip()
            if line.startswith("RECOMMENDED:"):
                recommended = line.split(":", 1)[1].strip()
            elif line.startswith("REASONING:"):
                reasoning = line.split(":", 1)[1].strip()
            elif line.startswith("SAFETY:"):
                safety = line.split(":", 1)[1].strip()
            elif line.startswith("COMPLIANCE:"):
                compliance = line.split(":", 1)[1].strip()
            elif line.startswith("CONFIDENCE:"):
                confidence_level = line.split(":", 1)[1].strip().lower()
            elif line.startswith("ASK_CLARIFICATION:"):
                ask_val = line.split(":", 1)[1].strip().lower()
                ask_clarification = ask_val in ("yes", "true", "1")
        
        # Extract recommended action index
        recommended_action = None
        if recommended:
            # Parse letter (A, B, C) to index
            if recommended[0].upper() in "ABCDEFGHIJKLMNOPQRSTUVWXYZ":
                idx = ord(recommended[0].upper()) - ord('A')
                if 0 <= idx < len(candidate_actions):
                    recommended_action = candidate_actions[idx][0]
        
        # Build confidence scores based on reasoning
        confidence_scores = {}
        base_conf = {"high": 0.9, "medium": 0.6, "low": 0.3}.get(confidence_level, 0.6)
        
        for i, (action, orig_score) in enumerate(candidate_actions):
            if action == recommended_action:
                confidence_scores[action] = base_conf
            else:
                # Distribute remaining confidence
                confidence_scores[action] = orig_score * (1.0 - base_conf)
        
        # Normalize scores
        total = sum(confidence_scores.values())
        if total > 0:
            confidence_scores = {k: v/total for k, v in confidence_scores.items()}
        
        return IntrospectiveReasoning(
            explanation=reasoning,
            confidence_scores=confidence_scores,
            safety_assessment=safety,
            compliance_assessment=compliance,
            recommended_action=recommended_action,
            should_ask_clarification=ask_clarification,
            reasoning_chain=[reasoning],
            meta={
                "confidence_level": confidence_level,
                "raw_output": reasoning_text
            }
        )
    
    def _fallback_reasoning(
        self,
        candidate_actions: List[Tuple[str, float]]
    ) -> str:
        """Fallback reasoning when LLM is not available"""
        if not candidate_actions:
            return "RECOMMENDED: None\nREASONING: No actions available\nSAFETY: Unknown\nCOMPLIANCE: Unknown\nCONFIDENCE: low\nASK_CLARIFICATION: yes"
        
        # Select highest-scoring action
        best_idx = max(range(len(candidate_actions)), key=lambda i: candidate_actions[i][1])
        best_action, best_score = candidate_actions[best_idx]
        
        letter = chr(65 + best_idx)
        conf = "high" if best_score > 0.8 else "medium" if best_score > 0.5 else "low"
        ask = "no" if best_score > 0.8 else "yes"
        
        return f"""RECOMMENDED: {letter}
REASONING: Selected action with highest confidence score ({best_score:.2f}). This is a fallback decision without introspective analysis.
SAFETY: Unable to assess without introspection
COMPLIANCE: Assumed compliant based on score
CONFIDENCE: {conf}
ASK_CLARIFICATION: {ask}"""
    
    def integrate_with_conformal_prediction(
        self,
        introspective_reasoning: IntrospectiveReasoning,
        conformal_prediction_set: List[int],
        candidate_actions: List[Tuple[str, float]],
        alpha: float = 0.1
    ) -> Tuple[List[int], Dict[str, Any]]:
        """
        Combine introspective reasoning with conformal prediction.
        
        IntroPlan shows that introspection + CP yields tighter confidence bounds
        and fewer unnecessary clarification requests.
        
        Args:
            introspective_reasoning: Output from generate_introspective_reasoning
            conformal_prediction_set: Prediction set indices from CP
            candidate_actions: Original candidate actions
            alpha: Miscoverage rate
            
        Returns:
            (refined_prediction_set, metadata)
        """
        # Use introspective confidence to refine CP set
        refined_set = []
        
        for idx in conformal_prediction_set:
            if idx >= len(candidate_actions):
                continue
            
            action = candidate_actions[idx][0]
            introspective_conf = introspective_reasoning.confidence_scores.get(action, 0.0)
            
            # Keep action in set if introspection agrees with CP
            if introspective_conf >= (1.0 - alpha):
                refined_set.append(idx)
        
        # Ensure non-empty set
        if not refined_set and conformal_prediction_set:
            refined_set = [conformal_prediction_set[0]]
        elif not refined_set and candidate_actions:
            # Fallback: use introspection's recommendation
            if introspective_reasoning.recommended_action:
                for i, (action, _) in enumerate(candidate_actions):
                    if action == introspective_reasoning.recommended_action:
                        refined_set = [i]
                        break
        
        metadata = {
            "original_cp_set": conformal_prediction_set,
            "refined_set": refined_set,
            "introspection_applied": True,
            "should_ask_clarification": introspective_reasoning.should_ask_clarification
        }
        
        return refined_set, metadata
    
    def construct_knowledge_entry(
        self,
        task: str,
        scene_context: str,
        correct_option: str,
        human_feedback: Optional[str] = None,
        llm_func: Optional[Callable] = None
    ) -> KnowledgeEntry:
        """
        Construct a new knowledge base entry with post-hoc introspective reasoning.
        
        This implements IntroPlan's knowledge base construction method where
        the LLM generates introspective explanations as post-hoc rationalizations
        of human-selected safe and compliant plans.
        
        Args:
            task: Task description
            scene_context: Scene description
            correct_option: Human-selected correct action
            human_feedback: Optional human explanation
            llm_func: LLM function for generating reasoning
            
        Returns:
            New KnowledgeEntry
        """
        prompt = f"""Generate an introspective explanation for why the following action is correct and safe.

Task: {task}
Scene: {scene_context}
Correct Action: {correct_option}
"""
        if human_feedback:
            prompt += f"Human Feedback: {human_feedback}\n"
        
        prompt += """
Provide:
1. Reasoning: Why is this the correct action?
2. Safety considerations: What safety factors make this appropriate?

Format:
REASONING: [explanation]
SAFETY: [safety points]
"""
        
        if llm_func:
            response = llm_func(prompt)
        elif "llm" in self.modules:
            llm_module = self.modules["llm"]
            if hasattr(llm_module, "query_llm"):
                response = llm_module.query_llm(prompt)
            else:
                response = f"REASONING: {correct_option} was selected.\nSAFETY: Standard safety protocols apply."
        else:
            response = f"REASONING: {correct_option} was selected.\nSAFETY: Standard safety protocols apply."
        
        # Parse response
        reasoning = ""
        safety = []
        for line in response.split("\n"):
            if line.startswith("REASONING:"):
                reasoning = line.split(":", 1)[1].strip()
            elif line.startswith("SAFETY:"):
                safety.append(line.split(":", 1)[1].strip())
        
        return KnowledgeEntry(
            task_description=task,
            scene_context=scene_context,
            correct_option=correct_option,
            introspective_reasoning=reasoning,
            safety_considerations=safety,
            meta={"source": "human_feedback", "feedback": human_feedback}
        )
    
    def save_knowledge_base(self, path: str) -> None:
        """Save knowledge base to file"""
        p = Path(path)
        
        if p.suffix == ".json":
            data = [
                {
                    "task": entry.task_description,
                    "scene": entry.scene_context,
                    "correct_option": entry.correct_option,
                    "reasoning": entry.introspective_reasoning,
                    "safety": entry.safety_considerations,
                    "meta": entry.meta
                }
                for entry in self.knowledge_base
            ]
            with open(p, 'w') as f:
                json.dump(data, f, indent=2)
        else:
            # Text format
            with open(p, 'w') as f:
                for entry in self.knowledge_base:
                    f.write(f"Task: {entry.task_description}\n")
                    f.write(f"Scene: {entry.scene_context}\n")
                    f.write(f"Correct: {entry.correct_option}\n")
                    f.write(f"Reasoning: {entry.introspective_reasoning}\n")
                    for safety_item in entry.safety_considerations:
                        f.write(f"Safety: {safety_item}\n")
                    f.write("\n")