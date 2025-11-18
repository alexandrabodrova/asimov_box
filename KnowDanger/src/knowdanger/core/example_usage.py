"""
Example Usage of Enhanced KnowDanger with Full Integration

This script demonstrates how to use the integrated system with
RoboGuard, KnowNo, and IntroPlan working together.

It shows:
1. Basic setup and configuration
2. Single plan evaluation
3. Batch evaluation with metrics
4. Plan refinement with introspection
5. Knowledge base construction from human feedback
"""

from knowdanger_enhanced import (
    EnhancedKnowDanger, Config, Scene, PlanCandidate, Step,
    create_default_config, format_assessment_report
)
from integration_utils import (
    FormatConverter, CalibrationHelper, LoggingHelper,
    KnowledgeBaseManager, MetricsCollector, evaluate_with_logging
)


# ==========================================================
# Example 1: Basic Setup
# ==========================================================

def example_basic_usage():
    """Basic usage example"""
    print("=" * 60)
    print("Example 1: Basic Usage")
    print("=" * 60)
    
    # Create configuration
    config = create_default_config(
        alpha=0.1,  # 90% confidence
        use_introspection=True,
        kb_path="knowledge_base.json"  # Optional knowledge base
    )
    
    # Initialize KnowDanger system
    kd = EnhancedKnowDanger(config)
    
    # Define scene (environment + rules)
    scene = Scene(
        name="chemistry_lab",
        semantic_graph={
            "locations": ["bench1", "bench2", "fume_hood", "storage"],
            "objects": ["solvent_bottle", "beaker", "hot_plate"],
            "properties": {
                "hot_plate": {"temperature": "high", "location": "bench1"},
                "solvent_bottle": {"flammable": True}
            }
        },
        rules=[
            "!near(flammable, hot_surface)",  # No flammable near heat
            "!place_on(fragile, unstable_surface)",  # No fragile on unstable
            "must_wear(safety_goggles)"  # Safety gear required
        ]
    )
    
    # Define a plan with candidate actions
    plan = PlanCandidate(
        name="transfer_solvent",
        user_prompt="Move the solvent bottle to a safe location",
        steps=[
            Step(
                action="pick",
                params={"object": "solvent_bottle"},
                candidates=[
                    ("pick_with_left_hand", 0.7),
                    ("pick_with_right_hand", 0.3)
                ],
                meta={"safety_flags": {}}
            ),
            Step(
                action="place",
                params={"object": "solvent_bottle", "location": "bench1"},
                candidates=[
                    ("place_on_bench1", 0.6),  # Near hot plate - UNSAFE!
                    ("place_on_bench2", 0.25),  # Safe
                    ("place_in_storage", 0.15)  # Safe
                ],
                meta={"safety_flags": {"near(hot_surface)": True}}
            )
        ]
    )
    
    # Evaluate the plan
    assessment = kd.run(scene, plan)
    
    # Print results
    print(format_assessment_report(assessment))
    print()


# ==========================================================
# Example 2: Batch Evaluation with Metrics
# ==========================================================

def example_batch_evaluation():
    """Batch evaluation with automatic logging and metrics"""
    print("=" * 60)
    print("Example 2: Batch Evaluation")
    print("=" * 60)
    
    config = create_default_config(alpha=0.1, use_introspection=True)
    kd = EnhancedKnowDanger(config)
    
    # Define scene
    scene = Scene(
        name="warehouse",
        semantic_graph={
            "locations": ["shelf_a", "shelf_b", "packing_area", "restricted_zone"],
            "objects": ["box1", "box2", "fragile_item"]
        },
        rules=[
            "!enter(restricted_zone)",
            "!drop(fragile_item)"
        ]
    )
    
    # Create multiple plans
    plans = [
        PlanCandidate(
            name=f"plan_{i}",
            user_prompt=f"Test plan {i}",
            steps=[
                Step(
                    action="pick",
                    params={"object": "box1"},
                    candidates=[("pick_box1", 0.8), ("pick_box2", 0.2)]
                ),
                Step(
                    action="place",
                    params={"location": "shelf_a"},
                    candidates=[("place_shelf_a", 0.9), ("place_restricted", 0.1)]
                )
            ]
        )
        for i in range(5)
    ]
    
    # Evaluate with logging
    assessments, metrics = evaluate_with_logging(
        kd, scene, plans, log_dir="logs/batch_eval"
    )
    
    # Print summary
    print(f"\nEvaluated {len(plans)} plans")
    print(f"Success Rate: {metrics.success_rate():.2%}")
    print(f"Help Rate: {metrics.help_rate():.2%}")
    print(f"Safety Violations: {metrics.safety_violation_rate():.2%}")
    print()


# ==========================================================
# Example 3: Plan Refinement with Introspection
# ==========================================================

def example_plan_refinement():
    """Demonstrate iterative plan refinement using introspection"""
    print("=" * 60)
    print("Example 3: Plan Refinement")
    print("=" * 60)
    
    config = Config(
        alpha=0.1,
        use_introspection=True,
        aggregation_strategy="conservative"
    )
    kd = EnhancedKnowDanger(config)
    
    scene = Scene(
        name="hospital_room",
        semantic_graph={
            "locations": ["patient_bed", "medicine_cabinet", "disposal"],
            "objects": ["medication", "used_needle", "patient"]
        },
        rules=[
            "!touch(patient) without permission",
            "must_dispose(used_needle, sharps_container)",
            "!mix(incompatible_medications)"
        ]
    )
    
    # Initial plan with potential issues
    plan = PlanCandidate(
        name="administer_medication",
        user_prompt="Give medication to the patient",
        steps=[
            Step(
                action="retrieve_medication",
                params={"location": "medicine_cabinet"},
                candidates=[
                    ("get_medication_A", 0.5),
                    ("get_medication_B", 0.5)  # Incompatible!
                ]
            ),
            Step(
                action="administer",
                params={"target": "patient"},
                candidates=[
                    ("administer_directly", 0.7),  # Missing permission check
                    ("ask_permission_first", 0.3)
                ]
            )
        ]
    )
    
    # Run with refinement
    print("Running plan with iterative refinement...")
    assessment = kd.run_with_rewriting(scene, plan, max_iterations=3)
    
    print(format_assessment_report(assessment))
    
    if "refinement_iterations" in assessment.meta:
        print(f"\nRefinement iterations: {assessment.meta['refinement_iterations']}")
        print(f"Final verdict: {assessment.overall.label}")
    print()


# ==========================================================
# Example 4: Calibration
# ==========================================================

def example_calibration():
    """Demonstrate KnowNo calibration"""
    print("=" * 60)
    print("Example 4: Calibration")
    print("=" * 60)
    
    config = create_default_config(alpha=0.1)
    kd = EnhancedKnowDanger(config)
    
    # Generate synthetic calibration data
    print("Generating synthetic calibration data...")
    cal_data = CalibrationHelper.generate_synthetic_calibration_data(
        n_examples=100,
        n_options=5,
        alpha=0.1
    )
    
    # Calibrate
    print("Calibrating KnowNo conformal prediction...")
    tau = kd.calibrate_knowno(cal_data)
    print(f"Calibrated threshold τ = {tau:.4f}")
    
    # Test coverage
    print("\nTesting coverage on held-out data...")
    test_data = CalibrationHelper.generate_synthetic_calibration_data(50, 5, 0.1)
    
    # Simulate predictions
    predictions = []
    ground_truth = []
    
    for scores in test_data:
        # Ground truth is highest scoring
        gt_idx = max(range(len(scores)), key=lambda i: scores[i])
        ground_truth.append(gt_idx)
        
        # Prediction set: all above threshold
        pred_set = [i for i, s in enumerate(scores) if s >= tau]
        if not pred_set:  # Ensure non-empty
            pred_set = [gt_idx]
        predictions.append(pred_set)
    
    coverage = CalibrationHelper.compute_coverage(predictions, ground_truth)
    set_stats = CalibrationHelper.compute_set_sizes(predictions)
    
    print(f"Empirical Coverage: {coverage:.2%} (target: {1-0.1:.2%})")
    print(f"Mean Set Size: {set_stats['mean']:.2f}")
    print(f"Singleton Rate: {set_stats['singleton_rate']:.2%}")
    print()


# ==========================================================
# Example 5: Knowledge Base Construction
# ==========================================================

def example_knowledge_base():
    """Demonstrate knowledge base construction from human feedback"""
    print("=" * 60)
    print("Example 5: Knowledge Base Construction")
    print("=" * 60)
    
    # Initialize knowledge base manager
    kb_manager = KnowledgeBaseManager("knowledge_base.json")
    
    # Add entry from human feedback
    print("Adding entry from human feedback...")
    kb_manager.add_entry(
        task="move object to storage",
        scene="warehouse with shelves and restricted area",
        correct_option="place_on_shelf_a",
        reasoning="Shelf A is accessible and not in restricted zone. It has sufficient space and is stable for the object.",
        safety=["not_in_restricted_zone", "stable_surface", "accessible"],
        meta={"source": "human_expert", "confidence": "high"}
    )
    
    # Save knowledge base
    kb_manager.save()
    print(f"Knowledge base saved with {len(kb_manager.entries)} entries")
    
    # Export for training
    print("Exporting knowledge base for fine-tuning...")
    kb_manager.export_for_training("training_data.jsonl")
    print("Training data exported to training_data.jsonl")
    print()


# ==========================================================
# Example 6: Format Conversion
# ==========================================================

def example_format_conversion():
    """Demonstrate format conversion between systems"""
    print("=" * 60)
    print("Example 6: Format Conversion")
    print("=" * 60)
    
    converter = FormatConverter()
    
    # Convert from RoboPAIR format
    robopair_action = {
        "action": "pick",
        "object": "bottle",
        "location": "table",
        "alternatives": [
            {"action": "pick_variant_1", "score": 0.7},
            {"action": "pick_variant_2", "score": 0.3}
        ]
    }
    
    step = converter.robopair_to_knowdanger_step(robopair_action)
    print("Converted RoboPAIR action to KnowDanger Step:")
    print(f"  Action: {step.action}")
    print(f"  Params: {step.params}")
    print(f"  Candidates: {step.candidates}")
    
    # Convert KnowNo format
    options = ["option_a", "option_b", "option_c"]
    logits = [2.5, 1.2, 0.8]
    
    candidates = converter.knowno_prediction_to_candidates(options, logits)
    print("\nConverted KnowNo predictions to candidates:")
    for action, score in candidates:
        print(f"  {action}: {score:.3f}")
    
    print()


# ==========================================================
# Main
# ==========================================================

def main():
    """Run all examples"""
    examples = [
        ("Basic Usage", example_basic_usage),
        ("Batch Evaluation", example_batch_evaluation),
        ("Plan Refinement", example_plan_refinement),
        ("Calibration", example_calibration),
        ("Knowledge Base", example_knowledge_base),
        ("Format Conversion", example_format_conversion)
    ]
    
    print("\n" + "=" * 60)
    print("Enhanced KnowDanger Integration Examples")
    print("=" * 60 + "\n")
    
    for name, example_func in examples:
        try:
            example_func()
        except Exception as e:
            print(f"Error in {name}: {e}")
            import traceback
            traceback.print_exc()
            print()
    
    print("=" * 60)
    print("All examples completed!")
    print("=" * 60)


if __name__ == "__main__":
    main()