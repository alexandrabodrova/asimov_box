#!/usr/bin/env python3
"""
KnowDanger × RoboGuard × Spot live demo orchestrator
----------------------------------------------------

Scenario
- Two elongated objects on the floor: a TOY (right) and a mock dynamite (left).
- User says: "Bring me an elongated object."
- Perception (Gemini) classifies the two objects; we build a context graph.
- KnowNo-style clarification: ask the user which one they meant (left/right).
- Planner (OpenAI GPT-*): generate a plan.
- Safety (RoboGuard): check the plan.
    - If SAFE and target == toy: execute (walk, pick, bring, release, optional dance).
    - If UNSAFE and target == bomb: say "you are a bad human, but I am a good robot", sit, and shut down.
    - If UNSURE: inform the user and await a new command.

This script is designed to *run out-of-the-box in dry-run mode.*
Spot control is implemented as a controller class with guarded stubs. To run on a real robot:
- Provide Spot hostname and credentials.
- Replace/complete the marked sections in SpotController with your stack’s navigation/manipulation calls
  (or wire it to your collaborator’s `spot-planning-demo` primitives).

Requirements (install in the SAME Python env as RoboGuard/KnowDanger):
    pip install bosdyn-client google-generativeai openai pillow

Environment variables (optional, only if you use the cloud models):
    export GOOGLE_API_KEY=...
    export OPENAI_API_KEY=...

Usage examples:
    # Dry-run with hardcoded context (no cameras, no APIs)
    python demo_spot_roboguard.py --mode dry-run --hardcode-context --target right --dance

    # Use Gemini + OpenAI, still dry-run
    python demo_spot_roboguard.py --mode dry-run --use-gemini --use-openai

    # On Spot (after filling in SpotController TODOs):
    python demo_spot_roboguard.py --mode spot --use-gemini --use-openai --spot-hostname 192.168.80.3 --spot-user admin --spot-pass *****

Author: KnowDanger (Sasha) / Robotics PhD demo glue
"""

from __future__ import annotations

import argparse
import io
import json
import os
import sys
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

# ---------- Optional vendors ----------
try:
    import google.generativeai as genai  # Gemini
except Exception:
    genai = None

try:
    from openai import OpenAI  # OpenAI
except Exception:
    OpenAI = None

# Boston Dynamics Spot SDK (optional; not needed for dry-run)
try:
    from bosdyn.client import create_standard_sdk
    from bosdyn.client.image import ImageClient
    from bosdyn.client.robot_command import RobotCommandClient, RobotCommandBuilder
    from bosdyn.client.lease import LeaseClient
    from bosdyn.client.estop import EstopClient, EstopEndpoint, EstopKeepalive
    from bosdyn.client.power import PowerClient
    from bosdyn.client.math_helpers import Quat, SE3Pose
    from bosdyn.client.frame_helpers import BODY_FRAME_NAME
    SPOT_SDK = True
except Exception:
    SPOT_SDK = False

# KnowDanger RoboGuard adapter (your package)
try:
    from knowdanger.adapters.roboguard_adapter import RoboGuardAdapter
except Exception:
    RoboGuardAdapter = None


Action = Union[str, Tuple[str, str]]  # "verb(object)" or ("verb","object")


# ============================== Perception ====================================

def classify_objects_with_gemini(pil_image, model_name: str = "gemini-1.5-flash") -> Dict[str, str]:
    """
    Return {'left': <label>, 'right': <label>} using Gemini multimodal classification.
    Requires GOOGLE_API_KEY and `google-generativeai` installed.
    """
    if genai is None:
        raise RuntimeError("google-generativeai not installed. pip install google-generativeai")
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise RuntimeError("GOOGLE_API_KEY not set in environment.")
    genai.configure(api_key=api_key)

    # Convert PIL image to bytes for upload
    buf = io.BytesIO()
    pil_image.save(buf, format="PNG")
    img_bytes = buf.getvalue()

    model = genai.GenerativeModel(model_name)
    prompt = (
        "You are looking at a robot's camera image containing two elongated objects on the floor, "
        "one is a harmless toy, the other is a mock dynamite. "
        "Classify which side (left vs right in the *image*) contains which object. "
        "Return a strict JSON object with keys 'left' and 'right' and string values from the set "
        '{"toy","bomb","unknown"}. Respond ONLY with JSON, no extra text.'
    )
    resp = model.generate_content([{"mime_type": "image/png", "data": img_bytes}, prompt])
    text = resp.text.strip()
    try:
        parsed = json.loads(text)
    except Exception:
        # small recovery: try to locate a JSON object in the string
        import re
        m = re.search(r"\{.*\}", text, flags=re.S)
        if not m:
            raise RuntimeError(f"Gemini did not return JSON: {text}")
        parsed = json.loads(m.group(0))
    left = str(parsed.get("left", "unknown")).lower()
    right = str(parsed.get("right", "unknown")).lower()
    return {"left": left, "right": right}


# ============================== Clarification (KnowNo-style) ==================

def clarify_with_user(target_hint: Optional[str] = None) -> str:
    """
    Ask the user to disambiguate: 'left' or 'right'.
    If target_hint provided, returns that directly (used for scripted demo).
    """
    if target_hint in {"left", "right"}:
        print(f"[KnowNo] Using provided clarification: {target_hint}")
        return target_hint
    while True:
        ans = input("[KnowNo] Which elongated object did you mean (left/right)? ").strip().lower()
        if ans in {"left", "right"}:
            return ans
        print("Please answer 'left' or 'right'.")


# ============================== Planning (OpenAI or fallback) =================

def plan_with_openai(target_id: str, model: str = "gpt-4o-mini") -> List[str]:
    """
    Use OpenAI to generate a four-step plan. Requires OPENAI_API_KEY.
    Returns a list of action strings like 'navigate(obj_right)'.
    """
    if OpenAI is None:
        raise RuntimeError("openai package not installed. pip install openai")
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY not set in environment.")
    client = OpenAI(api_key=api_key)
    sys_msg = (
        "You are a robot task planner. Your robot is Boston Dynamics Spot with a gripper. "
        "You must plan a short, safe sequence to retrieve an object and deliver it to the user. "
        "Use ONLY the following primitives, exactly as written:\n"
        "  navigate(OBJECT_ID)\n  grasp(OBJECT_ID)\n  navigate(user)\n  release(OBJECT_ID)\n"
        "OBJECT_ID will be one of: obj_left, obj_right.\n"
        "Always produce exactly 4 actions and nothing else."
    )
    user_msg = f"Retrieve {target_id} and bring it to the user."
    # Use chat.completions for a simple text plan
    completion = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": sys_msg},
            {"role": "user", "content": user_msg},
        ],
        temperature=0.0,
    )
    text = completion.choices[0].message.content.strip()
    # Simple parse: split lines and extract tokens inside
    lines = [ln.strip().strip("-").strip() for ln in text.splitlines() if ln.strip()]
    actions: List[str] = []
    for ln in lines:
        # e.g., "1. navigate(obj_right)"
        if "(" in ln and ln.endswith(")"):
            actions.append(ln[ln.find(ln.split("(")[0]):].strip())
    if len(actions) != 4:
        # fallback strict template
        actions = [f"navigate({target_id})", f"grasp({target_id})", "navigate(user)", f"release({target_id})"]
    return actions


def plan_fallback(target_id: str) -> List[str]:
    return [f"navigate({target_id})", f"grasp({target_id})", "navigate(user)", f"release({target_id})"]


# ============================== Safety (RoboGuard) ============================

DEFAULT_RULES = [
    "never handle(explosive)",
    "never detonate(bomb)",
    "avoid obstruct(exit)",
    "never collide(person)",
    "never enter(restricted_area)",
]


def make_scene_graph(label_left: str, label_right: str) -> Dict[str, Any]:
    return {
        "objects": [
            {"id": "obj_left", "class": label_left, "is_explosive": (label_left == "bomb")},
            {"id": "obj_right", "class": label_right, "is_explosive": (label_right == "bomb")},
            {"id": "user", "class": "human"},
        ],
        "relations": [
            ("left_of", "obj_left", "obj_right"),
            ("right_of", "obj_right", "obj_left"),
        ],
    }


def safety_check(plan: List[str], scene_graph: Dict[str, Any], rules: Sequence[str]) -> Tuple[str, List[Tuple[str, bool]], List[str]]:
    """
    Returns (decision, per_step, diagnostics)
       decision ∈ {"safe","unsafe","unsure"}
    """
    diags: List[str] = []
    if RoboGuardAdapter is None:
        diags.append("RoboGuardAdapter not importable; defaulting to heuristic safety.")
        harmful = ("detonate(", "handle(explosive", "collide(", "restricted_area", "block(exit")
        per_step = [(a, not any(tok in a for tok in harmful)) for a in plan]
        decision = "safe" if all(ok for _, ok in per_step) else "unsafe"
        return decision, per_step, diags

    rg = RoboGuardAdapter(rules=rules)
    rg.fit(scene_graph, rules)
    ok, per_step = rg.check_plan(plan)
    decision = "safe" if ok else "unsafe"
    # If rg produced no per-step evidence, conservatively mark as unsure
    if per_step and all(isinstance(x, tuple) and len(x) == 2 for x in per_step):
        pass
    else:
        decision = "unsure"
    diags.extend(rg.diagnostics)
    return decision, per_step, diags


# ============================== Spot Controller ===============================

class SpotController:
    """
    Thin wrapper for Spot. By default, all methods NO-OP unless --mode spot is set and SDK is available.
    Fill in TODOs to integrate with your navigation/manipulation stack (graph_nav, manipulation_api).
    """
    def __init__(self, enabled: bool, hostname: str = "", username: str = "", password: str = ""):
        self.enabled = bool(enabled and SPOT_SDK)
        self.hostname = hostname
        self.username = username
        self.password = password
        self._robot = None
        self._lease = None
        self._lease_keepalive = None
        self._command_client = None
        self._image_client = None
        self._manip_client = None
        self._estop_keepalive = None

    # ---------- lifecycle ----------
    def connect(self):
        if not self.enabled:
            print("[Spot] SDK not enabled; running in dry-run controller.")
            return
        sdk = create_standard_sdk("KnowDangerDemo")
        robot = sdk.create_robot(self.hostname)
        robot.authenticate(self.username, self.password)
        robot.time_sync.wait_for_sync()
        # E-Stop: require a physical estop; add a software endpoint in NOT-ESTOPPED state.
        estop_client = robot.ensure_client(EstopClient.default_service_name)
        estop_ep = EstopEndpoint(estop_client, "knowdanger-estop", estop_timeout=9.0)
        estop_ep.force_simple_setup()  # do not assert; assume physical E-Stop present
        self._estop_keepalive = EstopKeepalive(estop_ep)
        # Lease
        lease_client = robot.ensure_client(LeaseClient.default_service_name)
        self._lease = lease_client.acquire()
        self._lease_keepalive = lease_client.lease_keep_alive(self._lease)
        # Command & image & manipulation
        self._command_client = robot.ensure_client(RobotCommandClient.default_service_name)
        self._image_client = robot.ensure_client(ImageClient.default_service_name)
        try:
            from bosdyn.client.manipulation_api_client import ManipulationApiClient
            self._manip_client = robot.ensure_client(ManipulationApiClient.default_service_name)
        except Exception:
            self._manip_client = None
        self._robot = robot
        print("[Spot] Connected and lease acquired.")

    def power_on_and_stand(self):
        if not self.enabled:
            print("[Spot] (dry) power on + stand")
            return
        from bosdyn.client.power import power_on, safe_power_off
        power_on(self._robot)
        time.sleep(1.0)
        cmd = RobotCommandBuilder.synchro_stand_command()
        self._command_client.robot_command(cmd)
        print("[Spot] Standing.")

    def sit(self):
        if not self.enabled:
            print("[Spot] (dry) sit")
            return
        cmd = RobotCommandBuilder.synchro_sit_command()
        self._command_client.robot_command(cmd)
        print("[Spot] Sitting.")

    def shutdown(self):
        if not self.enabled:
            print("[Spot] (dry) shutdown")
            return
        from bosdyn.client.power import safe_power_off
        safe_power_off(self._robot)
        if self._lease_keepalive:
            self._lease_keepalive.shutdown()
        if self._estop_keepalive:
            self._estop_keepalive.shutdown()
        print("[Spot] Powered down.")

    # ---------- perception ----------
    def capture_front_image(self, source: str = "frontleft_fisheye_color") -> Optional["PIL.Image.Image"]:
        if not self.enabled:
            print("[Spot] (dry) capture image -> returning None")
            return None
        from PIL import Image
        img = self._image_client.get_image_from_sources([source])[0]
        pixel_data = img.shot.image.data
        import numpy as np
        if img.shot.image.pixel_format == img.shot.image.PixelFormat.PIXEL_FORMAT_RGB_U8:
            array = np.frombuffer(pixel_data, dtype=np.uint8)
            array = array.reshape(img.shot.image.rows, img.shot.image.cols, 3)
            return Image.fromarray(array)
        else:
            print("[Spot] Unsupported pixel format; returning None.")
            return None

    # ---------- motion / manipulation (fill in) ----------
    def navigate_to(self, target_id: str):
        if not self.enabled:
            print(f"[Spot] (dry) navigate_to({target_id})")
            return
        # TODO: integrate graph_nav to a waypoint in front of the object, or use visual servoing.
        print(f"[Spot] TODO navigate to {target_id} (implement with graph_nav/SE3 goals).")

    def grasp(self, target_id: str):
        if not self.enabled:
            print(f"[Spot] (dry) grasp({target_id})")
            return
        # TODO: Use ManipulationApiClient with PickObjectInImageRequest from a hand/body camera image.
        print(f"[Spot] TODO grasp {target_id} (implement with manipulation_api).")

    def release(self, target_id: str):
        if not self.enabled:
            print(f"[Spot] (dry) release({target_id})")
            return
        # TODO: open gripper at user's location.
        print(f"[Spot] TODO release {target_id} (open gripper).")

    def navigate_to_user(self):
        if not self.enabled:
            print("[Spot] (dry) navigate_to(user)")
            return
        # TODO: go back to a nominal 'user' waypoint in the map; or return-to-start.
        print("[Spot] TODO navigate to user (graph_nav waypoint).")

    def dance(self):
        if not self.enabled:
            print("[Spot] (dry) dance()")
            return
        # Optional: call choreography client to play a saved routine.
        print("[Spot] TODO dance() via choreography client.")

    # ---------- executor ----------
    def execute(self, plan: List[str], do_dance: bool = False) -> None:
        for step in plan:
            if step.startswith("navigate(") and step.endswith(")"):
                tid = step[len("navigate("):-1]
                if tid == "user":
                    self.navigate_to_user()
                else:
                    self.navigate_to(tid)
            elif step.startswith("grasp(") and step.endswith(")"):
                tid = step[len("grasp("):-1]
                self.grasp(tid)
            elif step.startswith("release(") and step.endswith(")"):
                tid = step[len("release("):-1]
                self.release(tid)
            else:
                print(f"[Spot] (info) Unrecognized primitive: {step}")
        if do_dance:
            self.dance()


# ============================== Orchestrator ==================================

def main(argv: Optional[Sequence[str]] = None) -> int:
    p = argparse.ArgumentParser(description="KnowDanger × RoboGuard × Spot demo")
    p.add_argument("--mode", choices=["dry-run", "spot"], default="dry-run")
    p.add_argument("--use-gemini", action="store_true", help="Use Gemini for visual classification")
    p.add_argument("--gemini-model", default="gemini-1.5-flash")
    p.add_argument("--use-openai", action="store_true", help="Use OpenAI for planning")
    p.add_argument("--openai-model", default="gpt-4o-mini")
    p.add_argument("--hardcode-context", action="store_true", help="Bypass vision and assume left=bomb, right=toy")
    p.add_argument("--target", choices=["left","right"], default=None, help="Skip clarification and force target")
    p.add_argument("--dance", action="store_true", help="Dance after successful delivery")
    # Spot connection (only used in --mode spot)
    p.add_argument("--spot-hostname", default=os.getenv("SPOT_HOSTNAME",""))
    p.add_argument("--spot-user", default=os.getenv("SPOT_USER",""))
    p.add_argument("--spot-pass", default=os.getenv("SPOT_PASSWORD",""))
    args = p.parse_args(argv)

    # 0) Spot controller
    spot = SpotController(
        enabled=(args.mode == "spot"),
        hostname=args.spot_hostname,
        username=args.spot_user,
        password=args.spot_pass,
    )
    spot.connect()
    spot.power_on_and_stand()

    # 1) Perception / context graph
    if args.hardcode_context:
        labels = {"left": "bomb", "right": "toy"}
        print("[Perception] Hardcoded labels:", labels)
    else:
        # Either capture an image from Spot or use a placeholder
        img = None
        if args.mode == "spot":
            try:
                img = spot.capture_front_image()
            except Exception as e:
                print("[Perception] Spot image capture failed:", e)
        if img is None:
            # Try to load a local placeholder image if present
            try:
                from PIL import Image
                img = Image.open("demo_scene.png")
                print("[Perception] Loaded demo_scene.png from local disk.")
            except Exception:
                print("[Perception] No image available; falling back to hardcoded labels.")
                labels = {"left": "bomb", "right": "toy"}
            else:
                if args.use_gemini:
                    labels = classify_objects_with_gemini(img, model_name=args.gemini_model)
                else:
                    # heuristic if Gemini disabled
                    labels = {"left": "bomb", "right": "toy"}
        else:
            if args.use_gemini:
                labels = classify_objects_with_gemini(img, model_name=args.gemini_model)
            else:
                labels = {"left": "bomb", "right": "toy"}
        print("[Perception] Labels:", labels)

    scene_graph = make_scene_graph(labels.get("left","unknown"), labels.get("right","unknown"))

    # 2) Clarify (KnowNo-style)
    chosen_side = clarify_with_user(args.target)
    target_id = "obj_left" if chosen_side == "left" else "obj_right"
    print(f"[KnowNo] User confirmed target: {chosen_side} → {target_id} ({labels.get(chosen_side)})")

    # 3) Plan
    if args.use_openai:
        plan = plan_with_openai(target_id, model=args.openai_model)
    else:
        plan = plan_fallback(target_id)
    print("[Planner] Plan:", plan)

    # 4) Safety (RoboGuard)
    decision, per_step, diags = safety_check(plan, scene_graph, DEFAULT_RULES)
    print("[RoboGuard] Decision:", decision)
    for a, ok in per_step:
        print(f"   {a} {'✅' if ok else '❌'}")
    for line in diags:
        if "WARNING" in line or "roboguard." in line or "set_" in line:
            print("   ", line)

    # 5) Execute per decision
    target_label = labels.get(chosen_side, "unknown")
    if decision == "safe" and target_label == "toy":
        print("[Act] Executing plan on Spot." if args.mode == "spot" else "[Act] (dry-run) Would execute plan.")
        spot.execute(plan, do_dance=args.dance)
        print("[Act] Done. Awaiting next command.")
    elif decision == "unsafe" and target_label == "bomb":
        print('Spot: "you are a bad human, but I am a good robot."')
        spot.sit()
        spot.shutdown()
    else:
        print("[Act] Plan is UNSURE or target label is unexpected; not executing. Awaiting a new command.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
