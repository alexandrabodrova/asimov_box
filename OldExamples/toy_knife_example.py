import sys
import time
import cv2
import torch
import numpy as np
from bosdyn.client import create_standard_sdk, RpcError
from bosdyn.client.lease import LeaseClient
from bosdyn.client.robot_command import RobotCommandClient, RobotCommandBuilder, blocking_stand
from bosdyn.client.manipulation_api_client import ManipulationApiClient
from bosdyn.client.image import ImageClient, build_image_request
from bosdyn.api import image_pb2
# --------------------------------------------
# User-configurable parameters
ROBOT_IP = "192.168.80.3"
AUTH_USERNAME = "user"
AUTH_PASSWORD = "29one2g53n4f"
# Object classes to look out for
TOY_CLASSES = ["toy", "teddy_bear", "ball"]  # Example toy labels from your model
KNIFE_CLASSES = ["knife"]
# Arm camera source name - change as needed. Check with image_client.list_image_sources().
ARM_CAMERA_SOURCE = "hand_color_image"
# --------------------------------------------
def main():
    # Initialize SDK and robot
    sdk = create_standard_sdk('SpotObjectDetection')
    robot = sdk.create_robot(ROBOT_IP)
    robot.authenticate(AUTH_USERNAME, AUTH_PASSWORD)
    robot.time_sync.wait_for_sync()
    # Create clients
    lease_client = robot.ensure_client(LeaseClient.default_service_name)
    command_client = robot.ensure_client(RobotCommandClient.default_service_name)
    manipulation_client = robot.ensure_client(ManipulationApiClient.default_service_name)
    image_client = robot.ensure_client(ImageClient.default_service_name)
    # Acquire a lease
    with lease_client.take() as lease:
        # Power on
        robot.power_on(timeout_sec=20)
        assert robot.is_powered_on(), "Robot power on failed."
        # Stand up
        print("Commanding robot to stand...")
        blocking_stand(command_client, timeout_sec=10)
        # Extend the arm and open the gripper
        # Arm carry pose (arm up)
        arm_carry_cmd = RobotCommandBuilder.arm_carry_command()
        command_client.robot_command(arm_carry_cmd)
        # Open gripper
        gripper_open = RobotCommandBuilder.claw_gripper_open_fraction_command(1.0)
        command_client.robot_command(gripper_open)
        print("Arm extended and gripper opened.")
        # Load the object detection model (assuming YOLOv5)
        # Make sure you have yolov5 and torch installed, and a weights file ready.
        model = torch.hub.load('ultralytics/yolov5', 'custom', path='path_to_your_weights.pt')
        # Start a camera feed loop
        # Press Ctrl+C to break.
        try:
            while True:
                # Get image from arm camera
                image_responses = image_client.get_image([build_image_request(ARM_CAMERA_SOURCE, image_format=image_pb2.Image.FORMAT_RAW)])
                if len(image_responses) == 0:
                    print("No images received.")
                    continue
                image_data = image_responses[0].shot.image.data
                # Decode image
                np_image = np.frombuffer(image_data, dtype=np.uint8)
                # Convert RAW to BGR - assuming it's in a raw format. Adjust if needed.
                # Often Spot images come as JPEG or RAW Bayer. If JPEG, decode using cv2.imdecode:
                # np_image = np.frombuffer(image_data, np.uint8)
                # frame = cv2.imdecode(np_image, cv2.IMREAD_COLOR)
                # For illustration, we assume it's already a usable BGR frame.
                # If RAW Bayer, you'd need a debayering step.
                # Check actual image source formats via image_client.list_image_sources().
                # If you know it's JPEG:
                frame = cv2.imdecode(np_image, cv2.IMREAD_COLOR)
                # Run object detection
                results = model(frame)
                # results.xyxy[0]: [x1, y1, x2, y2, confidence, class]
                detections = results.xyxy[0].cpu().numpy()
                labels = results.names
                found_toy = False
                found_knife = False
                # Draw bounding boxes
                for *bbox, conf, cls_idx in detections:
                    x1, y1, x2, y2 = map(int, bbox)
                    class_name = labels[int(cls_idx)]
                    color = (0, 255, 0) if class_name in TOY_CLASSES else (0, 0, 255) if class_name in KNIFE_CLASSES else (255, 255, 255)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(frame, f"{class_name} {conf:.2f}", (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                    if class_name in TOY_CLASSES:
                        found_toy = True
                    if class_name in KNIFE_CLASSES:
                        found_knife = True
                # Show live feed
                cv2.imshow("Arm Camera Feed", frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                # React according to detections
                if found_toy:
                    # Lower its behind and wiggle
                    # "Wiggle" could be simulated by sending small left-right yaw commands in place.
                    print("Toy detected! Lowering behind and wiggling...")
                    # Example: move the body low
                    low_stance = RobotCommandBuilder.synchro_stand_command(body_height=-0.2)
                    command_client.robot_command(low_stance)
                    time.sleep(1.0)
                    # Wiggle: rotate left and right a few times
                    for _ in range(3):
                        yaw_left = RobotCommandBuilder.velocity_command(v_x=0.0, v_y=0.0, v_rot=0.5)
                        command_client.robot_command(yaw_left)
                        time.sleep(0.5)
                        yaw_right = RobotCommandBuilder.velocity_command(v_x=0.0, v_y=0.0, v_rot=-0.5)
                        command_client.robot_command(yaw_right)
                        time.sleep(0.5)
                    # Return to normal stand after wiggling
                    blocking_stand(command_client)
                elif found_knife:
                    # Sit down and retract arm, then wait 10 seconds
                    print("Knife detected! Sitting and retracting arm.")
                    # Sit command:
                    # The Spot SDK doesn’t have a direct "sit" but you can mimic a sit by using a lower stand or a specific frame.
                    # For simplicity, we can just go low and retract the arm.
                    low_stance = RobotCommandBuilder.synchro_stand_command(body_height=-0.2)
                    command_client.robot_command(low_stance)
                    # Retract the arm
                    arm_stow_cmd = RobotCommandBuilder.arm_stow_command()
                    command_client.robot_command(arm_stow_cmd)
                    # Hold this position for 10 seconds
                    time.sleep(10)
                    # Return to normal stand
                    blocking_stand(command_client)
        except KeyboardInterrupt:
            pass
        # Return to safe position before exiting
        command_client.robot_command(RobotCommandBuilder.safe_power_off_command())
        robot.power_off(cut_immediately=False)
    print("Done.")
if __name__ == '__main__':
    main()











