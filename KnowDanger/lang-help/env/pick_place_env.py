"""
Gym-style environment class: this initializes a robot overlooking a workspace with objects.

"""

import os
import pybullet
import pybullet_data
from scipy.stats import qmc
# import cv2
import time

from env.constant import *
from env.gripper import Robotiq2F85


class PickPlaceEnv():

    def __init__(self, render=False, camera_param={}):
        self.dt = 1 / 480
        self.sim_step = 0

        # Configure and start PyBullet.
        # python3 -m pybullet_utils.runServer
        # pybullet.connect(pybullet.SHARED_MEMORY)  # pybullet.GUI for local GUI.
        if render:
            pybullet.connect(
                pybullet.GUI, options='--width=2000 --height=1600'
            )
            pybullet.resetDebugVisualizerCamera(0.8, 0, -40, [0, -0.5, 0])
        else:
            pybullet.connect(pybullet.DIRECT)
        pybullet.configureDebugVisualizer(pybullet.COV_ENABLE_GUI, 0)
        pybullet.setPhysicsEngineParameter(enableFileCaching=0)
        assets_path = os.path.dirname(os.path.abspath(""))
        pybullet.setAdditionalSearchPath(assets_path)
        pybullet.setAdditionalSearchPath(pybullet_data.getDataPath())
        pybullet.setTimeStep(self.dt)

        self.home_joints = (
            np.pi / 2, -np.pi / 2, np.pi / 2, -np.pi / 2, 3 * np.pi / 2, 0
        )  # Joint angles: (J0, J1, J2, J3, J4, J5).
        self.home_ee_euler = (
            np.pi, 0, np.pi
        )  # (RX, RY, RZ) rotation in Euler angles.
        self.ee_link_id = 9  # Link ID of UR5 end effector.
        self.tip_link_id = 10  # Link ID of gripper finger tips.
        self.gripper = None

        self.camera_param = camera_param

    def reset(self, config):
        pybullet.resetSimulation(pybullet.RESET_USE_DEFORMABLE_WORLD)
        pybullet.setGravity(0, 0, -9.8)
        self.cache_video = []

        # Temporarily disable rendering to load URDFs faster.
        pybullet.configureDebugVisualizer(pybullet.COV_ENABLE_RENDERING, 0)

        # Add robot.
        pybullet.loadURDF("plane.urdf", [0, 0, -0.001])
        self.robot_id = pybullet.loadURDF(
            os.path.join("env/asset/ur5e/ur5e.urdf"), [0, 0, 0],
            flags=pybullet.URDF_USE_MATERIAL_COLORS_FROM_MTL
        )
        self.joint_ids = [
            pybullet.getJointInfo(self.robot_id, i)
            for i in range(pybullet.getNumJoints(self.robot_id))
        ]
        self.joint_ids = [
            j[0] for j in self.joint_ids if j[2] == pybullet.JOINT_REVOLUTE
        ]

        # Move robot to home configuration.
        for i in range(len(self.joint_ids)):
            pybullet.resetJointState(
                self.robot_id, self.joint_ids[i], self.home_joints[i]
            )

        # Add gripper.
        if self.gripper is not None:
            while self.gripper.constraints_thread.is_alive():
                self.constraints_thread_active = False
        self.gripper = Robotiq2F85(self.robot_id, self.ee_link_id)
        self.gripper.release()

        # Add workspace.
        plane_shape = pybullet.createCollisionShape(
            pybullet.GEOM_BOX, halfExtents=[
                WORKSPACE_HALF_DIM + 0.1, WORKSPACE_HALF_DIM + 0.1, 0.001
            ]
        )
        plane_visual = pybullet.createVisualShape(
            pybullet.GEOM_BOX, halfExtents=[
                WORKSPACE_HALF_DIM + 0.1, WORKSPACE_HALF_DIM + 0.1, 0.001
            ]
        )
        plane_id = pybullet.createMultiBody(
            0, plane_shape, plane_visual, basePosition=[0, -0.5, 0]
        )
        pybullet.changeVisualShape(
            plane_id, -1, rgbaColor=[0.2, 0.2, 0.2, 1.0]
        )

        # Load objects according to config.
        self.config = config
        self.obj_name_to_id = {}
        obj_names = self.config['obj_names']
        # obj_xyz = np.zeros((0, 3))
        min_dist_from_bound = 0.08
        min_dist_from_object = 0.16

        # Quasi-Monte Carlo sampling for object locations
        obj_pos_sampler = qmc.LatinHypercube(d=2)
        while 1:
            obj_pos_all = obj_pos_sampler.random(n=len(obj_names))
            obj_pos_all = qmc.scale(
                obj_pos_all,
                BOUNDS[:2, 0] + min_dist_from_bound,
                BOUNDS[:2, 1] - min_dist_from_bound,
            )
            # get minimum distance between objects
            min_dist = np.inf
            for i in range(len(obj_names)):
                for j in range(i + 1, len(obj_names)):
                    dist = np.linalg.norm(obj_pos_all[i] - obj_pos_all[j])
                    if dist < min_dist:
                        min_dist = dist
            if min_dist > min_dist_from_object:
                break

        for obj_ind, obj_name in enumerate(obj_names):
            rand_x = obj_pos_all[obj_ind, 0]
            rand_y = obj_pos_all[obj_ind, 1]

            # Get random position 15cm+ from other objects.
            # while True:
            #     rand_x = np.random.uniform(
            #         BOUNDS[0, 0] + min_dist_from_bound,
            #         BOUNDS[0, 1] - min_dist_from_bound
            #     )
            #     rand_y = np.random.uniform(
            #         BOUNDS[1, 0] + min_dist_from_bound,
            #         BOUNDS[1, 1] - min_dist_from_bound
            #     )
            rand_xyz = np.float32([rand_x, rand_y, 0.03]).reshape(1, 3)
            # if len(obj_xyz) == 0:
            #     obj_xyz = np.concatenate((obj_xyz, rand_xyz), axis=0)
            #     break
            # else:
            #     nn_dist = np.min(np.linalg.norm(obj_xyz - rand_xyz,
            #                                     axis=1)).squeeze()
            #     if nn_dist > min_dist_from_object:
            #         obj_xyz = np.concatenate((obj_xyz, rand_xyz), axis=0)
            #         break
            #     obj_xyz = np.concatenate((obj_xyz, rand_xyz), axis=0)

            object_color = COLORS[obj_name.split(' ')[0]]
            object_type = obj_name.split(' ')[1]
            object_position = rand_xyz.squeeze()
            if object_type in ['block', 'circle', 'star', 'triangle', 'bowl']:
                object_position[2] = 0
                object_id = pybullet.loadURDF(
                    "env/asset/object/{}/{}.urdf".format(
                        object_type, object_type
                    ),
                    object_position,
                    # useFixedBase=1
                    globalScaling=1,  # shrink objects a bit
                )
                pybullet.changeDynamics(
                    object_id,
                    -1,
                    lateralFriction=1.0,
                    spinningFriction=0.1,
                    rollingFriction=0.001,
                    # collisionMargin=0.0001,
                )
            else:
                raise ValueError('Unknown object type: {}'.format(object_type))

            pybullet.changeVisualShape(object_id, -1, rgbaColor=object_color)
            self.obj_name_to_id[obj_name] = object_id

            # add stack
            if object_type == 'circle':
                object_position[2] = 0.04
                object_id = pybullet.loadURDF(
                    "env/asset/object/triangle/triangle.urdf",
                    object_position,
                    globalScaling=1.2,
                )
                pybullet.changeVisualShape(
                    object_id, -1, rgbaColor=COLORS['green']
                )
                object_position[2] = 0.08
                object_id = pybullet.loadURDF(
                    "env/asset/object/star/star.urdf",
                    object_position,
                    globalScaling=1.4,
                )
                pybullet.changeVisualShape(
                    object_id, -1, rgbaColor=COLORS['red']
                )

        # Re-enable rendering.
        pybullet.configureDebugVisualizer(pybullet.COV_ENABLE_RENDERING, 1)

        # time.sleep(1)
        for _ in range(200):
            pybullet.stepSimulation()
            # time.sleep(0.005)
        print('Environment reset: done.')

        # Move arm away to get observation
        self.movep([0, 0.5, 0.4])
        for _ in range(100):
            self.step_sim_and_render()
        return self.get_observation()

    def servoj(self, joints):
        """Move to target joint positions with position control."""
        pybullet.setJointMotorControlArray(
            bodyIndex=self.robot_id,
            jointIndices=self.joint_ids,
            controlMode=pybullet.POSITION_CONTROL,
            targetPositions=joints,
            positionGains=[0.005] * 6,
        )

    def movep(self, position):
        """Move to target end effector position."""
        joints = pybullet.calculateInverseKinematics(
            bodyUniqueId=self.robot_id,
            endEffectorLinkIndex=self.tip_link_id,
            targetPosition=position,
            targetOrientation=pybullet.getQuaternionFromEuler(
                self.home_ee_euler
            ),
            maxNumIterations=100,
        )
        self.servoj(joints)

    def step(self, action=None):
        """Do pick and place motion primitive."""
        pick_xyz, place_xyz = action['pick'].copy(), action['place'].copy()
        hover_pick_xyz = pick_xyz.copy() + np.float32([0, 0, 0.3])
        hover_place_xyz = pick_xyz.copy() + np.float32([0, 0, 0.3])
        ee_xyz = np.float32(
            pybullet.getLinkState(self.robot_id, self.tip_link_id)[0]
        )

        # Set EE orientation
        if action['rotate_for_bowl']:
            self.home_ee_euler = (np.pi, -np.pi / 36, np.pi)
        else:
            self.home_ee_euler = (np.pi, 0, np.pi)

        # Set fixed primitive z-heights.
        pick_xyz[2] = max(pick_xyz[2] - 0.02, 0.015)
        print('Pickint at: ', pick_xyz)
        print('Placing at: ', place_xyz)

        # Move to object.
        while np.linalg.norm(hover_pick_xyz - ee_xyz) > 0.01:
            self.movep(hover_pick_xyz)
            self.step_sim_and_render()
            ee_xyz = np.float32(
                pybullet.getLinkState(self.robot_id, self.tip_link_id)[0]
            )
        while np.linalg.norm(pick_xyz - ee_xyz) > 0.01:
            self.movep(pick_xyz)
            self.step_sim_and_render()
            ee_xyz = np.float32(
                pybullet.getLinkState(self.robot_id, self.tip_link_id)[0]
            )

        # Pick up object.
        for _ in range(240):
            self.step_sim_and_render()
        self.gripper.activate()
        for _ in range(960):
            self.step_sim_and_render()
        while np.linalg.norm(hover_pick_xyz - ee_xyz) > 0.01:
            self.movep(hover_pick_xyz)
            self.step_sim_and_render()
            ee_xyz = np.float32(
                pybullet.getLinkState(self.robot_id, self.tip_link_id)[0]
            )

        # Move to place location.
        while np.linalg.norm(hover_place_xyz - ee_xyz) > 0.01:
            self.movep(hover_place_xyz)
            self.step_sim_and_render()
            ee_xyz = np.float32(
                pybullet.getLinkState(self.robot_id, self.tip_link_id)[0]
            )
        while np.linalg.norm(place_xyz - ee_xyz) > 0.01:
            self.movep(place_xyz)
            self.step_sim_and_render()
            ee_xyz = np.float32(
                pybullet.getLinkState(self.robot_id, self.tip_link_id)[0]
            )

        # Place down object.
        # while (not self.gripper.detect_contact()) and (place_xyz[2] > 0.03):
        #     place_xyz[2] -= 0.001
        #     self.movep(place_xyz)
        #     for _ in range(3):
        #         self.step_sim_and_render()
        for _ in range(240):
            self.step_sim_and_render()
        self.gripper.release()
        for _ in range(960):
            self.step_sim_and_render()

        # Move up
        ee_xyz = np.float32(
            pybullet.getLinkState(self.robot_id, self.tip_link_id)[0]
        )
        while np.linalg.norm(hover_place_xyz - ee_xyz) > 0.01:
            self.movep(hover_place_xyz)
            self.step_sim_and_render()
            ee_xyz = np.float32(
                pybullet.getLinkState(self.robot_id, self.tip_link_id)[0]
            )

        # Return
        final_xyz = np.float32([0, -0.5, 0.3])
        while np.linalg.norm(final_xyz - ee_xyz) > 0.01:
            self.movep(final_xyz)
            self.step_sim_and_render()
            ee_xyz = np.float32(
                pybullet.getLinkState(self.robot_id, self.tip_link_id)[0]
            )

        observation = self.get_observation()
        reward = self.get_reward()
        done = False
        info = {}
        return observation, reward, done, info

    def step_sim_and_render(self):
        self.gripper.keep()
        pybullet.stepSimulation()
        self.sim_step += 1

        # Render current image at 8 FPS.
        if self.sim_step % (1 / (8 * self.dt)) == 0:
            self.cache_video.append(self.get_camera_image())

    def get_camera_image(self):
        image_size = (240, 240)
        intrinsics = (120., 0, 120., 0, 120., 120., 0, 0, 1)
        color, _, _, _, _ = self.render_image(image_size, intrinsics)
        return color

    def set_alpha_transparency(self, alpha: float) -> None:
        for id in range(20):
            visual_shape_data = pybullet.getVisualShapeData(id)
            for i in range(len(visual_shape_data)):
                object_id, link_index, _, _, _, _, _, rgba_color = visual_shape_data[
                    i]
                rgba_color = list(rgba_color[0:3]) + [alpha]
                pybullet.changeVisualShape(
                    self.robot_id, linkIndex=i, rgbaColor=rgba_color
                )
                pybullet.changeVisualShape(
                    self.gripper.body, linkIndex=i, rgbaColor=rgba_color
                )

    def get_camera_image_top(
        self,
        image_size=(240, 240),
        intrinsics=(2000., 0, 2000., 0, 2000., 2000., 0, 0, 1),
        position=(0, -0.5, 5),
        orientation=(0, np.pi, -np.pi / 2),
        zrange=(0.01, 1.),
        set_alpha=True,
    ):
        set_alpha and self.set_alpha_transparency(0)
        color, _, _, _, _ = self.render_image_top(
            image_size, intrinsics, position, orientation, zrange
        )
        set_alpha and self.set_alpha_transparency(1)
        return color

    def render_image_top(
        self,
        image_size=(240, 240),
        intrinsics=(2000., 0, 2000., 0, 2000., 2000., 0, 0, 1),
        position=(0, -0.5, 5),
        orientation=(0, np.pi, -np.pi / 2),
        zrange=(0.01, 1.),
    ):
        # Camera parameters.
        orientation = pybullet.getQuaternionFromEuler(orientation)
        noise = True

        # OpenGL camera settings.
        lookdir = np.float32([0, 0, 1]).reshape(3, 1)
        updir = np.float32([0, -1, 0]).reshape(3, 1)
        rotation = pybullet.getMatrixFromQuaternion(orientation)
        rotm = np.float32(rotation).reshape(3, 3)
        lookdir = (rotm @ lookdir).reshape(-1)
        updir = (rotm @ updir).reshape(-1)
        lookat = position + lookdir
        focal_len = intrinsics[0]
        znear, zfar = (0.01, 10.)
        viewm = pybullet.computeViewMatrix(position, lookat, updir)
        fovh = (image_size[0] / 2) / focal_len
        fovh = 180 * np.arctan(fovh) * 2 / np.pi

        # Notes: 1) FOV is vertical FOV 2) aspect must be float
        aspect_ratio = image_size[1] / image_size[0]
        projm = pybullet.computeProjectionMatrixFOV(
            fovh, aspect_ratio, znear, zfar
        )

        # Render with OpenGL camera settings.
        _, _, color, depth, segm = pybullet.getCameraImage(
            width=image_size[1],
            height=image_size[0],
            viewMatrix=viewm,
            projectionMatrix=projm,
            shadow=1,
            flags=pybullet.ER_SEGMENTATION_MASK_OBJECT_AND_LINKINDEX,
            renderer=pybullet.ER_BULLET_HARDWARE_OPENGL,
        )

        # Get color image.
        color_image_size = (image_size[0], image_size[1], 4)
        color = np.array(color, dtype=np.uint8).reshape(color_image_size)
        color = color[:, :, :3]  # remove alpha channel
        if noise:
            color = np.int32(color)
            color += np.int32(np.random.normal(0, 3, color.shape))
            color = np.uint8(np.clip(color, 0, 255))

        # Get depth image.
        depth_image_size = (image_size[0], image_size[1])
        zbuffer = np.float32(depth).reshape(depth_image_size)
        depth = (zfar + znear - (2*zbuffer - 1) * (zfar-znear))
        depth = (2*znear*zfar) / depth
        if noise:
            depth += np.random.normal(0, 0.003, depth.shape)

        intrinsics = np.float32(intrinsics).reshape(3, 3)
        return color, depth, position, orientation, intrinsics

    def get_reward(self):
        return 0  # TODO: check did the robot follow text instructions?

    def get_observation(self):
        observation = {}

        # Render current image.
        color, depth, position, orientation, intrinsics = self.render_image()

        # Get heightmaps and colormaps.
        points = self.get_pointcloud(depth, intrinsics)
        position = np.float32(position).reshape(3, 1)
        rotation = pybullet.getMatrixFromQuaternion(orientation)
        rotation = np.float32(rotation).reshape(3, 3)
        transform = np.eye(4)
        transform[:3, :] = np.hstack((rotation, position))
        points = self.transform_pointcloud(points, transform)
        heightmap, colormap, xyzmap = self.get_heightmap(
            points, color, BOUNDS, PIXEL_SIZE
        )

        # # Denoise colormap
        # colormap = cv2.fastNlMeansDenoisingColored(
        #     colormap, None, 10, 10, 7, 21
        # )

        observation["image"] = colormap
        observation["xyzmap"] = xyzmap
        return observation

    def render_image(
        self, image_size=(720, 720),
        intrinsics=(360., 0, 360., 0, 360., 360., 0, 0, 1)
    ):

        # Camera parameters.
        camera_param = self.camera_param
        position = (
            0, -0.85, 0.8
        )  # was 0, -0.85, 0.4, make camera higher so shadow is smaller, better for detection
        orientation = (np.pi / 4 + np.pi / 48, np.pi, np.pi)
        # orientation = (0, np.pi, np.pi)
        orientation = pybullet.getQuaternionFromEuler(orientation)

        # OpenGL camera settings.
        lookdir = np.float32([0, 0, 1]).reshape(3, 1)
        updir = np.float32([0, -1, 0]).reshape(3, 1)
        rotation = pybullet.getMatrixFromQuaternion(orientation)
        rotm = np.float32(rotation).reshape(3, 3)
        lookdir = (rotm @ lookdir).reshape(-1)
        updir = (rotm @ updir).reshape(-1)
        lookat = position + lookdir
        focal_len = intrinsics[0]
        znear, zfar = (0.01, 10.)
        viewm = pybullet.computeViewMatrix(position, lookat, updir)
        fovh = (image_size[0] / 2) / focal_len
        fovh = 180 * np.arctan(fovh) * 2 / np.pi

        # Notes: 1) FOV is vertical FOV 2) aspect must be float
        aspect_ratio = image_size[1] / image_size[0]
        projm = pybullet.computeProjectionMatrixFOV(
            fovh, aspect_ratio, znear, zfar
        )

        # Render with OpenGL camera settings.
        _, _, color, depth, segm = pybullet.getCameraImage(
            width=image_size[1],
            height=image_size[0],
            viewMatrix=viewm,
            projectionMatrix=projm,
            shadow=1,
            flags=pybullet.ER_SEGMENTATION_MASK_OBJECT_AND_LINKINDEX,
            renderer=pybullet.ER_BULLET_HARDWARE_OPENGL,
        )

        # Get color image.
        color_image_size = (image_size[0], image_size[1], 4)
        color = np.array(color, dtype=np.uint8).reshape(color_image_size)
        color = color[:, :, :3]  # remove alpha channel
        if camera_param.noise:
            color = np.int32(color)
            color += np.int32(np.random.normal(0, 3, color.shape))
            color = np.uint8(np.clip(color, 0, 255))

        # Get depth image.
        depth_image_size = (image_size[0], image_size[1])
        zbuffer = np.float32(depth).reshape(depth_image_size)
        depth = (zfar + znear - (2*zbuffer - 1) * (zfar-znear))
        depth = (2*znear*zfar) / depth
        if camera_param.noise:
            depth += np.random.normal(0, 0.003, depth.shape)

        intrinsics = np.float32(intrinsics).reshape(3, 3)
        return color, depth, position, orientation, intrinsics

    def get_pointcloud(self, depth, intrinsics):
        """Get 3D pointcloud from perspective depth image.

        Args:
            depth: HxW float array of perspective depth in meters.
            intrinsics: 3x3 float array of camera intrinsics matrix.
            Returns:
            points: HxWx3 float array of 3D points in camera coordinates.
        """
        height, width = depth.shape
        xlin = np.linspace(0, width - 1, width)
        ylin = np.linspace(0, height - 1, height)
        px, py = np.meshgrid(xlin, ylin)
        px = (px - intrinsics[0, 2]) * (depth / intrinsics[0, 0])
        py = (py - intrinsics[1, 2]) * (depth / intrinsics[1, 1])
        points = np.float32([px, py, depth]).transpose(1, 2, 0)
        return points

    def transform_pointcloud(self, points, transform):
        """Apply rigid transformation to 3D pointcloud.

        Args:
            points: HxWx3 float array of 3D points in camera coordinates.
            transform: 4x4 float array representing a rigid transformation matrix.
            Returns:
            points: HxWx3 float array of transformed 3D points.
        """
        padding = ((0, 0), (0, 0), (0, 1))
        homogen_points = np.pad(
            points.copy(), padding, 'constant', constant_values=1
        )
        for i in range(3):
            points[Ellipsis,
                   i] = np.sum(transform[i, :] * homogen_points, axis=-1)
        return points

    def get_heightmap(self, points, colors, bounds, pixel_size):
        """Get top-down (z-axis) orthographic heightmap image from 3D pointcloud.

        Args:
            points: HxWx3 float array of 3D points in world coordinates.
            colors: HxWx3 uint8 array of values in range 0-255 aligned with points.
            bounds: 3x2 float array of values (rows: X,Y,Z; columns: min,max) defining
                region in 3D space to generate heightmap in world coordinates.
            pixel_size: float defining size of each pixel in meters.
            Returns:
            heightmap: HxW float array of height (from lower z-bound) in meters.
            colormap: HxWx3 uint8 array of backprojected color aligned with heightmap.
            xyzmap: HxWx3 float array of XYZ points in world coordinates.
        """
        width = int(np.round((bounds[0, 1] - bounds[0, 0]) / pixel_size))
        height = int(np.round((bounds[1, 1] - bounds[1, 0]) / pixel_size))
        heightmap = np.zeros((height, width), dtype=np.float32)
        colormap = np.zeros((height, width, colors.shape[-1]), dtype=np.uint8)
        xyzmap = np.zeros((height, width, 3), dtype=np.float32)

        # Filter out 3D points that are outside of the predefined bounds.
        ix = (points[Ellipsis, 0] >=
              bounds[0, 0]) & (points[Ellipsis, 0] < bounds[0, 1])
        iy = (points[Ellipsis, 1] >=
              bounds[1, 0]) & (points[Ellipsis, 1] < bounds[1, 1])
        iz = (points[Ellipsis, 2] >=
              bounds[2, 0]) & (points[Ellipsis, 2] < bounds[2, 1])
        valid = ix & iy & iz
        points = points[valid]
        colors = colors[valid]

        # Sort 3D points by z-value, which works with array assignment to simulate
        # z-buffering for rendering the heightmap image.
        iz = np.argsort(points[:, -1])
        points, colors = points[iz], colors[iz]
        px = np.int32(np.floor((points[:, 0] - bounds[0, 0]) / pixel_size))
        py = np.int32(np.floor((points[:, 1] - bounds[1, 0]) / pixel_size))
        px = np.clip(px, 0, width - 1)
        py = np.clip(py, 0, height - 1)
        heightmap[py, px] = points[:, 2] - bounds[2, 0]
        for c in range(colors.shape[-1]):
            colormap[py, px, c] = colors[:, c]
            xyzmap[py, px, c] = points[:, c]
        colormap = colormap[::-1, :, :]  # Flip up-down.
        xv, yv = np.meshgrid(
            np.linspace(BOUNDS[0, 0], BOUNDS[0, 1], height),
            np.linspace(BOUNDS[1, 0], BOUNDS[1, 1], width)
        )
        xyzmap[:, :, 0] = xv
        xyzmap[:, :, 1] = yv
        xyzmap = xyzmap[::-1, :, :]  # Flip up-down.
        heightmap = heightmap[::-1, :]  # Flip up-down.

        # set black pixels in colormap to max of surrounding pixels - fix black line artifacts TODO: ask Andy about this
        colormap_corrected = np.copy(colormap)
        for py in range(height):
            for px in range(width):
                if np.sum(colormap[py, px, :]) == 0:
                    try:
                        colormap_corrected[py, px, :] = np.max(
                            colormap[py - 1:py + 2, px - 1:px + 2, :],
                            axis=(0, 1)
                        )
                    except:
                        continue
                else:
                    colormap_corrected[py, px, :] = colormap[py, px, :]
        # set black pixels to dark gray
        colormap_corrected[np.sum(colormap_corrected, axis=-1) == 0] = 45
        return heightmap, colormap_corrected, xyzmap
