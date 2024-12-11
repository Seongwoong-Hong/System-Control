"""
Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.

NVIDIA CORPORATION and its licensors retain all intellectual property
and proprietary rights in and to this software, related documentation
and any modifications thereto. Any use, reproduction, disclosure or
distribution of this software and related documentation without an express
license agreement from NVIDIA CORPORATION is strictly prohibited.


Apply Forces (apply_forces.py)
----------------------------
This example shows how to apply forces and torques to rigid bodies using the tensor API.
"""
import os
import random

from isaacgym import gymutil
from isaacgym import gymapi
from isaacgym import gymtorch
from isaacgymenvs.utils.torch_jit_utils import *

import numpy as np
import torch
from matplotlib import pyplot as plt

from common.path_config import MAIN_DIR

# initialize gym
gym = gymapi.acquire_gym()

# parse arguments
args = gymutil.parse_arguments(
    description="Example of applying forces and torques to bodies")

# configure sim
sim_params = gymapi.SimParams()
sim_params.up_axis = gymapi.UP_AXIS_Z
sim_params.gravity = gymapi.Vec3(0.0, 0.0, -9.81)
if args.physics_engine == gymapi.SIM_PHYSX:
    sim_params.substeps = 1
    sim_params.physx.solver_type = 1
    sim_params.physx.num_position_iterations = 4
    sim_params.physx.num_velocity_iterations = 1
    sim_params.physx.num_threads = args.num_threads
    sim_params.physx.use_gpu = args.use_gpu
elif args.physics_engine == gymapi.SIM_FLEX and not args.use_gpu_pipeline:
    sim_params.flex.shape_collision_margin = 0.25
    sim_params.flex.num_outer_iterations = 4
    sim_params.flex.num_inner_iterations = 10
else:
    raise Exception("GPU pipeline is only available with PhysX")

# set random seed
seed = 100
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
os.environ['PYTHONHASHSEED'] = str(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

sim_params.use_gpu_pipeline = args.use_gpu_pipeline
device = args.sim_device if args.use_gpu_pipeline else 'cpu'

sim = gym.create_sim(args.compute_device_id, args.graphics_device_id, args.physics_engine, sim_params)
if sim is None:
    raise Exception("Failed to create sim")

# add ground plane
plane_params = gymapi.PlaneParams()
plane_params.normal = gymapi.Vec3(0, 0, 1)
gym.add_ground(sim, plane_params)

# set up the env grid
num_envs = 10
num_per_row = int(np.sqrt(num_envs))
env_spacing = 2.0
env_lower = gymapi.Vec3(-env_spacing, -env_spacing, 0.0)
env_upper = gymapi.Vec3(env_spacing, env_spacing, env_spacing)

# load ball asset
asset_root = str(MAIN_DIR / "gym_envs" / "envs" / "assets")
asset_file = "IDP_isaacgym.xml"

asset_options = gymapi.AssetOptions()
asset_options.fix_base_link = True
asset = gym.load_asset(sim, asset_root, asset_file, asset_options)

num_dof = gym.get_asset_dof_count(asset)
num_bodies = gym.get_asset_rigid_body_count(asset)

actuator_props = gym.get_asset_actuator_properties(asset)
motor_efforts = [prop.motor_effort for prop in actuator_props]
joint_gears = to_torch(motor_efforts, device=device)

# default pose
pose = gymapi.Transform()

envs = []
handles = []
for i in range(num_envs):
    # create env
    env = gym.create_env(sim, env_lower, env_upper, num_per_row)
    ahandle = gym.create_actor(env, asset, pose, "actor", i, 1)
    dof_props = gym.get_actor_dof_properties(env, ahandle)
    dof_props['driveMode'][0] = gymapi.DOF_MODE_EFFORT
    dof_props['driveMode'][1] = gymapi.DOF_MODE_EFFORT

    gym.set_actor_dof_properties(env, ahandle, dof_props)

    envs.append(env)
    handles.append(ahandle)

gym.prepare_sim(sim)

torque_amt = 100

frame_count = 0

from pyvirtualdisplay.smartdisplay import SmartDisplay
virtual_display = SmartDisplay(size=(1000, 800))
virtual_display.start()

# create viewer
viewer = gym.create_viewer(sim, gymapi.CameraProperties())
if viewer is None:
    raise Exception("Failed to create viewer")
cam_pos = gymapi.Vec3(0.0, -2.0, 0.8)
cam_target = gymapi.Vec3(0.0, 0.0, 0.8)
gym.viewer_camera_look_at(viewer, None, cam_pos, cam_target)

_ptb_range = np.arange(0.0, 0.0 + 12*0.015, 0.015)
_ptb_act_time = 0.3333333
dt = sim_params.dt
dof_state_tensor = gym.acquire_dof_state_tensor(sim)
dof_state = gymtorch.wrap_tensor(dof_state_tensor)
dof_pos = dof_state.view(num_envs, num_dof, 2)[..., 0]
dof_vel = dof_state.view(num_envs, num_dof, 2)[..., 1]
print(dof_state)
gym.simulate(sim)

env_ids = torch.arange(num_envs, device=device)

_ptb_idx = np.random.choice(range(len(_ptb_range)), size=len(env_ids), replace=True)
x_max = _ptb_range[_ptb_idx][None, ...]  # Backward direction(+)
t = _ptb_act_time * torch.linspace(0, 1, round(_ptb_act_time / dt))

A = np.array([[_ptb_act_time ** 3, _ptb_act_time ** 4, _ptb_act_time ** 5],
              [3, 4 * _ptb_act_time, 5 * _ptb_act_time ** 2],
              [6, 12 * _ptb_act_time, 20 * _ptb_act_time ** 2]])
b = np.concatenate([x_max, np.zeros_like(x_max), np.zeros_like(x_max)], axis=0)
a = torch.tensor(np.linalg.inv(A) @ b, dtype=torch.float)

t_con_mat = torch.concat([torch.ones([round(_ptb_act_time / dt), 1]), t.reshape(-1, 1), (t ** 2).reshape(-1, 1)], dim=1)
fddx = t * (t_con_mat @ torch.diag(torch.tensor([6.0, 12.0, 20.0])) @ a).T  # 가속도 5차 regression
# st_time_idx = th.randint(0, round(self.max_episode_length - self._ptb_act_time/self.dt), (len(env_ids),))
st_time_idx = torch.zeros(len(env_ids)).to(torch.int)
offsets = torch.arange(round(_ptb_act_time / dt)).unsqueeze(0) + st_time_idx.unsqueeze(1)
_ptb_acc = torch.zeros([len(env_ids), 600 + 1]).to(device)
_ptb_acc[torch.arange(len(env_ids)).unsqueeze(1), offsets] = fddx.to(device)

# self.dof_pos[env_ids, :] = (th.rand([len(env_ids), 2], dtype=th.float) * th.deg2rad(th.tensor([2.5, 5]))).to(self.device)
dof_pos[env_ids, :] = torch.deg2rad(torch.tensor([2.5, 5])).to(device)
dof_vel[env_ids, :] = torch.zeros(2, device=device, dtype=torch.float)
env_ids_int32 = env_ids.to(dtype=torch.int32)

gym.set_dof_state_tensor_indexed(sim,
                                 gymtorch.unwrap_tensor(dof_state),
                                 gymtorch.unwrap_tensor(env_ids_int32), len(env_ids_int32))

print(dof_state)
gym.refresh_dof_state_tensor(sim)
print(dof_state)
while not gym.query_viewer_has_closed(viewer):
    # update the viewer
    # gym.step_graphics(sim)
    # gym.draw_viewer(viewer, sim, True)

    # Wait for dt to elapse in real time.
    # This synchronizes the physics simulation with the rendering rate.
    # gym.sync_frame_time(sim)
    # step the physics
    gym.simulate(sim)
    gym.fetch_results(sim, True)

    gym.refresh_dof_state_tensor(sim)
    print(dof_state)
    frame_count += 1

gym.destroy_viewer(viewer)
gym.destroy_sim(sim)
