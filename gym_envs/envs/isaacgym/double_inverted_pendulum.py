import random
import string
from pathlib import Path
from typing import List, Union
from xml.etree.ElementTree import parse, ElementTree

from gym import spaces
from isaacgym import gymutil, gymtorch, gymapi
from isaacgymenvs.utils.torch_jit_utils import *
from isaacgymenvs.tasks.base.vec_task import VecTask

from scipy import io

import torch
import numpy as np
import os


class IDPMinEffort(VecTask):
    def __init__(self, cfg, rl_device, sim_device, graphics_device_id, headless, virtual_screen_capture, force_render):
        self.cfg = cfg
        self._set_env_cfg(**self.cfg['env'])
        if "delayed_time" in cfg['env']:
            self.act_delay_time = cfg['env']['delayed_time']
        else:
            self.act_delay_time = 0.1
        self.cfg["env"]["numObservations"] = 4
        self.cfg["env"]["numActions"] = 2
        if self.action_as_state:
            self.cfg['env']['numObservations'] += self.cfg['env']['numActions'] * round(self.act_delay_time / self.cfg['sim']['dt'])
        super().__init__(config=self.cfg, rl_device=rl_device, sim_device=sim_device, graphics_device_id=graphics_device_id, headless=headless, virtual_screen_capture=virtual_screen_capture, force_render=force_render)
        self.act_space = spaces.Box(np.ones(self.num_actions) * -1., np.ones(self.num_actions) * 1.)

        self.high = to_torch(np.deg2rad([45, 120]), dtype=torch.float, device=self.device)
        self.low = to_torch(np.deg2rad([-45, -60]), dtype=torch.float, device=self.device)
        if "exp_time" in self.cfg['env']:
            self.max_episode_length = round(self.cfg['env']['exp_time'] / self.dt)
        else:
            self.max_episode_length = round(5 / self.dt)
        self.obs_traj = to_torch(np.zeros([self.num_envs, self.max_episode_length + 1, self.cfg['env']['numObservations']]), device=self.device)
        self.act_traj = to_torch(np.zeros([self.num_envs, self.max_episode_length + 1, self.cfg['env']['numActions']]), device=self.device)
        self.tqr_traj = to_torch(np.zeros([self.num_envs, self.max_episode_length + 1, self.cfg['env']['numActions']]), device=self.device)

        self.cuda_arange = to_torch(np.arange(self.max_episode_length), dtype=torch.int64, device=self.device)
        self._ptb = to_torch(np.zeros([self.num_envs, self.max_episode_length + 1]), device=self.device)
        self.max_episode_length = to_torch(self.max_episode_length, dtype=torch.int64, device=self.device)
        self.ptb_forces = to_torch(np.zeros([self.num_envs, self.num_bodies, 3]), device=self.device)
        self._ptb_range = to_torch(self._cal_ptb_acc(-self._ptb_range.reshape(1, -1)), device=self.device)
        self._ptb_act_idx = to_torch(round(self._ptb_act_time / self.dt), dtype=torch.int64, device=self.device)
        # self._ptb_act_time = to_torch(self._ptb_act_time, device=self.device)
        self.ptb_st_idx = to_torch(np.zeros([self.num_envs, 1]), dtype=torch.int64, device=self.device)

        if 'upright_type' in cfg['env']:
            self.lean_angle = np.deg2rad(cfg['env']['upright_type'] * 1.0)
        else:
            self.lean_angle = 0.0
        self.lean_angle_torch = to_torch([self.lean_angle, self.lean_angle], device=self.device).repeat(self.num_envs, 1)
        self.act_delay_idx = to_torch(round(self.act_delay_time / self.dt) * np.ones([self.num_envs, 1]), dtype=torch.int64, device=self.device)

        self.is_act_delayed = to_torch(self.is_act_delayed, device=self.device)
        self._jnt_stiffness = to_torch(self._jnt_stiffness, device=self.device)
        self._jnt_damping = to_torch(self._jnt_damping, device=self.device)

        self.stcost_ratio = to_torch(self.stcost_ratio, device=self.device)
        self.tqcost_ratio = to_torch(self.tqcost_ratio, device=self.device)
        self.tqrate_ratio = to_torch(self.tqrate_ratio, device=self.device)
        self.const_ratio = to_torch(self.const_ratio, device=self.device)
        self.ank_ratio = to_torch([self.ank_ratio, 1 - self.ank_ratio], device=self.device)
        self.vel_ratio = to_torch(self.vel_ratio, device=self.device)
        self.tq_ratio = to_torch([self.tq_ratio, 1 - self.tq_ratio], device=self.device)
        self.ankle_limit = to_torch(self.ankle_limit, device=self.device)
        self.const_type = to_torch(self.const_type, device=self.device)
        self.cost_type = to_torch(self.cost_type, device=self.device)
        self.limLevel = to_torch(self.limLevel, device=self.device)
        if self.tqr_limit is not None:
            self.tqr_limit = to_torch([self.tqr_limit], device=self.device)

        if self.const_type == 0:
            self.const_max_val = to_torch([0.16, -0.04], device=self.device)
        elif self.const_type == 1:
            self.const_max_val = to_torch(self.clip_actions * np.array([0.4*self.ankle_torque_max, -self.ankle_torque_max]), device=self.device) / self.joint_gears[0]
        else:
            raise Exception("undefined constraint type")

        self.delayed_act_buf = to_torch(
            np.zeros([self.num_envs, self.action_space.shape[0], round((1 + self.delay_randomize) * self.act_delay_time / self.dt) + 1]), device=self.device)

        if self.viewer != None:
            cam_pos = gymapi.Vec3(0.0, -2.0, 0.8)
            cam_target = gymapi.Vec3(0.0, 0.0, 0.8)
            self.gym.viewer_camera_look_at(self.viewer, None, cam_pos, cam_target)

        self.actions = to_torch(np.zeros([self.num_envs, 2]), device=self.device)
        self.prev_actions = self.actions.clone()
        self.prev_passive_actions = self.actions.clone()
        dof_state_tensor = self.gym.acquire_dof_state_tensor(self.sim)
        self.dof_state = gymtorch.wrap_tensor(dof_state_tensor)
        self.dof_pos = self.dof_state.view(self.num_envs, self.num_dof, 2)[..., 0]
        self.dof_vel = self.dof_state.view(self.num_envs, self.num_dof, 2)[..., 1]
        force_state_tensor = self.gym.acquire_force_sensor_tensor(self.sim)
        self.foot_forces = gymtorch.wrap_tensor(force_state_tensor)
        self.extras["foot_forces"] = self.foot_forces
        self.extras["ptb_forces"] = self.ptb_forces
        self.extras["torque_rate"] = (self.actions - self.prev_actions) / self.dt
        self.extras["ddtq"] = (self.actions - self.prev_actions) / self.dt - self.extras["torque_rate"]
        self.extras["dd_acts"] = (self.actions - self.prev_actions) / self.dt - self.extras["torque_rate"]
        self.extras["sampled_action"] = self.actions.clone()

    def _set_env_cfg(
            self,
            bsp_path: Union[Path, str] = None,
            ptb_act_time: float = 1 / 3,
            ankle_limit: str = "satu",
            tqr_regularize_type: str = "torque_rate",
            stptb: float = None,
            edptb: float = None,
            ptb_step: float = None,
            ptb_type: str = "5th",
            stcost_ratio: float = 1.0,
            tqcost_ratio: float = 0.5,
            tqrate_ratio: float = 0.0,
            const_ratio: float = 1.0,
            ank_ratio: float = 0.5,
            vel_ratio: float = 0.5,
            tq_ratio: float = 0.5,
            avg_coeff: float = 1.0,
            const_type: str = "cop",
            limLevel: float = 0.0,
            use_seg_ang: bool = False,
            action_as_state: bool = False,
            delay: bool = False,
            delay_randomize: float = None,
            ankle_torque_max: Union[int, float] = None,
            stiff_ank: float = 0.,
            damp_ank: float = 0.,
            stiff_hip: float = 0.,
            damp_hip: float = 0.,
            tqr_limit: float = None,
            use_curriculum = False,
            cost_type = None,
            **kwargs
    ):
        self.limLevel = 10 ** (limLevel * ((-5) - (-2)) + (-2))
        self.asset_path = os.path.join(os.path.dirname(__file__), "..", "assets", "IDP_isaacgym.xml")

        self._ptb_idx = 0
        self._next_ptb_idx = self._ptb_idx
        self.ankle_limit = np.arange(3)[np.array(['satu', 'soft', 'hard']) == ankle_limit].item()
        self.const_type = np.arange(2)[np.array(['cop', 'ankle_torque']) == const_type].item()
        self.cost_type = np.arange(4)[np.array(['normal', 'com', 'reduced', 'cop']) == cost_type].item()
        if tqr_regularize_type not in ["torque_rate", "ddtq", "dd_acts"]:
            raise Exception("undefined tqr_regularize_type")
        self.tqr_regularize_type = tqr_regularize_type
        self.ankle_torque_max = ankle_torque_max
        self.use_seg_ang = use_seg_ang
        self.action_as_state = action_as_state
        self.is_act_delayed = delay
        self.delay_randomize = delay_randomize
        if self.delay_randomize is None:
            self.delay_randomize = 0.0
        elif not isinstance(self.delay_randomize, float):
            raise TypeError("delay_randomize는 float type 입니다.")

        self._jnt_stiffness = [stiff_ank, stiff_hip]
        self._jnt_damping = [damp_ank, damp_hip]
        self.tqr_limit = tqr_limit
        self.bsp = None
        if bsp_path is not None:
            if isinstance(bsp_path, str):
                bsp_path = Path(bsp_path)
            assert bsp_path.parent.exists()
            self.bsp = io.loadmat(str(bsp_path))['bsp']
        self._ptb_data_range = np.array([0.03, 0.045, 0.06, 0.075, 0.09, 0.12, 0.15])
        self._ptb_act_time = ptb_act_time
        self._ptb_type = ptb_type

        self.stcost_ratio = stcost_ratio
        self.tqcost_ratio = tqcost_ratio
        self.tqrate_ratio = tqrate_ratio
        self.const_ratio = const_ratio
        self.ank_ratio = ank_ratio
        self.vel_ratio = vel_ratio
        self.tq_ratio = tq_ratio
        self.avg_coeff = avg_coeff
        self.use_curriculum = use_curriculum
        assert 0 <= self.ank_ratio <= 1 and 0 <= self.tq_ratio <= 1 and 0 <= self.vel_ratio <= 1
        self._ptb_range = self._ptb_data_range.copy()
        if stptb is not None:
            self._ptb_range = np.arange(0, round((edptb - stptb)/ptb_step) + 1)*ptb_step + stptb

    def create_sim(self):
        # set the up axis to be z-up given that assets are y-up by default
        self.up_axis = self.cfg["sim"]["up_axis"]
        if self.bsp is not None:
            self.asset_path = self._set_body_config(self.asset_path, self.bsp)
        self.sim = super().create_sim(self.device_id, self.graphics_device_id, self.physics_engine, self.sim_params)
        self._create_ground_plane()
        self._create_envs(self.num_envs, self.cfg["env"]['envSpacing'], int(np.sqrt(self.num_envs)))
        if self.bsp is not None:
            os.remove(self.asset_path)

    def _create_ground_plane(self):
        plane_params = gymapi.PlaneParams()
        # set the normal force to be z dimension
        plane_params.normal = gymapi.Vec3(0.0, 0.0, 1.0) if self.up_axis == 'z' else gymapi.Vec3(0.0, 1.0, 0.0)
        self.gym.add_ground(self.sim, plane_params)

    def _create_envs(self, num_envs, spacing, num_per_row):
        # define plane on which environments are initialized
        lower = gymapi.Vec3(-spacing, -spacing, 0.0) if self.up_axis == 'z' else gymapi.Vec3(-spacing, 0.0, -spacing)
        upper = gymapi.Vec3(spacing, spacing, spacing)

        asset_root = os.path.dirname(self.asset_path)
        asset_file = os.path.basename(self.asset_path)

        asset_options = gymapi.AssetOptions()
        asset_options.fix_base_link = True
        idp_asset = self.gym.load_asset(self.sim, asset_root, asset_file, asset_options)
        self.num_dof = self.gym.get_asset_dof_count(idp_asset)
        self.num_bodies = self.gym.get_asset_rigid_body_count(idp_asset)
        # Note - for this asset we are loading the actuator info from the MJCF
        actuator_props = self.gym.get_asset_actuator_properties(idp_asset)
        motor_efforts = [prop.motor_effort for prop in actuator_props]
        self.joint_gears = to_torch(motor_efforts, device=self.device)

        pose = gymapi.Transform()

        body_idx = self.gym.find_asset_rigid_body_index(idp_asset, "foot")
        sensor_pose = gymapi.Transform(gymapi.Vec3(0.0, 0.0, 0.0))
        sensor_idx = self.gym.create_asset_force_sensor(idp_asset, body_idx, sensor_pose)

        self.idp_handles = []
        self.envs = []
        self.sensors = []
        for i in range(self.num_envs):
            # create env instance
            env_ptr = self.gym.create_env(
                self.sim, lower, upper, num_per_row
            )
            idp_handle = self.gym.create_actor(env_ptr, idp_asset, pose, "idp", i, 1, 0)
            self.gym.enable_actor_dof_force_sensors(env_ptr, idp_handle)
            dof_props = self.gym.get_actor_dof_properties(env_ptr, idp_handle)
            dof_props['driveMode'][0] = gymapi.DOF_MODE_EFFORT
            dof_props['driveMode'][1] = gymapi.DOF_MODE_EFFORT

            sensor = self.gym.get_actor_force_sensor(env_ptr, idp_handle, sensor_idx)
            self.gym.set_actor_dof_properties(env_ptr, idp_handle, dof_props)
            self.envs.append(env_ptr)
            self.idp_handles.append(idp_handle)
            self.sensors.append(sensor)

    def compute_reward(self):
        self.rew_buf[:], self.reset_buf[:] = compute_postural_reward(
            self.obs_buf,
            self.actions,
            self.extras[self.tqr_regularize_type],
            self.foot_forces,
            self.reset_buf, self.progress_buf, self.ptb_st_idx,
            self.stcost_ratio, self.tqcost_ratio, self.tqrate_ratio, self.const_ratio,
            self.ank_ratio, self.vel_ratio, self.tq_ratio,
            self.ankle_limit, self.const_max_val, self.limLevel,
            self.high, self.low, self.max_episode_length,
            self.com, self.mass, self.len, self.const_type, self.cost_type, self.tqr_limit
        )

    def compute_observations(self, env_ids=None):
        if env_ids is None:
            env_ids = np.arange(self.num_envs)

        self.gym.refresh_dof_state_tensor(self.sim)
        self.obs_buf[env_ids, 0] = self.dof_pos[env_ids, 0].squeeze()
        self.obs_buf[env_ids, 1] = self.dof_pos[env_ids, 1].squeeze()
        self.obs_buf[env_ids, 2] = self.dof_vel[env_ids, 0].squeeze()
        self.obs_buf[env_ids, 3] = self.dof_vel[env_ids, 1].squeeze()
        if self.action_as_state:
            self.obs_buf[env_ids, 4:] = self.delayed_act_buf[env_ids, :, :-1].permute(0, 2, 1).reshape(len(env_ids), -1)
        self.gym.refresh_force_sensor_tensor(self.sim)

        self.obs_traj[env_ids, self.progress_buf, :] = self.obs_buf
        self.act_traj[env_ids, self.progress_buf, :] = self.actions
        self.tqr_traj[env_ids, self.progress_buf, :] = self.extras['torque_rate']

        return self.obs_buf

    def reset(self):
        self.reset_idx(torch.arange(self.num_envs).to(self.device))
        return self.compute_observations()

    def reset_idx(self, env_ids):
        self._ptb[env_ids, :], self.ptb_st_idx[env_ids] = reset_ptb_acc(
            env_ids,
            self._ptb[env_ids],
            self._ptb_range,
            self._ptb_act_idx,
            self.max_episode_length,
            self.cuda_arange,
        )

        lean_angle = torch_rand_float(0, self.lean_angle, shape=(len(env_ids), 1), device=self.device)
        hip_ratio = torch_rand_float(0.0, 2.0, shape=(len(env_ids), 1), device=self.device)
        self.lean_angle_torch[env_ids, 0] = lean_angle[:, 0]
        self.lean_angle_torch[env_ids, 1] = hip_ratio[:, 0] * lean_angle[:, 0]
        self.dof_pos[env_ids, :] = self.lean_angle_torch[env_ids, :]
        self.dof_vel[env_ids, :] = 0.0
        env_ids_int32 = env_ids.to(dtype=torch.int32)

        self.gym.set_dof_state_tensor_indexed(self.sim,
                                              gymtorch.unwrap_tensor(self.dof_state),
                                              gymtorch.unwrap_tensor(env_ids_int32), len(env_ids_int32))

        act_delay_time = torch_rand_float(
            self.act_delay_time*(1 - self.delay_randomize),
            self.act_delay_time*(1 + self.delay_randomize),
            shape=(len(env_ids), 1), device=self.device,
        )
        self.act_delay_idx[env_ids] = (act_delay_time / self.dt).round().to(dtype=torch.int64, device=self.device)
        noise_time = torch_rand_float(-self.act_delay_time, 0, shape=(len(env_ids), 1), device=self.device)
        self.ptb_st_idx[env_ids] += (noise_time / self.dt - 1).round().to(dtype=torch.int64, device=self.device)

        self.delayed_act_buf[env_ids, ...] = fill_delayed_act_buf(
            self.dof_pos[env_ids, :],
            self.dof_vel[env_ids, :],
            self.mass,
            self.com,
            self.len,
            self._jnt_stiffness,
            self._jnt_damping,
            self.joint_gears,
        )
        self.reset_param_buffers(env_ids)

    def _cal_ptb_acc(self, x_max):
        if self._ptb_type == "exp":
            t1 = self._ptb_act_time * 1 / 5
            num_con_vel = round((self._ptb_act_time - 2 * t1) / self.dt)
            ts = np.arange(0, round(t1 / self.dt)) * self.dt
            A = np.array([[t1 ** 4 / 2 + 5 * t1 ** 3 / 24, 2 * t1 ** 3 / 3 + 5 * t1 ** 2 / 24], [3 * t1 ** 2, 2 * t1]])
            b = np.concatenate([x_max, np.zeros_like(x_max)], axis=0)
            a = np.linalg.inv(A) @ b
            t_con_mat = np.concatenate([3 * (ts ** 2).reshape(-1, 1), 2 * ts.reshape(-1, 1)], axis=1)
            fddx1 = t_con_mat @ a
            fddx = np.concatenate([fddx1, np.zeros([num_con_vel, x_max.shape[1]]), -fddx1], axis=0).T
            return fddx

        t = self._ptb_act_time * np.linspace(0, 1, round(self._ptb_act_time / self.dt))
        if self._ptb_type == "5th":
            A = np.array([[self._ptb_act_time ** 3, self._ptb_act_time ** 4, self._ptb_act_time ** 5],
                          [3, 4 * self._ptb_act_time, 5 * self._ptb_act_time ** 2],
                          [6, 12 * self._ptb_act_time, 20 * self._ptb_act_time ** 2]])
            b = np.concatenate([x_max, np.zeros_like(x_max), np.zeros_like(x_max)], axis=0)
            a = np.linalg.inv(A)@b

            t_con_mat = np.concatenate([np.ones([round(self._ptb_act_time/self.dt), 1]), t.reshape(-1, 1), (t**2).reshape(-1, 1)], axis=1)
            fddx = t*(t_con_mat @ np.diag(np.array([6.0, 12.0, 20.0])) @ a).T  # 가속도 5차 regression
        elif self._ptb_type == "3rd":
            fddx = ((6 * (1 - 2*t / self._ptb_act_time) / self._ptb_act_time**2)[:,None] * x_max).T
        else:
            raise Exception("Perturbation type not supported")

        return fddx

    def pre_physics_step(self, actions):
        env_ids = self.reset_buf.nonzero(as_tuple=False).squeeze(-1)

        if len(env_ids) > 0:
            self.reset_idx(env_ids)

        self.get_current_ptbs()
        actions = self.process_actions(actions)
        current_actions, self.delayed_act_buf[...] = compute_current_action(
            self.dof_pos,
            self.dof_vel,
            actions.to(self.device).clone(),
            self.delayed_act_buf,
            self.is_act_delayed,
            self.ptb_st_idx,
            self.progress_buf,
            self._jnt_stiffness,
            self._jnt_damping,
            self.joint_gears,
            self.lean_angle_torch,
            self.act_delay_idx
        )
        self.actions = current_actions.to(self.device).clone()
        self.extras['ddtq'], self.extras['dd_acts'], self.extras['torque_rate'] = _compute_extras_jit(
            self.delayed_act_buf,
            self.act_delay_idx,
            self.actions,
            self.prev_actions,
            self.extras['torque_rate'],
            self.extras['dd_acts'],
            self.dt,
            self.avg_coeff,
            self.device
        )
        forces = self.actions * self.joint_gears
        self.prev_actions = self.actions.clone()
        self.gym.apply_rigid_body_force_tensors(self.sim, gymtorch.unwrap_tensor(self.ptb_forces), None, gymapi.ENV_SPACE)
        self.gym.set_dof_actuation_force_tensor(self.sim, gymtorch.unwrap_tensor(forces))

    def post_physics_step(self):
        self.progress_buf += 1

        self.compute_observations()
        self.compute_reward()

    def _set_body_config(self, filepath, bsp):
        tl_f = bsp[0, 0] * 0.01
        m_f, l_f, com_f, I_f = bsp[1, :]
        m_u, l_u, com_u, I_u = bsp[6, :]
        m_s, l_s, com_s, I_s = bsp[2, :]
        m_t, l_t, com_t, I_t = bsp[3, :]
        l_l = l_s + l_t
        m_l = 2 * (m_s + m_t)
        com_l = (m_s * com_s + m_t * (l_s + com_t)) / (m_s + m_t)
        I_l = 2 * (I_s + m_s * (com_l - com_s) ** 2 + I_t + m_t * (com_l - (l_s + com_t)) ** 2)
        tree = parse(filepath)
        root = tree.getroot()
        offset = 0.1
        if self.use_seg_ang:
            root.find("compiler").attrib["coordinate"] = "global"
        f_body = root.find("worldbody").find("body")
        f_body.find('geom').attrib['fromto'] = f"{l_f - tl_f:.4f} 0 0 {l_f:.4f} 0 0"
        f_body.find('inertial').attrib['diaginertia'] = f"{2*I_f:.6f} {2*I_f:.6f} 0.001"
        f_body.find('inertial').attrib['mass'] = f"{2*m_f:.4f}"
        l_body = f_body.find("body")
        l_body.attrib["pos"] = f"0 0 {com_l + offset:.4f}"
        l_body.find("joint").attrib["pos"] = f"0 0 {-com_l:.4f}"
        l_body.find('geom').attrib['fromto'] = f"0 0 {-com_l:.4f} 0 0 {l_l - com_l :.4f}"
        l_body.find('inertial').attrib['diaginertia'] = f"{I_l:.6f} {I_l:.6f} 0.001"
        l_body.find('inertial').attrib['mass'] = f"{m_l:.4f}"
        u_body = l_body.find("body")
        u_body.attrib["pos"] = f"0 0 {com_u + l_l - com_l:.4f}"
        u_body.find("joint").attrib["pos"] = f"0 0 {-com_u:.4f}"
        u_body.find("geom").attrib["fromto"] = f"0 0 {-com_u:.4f} 0 0 {l_u - com_u:.4f}"
        u_body.find("inertial").attrib['diaginertia'] = f"{I_u:.6f} {I_u:.6f} 0.001"
        u_body.find("inertial").attrib['mass'] = f"{m_u:.4f}"

        if self.ankle_torque_max is not None:
            for motor in root.find('actuator').findall("motor"):
                if motor.get('name') == 'ank':
                    motor.attrib['gear'] = str(self.ankle_torque_max)
        m_tree = ElementTree(root)

        self.com = to_torch([com_f, com_l, com_u], device=self.device)
        self.mass = to_torch([m_f, m_l, m_u], device=self.device)
        self.len = to_torch([l_f, l_l, l_u], device=self.device)

        tmpname = ''.join(random.choice(string.ascii_letters + string.digits) for _ in range(6))
        dirname = Path(filepath).parent / "tmp"
        dirname.mkdir(parents=False, exist_ok=True)
        while (dirname / (tmpname + ".xml")).is_file():
            tmpname = ''.join(random.choice(string.ascii_letters + string.digits) for _ in range(6))
        filepath = str(dirname / (tmpname + ".xml"))

        m_tree.write(filepath)
        return filepath

    def process_actions(self, actions):
        # i, j, k = np.arange(self.num_envs).reshape(-1, 1, 1), np.arange(2).reshape(1, -1, 1), self.act_delay_idx.reshape(-1, 1, 1) - 1
        # self.extras['sampled_action'] = actions.clone()
        # return actions * self.avg_coeff + (1 - self.avg_coeff) * self.delayed_act_buf[i, j, k].to(self.device).clone().squeeze(-1)
        self.extras['sampled_action'] = actions.clone()
        return _process_actions_jit(
            actions,
            self.delayed_act_buf,
            self.act_delay_idx,
            self.avg_coeff,
            self.num_envs,
            self.device
        )

    def get_current_ptbs(self):
        self.ptb_forces = _get_current_ptbs_jit(self.mass, self._ptb, self.progress_buf, self.ptb_forces)

    def reset_param_buffers(self, env_ids):
        self.actions[env_ids] = self.delayed_act_buf[env_ids, :, 0].clone() - (self._jnt_stiffness*self.dof_pos[env_ids] + self._jnt_damping*self.dof_vel[env_ids])/self.joint_gears
        self.prev_actions[env_ids] = self.actions[env_ids].clone()
        if self.action_as_state:
            self.obs_buf[env_ids, 4:] = self.delayed_act_buf[env_ids, :, :-1].permute(0, 2, 1).reshape(len(env_ids), -1)
        self.reset_buf[env_ids] = 0
        self.progress_buf[env_ids] = 0
        self.obs_traj[env_ids] = 0
        self.act_traj[env_ids] = 0
        self.tqr_traj[env_ids] = 0
        self.extras['torque_rate'][env_ids] = 0
        self.extras['ddtq'][env_ids] = 0
        self.extras['sampled_action'][env_ids] = 0

    def update_curriculum(self, **kwargs):
        for k, v in kwargs.items():
            try:
                if isinstance(getattr(self, k), torch.Tensor):
                    if k == "_ptb_range":
                        self._ptb_range = to_torch(self._cal_ptb_acc(-v.reshape(1, -1)), device=self.device)
                    elif isinstance(v, (int, float)):
                        setattr(self, k, to_torch(v, dtype=torch.float32, device=self.device))
                    else:
                        if v.shape == getattr(self, k).shape:
                            setattr(self, k, to_torch(v, dtype=torch.float32, device=self.device))
                        else:
                            raise ValueError("Size didn't match")
                else:
                    setattr(self, k, v)
            except AttributeError:
                print(f"self.{k}: Attribute not found")


class IDPMinEffortDet(IDPMinEffort):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.max_episode_length = round(3 / self.dt)
        self._ptb = to_torch(np.zeros([self.num_envs, self.max_episode_length + 1]), device=self.device)
        self.max_episode_length = to_torch(self.max_episode_length, dtype=torch.int64, device=self.device)
        if self.lean_angle > 0.0:
            self._ptb_range = to_torch(
                self._cal_ptb_acc(-np.array([0.012, 0.024, 0.036, 0.048, 0.07, 0.09, 0.12]).reshape(1, -1)),
                device=self.device,
            )
        else:
            self._ptb_range = to_torch(
                self._cal_ptb_acc(-np.array([0.03, 0.045, 0.06, 0.075, 0.09, 0.12, 0.15]).reshape(1, -1)),
                # self._cal_ptb_acc(-np.array([0.12, 0.135, 0.15, 0.165, 0.18, 0.195, 0.21]).reshape(1, -1)),
                device=self.device,
            )
        self.ptb_idx = to_torch(np.arange(self.num_envs) % self._ptb_range.shape[0], dtype=torch.int64, device=self.device)
        self.delayed_time = 0.1
        if "st_ptb_idx" in kwargs['cfg']['env'].keys():
            self.ptb_idx += kwargs['cfg']['env']["st_ptb_idx"]

    def reset_idx(self, env_ids):
        st_idx = round(1 /(3 * self.dt))
        ed_idx = st_idx + self._ptb_range.shape[1]
        self.ptb_st_idx[env_ids] = st_idx
        self._ptb[env_ids] = 0
        self._ptb[env_ids, st_idx:ed_idx] = self._ptb_range[self.ptb_idx[env_ids.unsqueeze(1)], np.arange(self._ptb_range.shape[1])]
        self.ptb_idx[env_ids] = (self.ptb_idx[env_ids] + self.num_envs) % self._ptb_range.shape[0]

        self.dof_pos[env_ids, :] = self.lean_angle_torch[env_ids, :]
        self.dof_vel[env_ids, :] = 0.0
        env_ids_int32 = env_ids.to(dtype=torch.int32)

        self.gym.set_dof_state_tensor_indexed(self.sim,
                                              gymtorch.unwrap_tensor(self.dof_state),
                                              gymtorch.unwrap_tensor(env_ids_int32), len(env_ids_int32))

        self.delayed_act_buf[env_ids, ...] = fill_delayed_act_buf(
            self.dof_pos[env_ids, :],
            self.dof_vel[env_ids, :],
            self.mass,
            self.com,
            self.len,
            self._jnt_stiffness,
            self._jnt_damping,
            self.joint_gears,
        )
        self.reset_param_buffers(env_ids)


class IDPMinEffortHumanDet(IDPMinEffortDet):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._ptb_range = to_torch(
            self._cal_ptb_acc(-np.array([0.03, 0.045, 0.06, 0.075, 0.09, 0.12, 0.15]).reshape(1, -1)),
            device=self.device,
        )
        from common.path_config import MAIN_DIR
        subpath = MAIN_DIR / "demos" / "IDP" / "sub10" / "sub10"
        init_state = []
        for i in range(5):
            for j in range(7):
                humanData = io.loadmat(str(subpath) + f"i{(i + j*5 + 1)}.mat")
                init_state.append(-humanData["state"][:40, :2].mean(axis=0).squeeze())
        self.lean_angle_torch = to_torch(init_state, device=self.device)
        self.trial_idx = to_torch(np.arange(self.num_envs) % 35, dtype=torch.int64, device=self.device)


class IDPMinEffortHumanLeanDet(IDPMinEffortDet):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._ptb_range = to_torch(
            self._cal_ptb_acc(-np.array([0.012, 0.024, 0.036, 0.048, 0.07, 0.09, 0.12]).reshape(1, -1)),
            device=self.device,
        )
        from common.path_config import MAIN_DIR
        subpath = MAIN_DIR / "demos" / "IDPLean" / "sub10" / "sub10"
        init_state = []
        for i in range(7):
            st = np.zeros(2)
            for j in range(5):
                humanData = io.loadmat(str(subpath) + f"i{(i*5 + j + 1)}.mat")
                st += -humanData["state"][:40, :2].mean(axis=0).squeeze() / 5
            init_state.append(st)
        self.lean_angle_torch = to_torch(init_state, device=self.device)
        self.trial_idx = to_torch(np.arange(self.num_envs) % 7, dtype=torch.int64, device=self.device)


    def reset_idx(self, env_ids):
        st_idx = round(1 /(3 * self.dt))
        ed_idx = st_idx + self._ptb_range.shape[1]
        self.ptb_st_idx[env_ids] = st_idx
        self._ptb[env_ids] = 0
        self._ptb[env_ids, st_idx:ed_idx] = self._ptb_range[self.trial_idx[env_ids.unsqueeze(1)] % 7, np.arange(self._ptb_range.shape[1])]

        self.dof_pos[env_ids, :] = self.lean_angle_torch[self.trial_idx[env_ids], :]
        self.dof_vel[env_ids, :] = 0.0

        self.trial_idx[env_ids] = (self.trial_idx[env_ids] + self.num_envs) % self._ptb_range.shape[0]
        env_ids_int32 = env_ids.to(dtype=torch.int32)

        self.gym.set_dof_state_tensor_indexed(self.sim,
                                              gymtorch.unwrap_tensor(self.dof_state),
                                              gymtorch.unwrap_tensor(env_ids_int32), len(env_ids_int32))

        self.delayed_act_buf[env_ids, ...] = fill_delayed_act_buf(
            self.dof_pos[env_ids, :],
            self.dof_vel[env_ids, :],
            self.mass,
            self.com,
            self.len,
            self._jnt_stiffness,
            self._jnt_damping,
            self.joint_gears,
        )
        self.reset_param_buffers(env_ids)


class IDPLeanAndRelease(IDPMinEffort):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.max_episode_length = round(3 / self.dt)
        self.max_episode_length = to_torch(self.max_episode_length, dtype=torch.int64, device=self.device)
        self.init_vel = 10

    def reset_idx(self, env_ids):
        self.ptb_st_idx[...] = 0
        self._ptb[...] = 0

        th1 = torch_rand_float(np.deg2rad(0), np.deg2rad(2.5), shape=(len(env_ids), 1), device=self.device)
        th2 = torch_rand_float(np.deg2rad(-5), np.deg2rad(10), shape=(len(env_ids), 1), device=self.device)
        dth1 = torch_rand_float(np.deg2rad(-0.5*self.init_vel), np.deg2rad(self.init_vel), shape=(len(env_ids), 1), device=self.device)
        dth2 = torch_rand_float(np.deg2rad(-self.init_vel), np.deg2rad(2*self.init_vel), shape=(len(env_ids), 1), device=self.device)
        self.dof_pos[env_ids, 0] = th1[:, 0]
        self.dof_pos[env_ids, 1] = th2[:, 0]
        self.dof_vel[env_ids, 0] = dth1[:, 0]
        self.dof_vel[env_ids, 1] = dth2[:, 0]

        env_ids_int32 = env_ids.to(dtype=torch.int32)

        self.gym.set_dof_state_tensor_indexed(self.sim,
                                              gymtorch.unwrap_tensor(self.dof_state),
                                              gymtorch.unwrap_tensor(env_ids_int32), len(env_ids_int32))

        act_delay_time = torch_rand_float(
            self.act_delay_time*(1 - self.delay_randomize),
            self.act_delay_time*(1 + self.delay_randomize),
            shape=(len(env_ids), 1), device=self.device,
        )
        self.act_delay_idx[env_ids] = (act_delay_time / self.dt).round().to(dtype=torch.int64, device=self.device)
        noise_time = torch_rand_float(-self.act_delay_time, 0, shape=(len(env_ids), 1), device=self.device)
        self.ptb_st_idx[env_ids] += (noise_time / self.dt - 1).round().to(dtype=torch.int64, device=self.device)

        self.delayed_act_buf[env_ids, ...] = fill_delayed_act_buf(
            self.dof_pos[env_ids, :],
            self.dof_vel[env_ids, :],
            self.mass,
            self.com,
            self.len,
            self._jnt_stiffness,
            self._jnt_damping,
            self.joint_gears,
        )
        self.reset_param_buffers(env_ids)


class IDPLeanAndReleaseDet(IDPMinEffortDet):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._ptb_range = to_torch(self._cal_ptb_acc(np.array([0.0]).reshape(1, -1)), device=self.device)
        self.target_lean_angle = to_torch(np.array(np.deg2rad(
            [[0.5, 1.0], [1.0, 2.0], [1.5, 3.0], [2, 4], [2.5, 5], [3.0, 6.0], [3.5, 7]]
        )), device=self.device)
        self.ptb_idx = to_torch(np.arange(self.num_envs) % self._ptb_range.shape[0], dtype=torch.int64, device=self.device)
        self.lean_idx = to_torch(np.arange(self.num_envs) % self.target_lean_angle.shape[0], dtype=torch.int64, device=self.device)
        self.lean_angle_torch = self.target_lean_angle[self.lean_idx.reshape(-1, 1), np.arange(2)]

    def reset_idx(self, env_ids):
        self.ptb_st_idx[...] = round(1 /(3 * self.dt))
        self._ptb[...] = 0
        self.lean_angle_torch[env_ids, :] = self.target_lean_angle[self.lean_idx[env_ids.unsqueeze(1)], np.arange(2)]

        self.dof_pos[env_ids, :] = self.lean_angle_torch[env_ids, :]
        self.dof_vel[env_ids, :] = 0.0
        env_ids_int32 = env_ids.to(dtype=torch.int32)

        self.gym.set_dof_state_tensor_indexed(self.sim,
                                              gymtorch.unwrap_tensor(self.dof_state),
                                              gymtorch.unwrap_tensor(env_ids_int32), len(env_ids_int32))

        self.delayed_act_buf[env_ids, ...] = fill_delayed_act_buf(
            self.dof_pos[env_ids, :],
            self.dof_vel[env_ids, :],
            self.mass,
            self.com,
            self.len,
            self._jnt_stiffness,
            self._jnt_damping,
            self.joint_gears,
        )
        self.reset_param_buffers(env_ids)


class IDPForwardPushDet(IDPMinEffortDet):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._ptb = to_torch(np.zeros([self.num_envs, self.max_episode_length + 1]), device=self.device)
        self.max_episode_length = to_torch(self.max_episode_length, dtype=torch.int64, device=self.device)
        self._ptb_range = to_torch(
            self._cal_ptb_force(np.array([40, 60, 80, 100, 120]).reshape(-1, 1)),
            device=self.device,
        )
        self.ptb_idx = to_torch(np.arange(self.num_envs) % self._ptb_range.shape[0], dtype=torch.int64, device=self.device)
        self.delayed_time = 0.1
        if "st_ptb_idx" in kwargs['cfg']['env'].keys():
            self.ptb_idx += kwargs['cfg']['env']["st_ptb_idx"]

    def _cal_ptb_force(self, f_max):
        wt = 2*np.pi*1/(2*self._ptb_act_time)*np.arange(0, round(self._ptb_act_time/self.dt)) * self.dt
        force = f_max * np.sin(wt[None, :])
        return force

    def get_current_ptbs(self):
        self.ptb_forces[:, 2, 0] = self._ptb[np.arange(self.num_envs), self.progress_buf]

#####################################################################
###=========================jit functions=========================###
#####################################################################

@torch.jit.script
def compute_postural_reward(
        obs_buf, actions, torque_rate, foot_forces,
        reset_buf, progress_buf, ptb_st_idx,
        stcost_ratio, tqcost_ratio, tqrate_ratio, const_ratio,
        ank_ratio, vel_ratio, tq_ratio,
        ankle_limit_type, const_max_val, limLevel,
        high, low, max_episode_length,
        com_len, mass, seg_len, const_type, cost_type, tqr_limit
):
    need_update = progress_buf >= ptb_st_idx.view(-1)
    if const_type == 0:
        const_var = (foot_forces[:, 4] + 0.08*foot_forces[:, 0]) / -foot_forces[:, 2]
    elif const_type == 1:
        const_var = actions[:, 0]
    else:
        raise Exception("undefined constraint type")

    if cost_type == 0:
        rew = -stcost_ratio * torch.sum(torch.cat([
            (1 - vel_ratio) * ank_ratio * obs_buf[:, :2] ** 2,
            vel_ratio * obs_buf[:, 2:4] ** 2], dim=1),
            dim=1)
    elif cost_type == 1:
        com = (mass[1] * com_len[1] * torch.sin(obs_buf[:, [0]]) +
               mass[2] * (seg_len[1] * torch.sin(obs_buf[:, [0]]) + com_len[2] * torch.sin(obs_buf[:, :2].sum(dim=1, keepdim=True)))
               ) / mass[1:].sum()
        rew = -stcost_ratio * torch.sum(torch.cat([
            (1 - vel_ratio) * 0.5 * com ** 2,
            (1 - vel_ratio) * 0.5 * ank_ratio * obs_buf[:, :2] ** 2,
            vel_ratio * obs_buf[:, 2:4] ** 2], dim=1),
            dim=1)
    elif cost_type == 2:
        rew = 0.0 * torch.sum(torch.cat([
            (1 - vel_ratio) * ank_ratio * obs_buf[:, :2] ** 2,
            vel_ratio * obs_buf[:, 2:4] ** 2], dim=1),
            dim=1)
    elif cost_type == 3:
        cop = (foot_forces[:, [4]] + 0.08*foot_forces[:, [0]]) / -foot_forces[:, [2]]
        rew = -stcost_ratio * torch.sum(torch.cat([
            (1 - vel_ratio) * 0.5 * cop ** 2,
            (1 - vel_ratio) * 0.5 * ank_ratio * obs_buf[:, :2] ** 2,
            vel_ratio * obs_buf[:, 2:4] ** 2], dim=1),
            dim=1)
    else:
        raise Exception("undefined cost type")

    rew -= (1 - tqrate_ratio) * tqcost_ratio * torch.sum(tq_ratio * actions ** 2, dim=1)
    rew += 1

    r_penalty = torch.zeros_like(rew, dtype=torch.float)
    if ankle_limit_type == 0:
        reset = reset_buf.clone()
    elif ankle_limit_type == 1:
        reset = torch.zeros_like(reset_buf, dtype=torch.long)
        # reset = torch.where(const_max_val[1] >= const_var, torch.ones_like(reset_buf), reset_buf)
        # reset = torch.where(const_var >= const_max_val[0], torch.ones_like(reset_buf), reset)
        poscop = torch.clamp(const_var, min=0., max=const_max_val[0])
        negcop = torch.clamp(const_var, min=const_max_val[1], max=0.)

        # r_penalty = const_ratio * limLevel * (-2 / (limLevel + 1) + (
        #         1 / ((poscop / const_max_val[0] - 1) ** 2 + limLevel) + 1 / ((negcop / const_max_val[1] - 1) ** 2 + limLevel)))
        r_penalty = const_ratio * limLevel * (-1 / (limLevel + 1) + 1 / ((poscop / const_max_val[0] - 1) ** 2 + limLevel))
    elif ankle_limit_type == 2:
        # reset = torch.zeros_like(reset_buf, dtype=torch.long)
        reset = torch.where(const_max_val[1] >= const_var, torch.ones_like(reset_buf), reset_buf)
        reset = torch.where(const_var >= const_max_val[0], torch.ones_like(reset_buf), reset)
        r_penalty = const_ratio * torch.where(reset.to(torch.bool), torch.ones_like(rew), torch.zeros_like(rew))
    else:
        raise Exception(f"Unexpected ankle limit type")

    fall_reset = torch.where(low[0] > obs_buf[:, 0], torch.ones_like(reset), torch.zeros_like(reset))
    fall_reset = torch.where(obs_buf[:, 0] > high[0], torch.ones_like(reset), fall_reset)
    fall_reset = torch.where(low[1] > obs_buf[:, 1], torch.ones_like(reset), fall_reset)
    fall_reset = torch.where(obs_buf[:, 1] > high[1], torch.ones_like(reset), fall_reset)
    # comy_torso = seg_len[1] * torch.cos(obs_buf[:, 0]) + com_len[2] * torch.cos(obs_buf[:, :2].sum(dim=1))

    # fall_reset = torch.where(comy_torso < 1., torch.ones_like(reset), torch.zeros_like(reset))
    r_penalty = torch.where(fall_reset.to(torch.bool), torch.ones_like(r_penalty) + r_penalty, r_penalty)

    torque_rate_const = torch.max((torque_rate / tqr_limit) ** 2 - 1, torch.tensor(0.0))
    # reset = torch.where(torque_rate[:, 0] >= 3, torch.ones_like(reset), reset)
    # reset = torch.where(torque_rate[:, 1] >= 3, torch.ones_like(reset), reset)
    r_penalty += tqrate_ratio * torch.sum(torque_rate_const, dim=1)

    reset = torch.where(fall_reset.to(torch.bool), torch.ones_like(reset), reset)
    reset = torch.where(progress_buf >= max_episode_length, torch.ones_like(reset), reset)

    rew -= r_penalty
    rew[~need_update] = 0.0

    return rew, reset


@torch.jit.script
def fill_delayed_act_buf(
        dof_pos,
        dof_vel,
        mass,
        com,
        len,
        jnt_stiffness,
        jnt_damping,
        joint_gears,
):
    T2 = -mass[2] * 9.81 * com[2] * torch.sin(torch.sum(dof_pos, dim=1))
    T1 = T2 - mass[1] * 9.81 * com[1] * torch.sin(dof_pos[:, 0]) - mass[2] * 9.81 * len[1] * torch.sin(dof_pos[:, 0])
    Ts = torch.concat([T1.view(-1, 1), T2.view(-1, 1)], dim=1)
    pTs = - (jnt_stiffness * dof_pos + jnt_damping * dof_vel)
    return ((Ts - pTs) / joint_gears)[..., None]


@torch.jit.script
def compute_current_action(
        dof_pos, dof_vel, actions, delayed_act_buf,
        is_act_delayed, ptb_st_idx, progress_buf,
        jnt_stiffness, jnt_damping, joint_gears, lean_ang, delayed_idx
):
    need_update = progress_buf >= ptb_st_idx.view(-1)
    if is_act_delayed:
        assert actions.shape == delayed_act_buf[:, :, 0].shape
        i, j = torch.where(need_update)[0].view(-1, 1, 1), torch.arange(delayed_act_buf.shape[1]).view(1, -1, 1)
        delayed_act_buf[i, j, delayed_idx[need_update].view(-1, 1, 1)] = actions[need_update].unsqueeze(-1).clone()
        delayed_action = delayed_act_buf[need_update, :, 0].clone()
        tmp_buf = delayed_act_buf[need_update, :, 1:].clone()
        delayed_act_buf[need_update, :, :-1] = tmp_buf
        actions[need_update] = delayed_action
    actions[~need_update] = delayed_act_buf[~need_update, :, 0].clone()
    actions += (-(jnt_stiffness * dof_pos + jnt_damping * dof_vel) / joint_gears)
    return actions, delayed_act_buf


@torch.jit.script
def reset_ptb_acc(
        env_ids,
        ptb_acc,
        ptb_acc_range,
        ptb_act_idx,
        max_episode_length,
        cuda_arange,
):
    _ptb_idx = torch.randint(0, ptb_acc_range.shape[0], (len(env_ids), 1))
    # ptb_st_idx = torch.randint(0, 2 * ptb_act_idx, (len(env_ids), 1))
    ptb_st_idx = torch.randint(0, max_episode_length // 2 - ptb_act_idx, (len(env_ids), 1))
    offsets = torch.arange(ptb_act_idx).unsqueeze(0) + ptb_st_idx
    ptb_acc[:] = 0
    ptb_acc[torch.arange(len(env_ids)).unsqueeze(1), offsets] = ptb_acc_range[_ptb_idx, torch.arange(ptb_acc_range.shape[1]).unsqueeze(0)]
    return ptb_acc, cuda_arange[ptb_st_idx]


@torch.jit.script
def _compute_extras_jit(delayed_act_buf, act_delay_idx, actions, prev_actions, old_torque_rate, old_dd_acts, dt: float,
                        avg_coeff: float, device: torch.device):
    # ddtq와 torque_rate 계산
    torque_rate = (actions - prev_actions) / dt
    ddtq = torque_rate - old_torque_rate

    # dd_acts 계산, np.arange를 torch.arange로 변경
    i = torch.arange(delayed_act_buf.shape[0]).to(device).view(-1, 1, 1)
    j = torch.arange(delayed_act_buf.shape[1]).to(device).view(1, -1, 1)
    k = act_delay_idx.view(-1, 1, 1)

    # delayed_act_buf에서 값을 가져올 때 인덱싱 범위 확인
    # k-2가 0보다 작아지는 것을 방지하기 위해 k를 2 이상으로 클램핑하여 인덱싱
    k_safe = torch.clamp(k, min=2)

    dd_acts_new = (
            (delayed_act_buf[i, j, k_safe] - delayed_act_buf[i, j, k_safe - 1]) / dt -
            (delayed_act_buf[i, j, k_safe - 1] - delayed_act_buf[i, j, k_safe - 2]) / dt
    ).squeeze(-1)

    # k가 2보다 작은 경우, 원래 dd_acts 값을 유지하여 오류 방지
    # k<2인 인덱스를 찾아서 해당 위치의 dd_acts_new 값을 old_dd_acts 값으로 대체
    mask = (k < 2).squeeze()
    dd_acts_new[mask] = old_dd_acts[mask]

    dd_acts = avg_coeff * dd_acts_new + (1 - avg_coeff) * old_dd_acts

    return ddtq, dd_acts, torque_rate


@torch.jit.script
def _process_actions_jit(actions, delayed_act_buf, act_delay_idx, avg_coeff: float, num_envs: int, device: torch.device):
    i = torch.arange(num_envs).to(device).view(-1, 1, 1)
    j = torch.arange(actions.shape[1]).to(device).view(1, -1, 1)
    k = act_delay_idx.view(-1, 1, 1) - 1

    processed_actions = actions * avg_coeff + (1 - avg_coeff) * delayed_act_buf[i, j, k].clone().squeeze(-1)
    return processed_actions

@torch.jit.script
def _get_current_ptbs_jit(mass, ptb_buf, progress_buf, ptb_forces):
    env_ids = torch.arange(ptb_buf.shape[0]).to(ptb_buf.device)
    ptb_forces[:, :, 0] = -mass.view(1, -1) * ptb_buf[env_ids, progress_buf].view(-1, 1)
    return ptb_forces