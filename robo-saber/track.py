"""Track generated 3-point trajectories with a physics-based humanoid (PHC in Isaac Gym Preview 4)."""

import os
import random
from argparse import ArgumentParser, Namespace

from isaacgym import gymapi, gymutil  # isaacgym must be imported before torch
from isaacgym.torch_utils import to_torch

import h5py
import numpy as np
import torch
import xarray as xr
import yaml
from easydict import EasyDict
from rl_games.algos_torch import model_builder, torch_ext
from rl_games.common import env_configurations, object_factory, vecenv
from rl_games.torch_runner import Runner
from tqdm import tqdm

from beaty_common.pose_utils import interpolate_xyzquat, rotate_fingertip_to_thumb, rotate_thumb_to_fingertip, sixd_to_quat, unity_to_zup, zup_to_unity
from vendor.phc import ASSET_DIR, DATA_DIR, amp_models, amp_network_builder, humanoid_amp_task, im_amp_players, network_builder
from vendor.phc.flags import flags
from vendor.phc.humanoid_amp import HumanoidAMP
from vendor.phc.humanoid_im import HumanoidIm
from vendor.phc.vec_task_wrappers import VecTaskPythonWrapper
from vendor.poselib.poselib.skeleton.skeleton3d import SkeletonState

SIM_TIMESTEP = 1.0 / 60.0
TRACK_BODY_IDS = [13, 18, 23]  # Head, L_Hand, R_Hand

cfg = None
cfg_train = None


class MyPlayer(im_amp_players.IMAMPPlayerContinuous):
    def __init__(self, config):
        super().__init__(config)

        self.dof_prop = self.env.task.gym.get_asset_dof_properties(self.env.task.humanoid_assets[0])
        self.stiffness_np = to_torch(self.dof_prop["stiffness"]).detach().cpu().numpy()
        self.damping_np = to_torch(self.dof_prop["damping"]).detach().cpu().numpy()
        self.rb_prop = self.env.task.gym.get_actor_rigid_body_properties(self.env.task.envs[0], self.env.task.humanoid_handles[0])
        self.mass_np = to_torch([rb.mass for rb in self.rb_prop]).detach().cpu().numpy()

        self.strength_scale = 1.0
        self.hand_mass_scale = 1.0

    def run(self):
        for env_ptr, handle in zip(self.env.task.envs, self.env.task.humanoid_handles):
            dof_prop = self.env.task.gym.get_actor_dof_properties(env_ptr, handle)
            if cfg["env"]["control_mode"] == "force":
                dof_prop["stiffness"][:] = 0
                dof_prop["damping"][:] = 0
            else:
                dof_prop["stiffness"][:] = self.stiffness_np * self.strength_scale
                dof_prop["damping"][:] = self.damping_np * self.strength_scale
            self.env.task.gym.set_actor_dof_properties(env_ptr, handle, dof_prop)

            rb_prop = self.env.task.gym.get_actor_rigid_body_properties(env_ptr, handle)
            for p in range(len(rb_prop)):
                rb_prop[p].mass = self.mass_np[p] * self.hand_mass_scale
                rb_prop[p].invMass = 1.0 / rb_prop[p].mass
            self.env.task.gym.set_actor_rigid_body_properties(env_ptr, handle, rb_prop)

        with h5py.File(cfg.gen3p_path, "r") as f:
            group_names = sorted(f.keys(), key=int)
        if os.path.exists(cfg.out_path):
            os.remove(cfg.out_path)

        for out_i, group_name in enumerate(tqdm(group_names)):
            with xr.open_dataset(cfg.gen3p_path, group=group_name, engine="h5netcdf") as ds:
                three_p = torch.as_tensor(ds["3p"].values)[0, 0][::2]  # 60 Hz -> 30 Hz control
                attrs = dict(ds.attrs)

            xyzs, quat = unity_to_zup(three_p[..., :3], sixd_to_quat(three_p[..., 3:]))
            quat = torch.as_tensor(rotate_thumb_to_fingertip(quat.detach().cpu().numpy())).to(quat)
            n = three_p.shape[0]
            self.max_steps = n - 1

            custom_3p_ref_pos = torch.zeros((n, 24, 3), dtype=torch.float, device=self.device)
            custom_3p_ref_pos[:, TRACK_BODY_IDS] = xyzs.reshape(-1, 3, 3).to(device=self.device)
            custom_3p_ref_rot = torch.zeros((n, 24, 4), dtype=torch.float, device=self.device)
            custom_3p_ref_rot[..., -1] = 1
            custom_3p_ref_rot[:, TRACK_BODY_IDS] = quat.reshape(-1, 3, 4).to(device=self.device)
            self.env.task.custom_3p_ref_pos = custom_3p_ref_pos
            self.env.task.custom_3p_ref_rot = custom_3p_ref_rot

            rb_pos_history = []
            rb_rot_history = []
            self.env.task._state_init = HumanoidAMP.StateInit.Start
            self.env.task.custom_progress_buf = 0
            obs_dict = self.env_reset()
            self.get_batch_size(obs_dict["obs"], 1)
            if self.is_rnn:
                self.init_rnn()

            done_indices = []
            with torch.no_grad():
                for _ in range(self.max_steps):
                    obs_dict = self.env_reset(done_indices)
                    action = self.get_action(obs_dict, self.is_determenistic)
                    obs_dict, _, _, _ = self.env_step(self.env, action)
                    rb_pos_history.append(self.env.task._rigid_body_pos.detach().cpu())
                    rb_rot_history.append(self.env.task._rigid_body_rot.detach().cpu())
                    done_indices = self.env.task._terminate_buf.nonzero(as_tuple=False)[:, 0]
                    self.env.task._terminate_buf *= 0

            phys_xyzquat = torch.cat([torch.stack(rb_pos_history[2:])[:, 0], torch.stack(rb_rot_history[2:])[:, 0]], dim=-1)
            phys_xyzquat = interpolate_xyzquat(phys_xyzquat[None, None], 2)[0, 0]  # 30 Hz -> 60 Hz
            phys_xyz, phys_quat = phys_xyzquat[..., :3] * 1, phys_xyzquat[..., 3:] * 1

            sk_state = SkeletonState.from_rotation_and_root_translation(self.env.task.skeleton_trees[0], phys_quat, phys_xyz[:, 0], is_local=False)
            root_pos, local_rot = zup_to_unity(sk_state.global_translation[:, 0].cpu().numpy(), sk_state.local_rotation.cpu().numpy())

            phy3p_xyz, phy3p_quat = phys_xyz[:, TRACK_BODY_IDS], phys_quat[:, TRACK_BODY_IDS]
            phy3p_quat = torch.as_tensor(rotate_fingertip_to_thumb(phy3p_quat.detach().cpu().numpy())).to(phy3p_quat)
            phy3p_xyz, phy3p_quat = zup_to_unity(phy3p_xyz, phy3p_quat)

            xr.Dataset(
                {
                    "pos": (("frame", "xyz"), root_pos),
                    "rots": (("frame", "bone", "quat"), local_rot),
                    "phy3p": (("frame", "joint", "xyzquat"), torch.cat([phy3p_xyz, phy3p_quat], dim=-1).numpy()),
                },
                attrs=attrs,
            ).to_netcdf(cfg.out_path, mode="w" if out_i == 0 else "a", group=group_name, engine="h5netcdf")


class MyRunner(Runner):
    def __init__(self):
        super().__init__()
        self.algo_factory = object_factory.ObjectFactory()
        self.player_factory = object_factory.ObjectFactory()
        self.player_factory.register_builder("im_amp", lambda **kwargs: MyPlayer(**kwargs))
        self.model_builder = model_builder.ModelBuilder()
        self.model_builder.model_factory.register_builder("amp", lambda network, **kwargs: amp_models.ModelAMPContinuous(network))
        self.model_builder.network_factory.register_builder("amp", lambda **kwargs: amp_network_builder.AMPBuilder())
        self.network_builder = network_builder.NetworkBuilder()
        torch.backends.cudnn.benchmark = True

    def run(self, args):
        player = self.create_player()
        d = torch_ext.load_checkpoint(cfg["checkpoint"])
        checkpoint = d["state"]
        player.model.load_state_dict(checkpoint["model"])
        if "strength_scale" in d:
            player.strength_scale = d["strength_scale"]
        if "hand_mass_scale" in d:
            player.hand_mass_scale = d["hand_mass_scale"]
        if player.normalize_input:
            player.running_mean_std.load_state_dict(checkpoint["running_mean_std"])
        player.epoch_num = checkpoint["epoch"]
        if player._normalize_amp_input:
            player._amp_input_mean_std.load_state_dict(checkpoint["amp_input_mean_std"])
            if player._normalize_input:
                player.running_mean_std.load_state_dict(checkpoint["running_mean_std"])
        player.run()


def parse_sim_params(cfg):
    sim_params = gymapi.SimParams()
    sim_params.dt = SIM_TIMESTEP
    sim_params.num_client_threads = cfg.sim.slices

    if cfg.sim.use_flex:
        if cfg.sim.pipeline in ["gpu"]:
            print("WARNING: Using Flex with GPU instead of PHYSX!")
        sim_params.use_flex.shape_collision_margin = 0.01
        sim_params.use_flex.num_outer_iterations = 4
        sim_params.use_flex.num_inner_iterations = 10
    else:
        sim_params.physx.solver_type = 1
        sim_params.physx.num_position_iterations = 4
        sim_params.physx.num_velocity_iterations = 1
        sim_params.physx.num_threads = 4
        sim_params.physx.use_gpu = cfg.sim.pipeline in ["gpu"]
        sim_params.physx.num_subscenes = cfg.sim.subscenes
        if flags.test and not flags.im_eval:
            sim_params.physx.max_gpu_contact_pairs = 4 * 1024 * 1024
        else:
            sim_params.physx.max_gpu_contact_pairs = 16 * 1024 * 1024

    sim_params.use_gpu_pipeline = cfg.sim.pipeline in ["gpu"]
    sim_params.physx.use_gpu = cfg.sim.pipeline in ["gpu"]

    if "sim" in cfg:
        gymutil.parse_sim_config(cfg["sim"], sim_params)

    if not cfg.sim.use_flex and cfg.sim.physx.num_threads > 0:
        sim_params.physx.num_threads = cfg.sim.physx.num_threads

    return sim_params


def create_rlgpu_env(**kwargs):
    cfg["seed"] = cfg_train.get("seed", -1)
    cfg["env"]["seed"] = cfg["seed"]
    task = HumanoidIm(
        cfg=cfg,
        sim_params=parse_sim_params(cfg),
        physics_engine=gymapi.SIM_FLEX if cfg.sim.use_flex else gymapi.SIM_PHYSX,
        device_type=cfg.device,
        device_id=cfg.device_id,
        headless=cfg.headless,
    )
    return VecTaskPythonWrapper(task, cfg.rl_device, cfg_train.get("clip_observations", np.inf))


class RLGPUEnv(vecenv.IVecEnv):
    def __init__(self, config_name, num_actors, **kwargs):
        self.env = env_configurations.configurations[config_name]["env_creator"](**kwargs)
        self.use_global_obs = self.env.num_states > 0

        self.full_state = {}
        self.full_state["obs"] = self.reset()
        if self.use_global_obs:
            self.full_state["states"] = self.env.get_state()

    def step(self, action):
        next_obs, reward, is_done, info = self.env.step(action)
        self.full_state["obs"] = next_obs
        if self.use_global_obs:
            self.full_state["states"] = self.env.get_state()
            return self.full_state, reward, is_done, info
        return self.full_state["obs"], reward, is_done, info

    def reset(self, env_ids=None):
        self.full_state["obs"] = self.env.reset(env_ids)
        if self.use_global_obs:
            self.full_state["states"] = self.env.get_state()
            return self.full_state
        return self.full_state["obs"]

    def get_number_of_agents(self):
        return self.env.get_number_of_agents()

    def get_env_info(self):
        info = {}
        info["action_space"] = self.env.action_space
        info["observation_space"] = self.env.observation_space
        info["amp_observation_space"] = self.env.amp_observation_space
        info["enc_amp_observation_space"] = self.env.enc_amp_observation_space
        if isinstance(self.env.task, humanoid_amp_task.HumanoidAMPTask):
            info["task_obs_size"] = self.env.task.get_task_obs_size()
        else:
            info["task_obs_size"] = 0
        if self.use_global_obs:
            info["state_space"] = self.env.state_space
        return info


vecenv.register("RLGPU", lambda config_name, num_actors, **kwargs: RLGPUEnv(config_name, num_actors, **kwargs))
env_configurations.register("rlgpu", {"env_creator": lambda **kwargs: create_rlgpu_env(**kwargs), "vecenv_type": "RLGPU"})


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False


def main(args: Namespace) -> None:
    global cfg, cfg_train

    with open(f"{ASSET_DIR}/track.yaml") as f:
        cfg = EasyDict(yaml.safe_load(f))
    cfg.gen3p_path = args.gen3p_path
    cfg.out_path = args.out_path
    cfg.checkpoint = args.checkpoint
    cfg["env"]["motion_file"] = f"{DATA_DIR}/sample_data/amass_isaac_standing_upright_slim.pkl"
    os.makedirs(os.path.dirname(os.path.abspath(cfg.out_path)), exist_ok=True)

    (
        flags.debug,
        flags.follow,
        flags.fixed,
        flags.divide_group,
        flags.no_collision_check,
        flags.fixed_path,
        flags.real_path,
        flags.show_traj,
        flags.server_mode,
        flags.slow,
        flags.real_traj,
        flags.im_eval,
        flags.no_virtual_display,
        flags.render_o3d,
    ) = (cfg.debug, cfg.follow, False, False, False, False, False, True, cfg.server_mode, False, False, cfg.im_eval, cfg.no_virtual_display, cfg.render_o3d)
    flags.test = cfg.test
    flags.add_proj = cfg.add_proj
    flags.has_eval = cfg.has_eval
    flags.trigger_input = False

    set_seed(cfg.seed)
    cfg_train = cfg.learning
    runner = MyRunner()
    runner.load(cfg_train)
    runner.reset()
    runner.run(cfg)
    print(f"Saved to {cfg.out_path}")


if __name__ == "__main__":
    parser = ArgumentParser(allow_abbrev=False)
    parser.add_argument("--gen3p_path", type=str, default="out/gen3p.nc")
    parser.add_argument("--out_path", type=str, default="out/tracking.nc")
    parser.add_argument("--checkpoint", type=str, default="models/phc.pkl")
    main(parser.parse_args())
