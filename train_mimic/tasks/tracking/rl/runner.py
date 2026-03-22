import os
import pathlib
import statistics
import time
from typing import cast

import torch
from rsl_rl.env.vec_env import VecEnv
from torch import nn

from mjlab.rl import RslRlVecEnvWrapper
from mjlab.rl.runner import MjlabOnPolicyRunner
from rsl_rl.utils import check_nan
from train_mimic.tasks.tracking.mdp import MotionCommand


def _one_based_iteration_range(start_iteration: int, total_iterations: int) -> range:
    """Return the inclusive 1-based iteration range up to the target total."""
    if total_iterations < start_iteration:
        raise ValueError(
            "num_learning_iterations must be >= the completed iteration count when resuming. "
            f"Got total_iterations={total_iterations}, start_iteration={start_iteration}."
        )
    return range(start_iteration + 1, total_iterations + 1)


def _resolve_total_iterations(start_iteration: int, num_learning_iterations: int) -> int:
    """Return the cumulative 1-based target iteration after running more iterations."""
    if num_learning_iterations < 0:
        raise ValueError(
            "num_learning_iterations must be non-negative. "
            f"Got num_learning_iterations={num_learning_iterations}."
        )
    return start_iteration + num_learning_iterations


class _OnnxMotionModel(nn.Module):
    """ONNX-exportable model that wraps the policy and bundles motion reference data."""

    def __init__(self, actor, motion):
        super().__init__()
        self.policy = actor.as_onnx(verbose=False)
        # torch.from_numpy shares memory with the underlying numpy array
        # (zero-copy).  The arrays are already writable (loaded from .npz).
        self.register_buffer("joint_pos", torch.from_numpy(motion._joint_pos))
        self.register_buffer("joint_vel", torch.from_numpy(motion._joint_vel))
        self.register_buffer("body_pos_w", torch.from_numpy(motion._body_pos_w))
        self.register_buffer("body_quat_w", torch.from_numpy(motion._body_quat_w))
        self.register_buffer("body_lin_vel_w", torch.from_numpy(motion._body_lin_vel_w))
        self.register_buffer("body_ang_vel_w", torch.from_numpy(motion._body_ang_vel_w))
        self.time_step_total: int = self.joint_pos.shape[0]  # type: ignore[index]

    def forward(self, *args):
        # Last arg is always time_step; preceding args are policy inputs.
        *policy_args, time_step = args
        time_step_clamped = torch.clamp(
            time_step.long().squeeze(-1), max=self.time_step_total - 1
        )
        if len(policy_args) == 1:
            policy_out = self.policy(policy_args[0])
        else:
            policy_out = self.policy(*policy_args)
        return (
            policy_out,
            self.joint_pos[time_step_clamped],  # type: ignore[index]
            self.joint_vel[time_step_clamped],  # type: ignore[index]
            self.body_pos_w[time_step_clamped],  # type: ignore[index]
            self.body_quat_w[time_step_clamped],  # type: ignore[index]
            self.body_lin_vel_w[time_step_clamped],  # type: ignore[index]
            self.body_ang_vel_w[time_step_clamped],  # type: ignore[index]
        )


class MotionTrackingOnPolicyRunner(MjlabOnPolicyRunner):
    env: RslRlVecEnvWrapper

    def __init__(
        self,
        env: VecEnv,
        train_cfg: dict,
        log_dir: str | None = None,
        device: str = "cpu",
        registry_name: str | None = None,
    ):
        super().__init__(env, train_cfg, log_dir, device)
        self.registry_name = registry_name

    def learn(self, num_learning_iterations: int, init_at_random_ep_len: bool = False) -> None:
        """Run the learning loop using 1-based iteration numbering."""
        if init_at_random_ep_len:
            self.env.episode_length_buf = torch.randint_like(
                self.env.episode_length_buf, high=int(self.env.max_episode_length)
            )

        obs = self.env.get_observations().to(self.device)
        self.alg.train_mode()

        if self.is_distributed:
            print(f"Synchronizing parameters for rank {self.gpu_global_rank}...")
            self.alg.broadcast_parameters()

        self.logger.init_logging_writer()

        start_it = self.current_learning_iteration
        total_it = _resolve_total_iterations(start_it, num_learning_iterations)
        for it in _one_based_iteration_range(start_it, total_it):
            start = time.time()
            with torch.inference_mode():
                for _ in range(self.cfg["num_steps_per_env"]):
                    actions = self.alg.act(obs)
                    obs, rewards, dones, extras = self.env.step(actions.to(self.env.device))
                    if self.cfg.get("check_for_nan", True):
                        check_nan(obs, rewards, dones)
                    obs, rewards, dones = (obs.to(self.device), rewards.to(self.device), dones.to(self.device))
                    self.alg.process_env_step(obs, rewards, dones, extras)
                    intrinsic_rewards = self.alg.intrinsic_rewards if self.cfg["algorithm"]["rnd_cfg"] else None
                    self.logger.process_env_step(rewards, dones, extras, intrinsic_rewards)

                stop = time.time()
                collect_time = stop - start
                start = stop
                self.alg.compute_returns(obs)

            loss_dict = self.alg.update()

            stop = time.time()
            learn_time = stop - start
            self.current_learning_iteration = it

            self._log_one_based_iteration(
                it=it,
                start_it=start_it,
                total_it=total_it,
                collect_time=collect_time,
                learn_time=learn_time,
                loss_dict=loss_dict,
                learning_rate=self.alg.learning_rate,
                action_std=self.alg.get_policy().output_std,
                rnd_weight=self.alg.rnd.weight if self.cfg["algorithm"]["rnd_cfg"] else None,
            )

            if self.logger.writer is not None and it % self.cfg["save_interval"] == 0:
                self.save(os.path.join(self.logger.log_dir, f"model_{it}.pt"))  # type: ignore[arg-type]

        if self.logger.writer is not None:
            self.save(os.path.join(self.logger.log_dir, f"model_{self.current_learning_iteration}.pt"))  # type: ignore[arg-type]
            self.logger.stop_logging_writer()

    def _log_one_based_iteration(
        self,
        *,
        it: int,
        start_it: int,
        total_it: int,
        collect_time: float,
        learn_time: float,
        loss_dict: dict,
        learning_rate: float,
        action_std: torch.Tensor,
        rnd_weight: float | None,
        print_minimal: bool = False,
        width: int = 80,
        pad: int = 40,
    ) -> None:
        logger = self.logger
        if logger.writer is None:
            return

        collection_size = logger.cfg["num_steps_per_env"] * logger.num_envs * logger.gpu_world_size
        iteration_time = collect_time + learn_time
        logger.tot_timesteps += collection_size
        logger.tot_time += iteration_time

        extras_string = ""
        if logger.ep_extras:
            for key in logger.ep_extras[0]:
                infotensor = torch.tensor([], device=logger.device)
                for ep_info in logger.ep_extras:
                    if key not in ep_info:
                        continue
                    if not isinstance(ep_info[key], torch.Tensor):
                        ep_info[key] = torch.Tensor([ep_info[key]])
                    if len(ep_info[key].shape) == 0:
                        ep_info[key] = ep_info[key].unsqueeze(0)
                    infotensor = torch.cat((infotensor, ep_info[key].to(logger.device)))
                value = torch.mean(infotensor)
                if "/" in key:
                    logger.writer.add_scalar(key, value, it)  # type: ignore[arg-type]
                    extras_string += f"""{f"{key}:":>{pad}} {value:.4f}
"""
                else:
                    logger.writer.add_scalar("Episode/" + key, value, it)  # type: ignore[arg-type]
                    extras_string += f"""{f"Mean episode {key}:":>{pad}} {value:.4f}
"""

        for key, value in loss_dict.items():
            logger.writer.add_scalar(f"Loss/{key}", value, it)
        logger.writer.add_scalar("Loss/learning_rate", learning_rate, it)
        logger.writer.add_scalar("Policy/mean_std", action_std.mean().item(), it)

        fps = int(collection_size / (collect_time + learn_time))
        logger.writer.add_scalar("Perf/total_fps", fps, it)
        logger.writer.add_scalar("Perf/collection_time", collect_time, it)
        logger.writer.add_scalar("Perf/learning_time", learn_time, it)

        if len(logger.rewbuffer) > 0:
            if logger.cfg["algorithm"]["rnd_cfg"]:
                logger.writer.add_scalar("Rnd/mean_extrinsic_reward", statistics.mean(logger.erewbuffer), it)
                logger.writer.add_scalar("Rnd/mean_intrinsic_reward", statistics.mean(logger.irewbuffer), it)
                logger.writer.add_scalar("Rnd/weight", rnd_weight, it)  # type: ignore[arg-type]
            logger.writer.add_scalar("Train/mean_reward", statistics.mean(logger.rewbuffer), it)
            logger.writer.add_scalar("Train/mean_episode_length", statistics.mean(logger.lenbuffer), it)
            if logger.logger_type != "wandb":
                logger.writer.add_scalar("Train/mean_reward/time", statistics.mean(logger.rewbuffer), int(logger.tot_time))
                logger.writer.add_scalar(
                    "Train/mean_episode_length/time", statistics.mean(logger.lenbuffer), int(logger.tot_time)
                )

        log_string = f"""{'#' * width}
"""
        log_string += f"""[1m{f' Learning iteration {it}/{total_it} '.center(width)}[0m 

"""

        run_name = logger.cfg.get("run_name")
        log_string += f"""{'Run name:':>{pad}} {run_name}
""" if run_name else ""
        log_string += (
            f"""{'Total steps:':>{pad}} {logger.tot_timesteps} 
"""
            f"""{'Steps per second:':>{pad}} {fps:.0f} 
"""
            f"""{'Collection time:':>{pad}} {collect_time:.3f}s 
"""
            f"""{'Learning time:':>{pad}} {learn_time:.3f}s 
"""
        )

        for key, value in loss_dict.items():
            log_string += f"""{f'Mean {key} loss:':>{pad}} {value:.4f}
"""

        if len(logger.rewbuffer) > 0:
            if logger.cfg["algorithm"]["rnd_cfg"]:
                log_string += f"""{'Mean extrinsic reward:':>{pad}} {statistics.mean(logger.erewbuffer):.2f}
"""
                log_string += f"""{'Mean intrinsic reward:':>{pad}} {statistics.mean(logger.irewbuffer):.2f}
"""
            log_string += f"""{'Mean reward:':>{pad}} {statistics.mean(logger.rewbuffer):.2f}
"""
            log_string += f"""{'Mean episode length:':>{pad}} {statistics.mean(logger.lenbuffer):.2f}
"""

        log_string += f"""{'Mean action std:':>{pad}} {action_std.mean().item():.2f}
"""
        if not print_minimal:
            log_string += extras_string

        done_it = it - start_it
        remaining_it = total_it - it
        eta = logger.tot_time / done_it * remaining_it if done_it > 0 else 0.0
        log_string += (
            f"""{'-' * width}
"""
            f"""{'Iteration time:':>{pad}} {iteration_time:.2f}s
"""
            f"""{'Time elapsed:':>{pad}} {time.strftime('%H:%M:%S', time.gmtime(logger.tot_time))}
"""
            f"""{'ETA:':>{pad}} {time.strftime('%H:%M:%S', time.gmtime(eta))}
"""
        )
        print(log_string)

        if logger.logger_type == "wandb":
            for video in pathlib.Path(logger.log_dir).rglob("*.mp4"):  # type: ignore[arg-type]
                logger.writer.save_video(video, it)  # type: ignore[arg-type]

        logger.ep_extras.clear()

    def export_policy_to_onnx(
        self,
        path: str,
        filename: str = "policy.onnx",
        verbose: bool = False,
        *,
        include_motion_labels: bool = False,
    ) -> None:
        os.makedirs(path, exist_ok=True)
        output_path = os.path.join(path, filename)
        if include_motion_labels:
            cmd = cast(MotionCommand, self.env.unwrapped.command_manager.get_term("motion"))
            model = _OnnxMotionModel(self.alg.get_policy(), cmd.motion)
            model.to("cpu")
            model.eval()
            dummy_inputs = model.policy.get_dummy_inputs()
            time_step = torch.zeros(1, 1)
            input_names = model.policy.input_names + ["time_step"]
            torch.onnx.export(
                model,
                (*dummy_inputs, time_step),
                output_path,
                export_params=True,
                opset_version=18,
                verbose=verbose,
                input_names=input_names,
                output_names=[
                    "actions",
                    "joint_pos",
                    "joint_vel",
                    "body_pos_w",
                    "body_quat_w",
                    "body_lin_vel_w",
                    "body_ang_vel_w",
                ],
                dynamic_axes={},
                dynamo=False,
            )
            return

        model = self.alg.get_policy().as_onnx(verbose=False)
        model.to("cpu")
        model.eval()
        dummy_inputs = model.get_dummy_inputs()
        torch.onnx.export(
            model,
            dummy_inputs,
            output_path,
            export_params=True,
            opset_version=18,
            verbose=verbose,
            input_names=model.input_names,
            output_names=model.output_names,
            dynamic_axes={},
            dynamo=False,
        )
