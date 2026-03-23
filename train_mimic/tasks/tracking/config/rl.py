"""PPO runner configuration for supported tracking tasks."""

from mjlab.rl import RslRlModelCfg, RslRlOnPolicyRunnerCfg, RslRlPpoAlgorithmCfg

from train_mimic.tasks.tracking.config.constants import (
    MOTION_TRACKING_DEPLOY_EXPERIMENT_NAME,
    VELCMD_HISTORY_LARGE_EXPERIMENT_NAME,
    VELCMD_HISTORY_REGULAR_EXPERIMENT_NAME,
    VELCMD_HISTORY_EXPERIMENT_NAME,
    VELCMD_REF_WINDOW_EXPERIMENT_NAME,
)

_TEMPORAL_CNN_MODEL_CLASS = (
    "train_mimic.tasks.tracking.rl.temporal_cnn_model:TemporalCNNModel"
)
_MLP_MODEL_CLASS = "rsl_rl.models.mlp_model:MLPModel"
_CNN_CFG: dict = {
    "output_channels": (64, 32),
    "kernel_size": 3,
    "activation": "elu",
    "global_pool": "avg",
}
_CNN_CFG_LARGE: dict = {
    "output_channels": (128, 64, 32),
    "kernel_size": 3,
    "activation": "elu",
    "global_pool": "avg",
}


def _make_temporal_runner_cfg(experiment_name: str) -> RslRlOnPolicyRunnerCfg:
    return RslRlOnPolicyRunnerCfg(
        actor=RslRlModelCfg(
            class_name=_TEMPORAL_CNN_MODEL_CLASS,
            hidden_dims=(512, 256, 128),
            activation="elu",
            obs_normalization=True,
            cnn_cfg=_CNN_CFG,
            distribution_cfg={
                "class_name": "GaussianDistribution",
                "init_std": 1.0,
                "std_type": "scalar",
            },
        ),
        critic=RslRlModelCfg(
            class_name=_TEMPORAL_CNN_MODEL_CLASS,
            hidden_dims=(512, 256, 128),
            activation="elu",
            obs_normalization=True,
            cnn_cfg=_CNN_CFG,
        ),
        algorithm=RslRlPpoAlgorithmCfg(
            value_loss_coef=1.0,
            use_clipped_value_loss=True,
            clip_param=0.2,
            entropy_coef=0.005,
            num_learning_epochs=5,
            num_mini_batches=4,
            learning_rate=1.0e-3,
            schedule="adaptive",
            gamma=0.99,
            lam=0.95,
            desired_kl=0.01,
            max_grad_norm=1.0,
        ),
        obs_groups={
            "actor": ("actor", "actor_history"),
            "critic": ("critic", "critic_history"),
        },
        experiment_name=experiment_name,
        save_interval=2000,
        num_steps_per_env=24,
        max_iterations=30_000,
        logger="tensorboard",
        upload_model=False,
    )


def make_velcmd_history_tracking_ppo_runner_cfg(
    experiment_name: str = VELCMD_HISTORY_EXPERIMENT_NAME,
) -> RslRlOnPolicyRunnerCfg:
    """Create RL runner configuration for VelCmdHistory."""
    return _make_temporal_runner_cfg(experiment_name)


def make_velcmd_history_large_tracking_ppo_runner_cfg(
    experiment_name: str = VELCMD_HISTORY_LARGE_EXPERIMENT_NAME,
) -> RslRlOnPolicyRunnerCfg:
    """Create RL runner configuration for VelCmdHistoryLarge (wider+deeper MLP, larger CNN)."""
    return RslRlOnPolicyRunnerCfg(
        actor=RslRlModelCfg(
            class_name=_TEMPORAL_CNN_MODEL_CLASS,
            hidden_dims=(1024, 512, 256, 256, 128),
            activation="elu",
            obs_normalization=True,
            cnn_cfg=_CNN_CFG_LARGE,
            distribution_cfg={
                "class_name": "GaussianDistribution",
                "init_std": 1.0,
                "std_type": "scalar",
            },
        ),
        critic=RslRlModelCfg(
            class_name=_TEMPORAL_CNN_MODEL_CLASS,
            hidden_dims=(1024, 512, 256, 256, 128),
            activation="elu",
            obs_normalization=True,
            cnn_cfg=_CNN_CFG_LARGE,
        ),
        algorithm=RslRlPpoAlgorithmCfg(
            value_loss_coef=1.0,
            use_clipped_value_loss=True,
            clip_param=0.2,
            entropy_coef=0.005,
            num_learning_epochs=5,
            num_mini_batches=4,
            learning_rate=1.0e-3,
            schedule="adaptive",
            gamma=0.99,
            lam=0.95,
            desired_kl=0.01,
            max_grad_norm=1.0,
        ),
        obs_groups={
            "actor": ("actor", "actor_history"),
            "critic": ("critic", "critic_history"),
        },
        experiment_name=experiment_name,
        save_interval=2000,
        num_steps_per_env=24,
        max_iterations=30_000,
        logger="tensorboard",
        upload_model=False,
    )


def make_velcmd_ref_window_tracking_ppo_runner_cfg(
    experiment_name: str = VELCMD_REF_WINDOW_EXPERIMENT_NAME,
) -> RslRlOnPolicyRunnerCfg:
    """Create RL runner configuration for VelCmdRefWindow.

    Uses TemporalCNNModel with two independent 3D groups:
    - actor_history / critic_history: proprio + mixed ref+proprio (10 frames)
    - actor_ref_window / critic_ref_window: pure-ref windowed (20 frames)
    """
    return RslRlOnPolicyRunnerCfg(
        actor=RslRlModelCfg(
            class_name=_TEMPORAL_CNN_MODEL_CLASS,
            hidden_dims=(512, 256, 128),
            activation="elu",
            obs_normalization=True,
            cnn_cfg=_CNN_CFG,
            distribution_cfg={
                "class_name": "GaussianDistribution",
                "init_std": 1.0,
                "std_type": "scalar",
            },
        ),
        critic=RslRlModelCfg(
            class_name=_TEMPORAL_CNN_MODEL_CLASS,
            hidden_dims=(512, 256, 128),
            activation="elu",
            obs_normalization=True,
            cnn_cfg=_CNN_CFG,
        ),
        algorithm=RslRlPpoAlgorithmCfg(
            value_loss_coef=1.0,
            use_clipped_value_loss=True,
            clip_param=0.2,
            entropy_coef=0.005,
            num_learning_epochs=5,
            num_mini_batches=4,
            learning_rate=1.0e-3,
            schedule="adaptive",
            gamma=0.99,
            lam=0.95,
            desired_kl=0.01,
            max_grad_norm=1.0,
        ),
        obs_groups={
            "actor": ("actor", "actor_history", "actor_ref_window"),
            "critic": ("critic", "critic_history", "critic_ref_window"),
        },
        experiment_name=experiment_name,
        save_interval=2000,
        num_steps_per_env=24,
        max_iterations=30_000,
        logger="tensorboard",
        upload_model=False,
    )


def make_velcmd_history_regular_tracking_ppo_runner_cfg(
    experiment_name: str = VELCMD_HISTORY_REGULAR_EXPERIMENT_NAME,
) -> RslRlOnPolicyRunnerCfg:
    """Create RL runner configuration for VelCmdHistoryRegular."""
    return _make_temporal_runner_cfg(experiment_name)


def make_motion_tracking_deploy_ppo_runner_cfg(
    experiment_name: str = MOTION_TRACKING_DEPLOY_EXPERIMENT_NAME,
) -> RslRlOnPolicyRunnerCfg:
    """Create RL runner configuration for the deploy-aligned motion-tracking task."""
    return RslRlOnPolicyRunnerCfg(
        actor=RslRlModelCfg(
            class_name=_MLP_MODEL_CLASS,
            hidden_dims=(1024, 512, 256),
            activation="elu",
            obs_normalization=True,
            distribution_cfg={
                "class_name": "GaussianDistribution",
                "init_std": 1.0,
                "std_type": "scalar",
            },
        ),
        critic=RslRlModelCfg(
            class_name=_MLP_MODEL_CLASS,
            hidden_dims=(1024, 512, 256),
            activation="elu",
            obs_normalization=True,
        ),
        algorithm=RslRlPpoAlgorithmCfg(
            value_loss_coef=1.0,
            use_clipped_value_loss=True,
            clip_param=0.2,
            entropy_coef=0.005,
            num_learning_epochs=5,
            num_mini_batches=4,
            learning_rate=1.0e-3,
            schedule="adaptive",
            gamma=0.99,
            lam=0.95,
            desired_kl=0.01,
            max_grad_norm=1.0,
        ),
        obs_groups={
            "actor": ("actor",),
            "critic": ("critic",),
        },
        experiment_name=experiment_name,
        save_interval=2000,
        num_steps_per_env=24,
        max_iterations=30_000,
        logger="tensorboard",
        upload_model=False,
    )
