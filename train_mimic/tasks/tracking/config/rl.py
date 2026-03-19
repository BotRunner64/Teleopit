"""PPO runner configuration for the single supported VelCmdHistory task."""

from mjlab.rl import RslRlModelCfg, RslRlOnPolicyRunnerCfg, RslRlPpoAlgorithmCfg

from train_mimic.tasks.tracking.config.constants import VELCMD_HISTORY_EXPERIMENT_NAME

_TEMPORAL_CNN_MODEL_CLASS = (
    "train_mimic.tasks.tracking.rl.temporal_cnn_model:TemporalCNNModel"
)
_CNN_CFG: dict = {
    "output_channels": (64, 32),
    "kernel_size": 3,
    "activation": "elu",
    "global_pool": "avg",
}


def make_velcmd_history_tracking_ppo_runner_cfg(
    experiment_name: str = VELCMD_HISTORY_EXPERIMENT_NAME,
) -> RslRlOnPolicyRunnerCfg:
    """Create RL runner configuration for VelCmdHistory."""
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
