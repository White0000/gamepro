import os
import sys
import json
import random
import time
import math
import shutil
import pathlib
import subprocess
import platform
import secrets
import uuid
import threading

class QLearningHyperparams:
    def __init__(self):
        self.learning_rate = 0.1
        self.discount_factor = 0.99
        self.epsilon_start = 1.0
        self.epsilon_end = 0.01
        self.epsilon_decay = 0.995
        self.min_epsilon = 0.0
        self.max_epsilon = 1.0
        self.clip_rewards = False
        self.reward_clip_value = 1.0
        self.update_target_interval = 100
        self.batch_size = 32
        self.max_memory_size = 50000
        self.max_episodes = 1000
        self.max_steps_per_episode = 200
        self.epsilon_anneal_steps = 10000
        self.start_learn_step = 1000
        self.target_sync_rate = 1.0
        self.reward_scaling_factor = 1.0
        self.reward_shaping = False
        self.shaping_function = "none"
        self.adaptive_learning_rate = False
        self.learning_rate_decay = 1.0
        self.learning_rate_min = 1e-5
        self.learning_rate_decay_steps = 50000
        self.gradient_clip_norm = 0.0
        self.optimization_method = "sgd"
        self.momentum = 0.0
        self.nesterov = False
        self.weight_decay = 0.0
        self.use_double_q = False
        self.use_prioritized_replay = False
        self.prioritized_alpha = 0.6
        self.prioritized_beta_start = 0.4
        self.prioritized_beta_end = 1.0
        self.prioritized_beta_decay_steps = 100000
        self.seed = None
        self.save_checkpoint_interval = 500
        self.early_stopping = False
        self.early_stopping_reward_threshold = None
        self.early_stopping_window = 100
        self.early_stopping_patience = 10
        self.initial_exploration_episodes = 0
        self.replay_burn_in = 1000
        self.max_grad_norm = None
        self.lr_schedule_type = "constant"
        self.lr_step_size = 10000
        self.lr_gamma = 0.9
        self.softmax_temperature = 1.0
        self.epsilon_schedule_type = "exponential"
        self.polyak_tau = 0.005
        self.target_update_interval = 1000
        self.reward_normalization = False
        self.reward_normalization_max_steps = 100000
        self.reward_scale_limit = 10.0
        self.reward_moving_avg_alpha = 0.01
        self.huber_loss_delta = 1.0
        self.exploration_final_frame = 100000
        self.exploration_anneal_type = "linear"
        self.reward_log_smoothing = False
        self.reward_log_smoothing_alpha = 0.3
        self.q_update_frequency = 1
        self.grad_accum_steps = 1
        self.use_noisy_nets = False
        self.noisy_net_sigma = 0.5
        self.multi_step_returns = 1
        self.enable_checkpointing = True
        self.checkpoint_path = "checkpoints"
        self.enable_tensorboard = False
        self.tensorboard_log_dir = "runs"
        self.random_action_probability = 0.0
        self.enable_pretraining = False
        self.pretraining_episodes = 0
        self.optimizer_eps = 1e-8
        self.optimizer_betas = (0.9, 0.999)
        self.use_categorical_dqn = False
        self.num_atoms = 51
        self.v_min = -10
        self.v_max = 10
        self.log_interval = 100
        self.eval_interval = 500
        self.eval_episodes = 10
        self.max_eval_steps = None
        self.dueling_dqn = False
        self.enable_curiosity = False
        self.curiosity_eta = 0.1
        self.curiosity_beta = 0.2
        self.curiosity_forward_loss_weight = 1.0
        self.curiosity_inverse_loss_weight = 0.8
        self.curiosity_buffer_size = 20000
        self.enable_feature_extraction = False
        self.feature_extractor_type = "mlp"
        self.feature_extractor_params = {}
        self.target_entropy = None
        self.entropy_coefficient = 0.0
        self.clip_grad_value = None
        self.obs_norm = False
        self.obs_clip_min = -10.0
        self.obs_clip_max = 10.0
        self.dqn_type = "standard"
        self.n_step = 1
        self.custom_loss_function = None
        self.custom_metrics_enabled = False
        self.custom_metrics = []
        self.auxiliary_tasks_enabled = False
        self.auxiliary_tasks = []
        self.max_episode_length_override = None
        self.save_best_model = False
        self.best_model_metric = "episode_reward"
        self.best_model_comparison = "max"
        self.best_model_path = "best_model.pt"
        self.rnd_enabled = False
        self.rnd_learning_rate = 1e-4
        self.rnd_update_proportion = 0.25
        self.rnd_obs_clip = None
        self.sarsa_enabled = False
        self.enable_momentum_exploration = False
        self.momentum_exploration_alpha = 0.9
        self.n_step_bootstrap = 1
        self.n_step_lambda = 1.0
        self.multi_agent = False
        self.num_agents = 1
        self.shared_replay_buffer = True
        self.agent_id_embeddings = False
        self.agent_id_emb_dim = 16
        self.enable_popart = False
        self.popart_beta = 0.999999
        self.test_exploration_rate = 0.0
        self.learning_rate_warmup = 0
        self.learning_rate_schedule_mode = "step"
        self.enable_q_value_clamping = False
        self.q_value_clip_min = -100.0
        self.q_value_clip_max = 100.0
        self.use_conservative_q = False
        self.conservative_q_lambda = 0.0
        self.enable_rainbow = False
        self.monitor_gpu_usage = False
        self.clip_actions = False
        self.action_clip_low = -1.0
        self.action_clip_high = 1.0
        self.reward_offset = 0.0
        self.episodic_life = False
        self.render_during_training = False
        self.render_interval = 1
        self.async_rollout = False
        self.async_rollout_workers = 2
        self.async_update = False
        self.async_update_frequency = 1
        self.idle_gpu_for_rollout = False
        self.agent_hierarchy = False
        self.hierarchy_levels = 1
        self.hierarchy_manager_index = None
        self.hierarchy_worker_indexes = []
        self.hierarchy_goal_tolerance = 0.0
        self.hindsight_experience_replay = False
        self.her_k = 4
        self.her_strategy = "future"
        self.enable_scheduled_target_update = False
        self.scheduled_target_update_times = []
        self.constant_reward = 0.0
        self.reward_penalty = 0.0
        self.decaying_negative_reward = False
        self.negative_reward_decay_rate = 1.0
        self.auto_adjust_critic_lr = False
        self.critic_lr_adjust_factor = 1.0
        self.max_steps_per_training_phase = None
        self.training_phases = {}
        self.enable_automatic_policy_switch = False
        self.policy_switch_interval = 0
        self.policy_pool = []
        self.policy_pool_weights = []
        self.enable_priority_loss = False
        self.priority_loss_alpha = 0.0
        self.priority_loss_beta = 0.0
        self.priority_loss_gamma = 0.0
        self.separate_target_network = True
        self.rnn_enabled = False
        self.rnn_hidden_size = 128
        self.rnn_num_layers = 1
        self.rnn_type = "lstm"
        self.max_seq_length = 20
        self.random_seed_sharing = False
        self.global_seed = None
        self.auto_export_models = False
        self.export_interval = 1000
        self.export_path = "exports"
        self.use_distributional_rl = False
        self.atoms = 51
        self.vmin = -10
        self.vmax = 10
        self.transition_storage = "list"
        self.gae_lambda = 1.0
        self.enable_advantage_normalization = False
        self.l2_reg = 0.0
        self.batch_norm = False
        self.layer_norm = False
        self.param_init_method = "xavier"
        self.param_init_gain = 1.0
        self.init_log_std = -1.0
        self.enable_trust_region = False
        self.trust_region_threshold = 0.0
        self.multi_headed_architecture = False
        self.num_heads = 1
        self.enable_switching_heads = False
        self.head_switch_interval = 0
        self.last_config_update_time = None
        self.accumulated_training_steps = 0
        self.amp_enabled = False
        self.distillation_enabled = False
        self.distillation_kl_coeff = 1.0
        self.distillation_teacher_path = None
        self.policy_integration_mode = "none"
        self.policy_integration_paths = []
        self.observation_padding = 0.0
        self.evaluation_metric = "mean_reward"
        self.model_save_format = "pt"
        self.enable_online_evaluation = False
        self.online_evaluation_interval = 1000
        self.online_evaluation_metric = "episode_length"
        self.online_evaluation_episodes = 1
        self.enable_checkpoint_rotation = False
        self.max_checkpoints_to_keep = 5
        self.optimizer_grad_clip_method = "none"
        self.custom_training_schedule = {}
        self.sparse_reward = False
        self.enable_mixed_precision = False
        self.reward_delay = 0
        self.distributed_training = False
        self.distributed_world_size = 1
        self.distributed_rank = 0
        self.distributed_backend = "nccl"
        self.strict_done_on_terminal_state = True
        self.q_risk_sensitive = False
        self.q_risk_factor = 1.0
        self.enable_episodic_exploration_bonus = False
        self.episodic_exploration_bonus_value = 0.0
        self.episodic_exploration_decay = 1.0
        self.no_reward_no_terminal = False
        self.rnn_stateful_across_episodes = False
        self.norm_advantages = False
        self.policy_clip_range = None
        self.enable_twin_q = False
        self.enable_clipped_double_q = False
        self.twin_q_decay_factor = 0.0
        self.enable_contextual_bandit_mode = False
        self.context_vector_size = 0
        self.context_embedding_type = None
        self.context_embedding_params = {}
        self.enable_bootstrapped_q = False
        self.bootstrapped_heads = 10
        self.bootstrapped_mask_probability = 0.5
        self.schedule_adaptation = False
        self.schedule_adaptation_metric = "reward"
        self.schedule_adaptation_threshold = 0.0
        self.schedule_adaptation_mode = "none"
        self.enable_gae = False
        self.gae_gamma = 0.99
        self.enable_reward_bonus_on_success = False
        self.success_reward_bonus = 0.0
        self.enable_inverse_dynamics = False
        self.inverse_dynamics_lr = 1e-3
        self.inverse_dynamics_beta = 0.2
        self.forward_dynamics_lr = 1e-3
        self.enable_proximal_updates = False
        self.proximal_clip_param = 0.2
        self.proximal_critic_loss_weight = 0.5
        self.proximal_entropy_weight = 0.01
        self.proximal_updates_per_step = 4
        self.proximal_max_grad_norm = 0.5
        self.eval_after_each_episode = False
        self.multi_env_training = False
        self.num_envs = 1
        self.enable_stochastic_rewards = False
        self.reward_variance = 0.0
        self.q_value_perturbation = False
        self.q_perturbation_std = 0.0
        self.exploration_mode = "epsilon_greedy"
        self.reinforce_baseline = False
        self.baseline_lr = 0.01

    def to_dict(self):
        return {
            "learning_rate": self.learning_rate,
            "discount_factor": self.discount_factor,
            "epsilon_start": self.epsilon_start,
            "epsilon_end": self.epsilon_end,
            "epsilon_decay": self.epsilon_decay,
            "min_epsilon": self.min_epsilon,
            "max_epsilon": self.max_epsilon,
            "clip_rewards": self.clip_rewards,
            "reward_clip_value": self.reward_clip_value,
            "update_target_interval": self.update_target_interval,
            "batch_size": self.batch_size,
            "max_memory_size": self.max_memory_size,
            "max_episodes": self.max_episodes,
            "max_steps_per_episode": self.max_steps_per_episode,
            "epsilon_anneal_steps": self.epsilon_anneal_steps,
            "start_learn_step": self.start_learn_step,
            "target_sync_rate": self.target_sync_rate,
            "reward_scaling_factor": self.reward_scaling_factor,
            "reward_shaping": self.reward_shaping,
            "shaping_function": self.shaping_function,
            "adaptive_learning_rate": self.adaptive_learning_rate,
            "learning_rate_decay": self.learning_rate_decay,
            "learning_rate_min": self.learning_rate_min,
            "learning_rate_decay_steps": self.learning_rate_decay_steps,
            "gradient_clip_norm": self.gradient_clip_norm,
            "optimization_method": self.optimization_method,
            "momentum": self.momentum,
            "nesterov": self.nesterov,
            "weight_decay": self.weight_decay,
            "use_double_q": self.use_double_q,
            "use_prioritized_replay": self.use_prioritized_replay,
            "prioritized_alpha": self.prioritized_alpha,
            "prioritized_beta_start": self.prioritized_beta_start,
            "prioritized_beta_end": self.prioritized_beta_end,
            "prioritized_beta_decay_steps": self.prioritized_beta_decay_steps,
            "seed": self.seed,
            "save_checkpoint_interval": self.save_checkpoint_interval,
            "early_stopping": self.early_stopping,
            "early_stopping_reward_threshold": self.early_stopping_reward_threshold,
            "early_stopping_window": self.early_stopping_window,
            "early_stopping_patience": self.early_stopping_patience,
            "initial_exploration_episodes": self.initial_exploration_episodes,
            "replay_burn_in": self.replay_burn_in,
            "max_grad_norm": self.max_grad_norm,
            "lr_schedule_type": self.lr_schedule_type,
            "lr_step_size": self.lr_step_size,
            "lr_gamma": self.lr_gamma,
            "softmax_temperature": self.softmax_temperature,
            "epsilon_schedule_type": self.epsilon_schedule_type,
            "polyak_tau": self.polyak_tau,
            "target_update_interval": self.target_update_interval,
            "reward_normalization": self.reward_normalization,
            "reward_normalization_max_steps": self.reward_normalization_max_steps,
            "reward_scale_limit": self.reward_scale_limit,
            "reward_moving_avg_alpha": self.reward_moving_avg_alpha,
            "huber_loss_delta": self.huber_loss_delta,
            "exploration_final_frame": self.exploration_final_frame,
            "exploration_anneal_type": self.exploration_anneal_type,
            "reward_log_smoothing": self.reward_log_smoothing,
            "reward_log_smoothing_alpha": self.reward_log_smoothing_alpha,
            "q_update_frequency": self.q_update_frequency,
            "grad_accum_steps": self.grad_accum_steps,
            "use_noisy_nets": self.use_noisy_nets,
            "noisy_net_sigma": self.noisy_net_sigma,
            "multi_step_returns": self.multi_step_returns,
            "enable_checkpointing": self.enable_checkpointing,
            "checkpoint_path": self.checkpoint_path,
            "enable_tensorboard": self.enable_tensorboard,
            "tensorboard_log_dir": self.tensorboard_log_dir,
            "random_action_probability": self.random_action_probability,
            "enable_pretraining": self.enable_pretraining,
            "pretraining_episodes": self.pretraining_episodes,
            "optimizer_eps": self.optimizer_eps,
            "optimizer_betas": self.optimizer_betas,
            "use_categorical_dqn": self.use_categorical_dqn,
            "num_atoms": self.num_atoms,
            "v_min": self.v_min,
            "v_max": self.v_max,
            "log_interval": self.log_interval,
            "eval_interval": self.eval_interval,
            "eval_episodes": self.eval_episodes,
            "max_eval_steps": self.max_eval_steps,
            "dueling_dqn": self.dueling_dqn,
            "enable_curiosity": self.enable_curiosity,
            "curiosity_eta": self.curiosity_eta,
            "curiosity_beta": self.curiosity_beta,
            "curiosity_forward_loss_weight": self.curiosity_forward_loss_weight,
            "curiosity_inverse_loss_weight": self.curiosity_inverse_loss_weight,
            "curiosity_buffer_size": self.curiosity_buffer_size,
            "enable_feature_extraction": self.enable_feature_extraction,
            "feature_extractor_type": self.feature_extractor_type,
            "feature_extractor_params": self.feature_extractor_params,
            "target_entropy": self.target_entropy,
            "entropy_coefficient": self.entropy_coefficient,
            "clip_grad_value": self.clip_grad_value,
            "obs_norm": self.obs_norm,
            "obs_clip_min": self.obs_clip_min,
            "obs_clip_max": self.obs_clip_max,
            "dqn_type": self.dqn_type,
            "n_step": self.n_step,
            "custom_loss_function": self.custom_loss_function,
            "custom_metrics_enabled": self.custom_metrics_enabled,
            "custom_metrics": self.custom_metrics,
            "auxiliary_tasks_enabled": self.auxiliary_tasks_enabled,
            "auxiliary_tasks": self.auxiliary_tasks,
            "max_episode_length_override": self.max_episode_length_override,
            "save_best_model": self.save_best_model,
            "best_model_metric": self.best_model_metric,
            "best_model_comparison": self.best_model_comparison,
            "best_model_path": self.best_model_path,
            "rnd_enabled": self.rnd_enabled,
            "rnd_learning_rate": self.rnd_learning_rate,
            "rnd_update_proportion": self.rnd_update_proportion,
            "rnd_obs_clip": self.rnd_obs_clip,
            "sarsa_enabled": self.sarsa_enabled,
            "enable_momentum_exploration": self.enable_momentum_exploration,
            "momentum_exploration_alpha": self.momentum_exploration_alpha,
            "n_step_bootstrap": self.n_step_bootstrap,
            "n_step_lambda": self.n_step_lambda,
            "multi_agent": self.multi_agent,
            "num_agents": self.num_agents,
            "shared_replay_buffer": self.shared_replay_buffer,
            "agent_id_embeddings": self.agent_id_embeddings,
            "agent_id_emb_dim": self.agent_id_emb_dim,
            "enable_popart": self.enable_popart,
            "popart_beta": self.popart_beta,
            "test_exploration_rate": self.test_exploration_rate,
            "learning_rate_warmup": self.learning_rate_warmup,
            "learning_rate_schedule_mode": self.learning_rate_schedule_mode,
            "enable_q_value_clamping": self.enable_q_value_clamping,
            "q_value_clip_min": self.q_value_clip_min,
            "q_value_clip_max": self.q_value_clip_max,
            "use_conservative_q": self.use_conservative_q,
            "conservative_q_lambda": self.conservative_q_lambda,
            "enable_rainbow": self.enable_rainbow,
            "monitor_gpu_usage": self.monitor_gpu_usage,
            "clip_actions": self.clip_actions,
            "action_clip_low": self.action_clip_low,
            "action_clip_high": self.action_clip_high,
            "reward_offset": self.reward_offset,
            "episodic_life": self.episodic_life,
            "render_during_training": self.render_during_training,
            "render_interval": self.render_interval,
            "async_rollout": self.async_rollout,
            "async_rollout_workers": self.async_rollout_workers,
            "async_update": self.async_update,
            "async_update_frequency": self.async_update_frequency,
            "idle_gpu_for_rollout": self.idle_gpu_for_rollout,
            "agent_hierarchy": self.agent_hierarchy,
            "hierarchy_levels": self.hierarchy_levels,
            "hierarchy_manager_index": self.hierarchy_manager_index,
            "hierarchy_worker_indexes": self.hierarchy_worker_indexes,
            "hierarchy_goal_tolerance": self.hierarchy_goal_tolerance,
            "hindsight_experience_replay": self.hindsight_experience_replay,
            "her_k": self.her_k,
            "her_strategy": self.her_strategy,
            "enable_scheduled_target_update": self.enable_scheduled_target_update,
            "scheduled_target_update_times": self.scheduled_target_update_times,
            "constant_reward": self.constant_reward,
            "reward_penalty": self.reward_penalty,
            "decaying_negative_reward": self.decaying_negative_reward,
            "negative_reward_decay_rate": self.negative_reward_decay_rate,
            "auto_adjust_critic_lr": self.auto_adjust_critic_lr,
            "critic_lr_adjust_factor": self.critic_lr_adjust_factor,
            "max_steps_per_training_phase": self.max_steps_per_training_phase,
            "training_phases": self.training_phases,
            "enable_automatic_policy_switch": self.enable_automatic_policy_switch,
            "policy_switch_interval": self.policy_switch_interval,
            "policy_pool": self.policy_pool,
            "policy_pool_weights": self.policy_pool_weights,
            "enable_priority_loss": self.enable_priority_loss,
            "priority_loss_alpha": self.priority_loss_alpha,
            "priority_loss_beta": self.priority_loss_beta,
            "priority_loss_gamma": self.priority_loss_gamma,
            "separate_target_network": self.separate_target_network,
            "rnn_enabled": self.rnn_enabled,
            "rnn_hidden_size": self.rnn_hidden_size,
            "rnn_num_layers": self.rnn_num_layers,
            "rnn_type": self.rnn_type,
            "max_seq_length": self.max_seq_length,
            "random_seed_sharing": self.random_seed_sharing,
            "global_seed": self.global_seed,
            "auto_export_models": self.auto_export_models,
            "export_interval": self.export_interval,
            "export_path": self.export_path,
            "use_distributional_rl": self.use_distributional_rl,
            "atoms": self.atoms,
            "vmin": self.vmin,
            "vmax": self.vmax,
            "transition_storage": self.transition_storage,
            "gae_lambda": self.gae_lambda,
            "enable_advantage_normalization": self.enable_advantage_normalization,
            "l2_reg": self.l2_reg,
            "batch_norm": self.batch_norm,
            "layer_norm": self.layer_norm,
            "param_init_method": self.param_init_method,
            "param_init_gain": self.param_init_gain,
            "init_log_std": self.init_log_std,
            "enable_trust_region": self.enable_trust_region,
            "trust_region_threshold": self.trust_region_threshold,
            "multi_headed_architecture": self.multi_headed_architecture,
            "num_heads": self.num_heads,
            "enable_switching_heads": self.enable_switching_heads,
            "head_switch_interval": self.head_switch_interval,
            "last_config_update_time": self.last_config_update_time,
            "accumulated_training_steps": self.accumulated_training_steps,
            "amp_enabled": self.amp_enabled,
            "distillation_enabled": self.distillation_enabled,
            "distillation_kl_coeff": self.distillation_kl_coeff,
            "distillation_teacher_path": self.distillation_teacher_path,
            "policy_integration_mode": self.policy_integration_mode,
            "policy_integration_paths": self.policy_integration_paths,
            "observation_padding": self.observation_padding,
            "evaluation_metric": self.evaluation_metric,
            "model_save_format": self.model_save_format,
            "enable_online_evaluation": self.enable_online_evaluation,
            "online_evaluation_interval": self.online_evaluation_interval,
            "online_evaluation_metric": self.online_evaluation_metric,
            "online_evaluation_episodes": self.online_evaluation_episodes,
            "enable_checkpoint_rotation": self.enable_checkpoint_rotation,
            "max_checkpoints_to_keep": self.max_checkpoints_to_keep,
            "optimizer_grad_clip_method": self.optimizer_grad_clip_method,
            "custom_training_schedule": self.custom_training_schedule,
            "sparse_reward": self.sparse_reward,
            "enable_mixed_precision": self.enable_mixed_precision,
            "reward_delay": self.reward_delay,
            "distributed_training": self.distributed_training,
            "distributed_world_size": self.distributed_world_size,
            "distributed_rank": self.distributed_rank,
            "distributed_backend": self.distributed_backend,
            "strict_done_on_terminal_state": self.strict_done_on_terminal_state,
            "q_risk_sensitive": self.q_risk_sensitive,
            "q_risk_factor": self.q_risk_factor,
            "enable_episodic_exploration_bonus": self.enable_episodic_exploration_bonus,
            "episodic_exploration_bonus_value": self.episodic_exploration_bonus_value,
            "episodic_exploration_decay": self.episodic_exploration_decay,
            "no_reward_no_terminal": self.no_reward_no_terminal,
            "rnn_stateful_across_episodes": self.rnn_stateful_across_episodes,
            "norm_advantages": self.norm_advantages,
            "policy_clip_range": self.policy_clip_range,
            "enable_twin_q": self.enable_twin_q,
            "enable_clipped_double_q": self.enable_clipped_double_q,
            "twin_q_decay_factor": self.twin_q_decay_factor,
            "enable_contextual_bandit_mode": self.enable_contextual_bandit_mode,
            "context_vector_size": self.context_vector_size,
            "context_embedding_type": self.context_embedding_type,
            "context_embedding_params": self.context_embedding_params,
            "enable_bootstrapped_q": self.enable_bootstrapped_q,
            "bootstrapped_heads": self.bootstrapped_heads,
            "bootstrapped_mask_probability": self.bootstrapped_mask_probability,
            "schedule_adaptation": self.schedule_adaptation,
            "schedule_adaptation_metric": self.schedule_adaptation_metric,
            "schedule_adaptation_threshold": self.schedule_adaptation_threshold,
            "schedule_adaptation_mode": self.schedule_adaptation_mode,
            "enable_gae": self.enable_gae,
            "gae_gamma": self.gae_gamma,
            "enable_reward_bonus_on_success": self.enable_reward_bonus_on_success,
            "success_reward_bonus": self.success_reward_bonus,
            "enable_inverse_dynamics": self.enable_inverse_dynamics,
            "inverse_dynamics_lr": self.inverse_dynamics_lr,
            "inverse_dynamics_beta": self.inverse_dynamics_beta,
            "forward_dynamics_lr": self.forward_dynamics_lr,
            "enable_proximal_updates": self.enable_proximal_updates,
            "proximal_clip_param": self.proximal_clip_param,
            "proximal_critic_loss_weight": self.proximal_critic_loss_weight,
            "proximal_entropy_weight": self.proximal_entropy_weight,
            "proximal_updates_per_step": self.proximal_updates_per_step,
            "proximal_max_grad_norm": self.proximal_max_grad_norm,
            "eval_after_each_episode": self.eval_after_each_episode,
            "multi_env_training": self.multi_env_training,
            "num_envs": self.num_envs,
            "enable_stochastic_rewards": self.enable_stochastic_rewards,
            "reward_variance": self.reward_variance,
            "q_value_perturbation": self.q_value_perturbation,
            "q_perturbation_std": self.q_perturbation_std,
            "exploration_mode": self.exploration_mode,
            "reinforce_baseline": self.reinforce_baseline,
            "baseline_lr": self.baseline_lr
        }

    def from_dict(self, data):
        self.learning_rate = data.get("learning_rate", self.learning_rate)
        self.discount_factor = data.get("discount_factor", self.discount_factor)
        self.epsilon_start = data.get("epsilon_start", self.epsilon_start)
        self.epsilon_end = data.get("epsilon_end", self.epsilon_end)
        self.epsilon_decay = data.get("epsilon_decay", self.epsilon_decay)
        self.min_epsilon = data.get("min_epsilon", self.min_epsilon)
        self.max_epsilon = data.get("max_epsilon", self.max_epsilon)
        self.clip_rewards = data.get("clip_rewards", self.clip_rewards)
        self.reward_clip_value = data.get("reward_clip_value", self.reward_clip_value)
        self.update_target_interval = data.get("update_target_interval", self.update_target_interval)
        self.batch_size = data.get("batch_size", self.batch_size)
        self.max_memory_size = data.get("max_memory_size", self.max_memory_size)
        self.max_episodes = data.get("max_episodes", self.max_episodes)
        self.max_steps_per_episode = data.get("max_steps_per_episode", self.max_steps_per_episode)
        self.epsilon_anneal_steps = data.get("epsilon_anneal_steps", self.epsilon_anneal_steps)
        self.start_learn_step = data.get("start_learn_step", self.start_learn_step)
        self.target_sync_rate = data.get("target_sync_rate", self.target_sync_rate)
        self.reward_scaling_factor = data.get("reward_scaling_factor", self.reward_scaling_factor)
        self.reward_shaping = data.get("reward_shaping", self.reward_shaping)
        self.shaping_function = data.get("shaping_function", self.shaping_function)
        self.adaptive_learning_rate = data.get("adaptive_learning_rate", self.adaptive_learning_rate)
        self.learning_rate_decay = data.get("learning_rate_decay", self.learning_rate_decay)
        self.learning_rate_min = data.get("learning_rate_min", self.learning_rate_min)
        self.learning_rate_decay_steps = data.get("learning_rate_decay_steps", self.learning_rate_decay_steps)
        self.gradient_clip_norm = data.get("gradient_clip_norm", self.gradient_clip_norm)
        self.optimization_method = data.get("optimization_method", self.optimization_method)
        self.momentum = data.get("momentum", self.momentum)
        self.nesterov = data.get("nesterov", self.nesterov)
        self.weight_decay = data.get("weight_decay", self.weight_decay)
        self.use_double_q = data.get("use_double_q", self.use_double_q)
        self.use_prioritized_replay = data.get("use_prioritized_replay", self.use_prioritized_replay)
        self.prioritized_alpha = data.get("prioritized_alpha", self.prioritized_alpha)
        self.prioritized_beta_start = data.get("prioritized_beta_start", self.prioritized_beta_start)
        self.prioritized_beta_end = data.get("prioritized_beta_end", self.prioritized_beta_end)
        self.prioritized_beta_decay_steps = data.get("prioritized_beta_decay_steps", self.prioritized_beta_decay_steps)
        self.seed = data.get("seed", self.seed)
        self.save_checkpoint_interval = data.get("save_checkpoint_interval", self.save_checkpoint_interval)
        self.early_stopping = data.get("early_stopping", self.early_stopping)
        self.early_stopping_reward_threshold = data.get("early_stopping_reward_threshold", self.early_stopping_reward_threshold)
        self.early_stopping_window = data.get("early_stopping_window", self.early_stopping_window)
        self.early_stopping_patience = data.get("early_stopping_patience", self.early_stopping_patience)
        self.initial_exploration_episodes = data.get("initial_exploration_episodes", self.initial_exploration_episodes)
        self.replay_burn_in = data.get("replay_burn_in", self.replay_burn_in)
        self.max_grad_norm = data.get("max_grad_norm", self.max_grad_norm)
        self.lr_schedule_type = data.get("lr_schedule_type", self.lr_schedule_type)
        self.lr_step_size = data.get("lr_step_size", self.lr_step_size)
        self.lr_gamma = data.get("lr_gamma", self.lr_gamma)
        self.softmax_temperature = data.get("softmax_temperature", self.softmax_temperature)
        self.epsilon_schedule_type = data.get("epsilon_schedule_type", self.epsilon_schedule_type)
        self.polyak_tau = data.get("polyak_tau", self.polyak_tau)
        self.target_update_interval = data.get("target_update_interval", self.target_update_interval)
        self.reward_normalization = data.get("reward_normalization", self.reward_normalization)
        self.reward_normalization_max_steps = data.get("reward_normalization_max_steps", self.reward_normalization_max_steps)
        self.reward_scale_limit = data.get("reward_scale_limit", self.reward_scale_limit)
        self.reward_moving_avg_alpha = data.get("reward_moving_avg_alpha", self.reward_moving_avg_alpha)
        self.huber_loss_delta = data.get("huber_loss_delta", self.huber_loss_delta)
        self.exploration_final_frame = data.get("exploration_final_frame", self.exploration_final_frame)
        self.exploration_anneal_type = data.get("exploration_anneal_type", self.exploration_anneal_type)
        self.reward_log_smoothing = data.get("reward_log_smoothing", self.reward_log_smoothing)
        self.reward_log_smoothing_alpha = data.get("reward_log_smoothing_alpha", self.reward_log_smoothing_alpha)
        self.q_update_frequency = data.get("q_update_frequency", self.q_update_frequency)
        self.grad_accum_steps = data.get("grad_accum_steps", self.grad_accum_steps)
        self.use_noisy_nets = data.get("use_noisy_nets", self.use_noisy_nets)
        self.noisy_net_sigma = data.get("noisy_net_sigma", self.noisy_net_sigma)
        self.multi_step_returns = data.get("multi_step_returns", self.multi_step_returns)
        self.enable_checkpointing = data.get("enable_checkpointing", self.enable_checkpointing)
        self.checkpoint_path = data.get("checkpoint_path", self.checkpoint_path)
        self.enable_tensorboard = data.get("enable_tensorboard", self.enable_tensorboard)
        self.tensorboard_log_dir = data.get("tensorboard_log_dir", self.tensorboard_log_dir)
        self.random_action_probability = data.get("random_action_probability", self.random_action_probability)
        self.enable_pretraining = data.get("enable_pretraining", self.enable_pretraining)
        self.pretraining_episodes = data.get("pretraining_episodes", self.pretraining_episodes)
        self.optimizer_eps = data.get("optimizer_eps", self.optimizer_eps)
        self.optimizer_betas = tuple(data.get("optimizer_betas", list(self.optimizer_betas)))
        self.use_categorical_dqn = data.get("use_categorical_dqn", self.use_categorical_dqn)
        self.num_atoms = data.get("num_atoms", self.num_atoms)
        self.v_min = data.get("v_min", self.v_min)
        self.v_max = data.get("v_max", self.v_max)
        self.log_interval = data.get("log_interval", self.log_interval)
        self.eval_interval = data.get("eval_interval", self.eval_interval)
        self.eval_episodes = data.get("eval_episodes", self.eval_episodes)
        self.max_eval_steps = data.get("max_eval_steps", self.max_eval_steps)
        self.dueling_dqn = data.get("dueling_dqn", self.dueling_dqn)
        self.enable_curiosity = data.get("enable_curiosity", self.enable_curiosity)
        self.curiosity_eta = data.get("curiosity_eta", self.curiosity_eta)
        self.curiosity_beta = data.get("curiosity_beta", self.curiosity_beta)
        self.curiosity_forward_loss_weight = data.get("curiosity_forward_loss_weight", self.curiosity_forward_loss_weight)
        self.curiosity_inverse_loss_weight = data.get("curiosity_inverse_loss_weight", self.curiosity_inverse_loss_weight)
        self.curiosity_buffer_size = data.get("curiosity_buffer_size", self.curiosity_buffer_size)
        self.enable_feature_extraction = data.get("enable_feature_extraction", self.enable_feature_extraction)
        self.feature_extractor_type = data.get("feature_extractor_type", self.feature_extractor_type)
        self.feature_extractor_params = data.get("feature_extractor_params", self.feature_extractor_params)
        self.target_entropy = data.get("target_entropy", self.target_entropy)
        self.entropy_coefficient = data.get("entropy_coefficient", self.entropy_coefficient)
        self.clip_grad_value = data.get("clip_grad_value", self.clip_grad_value)
        self.obs_norm = data.get("obs_norm", self.obs_norm)
        self.obs_clip_min = data.get("obs_clip_min", self.obs_clip_min)
        self.obs_clip_max = data.get("obs_clip_max", self.obs_clip_max)
        self.dqn_type = data.get("dqn_type", self.dqn_type)
        self.n_step = data.get("n_step", self.n_step)
        self.custom_loss_function = data.get("custom_loss_function", self.custom_loss_function)
        self.custom_metrics_enabled = data.get("custom_metrics_enabled", self.custom_metrics_enabled)
        self.custom_metrics = data.get("custom_metrics", self.custom_metrics)
        self.auxiliary_tasks_enabled = data.get("auxiliary_tasks_enabled", self.auxiliary_tasks_enabled)
        self.auxiliary_tasks = data.get("auxiliary_tasks", self.auxiliary_tasks)
        self.max_episode_length_override = data.get("max_episode_length_override", self.max_episode_length_override)
        self.save_best_model = data.get("save_best_model", self.save_best_model)
        self.best_model_metric = data.get("best_model_metric", self.best_model_metric)
        self.best_model_comparison = data.get("best_model_comparison", self.best_model_comparison)
        self.best_model_path = data.get("best_model_path", self.best_model_path)
        self.rnd_enabled = data.get("rnd_enabled", self.rnd_enabled)
        self.rnd_learning_rate = data.get("rnd_learning_rate", self.rnd_learning_rate)
        self.rnd_update_proportion = data.get("rnd_update_proportion", self.rnd_update_proportion)
        self.rnd_obs_clip = data.get("rnd_obs_clip", self.rnd_obs_clip)
        self.sarsa_enabled = data.get("sarsa_enabled", self.sarsa_enabled)
        self.enable_momentum_exploration = data.get("enable_momentum_exploration", self.enable_momentum_exploration)
        self.momentum_exploration_alpha = data.get("momentum_exploration_alpha", self.momentum_exploration_alpha)
        self.n_step_bootstrap = data.get("n_step_bootstrap", self.n_step_bootstrap)
        self.n_step_lambda = data.get("n_step_lambda", self.n_step_lambda)
        self.multi_agent = data.get("multi_agent", self.multi_agent)
        self.num_agents = data.get("num_agents", self.num_agents)
        self.shared_replay_buffer = data.get("shared_replay_buffer", self.shared_replay_buffer)
        self.agent_id_embeddings = data.get("agent_id_embeddings", self.agent_id_embeddings)
        self.agent_id_emb_dim = data.get("agent_id_emb_dim", self.agent_id_emb_dim)
        self.enable_popart = data.get("enable_popart", self.enable_popart)
        self.popart_beta = data.get("popart_beta", self.popart_beta)
        self.test_exploration_rate = data.get("test_exploration_rate", self.test_exploration_rate)
        self.learning_rate_warmup = data.get("learning_rate_warmup", self.learning_rate_warmup)
        self.learning_rate_schedule_mode = data.get("learning_rate_schedule_mode", self.learning_rate_schedule_mode)
        self.enable_q_value_clamping = data.get("enable_q_value_clamping", self.enable_q_value_clamping)
        self.q_value_clip_min = data.get("q_value_clip_min", self.q_value_clip_min)
        self.q_value_clip_max = data.get("q_value_clip_max", self.q_value_clip_max)
        self.use_conservative_q = data.get("use_conservative_q", self.use_conservative_q)
        self.conservative_q_lambda = data.get("conservative_q_lambda", self.conservative_q_lambda)
        self.enable_rainbow = data.get("enable_rainbow", self.enable_rainbow)
        self.monitor_gpu_usage = data.get("monitor_gpu_usage", self.monitor_gpu_usage)
        self.clip_actions = data.get("clip_actions", self.clip_actions)
        self.action_clip_low = data.get("action_clip_low", self.action_clip_low)
        self.action_clip_high = data.get("action_clip_high", self.action_clip_high)
        self.reward_offset = data.get("reward_offset", self.reward_offset)
        self.episodic_life = data.get("episodic_life", self.episodic_life)
        self.render_during_training = data.get("render_during_training", self.render_during_training)
        self.render_interval = data.get("render_interval", self.render_interval)
        self.async_rollout = data.get("async_rollout", self.async_rollout)
        self.async_rollout_workers = data.get("async_rollout_workers", self.async_rollout_workers)
        self.async_update = data.get("async_update", self.async_update)
        self.async_update_frequency = data.get("async_update_frequency", self.async_update_frequency)
        self.idle_gpu_for_rollout = data.get("idle_gpu_for_rollout", self.idle_gpu_for_rollout)
        self.agent_hierarchy = data.get("agent_hierarchy", self.agent_hierarchy)
        self.hierarchy_levels = data.get("hierarchy_levels", self.hierarchy_levels)
        self.hierarchy_manager_index = data.get("hierarchy_manager_index", self.hierarchy_manager_index)
        self.hierarchy_worker_indexes = data.get("hierarchy_worker_indexes", self.hierarchy_worker_indexes)
        self.hierarchy_goal_tolerance = data.get("hierarchy_goal_tolerance", self.hierarchy_goal_tolerance)
        self.hindsight_experience_replay = data.get("hindsight_experience_replay", self.hindsight_experience_replay)
        self.her_k = data.get("her_k", self.her_k)
        self.her_strategy = data.get("her_strategy", self.her_strategy)
        self.enable_scheduled_target_update = data.get("enable_scheduled_target_update", self.enable_scheduled_target_update)
        self.scheduled_target_update_times = data.get("scheduled_target_update_times", self.scheduled_target_update_times)
        self.constant_reward = data.get("constant_reward", self.constant_reward)
        self.reward_penalty = data.get("reward_penalty", self.reward_penalty)
        self.decaying_negative_reward = data.get("decaying_negative_reward", self.decaying_negative_reward)
        self.negative_reward_decay_rate = data.get("negative_reward_decay_rate", self.negative_reward_decay_rate)
        self.auto_adjust_critic_lr = data.get("auto_adjust_critic_lr", self.auto_adjust_critic_lr)
        self.critic_lr_adjust_factor = data.get("critic_lr_adjust_factor", self.critic_lr_adjust_factor)
        self.max_steps_per_training_phase = data.get("max_steps_per_training_phase", self.max_steps_per_training_phase)
        self.training_phases = data.get("training_phases", self.training_phases)
        self.enable_automatic_policy_switch = data.get("enable_automatic_policy_switch", self.enable_automatic_policy_switch)
        self.policy_switch_interval = data.get("policy_switch_interval", self.policy_switch_interval)
        self.policy_pool = data.get("policy_pool", self.policy_pool)
        self.policy_pool_weights = data.get("policy_pool_weights", self.policy_pool_weights)
        self.enable_priority_loss = data.get("enable_priority_loss", self.enable_priority_loss)
        self.priority_loss_alpha = data.get("priority_loss_alpha", self.priority_loss_alpha)
        self.priority_loss_beta = data.get("priority_loss_beta", self.priority_loss_beta)
        self.priority_loss_gamma = data.get("priority_loss_gamma", self.priority_loss_gamma)
        self.separate_target_network = data.get("separate_target_network", self.separate_target_network)
        self.rnn_enabled = data.get("rnn_enabled", self.rnn_enabled)
        self.rnn_hidden_size = data.get("rnn_hidden_size", self.rnn_hidden_size)
        self.rnn_num_layers = data.get("rnn_num_layers", self.rnn_num_layers)
        self.rnn_type = data.get("rnn_type", self.rnn_type)
        self.max_seq_length = data.get("max_seq_length", self.max_seq_length)
        self.random_seed_sharing = data.get("random_seed_sharing", self.random_seed_sharing)
        self.global_seed = data.get("global_seed", self.global_seed)
        self.auto_export_models = data.get("auto_export_models", self.auto_export_models)
        self.export_interval = data.get("export_interval", self.export_interval)
        self.export_path = data.get("export_path", self.export_path)
        self.use_distributional_rl = data.get("use_distributional_rl", self.use_distributional_rl)
        self.atoms = data.get("atoms", self.atoms)
        self.vmin = data.get("vmin", self.vmin)
        self.vmax = data.get("vmax", self.vmax)
        self.transition_storage = data.get("transition_storage", self.transition_storage)
        self.gae_lambda = data.get("gae_lambda", self.gae_lambda)
        self.enable_advantage_normalization = data.get("enable_advantage_normalization", self.enable_advantage_normalization)
        self.l2_reg = data.get("l2_reg", self.l2_reg)
        self.batch_norm = data.get("batch_norm", self.batch_norm)
        self.layer_norm = data.get("layer_norm", self.layer_norm)
        self.param_init_method = data.get("param_init_method", self.param_init_method)
        self.param_init_gain = data.get("param_init_gain", self.param_init_gain)
        self.init_log_std = data.get("init_log_std", self.init_log_std)
        self.enable_trust_region = data.get("enable_trust_region", self.enable_trust_region)
        self.trust_region_threshold = data.get("trust_region_threshold", self.trust_region_threshold)
        self.multi_headed_architecture = data.get("multi_headed_architecture", self.multi_headed_architecture)
        self.num_heads = data.get("num_heads", self.num_heads)
        self.enable_switching_heads = data.get("enable_switching_heads", self.enable_switching_heads)
        self.head_switch_interval = data.get("head_switch_interval", self.head_switch_interval)
        self.last_config_update_time = data.get("last_config_update_time", self.last_config_update_time)
        self.accumulated_training_steps = data.get("accumulated_training_steps", self.accumulated_training_steps)
        self.amp_enabled = data.get("amp_enabled", self.amp_enabled)
        self.distillation_enabled = data.get("distillation_enabled", self.distillation_enabled)
        self.distillation_kl_coeff = data.get("distillation_kl_coeff", self.distillation_kl_coeff)
        self.distillation_teacher_path = data.get("distillation_teacher_path", self.distillation_teacher_path)
        self.policy_integration_mode = data.get("policy_integration_mode", self.policy_integration_mode)
        self.policy_integration_paths = data.get("policy_integration_paths", self.policy_integration_paths)
        self.observation_padding = data.get("observation_padding", self.observation_padding)
        self.evaluation_metric = data.get("evaluation_metric", self.evaluation_metric)
        self.model_save_format = data.get("model_save_format", self.model_save_format)
        self.enable_online_evaluation = data.get("enable_online_evaluation", self.enable_online_evaluation)
        self.online_evaluation_interval = data.get("online_evaluation_interval", self.online_evaluation_interval)
        self.online_evaluation_metric = data.get("online_evaluation_metric", self.online_evaluation_metric)
        self.online_evaluation_episodes = data.get("online_evaluation_episodes", self.online_evaluation_episodes)
        self.enable_checkpoint_rotation = data.get("enable_checkpoint_rotation", self.enable_checkpoint_rotation)
        self.max_checkpoints_to_keep = data.get("max_checkpoints_to_keep", self.max_checkpoints_to_keep)
        self.optimizer_grad_clip_method = data.get("optimizer_grad_clip_method", self.optimizer_grad_clip_method)
        self.custom_training_schedule = data.get("custom_training_schedule", self.custom_training_schedule)
        self.sparse_reward = data.get("sparse_reward", self.sparse_reward)
        self.enable_mixed_precision = data.get("enable_mixed_precision", self.enable_mixed_precision)
        self.reward_delay = data.get("reward_delay", self.reward_delay)
        self.distributed_training = data.get("distributed_training", self.distributed_training)
        self.distributed_world_size = data.get("distributed_world_size", self.distributed_world_size)
        self.distributed_rank = data.get("distributed_rank", self.distributed_rank)
        self.distributed_backend = data.get("distributed_backend", self.distributed_backend)
        self.strict_done_on_terminal_state = data.get("strict_done_on_terminal_state", self.strict_done_on_terminal_state)
        self.q_risk_sensitive = data.get("q_risk_sensitive", self.q_risk_sensitive)
        self.q_risk_factor = data.get("q_risk_factor", self.q_risk_factor)
        self.enable_episodic_exploration_bonus = data.get("enable_episodic_exploration_bonus", self.enable_episodic_exploration_bonus)
        self.episodic_exploration_bonus_value = data.get("episodic_exploration_bonus_value", self.episodic_exploration_bonus_value)
        self.episodic_exploration_decay = data.get("episodic_exploration_decay", self.episodic_exploration_decay)
        self.no_reward_no_terminal = data.get("no_reward_no_terminal", self.no_reward_no_terminal)
        self.rnn_stateful_across_episodes = data.get("rnn_stateful_across_episodes", self.rnn_stateful_across_episodes)
        self.norm_advantages = data.get("norm_advantages", self.norm_advantages)
        self.policy_clip_range = data.get("policy_clip_range", self.policy_clip_range)
        self.enable_twin_q = data.get("enable_twin_q", self.enable_twin_q)
        self.enable_clipped_double_q = data.get("enable_clipped_double_q", self.enable_clipped_double_q)
        self.twin_q_decay_factor = data.get("twin_q_decay_factor", self.twin_q_decay_factor)
        self.enable_contextual_bandit_mode = data.get("enable_contextual_bandit_mode", self.enable_contextual_bandit_mode)
        self.context_vector_size = data.get("context_vector_size", self.context_vector_size)
        self.context_embedding_type = data.get("context_embedding_type", self.context_embedding_type)
        self.context_embedding_params = data.get("context_embedding_params", self.context_embedding_params)
        self.enable_bootstrapped_q = data.get("enable_bootstrapped_q", self.enable_bootstrapped_q)
        self.bootstrapped_heads = data.get("bootstrapped_heads", self.bootstrapped_heads)
        self.bootstrapped_mask_probability = data.get("bootstrapped_mask_probability", self.bootstrapped_mask_probability)
        self.schedule_adaptation = data.get("schedule_adaptation", self.schedule_adaptation)
        self.schedule_adaptation_metric = data.get("schedule_adaptation_metric", self.schedule_adaptation_metric)
        self.schedule_adaptation_threshold = data.get("schedule_adaptation_threshold", self.schedule_adaptation_threshold)
        self.schedule_adaptation_mode = data.get("schedule_adaptation_mode", self.schedule_adaptation_mode)
        self.enable_gae = data.get("enable_gae", self.enable_gae)
        self.gae_gamma = data.get("gae_gamma", self.gae_gamma)
        self.enable_reward_bonus_on_success = data.get("enable_reward_bonus_on_success", self.enable_reward_bonus_on_success)
        self.success_reward_bonus = data.get("success_reward_bonus", self.success_reward_bonus)
        self.enable_inverse_dynamics = data.get("enable_inverse_dynamics", self.enable_inverse_dynamics)
        self.inverse_dynamics_lr = data.get("inverse_dynamics_lr", self.inverse_dynamics_lr)
        self.inverse_dynamics_beta = data.get("inverse_dynamics_beta", self.inverse_dynamics_beta)
        self.forward_dynamics_lr = data.get("forward_dynamics_lr", self.forward_dynamics_lr)
        self.enable_proximal_updates = data.get("enable_proximal_updates", self.enable_proximal_updates)
        self.proximal_clip_param = data.get("proximal_clip_param", self.proximal_clip_param)
        self.proximal_critic_loss_weight = data.get("proximal_critic_loss_weight", self.proximal_critic_loss_weight)
        self.proximal_entropy_weight = data.get("proximal_entropy_weight", self.proximal_entropy_weight)
        self.proximal_updates_per_step = data.get("proximal_updates_per_step", self.proximal_updates_per_step)
        self.proximal_max_grad_norm = data.get("proximal_max_grad_norm", self.proximal_max_grad_norm)
        self.eval_after_each_episode = data.get("eval_after_each_episode", self.eval_after_each_episode)
        self.multi_env_training = data.get("multi_env_training", self.multi_env_training)
        self.num_envs = data.get("num_envs", self.num_envs)
        self.enable_stochastic_rewards = data.get("enable_stochastic_rewards", self.enable_stochastic_rewards)
        self.reward_variance = data.get("reward_variance", self.reward_variance)
        self.q_value_perturbation = data.get("q_value_perturbation", self.q_value_perturbation)
        self.q_perturbation_std = data.get("q_perturbation_std", self.q_perturbation_std)
        self.exploration_mode = data.get("exploration_mode", self.exploration_mode)
        self.reinforce_baseline = data.get("reinforce_baseline", self.reinforce_baseline)
        self.baseline_lr = data.get("baseline_lr", self.baseline_lr)

class RLConfig:
    def __init__(self):
        self.qlearning_params = QLearningHyperparams()
        self.rl_config_file_path = "rl_config.json"
        self.backup_on_save = True
        self.backup_folder = "rl_backups"
        self.last_load_time = None
        self.last_save_time = None
        self.enable_encryption = False
        self.encryption_key = None
        self.load_error_count = 0
        self.max_load_errors = 3
        self.lock_file_path = "rl_config.lock"
        self.enable_lock_file = False
        self.lock_file_handle = None

    def load(self):
        if not os.path.exists(self.rl_config_file_path):
            return
        try:
            with open(self.rl_config_file_path, "r", encoding="utf-8") as f:
                if self.enable_encryption and self.encryption_key:
                    raw = f.read()
                    decrypted = self.decrypt(raw)
                    data = json.loads(decrypted)
                else:
                    data = json.load(f)
            if "qlearning_params" in data:
                self.qlearning_params.from_dict(data["qlearning_params"])
            self.rl_config_file_path = data.get("rl_config_file_path", self.rl_config_file_path)
            self.backup_on_save = data.get("backup_on_save", self.backup_on_save)
            self.backup_folder = data.get("backup_folder", self.backup_folder)
            self.last_load_time = data.get("last_load_time", self.last_load_time)
            self.last_save_time = data.get("last_save_time", self.last_save_time)
            self.enable_encryption = data.get("enable_encryption", self.enable_encryption)
            self.encryption_key = data.get("encryption_key", self.encryption_key)
            self.load_error_count = data.get("load_error_count", self.load_error_count)
            self.max_load_errors = data.get("max_load_errors", self.max_load_errors)
            self.lock_file_path = data.get("lock_file_path", self.lock_file_path)
            self.enable_lock_file = data.get("enable_lock_file", self.enable_lock_file)
            self.last_load_time = time.time()
        except:
            self.load_error_count += 1

    def save(self):
        if self.load_error_count >= self.max_load_errors:
            return
        data = {
            "qlearning_params": self.qlearning_params.to_dict(),
            "rl_config_file_path": self.rl_config_file_path,
            "backup_on_save": self.backup_on_save,
            "backup_folder": self.backup_folder,
            "last_load_time": self.last_load_time,
            "last_save_time": self.last_save_time,
            "enable_encryption": self.enable_encryption,
            "encryption_key": self.encryption_key,
            "load_error_count": self.load_error_count,
            "max_load_errors": self.max_load_errors,
            "lock_file_path": self.lock_file_path,
            "enable_lock_file": self.enable_lock_file
        }
        if self.backup_on_save and os.path.exists(self.rl_config_file_path):
            if not os.path.exists(self.backup_folder):
                os.makedirs(self.backup_folder)
            backup_file = os.path.join(self.backup_folder, f"rl_config_backup_{int(time.time())}.json")
            shutil.copy(self.rl_config_file_path, backup_file)
        if self.enable_encryption and self.encryption_key:
            encoded = json.dumps(data)
            encrypted = self.encrypt(encoded)
            with open(self.rl_config_file_path, "w", encoding="utf-8") as f:
                f.write(encrypted)
        else:
            with open(self.rl_config_file_path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2)
        self.last_save_time = time.time()

    def encrypt(self, data):
        if not self.encryption_key:
            return data
        r = []
        for i, c in enumerate(data):
            r.append(chr(ord(c) ^ ord(self.encryption_key[i % len(self.encryption_key)])))
        return "".join(r)

    def decrypt(self, data):
        if not self.encryption_key:
            return data
        r = []
        for i, c in enumerate(data):
            r.append(chr(ord(c) ^ ord(self.encryption_key[i % len(self.encryption_key)])))
        return "".join(r)

    def acquire_lock(self):
        if not self.enable_lock_file:
            return
        if os.path.exists(self.lock_file_path):
            return
        self.lock_file_handle = open(self.lock_file_path, "w")
        self.lock_file_handle.write(str(os.getpid()))
        self.lock_file_handle.flush()

    def release_lock(self):
        if not self.enable_lock_file:
            return
        if self.lock_file_handle:
            self.lock_file_handle.close()
        if os.path.exists(self.lock_file_path):
            os.remove(self.lock_file_path)

    def initialize(self):
        self.acquire_lock()
        self.load()

    def finalize(self):
        self.release_lock()
