from __future__ import annotations

import pathlib
import time
from functools import partial

import equinox as eqx
import jax
import jax.numpy as jnp
import optax
from jax.flatten_util import ravel_pytree

from crazyplan import visualization as viz
from crazyplan.core.policy import OpenLoopPolicy, Policy
from crazyplan.core.rollout import Rollout
from crazyplan.core.sim import SimWrapper, make_initialized_sim


def save_policy(pol: Policy, path: pathlib.Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    eqx.tree_serialise_leaves(path, pol)


def _window_initial_states(
    *,
    policy: OpenLoopPolicy,
    rollout,
    initial_state,
    window_start,
):
    shape = (policy.n_worlds, policy.n_drones, 1)
    if window_start <= 0:
        zero = jnp.zeros(shape)
        return initial_state, zero, zero

    zero_theta = jnp.zeros(shape)
    final_state, final_theta, final_dtheta = rollout.advance_prefix(
        policy=policy,
        initial_state=initial_state,
        steps=jnp.asarray(window_start, dtype=jnp.int32),
        initial_theta=zero_theta,
        initial_dtheta=zero_theta,
    )
    return final_state, final_theta, final_dtheta


def train_policy(
    policy: OpenLoopPolicy,
    track,
    cfg,
    *,
    plot_enabled: bool,
    render_enabled: bool,
):
    settings = cfg.train
    cost_cfg = cfg.cost
    track_cfg = cfg.track_cfg
    checkpoint_dir = settings.checkpoint_dir
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    plotter = viz.TrajectoryPlotter(track, settings.plot_dir) if plot_enabled else None
    if plot_enabled:
        settings.plot_dir.mkdir(parents=True, exist_ok=True)

    trainer = WindowTrainer(policy=policy, track=track, cost_cfg=cost_cfg, settings=settings, plotter=plotter)

    total_step = 0

    for episode_idx, episode in enumerate(settings.episodes):
        raw_sim = make_initialized_sim(
            n_worlds=settings.n_worlds,
            n_drones=settings.n_drones,
            sim_cfg=settings.sim_cfg,
            initial_state=settings.initial_state,
        )
        sim = SimWrapper(sim=raw_sim)
        initial_state = jax.tree_util.tree_map(lambda x: x, raw_sim.data)
        total_step = trainer.run(
            sim=sim,
            initial_state=initial_state,
            episode_cfg=episode,
            episode_idx=episode_idx,
            total_step=total_step,
        )

    policy = trainer.current_policy

    if render_enabled:
        raw_sim.reset()
        render_sim = SimWrapper(sim=raw_sim)
        render_rollout = Rollout(sim=render_sim, track=track, traj_size=settings.traj_size, cost=cost_cfg)

        shape = (policy.n_worlds, policy.n_drones, 1)
        zero_theta = jnp.zeros(shape)
        states, _, _, _, _, _ = render_rollout.rollout(
            policy=policy,
            initial_state=initial_state,
            policy_offset=0,
            initial_theta=zero_theta,
            initial_dtheta=zero_theta,
        )

        n_pts = 250
        theta_grid = jnp.linspace(0.0, 1.0, n_pts)
        wps = render_rollout.track.position(theta=theta_grid)
        raw_sim = sim.sim
        raw_sim.reset()
        frames = viz.render_rollout_frames(
                                states=states,
                                sim=raw_sim,
                                config=track_cfg,
                                world=settings.render_world,
                                waypoints=wps,
                                traces=True,
                            )
        viz.save_video(frames, settings.render_output, fps=30)


    return policy


class WindowTrainer:
    def __init__(self, *, policy: Policy, track, cost_cfg, settings, plotter: viz.TrajectoryPlotter | None):
        self.policy_params, self.policy_static = eqx.partition(policy, eqx.is_array)
        self.track = track
        self.cost_cfg = cost_cfg
        self.settings = settings
        self.plotter = plotter

    def run(
        self,
        *,
        sim,
        initial_state,
        episode_cfg,
        episode_idx: int,
        total_step: int,
    ):
        policy_params = self.policy_params
        policy_static = self.policy_static
        window_limit = int(episode_cfg.window)
        traj_size = int(self.settings.traj_size)
        scale_penalty = episode_cfg.scale_penalty
        lr = episode_cfg.learning_rate
        steps_per_episode = int(self.settings.steps_per_episode)
        checkpoint_dir = self.settings.checkpoint_dir
        plot_every = int(self.settings.plot_every)
        log_interval = int(self.settings.log_interval)

        @jax.jit
        def loss_fn_window(
            pol_params,
            rollout,
            initial_state,
            initial_theta,
            initial_dtheta,
            window_size,
            scale_penalty,
            policy_offset,
        ):
            policy_obj = eqx.combine(pol_params, policy_static)

            frozen_state = jax.tree_util.tree_map(jax.lax.stop_gradient, initial_state)
            frozen_theta = jax.tree_util.tree_map(jax.lax.stop_gradient, initial_theta)
            frozen_dtheta = jax.tree_util.tree_map(jax.lax.stop_gradient, initial_dtheta)
            states, actions, cost, theta_hist, dtheta_hist, carry_next = rollout.rollout(
                policy=policy_obj,
                initial_state=frozen_state,
                scale_penalty=scale_penalty,
                policy_offset=policy_offset,
                initial_theta=frozen_theta,
                initial_dtheta=frozen_dtheta,
            )
            segment = jnp.sum(cost, axis=0) / jnp.maximum(window_size, 1)
            return jnp.mean(segment), (states, theta_hist, dtheta_hist, carry_next)

        optimizer = optax.chain(
            optax.clip_by_global_norm(1.0),
            optax.adam(learning_rate=lr),
        )
        opt_state = optimizer.init(policy_params)

        total_time = 0.0
        last_loss_time = 0.0
        last_fetch_time = 0.0

        rollout = Rollout(sim=sim, track=self.track, traj_size=window_limit, cost=self.cost_cfg)

        for step_in_episode in range(steps_per_episode):
            total_step += 1
            frac = (step_in_episode + 1) / steps_per_episode
            horizon = int(min(traj_size, frac * traj_size + 1))

            window_size = int(min(episode_cfg.window, horizon)) or 1
            window_start = int(max(horizon - window_size, 0))

            fetch_t0 = time.perf_counter()
            current_policy = eqx.combine(policy_params, policy_static)
            frozen_state, frozen_theta, frozen_dtheta = _window_initial_states(
                policy=current_policy,
                rollout=rollout,
                initial_state=initial_state,
                window_start=window_start,
            )
            fetch_time = time.perf_counter() - fetch_t0

            current_loss_fn = partial(
                loss_fn_window,
                rollout=rollout,
                initial_state=frozen_state,
                initial_theta=frozen_theta,
                initial_dtheta=frozen_dtheta,
                window_size=window_size,
                scale_penalty=scale_penalty,
                policy_offset=window_start,
            )

            t0 = time.perf_counter()
            loss_and_grad_fn = eqx.filter_value_and_grad(lambda p: current_loss_fn(p), has_aux=True)
            (loss_val, _), grads = loss_and_grad_fn(policy_params)
            updates, opt_state = optimizer.update(grads, opt_state, params=policy_params)
            policy_params = eqx.apply_updates(policy_params, updates)
            t1 = time.perf_counter()
            total_time += t1 - t0
            last_loss_time = t1 - t0
            last_fetch_time = fetch_time

            if total_step % log_interval == 0:
                current_policy = eqx.combine(policy_params, policy_static)
                save_policy(
                    current_policy,
                    path=checkpoint_dir / f"trained_policy_step_{total_step:06d}.eqx",
                )
                grad_vec, _ = ravel_pytree(grads)
                grad_norm = jnp.linalg.norm(grad_vec)
                timing_msg = (
                    f", loss_time={last_loss_time:.3f}s, fetch_time={last_fetch_time:.3f}s"
                    if self.settings.debug_timing
                    else ""
                )
                print(
                    f"[train] step={total_step}, episode={episode_idx}, window={window_limit}, "
                    f"window_start={window_start}, window_size={window_size}, horizon={horizon}, "
                    f"lr={lr:.2e}, Loss={loss_val:.4f}, grad_norm={grad_norm:.4f}, "
                    f"time/step={total_time / log_interval:.2f}s{timing_msg}",
                )
                total_time = 0.0
                if self.plotter and total_step % plot_every == 0:
                    current_policy = eqx.combine(policy_params, policy_static)
                    rollout_full = Rollout(sim=sim, track=self.track, traj_size=traj_size, cost=self.cost_cfg)
                    
                    initial_state, initial_theta, initial_dtheta = _window_initial_states(
                        policy=current_policy,
                        rollout=rollout_full,
                        initial_state=initial_state,
                        window_start=0,
                    )
                    states, _, _, _, _, _ = rollout_full.rollout(
                        policy=current_policy,
                        initial_state=initial_state,
                        policy_offset=0,
                        initial_theta=initial_theta,
                        initial_dtheta=initial_dtheta,
                    )
                    self.plotter.plot_window(
                        states=states,
                        step=total_step,
                        window_start=window_start,
                        window_size=window_size,
                        horizon=horizon,
                    )

        self.policy_params = policy_params
        self.policy_static = policy_static
        return total_step

    @property
    def current_policy(self) -> OpenLoopPolicy:
        return eqx.combine(self.policy_params, self.policy_static)
