import argparse
import pathlib

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import yaml

from crazyplan import visualization as viz
from crazyplan.core.policy import OpenLoopPolicy
from crazyplan.core.rollout import Rollout
from crazyplan.core.sim import SimWrapper, build_sim, create_initial_state, make_initialized_sim
from crazyplan.tracks.track_loader import TrackLoader
from crazyplan.core.config import load_train_config


def _make_policy_skeleton(traj_size: int, n_worlds: int, n_drones: int, policy_cfg: dict) -> OpenLoopPolicy:
    zeros_ctrl = jnp.zeros((traj_size, n_worlds, n_drones, 4), dtype=jnp.float32)
    zeros_dtheta = jnp.zeros((traj_size, n_worlds, n_drones, 1), dtype=jnp.float32)
    section = policy_cfg["policy"] if "policy" in policy_cfg else policy_cfg
    return OpenLoopPolicy(zeros_ctrl, zeros_dtheta, config=section)


def _drone_state_to_dict(state, drone_idx):
    s = state.states
    return {
        "pos": np.asarray(s.pos[0, drone_idx]).tolist(),
        "vel": np.asarray(s.vel[0, drone_idx]).tolist(),
        "quat": np.asarray(s.quat[0, drone_idx]).tolist(),
        "ang_vel": np.asarray(s.ang_vel[0, drone_idx]).tolist(),
        "force": np.asarray(s.force[0, drone_idx]).tolist(),
        "torque": np.asarray(s.torque[0, drone_idx]).tolist(),
        "rotor_vel": np.asarray(s.rotor_vel[0, drone_idx]).tolist(),
    }


def export_trajectory(
    *,
    cfg,
    output_dir: pathlib.Path,
    visualize_mode: str = "none",
    checkpoint: pathlib.Path | None = None,
):

    train = cfg.train
    track_cfg = cfg.track_cfg
    track_cfg_path = train.track_config
    policy_path = pathlib.Path(checkpoint) if checkpoint is not None else train.policy_output

    traj_size = train.traj_size
    n_worlds = train.n_worlds
    n_drones = train.n_drones
    seed = train.seed
    plot_dir = train.plot_dir
    render_output = train.render_output
    render_world = train.render_world

    skeleton = _make_policy_skeleton(traj_size, n_worlds, n_drones, cfg.raw)
    policy = eqx.tree_deserialise_leaves(policy_path, skeleton)

    key = jax.random.PRNGKey(seed)
    key, init_state_key = jax.random.split(key)
    template_sim = build_sim(
        n_worlds,
        n_drones,
        sim_cfg=train.sim_cfg,
    )
    template_sim.data = create_initial_state(train.initial_state_cfg, init_state_key, template_sim.data)
    template_sim.build_default_data()
    initial_state = jax.tree_util.tree_map(lambda x: x, template_sim.data)
    raw_sim = make_initialized_sim(
        n_worlds=n_worlds,
        n_drones=n_drones,
        sim_cfg=train.sim_cfg,
        initial_state=initial_state,
    )
    sim = SimWrapper(sim=raw_sim)
    track = TrackLoader(cfg=track_cfg).build()

    carry = policy.init_carry(raw_sim.data.states)
    state = raw_sim.data

    drones_records = [[] for _ in range(n_drones)]
    for t in range(traj_size):
        (cmd, _), carry = policy(state.states, carry)
        action = np.asarray(cmd).tolist()
        state = sim.step(state, cmd)
        for d in range(n_drones):
            drones_records[d].append({
                "t": t,
                "action": action[0][d],
                "state": _drone_state_to_dict(state, d),
            })

    output_dir.mkdir(parents=True, exist_ok=True)
    for d in range(n_drones):
        data = {
            "checkpoint": str(policy_path),
            "config": str(track_cfg_path),
            "traj_size": traj_size,
            "drone_index": d,
            "records": drones_records[d],
        }
        output_path = output_dir / f"trajectory_drone_{d:02d}.yaml"
        with output_path.open("w") as f:
            yaml.safe_dump(data, f, sort_keys=False)
        print(f"Saved trajectory for drone {d} to {output_path}")

        csv_path = output_dir / f"trajectory_drone_{d:02d}.csv"
        import csv
        state_keys = ["pos_x", "pos_y", "pos_z",
                      "vel_x", "vel_y", "vel_z",
                      "quat_w", "quat_x", "quat_y", "quat_z",
                      "ang_vel_x", "ang_vel_y", "ang_vel_z"]
        writer_header = ["t", "roll", "pitch", "yaw", "thrust"] + state_keys
        with csv_path.open("w", newline="") as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(writer_header)
            for rec in drones_records[d]:
                action = rec["action"]
                state_vals = rec["state"]
                row = [
                    rec["t"],
                    action[0], action[1], action[2], action[3],
                    state_vals["pos"][0], state_vals["pos"][1], state_vals["pos"][2],
                    state_vals["vel"][0], state_vals["vel"][1], state_vals["vel"][2],
                    state_vals["quat"][0], state_vals["quat"][1], state_vals["quat"][2], state_vals["quat"][3],
                    state_vals["ang_vel"][0], state_vals["ang_vel"][1], state_vals["ang_vel"][2],
                ]
                writer.writerow(row)
        print(f"Saved trajectory for drone {d} to {csv_path}")

    if visualize_mode != "none":
        render_raw_sim = make_initialized_sim(
            n_worlds=n_worlds,
            n_drones=n_drones,
            sim_cfg=train.sim_cfg,
            initial_state=initial_state,
        )
        render_sim = SimWrapper(sim=render_raw_sim)
        render_rollout = Rollout(sim=render_sim, track=track, traj_size=traj_size, cost=cfg.cost)
        theta_shape = render_raw_sim.data.states.pos.shape[:2] + (1,)
        zero_theta = jnp.zeros(theta_shape, dtype=render_raw_sim.data.states.pos.dtype)
        states, _, _, _, _, _ = render_rollout.rollout(
            policy=policy,
            initial_state=render_raw_sim.data,
            policy_offset=0,
            initial_theta=zero_theta,
            initial_dtheta=zero_theta,
        )
        n_pts = 250
        theta_grid = jnp.linspace(0.0, 1.0, n_pts)
        waypoints = track.position(theta=theta_grid)
        if visualize_mode in ("plot", "both"):
            viz.plot_rollout_trajectory(
                states=states,
                track=track,
                step=0,
                output_dir=plot_dir,
                active_start=0,
                active_end=traj_size,
                horizon=traj_size,
            )
        if visualize_mode in ("render", "both"):
            frames = viz.render_rollout_frames(
                states=states,
                sim=render_raw_sim,
                config=track_cfg,
                world=render_world,
                waypoints=waypoints,
                traces=True,
            )
            viz.save_video(frames, render_output, fps=30)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export rollout trajectory from a checkpoint to YAML.")
    parser.add_argument("--train-config", type=pathlib.Path, default=pathlib.Path("configs/train/default.toml"))
    parser.add_argument("--output-dir", type=pathlib.Path, default=pathlib.Path("outputs/export"))
    parser.add_argument("--checkpoint", type=pathlib.Path, default=None,
                        help="Optional checkpoint path; defaults to train.policy_output.")
    parser.add_argument(
        "--visualize",
        choices=["none", "plot", "render", "both"],
        default="none",
        help="Plot or render the checkpoint after exporting.",
    )
    args = parser.parse_args()

    cfg = load_train_config(args.train_config)
    checkpoint_path = pathlib.Path(args.checkpoint) if args.checkpoint else cfg.train.policy_output
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found at {checkpoint_path}. Run training first or provide --checkpoint.")

    export_trajectory(
        cfg=cfg,
        output_dir=args.output_dir,
        visualize_mode=args.visualize,
        checkpoint=checkpoint_path,
    )
