from __future__ import annotations

from pathlib import Path
from typing import Mapping
import warnings

import einops
import jax
import jax.numpy as jnp
import numpy as np
from gymnasium.envs.mujoco.mujoco_rendering import MujocoRenderer
from jax.scipy.spatial.transform import Rotation as R
from numpy.typing import NDArray
import mujoco

from crazyflow.sim.data import SimData
from crazyflow.sim import Sim as CrazySim
from crazyplan.tracks.track import Track
from mpl_toolkits.mplot3d.art3d import Poly3DCollection


ASSETS = Path(__file__).parent / "assets"
GATE_SPEC_PATH = ASSETS / "gate.xml"
OBSTACLE_SPEC_PATH = ASSETS / "obstacle.xml"
DEFAULT_CAM_CONFIG = dict(
    azimuth=180.0,
    elevation=-35.0,
    distance=5.0,
    lookat=jnp.array([0.0, 0.0, 0.0]),
)
DEFAULT_RENDER_WIDTH = 1088  # divisible by 16 to avoid macro block resizing
DEFAULT_RENDER_HEIGHT = 720


def plot_rollout_trajectory(
    states: SimData,
    track: Track,
    step: int,
    *,
    world_idx: int = 0,
    drone_idx: int = 0,
    output_dir: Path = Path("outputs/plots_window"),
    active_start: int | None = None,
    active_end: int | None = None,
    horizon: int | None = None,
):
    """Plot the 3D trajectory from a rollout with multiple viewpoints.

    Actions outside the active window are shown in gray to indicate that they
    are currently frozen while the highlighted segment is being optimized.
    """
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    positions = np.asarray(jax.device_get(states.states.pos))
    if positions.ndim < 4:
        return

    traj = positions[:, world_idx, drone_idx, :]
    output_dir.mkdir(parents=True, exist_ok=True)

    total_steps = traj.shape[0]
    active_start = 0 if active_start is None else int(active_start)
    active_end = total_steps if active_end is None else int(active_end)
    horizon = total_steps if horizon is None else int(horizon)
    horizon = max(0, min(total_steps, horizon))
    active_start = max(0, min(horizon, active_start))
    active_end = max(active_start, min(horizon, active_end))

    frozen_color = "0.75"
    active_color = "tab:orange"

    fig = plt.figure(figsize=(14, 10))
    views = [
        ("3D", None),
        ("View +X", dict(elev=0.0, azim=90.0)),
        ("View +Y", dict(elev=0.0, azim=0.0)),
        ("View +Z", dict(elev=90.0, azim=-90.0)),
    ]

    for idx, (title_suffix, view) in enumerate(views, start=1):
        ax = fig.add_subplot(2, 2, idx, projection="3d")
        plot_track_geometry(track, ax)
        # Plot the frozen segments before and after the active window in gray.
        if active_start > 0:
            frozen = traj[:active_start]
            ax.plot(frozen[:, 0], frozen[:, 1], frozen[:, 2], color=frozen_color, linewidth=1.0)
        if active_end < horizon:
            frozen = traj[active_end:horizon]
            ax.plot(frozen[:, 0], frozen[:, 1], frozen[:, 2], color=frozen_color, linewidth=1.0)
        # Plot the active window in orange to highlight the optimized portion.
        if active_start < active_end:
            active = traj[active_start:active_end]
            ax.plot(active[:, 0], active[:, 1], active[:, 2], color=active_color, linewidth=1.5)
        if view is not None:
            ax.view_init(**view)
        window_label = f"{active_start}-{active_end} / {horizon}"
        ax.set_title(f"Step {step} ({window_label}) - {title_suffix}")

    plot_path = output_dir / f"rollout_step_{step:06d}.png"
    fig.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_track_geometry(
    track: Track,
    ax,
    *,
    obs_color: str = "skyblue",
    obs_alpha: float = 0.5,
    wp_color: str = "green",
    wp_size: float = 50,
    n_samples: int = 200,
    cmap: str = "hot",
):
    """Draw obstacles, the cost-colored trajectory, and the waypoints."""
    obs = np.asarray(track.obstacles.value)
    for row in obs:
        centre = row[0:3]
        half = row[3:6]
        rot = row[6:15].reshape(3, 3)
        signs = np.array(
            [
                [-1, -1, -1],
                [1, -1, -1],
                [1, 1, -1],
                [-1, 1, -1],
                [-1, -1, 1],
                [1, -1, 1],
                [1, 1, 1],
                [-1, 1, 1],
            ]
        )
        corners = centre + (signs * half) @ rot.T
        faces = [[0, 1, 2, 3], [4, 5, 6, 7], [0, 1, 5, 4], [2, 3, 7, 6], [1, 2, 6, 5], [0, 3, 7, 4]]
        verts = [[corners[i] for i in f] for f in faces]
        ax.add_collection3d(Poly3DCollection(verts, facecolors=obs_color, edgecolors="none", alpha=obs_alpha))

    thetas = np.linspace(0.0, 1.0, n_samples)
    traj_t = jnp.array(thetas)
    traj_pos = track.position(traj_t)
    traj_np = np.asarray(traj_pos)
    importance_np = np.asarray(track.importance(traj_t[:, None]))

    sc = ax.scatter(traj_np[:, 0], traj_np[:, 1], traj_np[:, 2], c=importance_np, cmap=cmap, s=4)

    import matplotlib.pyplot as plt

    plt.colorbar(sc, ax=ax, shrink=0.5, label="Tracking weights")

    wp_theta = track.polys.waypoint_theta.value
    wp_pos = np.asarray(track.position(jnp.array(wp_theta)))
    ax.scatter(wp_pos[:, 0], wp_pos[:, 1], wp_pos[:, 2], color=wp_color, s=wp_size, label="Waypoints")
    ax.set_xlim([-2, 2])
    ax.set_ylim([-2, 2])
    ax.set_zlim([0, 2])

    ax.legend()


class TrajectoryPlotter:
    def __init__(
        self,
        track: Track,
        output_dir: Path,
        *,
        world_idx: int = 0,
        drone_idx: int = 0,
    ):
        self.track = track
        self.output_dir = output_dir
        self.world_idx = world_idx
        self.drone_idx = drone_idx
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def plot_window(self, *, states: SimData, step: int, window_start: int, window_size: int, horizon: int) -> None:
        plot_rollout_trajectory(
            states=states,
            track=self.track,
            step=step,
            world_idx=self.world_idx,
            drone_idx=self.drone_idx,
            output_dir=self.output_dir,
            active_start=window_start,
            active_end=window_start + window_size,
            horizon=horizon,
        )


def render_rollout_frames(
    states: SimData,
    sim: CrazySim,
    config: Mapping,
    *,
    world: int = 0,
    waypoints: np.ndarray | None = None,
    traces: bool = True,
    fps: int = 30,
    width: int = DEFAULT_RENDER_WIDTH,
    height: int = DEFAULT_RENDER_HEIGHT,
) -> list[np.ndarray]:
    """Render frames for a rollout using recorded simulation data."""
    gates = config["track"]["gates"]
    obstacles = config["track"]["obstacles"]
    quads = [R.from_euler("xyz", np.array(gate["rpy"])).as_quat().astype(np.float32) for gate in gates]

    sim.reset()
    _load_track_into_sim(sim, quads, gates, obstacles)
    sim.max_visual_geom = 100_000

    num_steps = states.states.pos.shape[0]

    frames: list[np.ndarray] = []
    trace_positions: list[np.ndarray] = []
    trace_rotations: list[R] = []
    trace_stride = 20

    control_freq = sim.control_freq

    def _select_state(idx):
        return jax.tree_util.tree_map(lambda arr: arr[idx], states)

    for idx in range(num_steps):
        state_t = _select_state(idx)
        state_t = state_t.replace(core=state_t.core.replace(mjx_synced=jnp.array(False)))
        sim.data = state_t

        if traces and idx % trace_stride == 0:
            pos_t = np.asarray(jax.device_get(state_t.states.pos[world]))
            trace_positions.append(pos_t)
            if len(trace_positions) > 1:
                trace_rotations.append(rotation_matrix_from_points(trace_positions[-2], trace_positions[-1]))

        if ((idx * fps) % control_freq) < fps:
            if traces and sim.viewer is not None:
                render_traces(sim.viewer, trace_positions, trace_rotations)
            if waypoints is not None and sim.viewer is not None:
                render_waypoints(sim.viewer, waypoints)

            frame = sim.render(
                mode="rgb_array",
                world=world,
                cam_config=DEFAULT_CAM_CONFIG,
                width=width,
                height=height,
            )
            frames.append(frame)

            if sim.viewer is not None:
                sim.viewer.viewer._markers.clear()

    sim.close()
    return frames


def _load_track_into_sim(sim: CrazySim, quads, gates, obstacles):
    gate_spec = mujoco.MjSpec.from_file(str(GATE_SPEC_PATH))
    obstacle_spec = mujoco.MjSpec.from_file(str(OBSTACLE_SPEC_PATH))

    frame = sim.spec.worldbody.add_frame()
    for idx, (gate_cfg, quat) in enumerate(zip(gates, quads)):
        gate_body = gate_spec.body("gate")
        if gate_body is None:
            raise ValueError("Gate body not found in gate spec")
        gate = frame.attach_body(gate_body, "", f":{idx}")
        gate.pos = gate_cfg["pos"]
        quat_np = np.asarray(jax.device_get(quat), dtype=np.float32)
        gate.quat = quat_np[[3, 0, 1, 2]]  # Convert from scipy order to MuJoCo order
        gate.mocap = True

    for idx, obstacle_cfg in enumerate(obstacles):
        obstacle_body = obstacle_spec.body("obstacle")
        if obstacle_body is None:
            raise ValueError("Obstacle body not found in obstacle spec")
        obstacle = frame.attach_body(obstacle_body, "", f":{idx}")
        obstacle.pos = obstacle_cfg["pos"]
        obstacle.mocap = True

    sim.build_mjx()


def render_waypoints(viewer: MujocoRenderer, waypoints):
    for point in np.asarray(waypoints):
        viewer.viewer.add_marker(
            type=mujoco.mjtGeom.mjGEOM_SPHERE,
            size=[0.01, 0.01, 0.01],
            pos=point,
            rgba=[0.0, 1.0, 0.0, 0.8],
        )


def render_traces(viewer: MujocoRenderer, positions: list[np.ndarray], rotations: list[R]):
    """Render traces of the drone trajectories."""
    if len(positions) < 2 or viewer is None:
        return

    n_trace, n_drones = len(rotations), len(positions[0])
    pos = np.array(positions)
    sizes = np.zeros((n_trace, n_drones, 3))
    sizes[..., 2] = np.linalg.norm(pos[1:] - pos[:-1], axis=-1)
    sizes[..., :2] = np.linspace(1.0, 5.0, n_trace)[:, None, None]
    mats = np.zeros((n_trace, n_drones, 9))
    for i in range(n_trace):
        mats[i, :] = einops.rearrange(rotations[i].as_matrix(), "d n m -> d (n m)")

    import matplotlib.pyplot as plt

    cmap = plt.get_cmap("inferno")
    color_samples = cmap(np.linspace(0.0, 1.0, n_drones))
    rgbas = np.zeros((n_trace, n_drones, 4))
    for j in range(n_drones):
        rgbas[:, j, :3] = color_samples[j, :3]
    rgbas[..., 3] = np.linspace(0, 1, n_trace)[:, None]

    for i in range(n_trace):
        for j in range(n_drones):
            viewer.viewer.add_marker(
                type=mujoco.mjtGeom.mjGEOM_LINE,
                size=sizes[i, j],
                pos=positions[i][j],
                mat=mats[i, j],
                rgba=rgbas[i, j],
            )


def rotation_matrix_from_points(p1: NDArray, p2: NDArray) -> R:
    z_axis = (v := p2 - p1) / np.linalg.norm(v, axis=-1, keepdims=True)
    random_vector = np.random.rand(*z_axis.shape)
    x_axis = (v := np.cross(random_vector, z_axis)) / np.linalg.norm(v, axis=-1, keepdims=True)
    y_axis = np.cross(z_axis, x_axis)
    return R.from_matrix(np.stack((x_axis, y_axis, z_axis), axis=-1))


def save_video(frames: list[np.ndarray], path: Path | str, *, fps: int = 30, macro_block_size: int = 16):
    """Save video frames while suppressing fork warnings from ffmpeg."""
    if not frames:
        return
    import imageio.v2 as imageio

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message=r"os\.fork\(\) was called",
            category=RuntimeWarning,
        )
        imageio.mimsave(output_path, frames, fps=fps, macro_block_size=macro_block_size)
