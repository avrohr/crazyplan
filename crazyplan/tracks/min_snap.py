import jax
import jax.numpy as jnp
from typing import NamedTuple, Tuple
from jaxtyping import Array, Float
import equinox as eqx
import dataclasses

import numpy as np
from jax.scipy.spatial.transform import Rotation as R
import minsnap_trajectories as ms

from mpl_toolkits.mplot3d.art3d import Poly3DCollection

from crazyplan.tracks.track import Track

    
@dataclasses.dataclass(frozen=True)
class FrozenArray:
    """Wrapper whose identity, not contents, is used for hashing/equality."""
    value: Array
    
class PiecewisePolynomialTrajectory(NamedTuple):
    time_reference: FrozenArray
    durations: FrozenArray
    coefficients: FrozenArray
    waypoint_theta: FrozenArray

def soft_clip(theta, min=0.0, max=1.0, alpha=10.0):

    a = alpha * theta
    b = alpha * (theta - 1.0)
    clipped = (jax.nn.softplus(a) - jax.nn.softplus(b)) / alpha
    return (max - min) * clipped + min

class MinSnap(Track):

    polys: PiecewisePolynomialTrajectory = eqx.field(static=True)
    obstacles: FrozenArray = eqx.field(static=True)
    safety_margin: float = eqx.field(static=True)

    def __init__(self, 
                 waypoints: Float[Array, "N 3"], 
                 velocities: Float[Array, "N 3"], 
                 margin: float = 0.0,
                 obstacles: None | Float[Array, "M 15"] = None
                 ):
        # Axis aligned boxes as obstacles only for now. Position + lengths (x,y,z,hx,hy,hzr)

        refs = []
        n_wps = waypoints.shape[0]
        waypoint_array = np.asarray(waypoints, dtype=float)
        diffs = np.linalg.norm(np.diff(waypoint_array, axis=0), axis=1)
        p = 0.6  # 0 < p < 1, p<1 compresses long segments
        scaled_diffs = diffs**p
        cumulative = np.concatenate(([0.0], np.cumsum(scaled_diffs)))
        total_length = cumulative[-1]

        times = (cumulative / total_length)

        for i in range(n_wps):
            refs.append(
                ms.Waypoint(
                    time=times[i],
                    position=np.array(waypoints[i]).flatten(),
                    velocity=np.array(velocities[i]).flatten(),
                )
            )
        polys = ms.generate_trajectory(
            refs,
            degree=7,  # Polynomial degree
            idx_minimized_orders=(3),  
            num_continuous_orders=4,  
            algorithm="closed-form",  # Or "constrained"
        )

        self.polys = PiecewisePolynomialTrajectory(
            time_reference=FrozenArray(jnp.array(polys.time_reference)),
            durations=FrozenArray(jnp.array(polys.durations)),
            coefficients=FrozenArray(jnp.array(polys.coefficients)),
            waypoint_theta=FrozenArray(jnp.array(times))
        )
        if obstacles is not None:
            self.obstacles = FrozenArray(value=jnp.array(obstacles))
        else:
            self.obstacles = FrozenArray(value=jnp.empty(shape=(0, 6)))

        self.safety_margin = margin

    def has_obstacles(self) -> bool:
        return True

    @jax.jit
    def position(self, theta: Float[Array, "N"]) -> Float[Array, "N 3"]:
        """
        theta: (N,) in arbitrary range; evaluated with straight-through clipping to [t_ref[0], t_ref[-1]].
        returns: (N, 3)
        """

        t_ref, _, coeffs, _ = self.polys
        t_ref = t_ref.value                  # shape: (S+1,)
        coeffs = coeffs.value                # shape: (S, n_cfs, 3)
        S, n_cfs, _ = coeffs.shape

        lo, hi = t_ref[0], t_ref[-1]
        theta_eval = self._clip_theta(theta, lo, hi)

        # segment index (clamped), local time
        idx = jnp.searchsorted(t_ref[1:], theta_eval, side="right")
        idx = jnp.clip(idx, 0, S - 1)                            # (N,)
        tau = theta_eval - t_ref[idx]                            # (N,)

        # gather segment coeffs per sample, reverse for Horner
        seg = coeffs[idx]                                        # (N, n_cfs, 3)
        seg_rev = seg[:, ::-1, :]                                # (N, n_cfs, 3)

        # vectorized Horner over coeff axis
        def body(i, acc):
            return acc * tau[:, None] + seg_rev[:, i, :]         # acc: (N,3)
        acc0 = jnp.zeros((seg_rev.shape[0], 3), seg_rev.dtype)
        pos = jax.lax.fori_loop(0, seg_rev.shape[1], body, acc0) # (N,3)

        return pos
    
    def _clip_theta(self, theta: Float[Array, "N 1"], lo: float | Array = 0., hi: float | Array = 1.) -> Float[Array, "N 1"]:

        # theta_clip = jnp.clip(theta, lo, hi)                     # (N,)
        # theta = theta + jax.lax.stop_gradient(theta_clip - theta)

        # theta = jax.lax.clamp(min=lo, x=theta, max=hi)

        return theta
    
    @jax.jit
    def importance(self, theta: Float[Array, "N 1"]) -> Float[Array, "N"]:

        t_ref, _, _, _ = self.polys
        t_ref = t_ref.value                  # shape: (S+1,)

        lo, hi = t_ref[0], t_ref[-1]
        theta = self._clip_theta(theta, lo, hi)
        wp_theta = self.polys.waypoint_theta.value #[1:]  # (K,)

        # Compute Gaussian bumps centered at each waypoint
        sigma = 0.03 # width of each bump — tune as needed
        diffs = theta - wp_theta[None, :]  # (N, K)
        gaussians = jnp.exp(-0.5 * (diffs / sigma)**2)  # (N, K)

        importance_vals = 0.0 + jnp.max(gaussians, axis=1)  # (N,)
        return importance_vals

    @jax.jit
    def obstacle(self, pos: Float[Array, "N 3"]) -> Float[Array, "N"]:
        """
        Smooth collision indicator for arbitrarily oriented boxes.
        Obstacles are (M,12): [cx,cy,cz, hx,hy,hz,  R11…R33].
        """
        obs        = self.obstacles.value                 # (M,12)
        centres    = obs[:, 0:3]                          # (M,3)
        half_sz    = obs[:, 3:6]                          # (M,3)
        Rmat       = obs[:, 6:15].reshape(-1, 3, 3)         # (M,3,3)

        # absolute distance along each axis
        diff   = pos[:, None, :] - centres[None, :, :]    # (N,M,3)
        local  = jnp.einsum('nmi,mij->nmj', diff, Rmat)   # world → local

        raw    = (half_sz[None, :, :] + self.safety_margin) - jnp.abs(local)     # (N,M,3)

        # 1) get a sigmoid that is 0 at raw=0, 1 deep inside:
        axis_sm = jax.nn.sigmoid(raw * 100)
        # 2) clamp negatives to 0 so that outside is exactly zero
        # axis_sm = soft_clip(axis_sm, min=0.0, max=1.0)
        inside_prob = jnp.prod(axis_sm, axis=-1)

        # ---- ground penalty ----
        z_min = 0.0 + self.safety_margin * 0.9
        k = 20.0                     # sharpness (bigger = harder barrier)

        violation = z_min - pos[:, 2]          # >0 when too low
        ground_penalty = jax.nn.softplus(k * violation) / k

        # finally, sum over all obstacles:
        return jnp.sum(inside_prob, axis=1) + ground_penalty            # (N,)


        # Old working code for circles
        # obstacles = self.obstacles.value
        # centers = obstacles[:, :3]  # (M, 3)
        # radii = obstacles[:, 3] # (M,)
        # diff = pos[:, None, :] - centers[None, :, :]
        # sq_dist = jnp.sum(diff**2, axis=-1)  # (N, M)
        # dist = jnp.sqrt(sq_dist + 1e-8)

        # # < 0 inside, > 0 outside
        # scale = (2 / radii)
        # dist_to_surface = (dist - radii) * scale  # (N, M)

        # temperature = 0.01  # tune this
        # soft_cost = -temperature * jax.nn.logsumexp(-dist_to_surface / temperature, axis=1)
        # return jax.nn.sigmoid(-soft_cost * 5)  # (N,)
    
