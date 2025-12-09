from typing import Callable, Tuple

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, Float
from jax.scipy.spatial.transform import Rotation as R

from crazyflow.sim.sim import SimData, SimState
from crazyplan.core.config import AttrDict

class Policy(eqx.Module):
    config: AttrDict = eqx.field(static=True)
    
    # Return initial policy-carry (e.g., time index, internal state)
    def init_carry(self, state) -> object:
        return None

    # One step: given state and carry -> (action, d_theta), new_carry
    def __call__(self, state, carry):
        raise NotImplementedError
    

class OpenLoopPolicy(Policy):
    """
    Fixed sequence of attitude/thrust commands with per-step track progress.
    Inputs shape: (T, n_worlds, n_drones, 4) for [roll, pitch, yaw, thrust].
    d_theta shape: (T, n_worlds, n_drones, 1) advances along the track.
    """

    n_worlds: int = eqx.field(static=True)
    n_drones: int = eqx.field(static=True)

    # [collective thrust, roll, pitch, yaw]
    input: Float[Array, "T W D 4"] # (T,n_worlds,n_drones,4)
    # Progress along the track
    d_theta: Float[Array, "T W D 1"] # (T,n_worlds,n_drones,1)

    def __init__(self, init_ctrl: Array, init_dtheta: Array, config: AttrDict):
        self.input = init_ctrl
        self.d_theta = init_dtheta

        self.n_worlds = init_ctrl.shape[1]
        self.n_drones = init_ctrl.shape[2]

        self.config = config

    @classmethod
    def from_config(cls, cfg: AttrDict, *, key: jax.Array) -> "OpenLoopPolicy":
        train = cfg.train
        init_params = cfg.policy_init
        ctrl_min = init_params["control_min"]
        ctrl_max = init_params["control_max"]
        thrust_bias = init_params["thrust_bias"]
        dtheta_min = init_params["dtheta_min"]
        dtheta_max = init_params["dtheta_max"]

        key, ctrl_key = jax.random.split(key)
        init_ctrl = jax.random.uniform(
            ctrl_key,
            shape=(train.traj_size, train.n_worlds, train.n_drones, 4),
            dtype=jnp.float32,
            minval=ctrl_min,
            maxval=ctrl_max,
        )
        init_ctrl = init_ctrl.at[..., 3].add(thrust_bias)

        key, dtheta_key = jax.random.split(key)
        init_dtheta = jax.random.uniform(
            dtheta_key,
            shape=(train.traj_size, train.n_worlds, train.n_drones, 1),
            minval=dtheta_min,
            maxval=dtheta_max,
        )

        section = cfg.raw.policy
        return cls(init_ctrl, init_dtheta, config=section)

    def init_carry(self, state):
        return jnp.array(0, dtype=jnp.int32)

    def __call__(self, state: SimState, carry):
        idx = carry
        next_idx = idx + 1
        idx = jnp.minimum(idx, self.input.shape[0] - 1)

        raw_action = jax.lax.dynamic_index_in_dim(self.input, idx, axis=0, keepdims=False)

        raw_angles = raw_action[..., 0:3]          # unconstrained r,p,y
        raw_thrust = raw_action[..., 3]            # unconstrained thrust

        d_theta_u = jax.lax.dynamic_index_in_dim(self.d_theta, idx, axis=0, keepdims=False)

        # Scale thrust
        thrust_range = self.config.max_thrust - self.config.min_thrust
        thrust = thrust_range * jax.nn.sigmoid(raw_thrust) + self.config.min_thrust

        # Scale angles
        rotvec = self.config.max_d_angle * jnp.tanh(raw_angles) # (n_worlds, n_drones, 3)
        rotate = jax.vmap(lambda quat, rotvec: 
                          (R.from_quat(quat) * R.from_rotvec(rotvec)).as_euler("xyz"), 
                          in_axes=0, out_axes=0
                          )
        
        angles = rotate(state.quat, rotvec)
        # Set yaw to 0
        angles = angles.at[..., 2].set(0.)
        # Scale progress along the track (theta)
        d_theta_range = self.config.max_d_theta - self.config.min_d_theta
        d_theta = d_theta_range * jax.nn.sigmoid(d_theta_u) + self.config.min_d_theta

        actions = jnp.concatenate([angles, thrust[..., None]], axis=-1)

        return (actions, d_theta), (next_idx)
    
