import jax
import jax.numpy as jnp
import equinox as eqx
from jaxtyping import Array, Float

from crazyflow.sim import Sim as CrazySim
from crazyflow.sim.data import SimData
from crazyflow.sim.physics import Physics
from crazyflow.control import Control


class SimWrapper(eqx.Module):
    sim: CrazySim = eqx.field(static=True)

    def step(self, state: SimData, action: Float[Array, "W D A"]) -> SimData:
        new_controls = state.controls.replace(
            attitude=state.controls.attitude.replace(staged_cmd=action)
        )
        updated_data = state.replace(controls=new_controls)
        return self.sim._step(updated_data, self.sim.freq // self.sim.control_freq)


def create_initial_state(cfg: dict, key: jax.Array, state_template: SimData) -> SimData:
    position = jnp.asarray(cfg.get("position", [-1.5, 0.75, 0.01]), dtype=state_template.states.pos.dtype)
    position = jnp.reshape(position, (1, 1, 3))
    position = jnp.broadcast_to(position, state_template.states.pos.shape)
    noise_mag = float(cfg.get("position_noise", 0.0))
    rand = jax.random.uniform(key, state_template.states.pos.shape, dtype=state_template.states.pos.dtype)
    noise = (rand * 2.0 - 1.0) * noise_mag
    new_pos = position + noise
    new_states = state_template.states.replace(pos=new_pos)
    return state_template.replace(states=new_states)


def build_sim(
    n_worlds: int,
    n_drones: int,
    sim_cfg: dict | None = None,
):
    sim_cfg = sim_cfg or {}
    drone_model = sim_cfg.get("drone_model", "cf21B_500")
    physics_name = sim_cfg.get("physics", "so_rpy_rotor_drag")
    control_frequency = int(sim_cfg.get("control_frequency", 200))
    device = sim_cfg.get("device", "cpu")

    physics = getattr(Physics, physics_name)

    return CrazySim(
        n_worlds,
        n_drones,
        drone_model=drone_model,
        physics=physics,
        control=Control.attitude,
        attitude_freq=control_frequency,
        device=device,
    )


def make_initialized_sim(
    n_worlds: int,
    n_drones: int,
    sim_cfg: dict,
    *,
    initial_state: SimData,
):
    sim = build_sim(
        n_worlds,
        n_drones,
        sim_cfg=sim_cfg,
    )
    sim.data = initial_state
    sim.build_default_data()
    return sim
