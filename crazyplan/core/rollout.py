import equinox as eqx
import jax
import jax.numpy as jnp

from crazyflow.sim.data import SimData

from crazyplan.core.config import AttrDict
from crazyplan.core.policy import Policy
from crazyplan.core.sim import SimWrapper
from crazyplan.tracks.track import Track


EPS = 1e-12


class Rollout(eqx.Module):
    sim: SimWrapper = eqx.field(static=True)
    track: Track = eqx.field(static=True)
    traj_size: int = eqx.field(static=True)
    cost: AttrDict = eqx.field(static=True)

    @jax.jit
    def rollout(
        self,
        policy: Policy,
        initial_state: SimData,
        scale_penalty=1.0,
        *,
        policy_offset,
        initial_theta,
        initial_dtheta,
    ):
        pol_carry = jnp.asarray(policy_offset)
        carry0 = (initial_state, initial_theta, initial_dtheta, pol_carry)

        def scan_fn(carry, _):
            next_pack, (new_state, cmd, cost) = self._policy_step(policy, carry, scale_penalty)
            _, theta_next, dtheta_next, pol_carry_next = next_pack
            return next_pack, (new_state, cmd, cost, theta_next, dtheta_next, pol_carry_next)

        (final_state, final_theta, final_dtheta, final_offset), (states, actions, costs, theta_hist, dtheta_hist, _) = jax.lax.scan(
            scan_fn,
            carry0,
            xs=None,
            length=self.traj_size,
        )

        return states, actions, costs, theta_hist, dtheta_hist, (final_state, final_theta, final_dtheta, final_offset)

    @eqx.filter_jit
    def advance_prefix(
        self,
        policy: Policy,
        initial_state,
        *,
        steps: jax.Array,
        initial_theta,
        initial_dtheta,
        scale_penalty=1.0,
    ):
        init_carry = policy.init_carry(initial_state)
        carry0 = (initial_state, initial_theta, initial_dtheta, init_carry, jnp.asarray(0, dtype=jnp.int32))

        def cond(carry):
            _, _, _, _, idx = carry
            return idx < steps

        def body(carry):
            state, theta, prev_dtheta, pol_carry, idx = carry
            next_pack, _ = self._policy_step(policy, (state, theta, prev_dtheta, pol_carry), scale_penalty)
            next_state, next_theta, next_dtheta, next_carry = next_pack
            return (next_state, next_theta, next_dtheta, next_carry, idx + 1)

        final_state, final_theta, final_dtheta, _, _ = jax.lax.while_loop(cond, body, carry0)
        return final_state, final_theta, final_dtheta

    def _policy_step(self, policy: Policy, carry_state, scale_penalty):
        state, theta, prev_dtheta, pol_carry = carry_state
        (cmd, d_theta), pol_carry = policy(state.states, pol_carry)
        new_state = self.sim.step(state, cmd)
        theta_next = jnp.minimum(theta + d_theta, jnp.ones_like(theta))
        cost = self._step_cost(new_state, cmd, theta_next, d_theta, prev_dtheta, scale_penalty)
        next_state = (new_state, theta_next, d_theta, pol_carry)
        return next_state, (new_state, cmd, cost)

    def _step_cost(self, new_state, cmd, theta, d_theta, prev_dtheta, scale_penalty):
        cfg = self.cost

        pos_t, tan_t = self._track_targets(theta)
        pos = new_state.states.pos
        e = pos - pos_t

        def project(x, y):
            yy = jnp.sum(y * y)
            return (jnp.sum(x * y) / (yy + EPS)) * y

        batch_drones = jax.vmap(project, in_axes=(0, 0), out_axes=0)
        batch_world = jax.vmap(batch_drones, in_axes=(0, 0), out_axes=0)
        e_l = batch_world(e, tan_t)
        e_c = e - e_l

        importance = jax.vmap(lambda thw: self.track.importance(thw), in_axes=0, out_axes=0)(theta)[..., None]
        scaled_countour_weight = importance * (cfg.contour_weight - cfg.contour_weight_min) + cfg.contour_weight_min

        e_c_sq = jnp.sum(e_c ** 2, axis=-1, keepdims=True)
        e_l_sq = jnp.sum(e_l ** 2, axis=-1, keepdims=True)
        error = scaled_countour_weight * e_c_sq + cfg.lag_weight * e_l_sq

        input_norm = cfg.action_weight * jnp.sum(cmd ** 2, axis=-1, keepdims=True)

        progress = cfg.progress_reward * d_theta
        smooth = cfg.smooth_progress_weight * (d_theta - prev_dtheta) ** 2

        soft_constraints = jax.vmap(lambda pw: self.track.obstacle(pw), in_axes=0, out_axes=0)(pos)[..., None]
        penalty = (scale_penalty * cfg.penalty_weight) * soft_constraints

        curtain = self._curtain_loss(pos, pos_t, tan_t, cfg.curtain_radius)
        curtain_loss = cfg.penalty_weight * importance * curtain

        return error + input_norm - progress + smooth + penalty + curtain_loss

    def _track_targets(self, theta):
        theta_flat = jnp.squeeze(theta, axis=-1)
        pd = jax.vmap(self.track.position, in_axes=0, out_axes=0)(theta_flat)
        tn = jax.vmap(self.track.tangent, in_axes=0, out_axes=0)(theta_flat)
        return pd, tn

    def _curtain_loss(self, p, pd, tn, radius):
        n = tn / (jnp.linalg.norm(tn, axis=-1, keepdims=True) + EPS)
        pc = p - pd
        h = jnp.sum(pc * n, axis=-1, keepdims=True)
        r = jnp.linalg.norm(pc - h * n, axis=-1, keepdims=True)
        outside = jnp.maximum(r - radius, 0.0)
        return outside**2
