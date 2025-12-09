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


class Track(eqx.Module):

    def position(self, theta: Float[Array, "N"]) -> Float[Array, "N 3"]:
        """Continuous position reference in (x,y,z)."""
        raise NotImplementedError

    def tangent(self, theta: Float[Array, "N"]) -> Float[Array, "N 3"]:
        """Continuous unit tangent of the reference."""

        def pos_scalar(t):
            return self.position(jnp.array([t]))[0] 
        
        grads = jax.vmap(jax.jacfwd(pos_scalar))(theta) 
        norms = jnp.linalg.norm(grads, axis=-1, keepdims=True)

        return grads / norms
    
    def sample(self, theta: Float[Array, "N"]) -> Tuple[Float[Array, "N 3"], Float[Array, "N 3"]]:

        return self.position(theta), self.tangent(theta)
    
    # Returns the importance of being close to the track. From (0, 1)
    def importance(self, theta: Float[Array, "N 1"]) -> Float[Array, "N"]:

        return jnp.zeros(shape=(theta.shape[0],))
    
    # Returns 0. if the postion is outside an obstacle. Soft.
    def obstacle(self, pos: Float[Array, "N 3"]) -> Float[Array, "N"]:

        return jnp.zeros(shape=(pos.shape[0]))
    
    def has_obstacles(self) -> bool:
        return False


class CircleTrack(Track):

    radius: float = eqx.field(static=True)
    z_ref:  float = eqx.field(static=True)

    def position(self, theta: Float[Array, "N"]) -> Float[Array, "N 3"]:
        """
        Theta from (0, infty] where theta = 1 is a full lap.
        Starting postion is (0,0,z_ref). 
        Moves clockwise around the point (radius, radius, z_ref).
        """
        
        # Convert arc‑length to angle ϕ: ϕ = θ / R
        phi = theta * (2 * np.pi)

        x = self.radius * jnp.cos(phi) - self.radius + 1.
        y = self.radius * jnp.sin(phi) + 1.
        z = jnp.zeros_like(x) + self.z_ref
        pos = jnp.stack([x, y, z], axis=-1)

        return pos

class Figure8Track(Track):

    radius: float = eqx.field(static=True)
    z_ref:  float = eqx.field(static=True)

    def position(self, theta: Float[Array, "N"]) -> Float[Array, "N 3"]:
        """
        Theta from (0, infty] where theta = 1 is a full lap.
        Starting postion is (0,0,z_ref).
        Moves clockwise around the point (radius, radius, z_ref).
        """
        
        # Convert arc‑length to angle ϕ: ϕ = θ / R
        phi = theta * (2 * np.pi)

        x = self.radius * jnp.cos(phi) - self.radius + 1.
        y = 1.25 * self.radius * jnp.sin(2 * phi) + 1.
        z = 0.1 * self.radius * jnp.sin(phi) + self.z_ref
        pos = jnp.stack([x, y, z], axis=-1)

        return pos

    

if __name__ == "__main__":
    # theta = jnp.array([.1231, 1])

    # track = CircleTrack(radius=1., z_ref=0.5)

    # assert track.position(theta=theta).shape == (2,3)
    # assert track.tangent(theta=theta).shape == (2,3)


    # phi = theta / track.radius
    # tan = jnp.stack([track.radius * -jnp.sin(phi), track.radius * jnp.cos(phi), jnp.zeros_like(phi)], axis=-1)
    # norms = jnp.linalg.norm(tan, axis=-1, keepdims=True)
    
    # assert jnp.allclose(track.tangent(theta=theta), tan / norms)
    # print(track.tangent(theta=theta))
    # print(tan / norms)


    track = Figure8Track(radius=1., z_ref=1.2)


    theta = jnp.array([0, .5, 1.])

    theta = jnp.linspace(0, 1.2, 100)

    positions = track.position(theta)
    pos_np = np.asarray(positions)


    tangents = track.tangent(theta)
    positions = track.position(theta)
    obst_vals = track.obstacle(positions)

    pos_tan_np = np.asarray(positions)
    tan_np = np.asarray(tangents)
    obst_vals_np = np.asarray(obst_vals)

    arrow_len = 0.2

    import numpy as np
    import matplotlib.pyplot as plt

    # Plot trajectories
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    # ax.plot(pos_np[:,0], pos_np[:,1], pos_np[:,2], label='MinSnap Path')

    sc = ax.scatter(pos_np[:,0], pos_np[:,1], pos_np[:,2], c=obst_vals_np,
                    cmap='hot', label='Obstacle Cost')


    ax.quiver(pos_tan_np[:, 0], pos_tan_np[:, 1], pos_tan_np[:, 2],     # bases
          tan_np[:, 0], tan_np[:, 1], tan_np[:, 2],     # directions
          length=arrow_len, normalize=True,
          color='green', linewidth=2)

    
    fig.colorbar(sc, ax=ax, shrink=0.5, label='Obstacle Soft Cost')

    ax.set_xlim(-2.0, 2.0)
    ax.set_ylim(-2.0, 2.0)
    ax.set_zlim(0.0, 1.4)

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.legend()
    plt.show()


