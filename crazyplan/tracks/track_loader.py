import toml
import numpy as np
import jax.numpy as jnp
from scipy.spatial.transform import Rotation as R
from jaxtyping import Float, Array
import minsnap_trajectories as ms

from crazyplan.tracks.min_snap import MinSnap

def rpy_from_config(rpy_cfg):
    """Map config rpy (yaw=0 → +Y) to internal rpy (yaw=0 → +X)."""
    roll, pitch, yaw = rpy_cfg
    # return jnp.array([roll, pitch, yaw - jnp.pi / 2.0])
    return jnp.array([roll, pitch, yaw])


class TrackLoader:
    def __init__(self, cfg: dict):
        cfg         = cfg['track']
        gate_cfg    = cfg.get('gate', {})
        self.clear_len = float(gate_cfg['length'])
        self.clear_h = float(gate_cfg['height'])

        self.bar_width = float(gate_cfg['bar_width'])
        self.bar_height = float(gate_cfg['bar_height'])

        self.start  = cfg.get('start', None)
        self.end    = cfg.get('end',   None)
        self.gates  = cfg.get('gates', [])
        self.obs_cfg= cfg.get('obstacles', [])
        self.margin = cfg.get('margin', [])

    def _gate_velocity(self, rpy, speed):
        roll, pitch, yaw = rpy
        x = jnp.cos(yaw)*jnp.cos(pitch)
        y = jnp.sin(yaw)*jnp.cos(pitch)
        z = jnp.sin(pitch)
        return speed*jnp.array([x,y,z])

    def _bars_for_gate(self, pos, rpy):
        """
        pos : world coordinates of gate center
        rpy : roll-pitch-yaw (xyz) rotation of gate

        Gate-local axes:
        x = depth/normal (out of plane)
        y = horizontal (left-right)
        z = vertical (down-up)
        """

        clear_len = self.clear_len
        clear_h   = self.clear_h
        bw = self.bar_width   # depth (out of plane)
        bh = self.bar_height  # visible thickness (in plane)

        # Half clear opening
        half_len = clear_len / 2.0
        half_h   = clear_h   / 2.0

        # Half cross-section
        half_bw = bw / 2.0   # along local x
        half_bh = bh / 2.0   # along local y or z depending on bar

        # Half bar lengths (your rule)
        half_side_len = (clear_h   + 2*bh) / 2.0   # vertical bars long in z
        half_top_len  = (clear_len + 2*bh) / 2.0   # horizontal bars long in y

        # Offsets of bar centers from gate center
        # NOTE: offset uses *bh* (in-plane thickness), NOT bw.
        y_off = half_len + half_bh   # left/right bars
        z_off = half_h   + half_bh   # top/bottom bars

        # Local bar centers (gate frame)
        local_centers = np.array([
            [0, +y_off, 0],   # right vertical bar
            [0, -y_off, 0],   # left vertical bar
            [0, 0, +z_off],   # top horizontal bar
            [0, 0, -z_off],   # bottom horizontal bar
        ])

        # Half-extents (hx, hy, hz) in gate-local axes
        # Vertical bars: size = [bw, bh, clear_h + 2*bh]
        # Horizontal bars: size = [bw, clear_len + 2*bh, bh]
        hx = np.array([
            half_bw, half_bw, half_bw, half_bw
        ])

        hy = np.array([
            half_bh,          # right vertical thickness in y
            half_bh,          # left vertical thickness in y
            half_top_len,     # top horizontal long in y
            half_top_len      # bottom horizontal long in y
        ])

        hz = np.array([
            half_side_len,    # right vertical long in z
            half_side_len,    # left vertical long in z
            half_bh,          # top horizontal thickness in z
            half_bh           # bottom horizontal thickness in z
        ])

        # Rotate + translate into world
        rot = R.from_euler('xyz', rpy).as_matrix()
        world_centers = pos + local_centers @ rot.T

        # Same orientation for all bars (they're aligned in gate frame)
        Rflat = np.tile(rot.reshape(1, 9), (4, 1))

        return np.hstack([
            world_centers,
            hx[:, None], hy[:, None], hz[:, None],
            Rflat
        ])



    def build(self) -> MinSnap:
        # collect waypoints/speeds in order: start → gates → end
        wp_list, rpy_list, s_list = [], [], []
        if self.start:
            wp_list.append(jnp.array(self.start['pos']))
            rpy_list.append(jnp.array(self.start['rpy']))
            s_list.append(self.start['speed'])
        for g in self.gates:
            wp_list.append(jnp.array(g['pos']))
            rpy_list.append(rpy_from_config(jnp.array(g['rpy'])))
            s_list.append(g['speed'])
        if self.end:
            wp_list.append(jnp.array(self.end['pos']))
            rpy_list.append(jnp.array(self.end['rpy']))
            s_list.append(self.end['speed'])

        wp  = jnp.stack(wp_list)                     # (N,3)
        rpy = jnp.stack(rpy_list)                    # (N,3)
        v   = jnp.stack([self._gate_velocity(r,s)
                         for r,s in zip(rpy_list,s_list)])

        # static obstacles
        static = []
        for o in self.obs_cfg:
            pos   = np.array(o['pos'], dtype=float)
            size_spec = o.get('size')
            if size_spec is None:
                raise KeyError("Each obstacle must define 'size' as [length, width, height].")
            size  = np.asarray(size_spec, dtype=float)
            if size.shape != (3,):
                raise ValueError("Obstacle 'size' must contain exactly three values [length, width, height].")
            half  = 0.5 * size
            Rflat = np.eye(3).reshape(9,)
            pos[2] += half[2] 
            static.append(np.concatenate([pos,half,Rflat]))
        static_obs = jnp.asarray(static) if static else jnp.zeros((0, 15))

        # gate‐bars only for actual gates (not start/end)
        gate_wp  = wp_list[self.start is not None :
                           len(self.gates)+(self.start is not None)]
        gate_rpy = rpy_list[self.start is not None :
                            len(self.gates)+(self.start is not None)]
        gate_blocks = [self._bars_for_gate(np.asarray(p),np.asarray(r))
                       for p,r in zip(gate_wp, gate_rpy)]
        gate_obs = jnp.asarray(np.vstack(gate_blocks)) if gate_blocks else jnp.zeros((0, 15))

        obs_parts = []
        if static_obs.shape[0]:
            obs_parts.append(static_obs)
        if gate_obs.shape[0]:
            obs_parts.append(gate_obs)
        if obs_parts:
            all_obs = jnp.vstack(obs_parts)
        else:
            all_obs = jnp.zeros((0, 15))

        return MinSnap(waypoints=wp,
                       velocities=v,
                       margin=self.margin,
                       obstacles=all_obs)
