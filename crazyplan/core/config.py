from __future__ import annotations

import pathlib
from typing import Any, Mapping

import toml


class AttrDict(dict):
    """Dict with attribute access for convenience."""

    def __getattr__(self, item: str) -> Any:  # pragma: no cover - trivial
        try:
            return self[item]
        except KeyError as exc:
            raise AttributeError(item) from exc

    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

    @classmethod
    def from_nested(cls, obj: Any) -> Any:
        if isinstance(obj, dict):
            return cls({k: cls.from_nested(v) for k, v in obj.items()})
        if isinstance(obj, list):
            return [cls.from_nested(v) for v in obj]
        return obj


def _build_episode_list(train_cfg: Mapping) -> list[AttrDict]:
    episodes_cfg = train_cfg.get("episodes")
    if episodes_cfg:
        return [
            AttrDict(
                window=max(int(ep["window"]), 1),
                scale_penalty=float(ep["scale_penalty"]),
                learning_rate=float(ep["learning_rate"]),
            )
            for ep in episodes_cfg
        ]

    # Backward-compatible with old parallel lists.
    windows = list(train_cfg["episode_windows"])
    penalties = list(train_cfg["scale_penalties"])
    lrs = list(train_cfg["episode_learning_rates"])
    if not (len(windows) == len(penalties) == len(lrs)):
        raise ValueError("episode_* lengths must match.")
    return [
        AttrDict(
            window=max(int(w), 1),
            scale_penalty=float(p),
            learning_rate=float(lr),
        )
        for w, p, lr in zip(windows, penalties, lrs, strict=True)
    ]


def _build_train_cfg(train_cfg: Mapping) -> AttrDict:
    traj_size = int(train_cfg["traj_size"])
    plot_every = int(train_cfg.get("plot_every", 50))
    log_interval = train_cfg.get("log_interval")
    log_interval = int(log_interval) if log_interval is not None else plot_every
    steps_per_episode = train_cfg.get("steps_per_episode")
    steps_per_episode = int(steps_per_episode) if steps_per_episode is not None else traj_size * 2

    return AttrDict(
        seed=int(train_cfg["seed"]),
        track_config=pathlib.Path(train_cfg["track_config"]),
        traj_size=traj_size,
        steps_per_episode=steps_per_episode,
        plot_every=plot_every,
        log_interval=log_interval,
        n_worlds=int(train_cfg["n_worlds"]),
        n_drones=int(train_cfg["n_drones"]),
        checkpoint_dir=pathlib.Path(train_cfg["checkpoint_dir"]),
        plot_dir=pathlib.Path(train_cfg["plot_dir"]),
        policy_output=pathlib.Path(train_cfg["policy_output"]),
        render_output=pathlib.Path(train_cfg["render_output"]),
        render_world=int(train_cfg.get("render_world", 0)),
        initial_state_cfg=dict(train_cfg.get("initial_state", {})),
        sim_cfg=dict(train_cfg.get("sim", {})),
        debug_timing=bool(train_cfg.get("debug_timing", True)),
        episodes=tuple(_build_episode_list(train_cfg)),
        initial_state=None,
    )


def load_train_config(path: pathlib.Path):
    raw_cfg = AttrDict.from_nested(toml.load(path))
    train_cfg = _build_train_cfg(raw_cfg.train)
    policy_init = dict(raw_cfg.policy_init)
    cost_cfg = AttrDict.from_nested(raw_cfg.cost if "cost" in raw_cfg else raw_cfg)
    track_cfg = toml.load(train_cfg.track_config)
    config = AttrDict(
        train=train_cfg,
        policy_init=policy_init,
        cost=cost_cfg,
        track_cfg=track_cfg,
        raw=raw_cfg,
    )
    return config
