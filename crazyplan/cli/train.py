import argparse
import pathlib

import jax

from crazyplan.core.policy import OpenLoopPolicy
from crazyplan.core.training import save_policy, train_policy
from crazyplan.core.sim import build_sim, create_initial_state
from crazyplan.core.config import load_train_config
from crazyplan.tracks.track_loader import TrackLoader

TRAIN_CONFIG_PATH = pathlib.Path("configs/train/default.toml")

def main():
    parser = argparse.ArgumentParser(description="Train rollout policy via window sweeps.")
    parser.add_argument("--train-config", type=pathlib.Path, default=TRAIN_CONFIG_PATH)
    parser.add_argument("--plot", action="store_true", help="Enable periodic trajectory plotting.")
    parser.add_argument("--render", action="store_true", help="Render the final rollout video.")
    args = parser.parse_args()

    cfg = load_train_config(args.train_config)
    train_params = cfg.train
    init_params = cfg.policy_init

    key = jax.random.PRNGKey(train_params.seed)

    key, init_policy_key = jax.random.split(key)
    policy = OpenLoopPolicy.from_config(cfg, key=init_policy_key)

    key, init_state_key = jax.random.split(key)
    template_sim = build_sim(
        train_params.n_worlds,
        train_params.n_drones,
        sim_cfg=train_params.sim_cfg,
    )
    template_sim.data = create_initial_state(train_params.initial_state_cfg, init_state_key, template_sim.data)
    template_sim.build_default_data()
    initial_state = jax.tree_util.tree_map(lambda x: x, template_sim.data)
    train_params.initial_state = initial_state
    loader = TrackLoader(cfg=cfg.track_cfg)
    track = loader.build()

    trained_policy = train_policy(
        policy,
        track,
        cfg,
        plot_enabled=args.plot,
        render_enabled=args.render,
    )
    save_policy(trained_policy, train_params.policy_output)


if __name__ == "__main__":
    main()
