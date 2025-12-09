import argparse

import toml
import matplotlib.pyplot as plt

from crazyplan import visualization as viz
from crazyplan.tracks.track_loader import TrackLoader


def main():
    parser = argparse.ArgumentParser(description="Plot track geometry.")
    parser.add_argument("--track-config", type=str, default="configs/tracks/drone_racing.toml")
    args = parser.parse_args()

    cfg = toml.load(args.track_config)
    loader = TrackLoader(cfg)
    track = loader.build()

    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    viz.plot_track_geometry(track, ax)
    plt.show()


if __name__ == "__main__":
    main()
