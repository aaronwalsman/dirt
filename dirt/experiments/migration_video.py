import argparse
import os
import glob
from typing import Tuple

import imageio
import numpy as np

import jax
import jax.numpy as jnp
import jax.random as jrng

from mechagogue import serial

from dirt.envs.tera_arium import make_tera_arium, TeraAriumParams
from dirt.gridworld2d.landscape import LandscapeParams
from dirt.bug import BugParams
from dirt.visualization.viewer import Viewer


def _stitch_frames(frames: np.ndarray, tile_dimensions: Tuple[int, int]) -> np.ndarray:
    tr, tc = tile_dimensions
    assert frames.shape[0] == tr * tc
    frames = frames.reshape(tr, tc, *frames.shape[1:])
    rows = [np.concatenate(list(frames[r]), axis=1) for r in range(tr)]
    return np.concatenate(rows, axis=0)


def build_env(tile_dimensions, world_size, max_players, initial_players):
    params = TeraAriumParams(
        distributed=True,
        tile_dimensions=tile_dimensions,
        world_size=world_size,
        initial_players=initial_players,
        max_players=max_players,
        include_audio=False,
        include_smell=False,
        include_wind=True,
        include_rain=False,
        include_temperature=True,
        include_light=True,
        include_water=True,
        include_energy=True,
        include_biomass=True,
        report_object_grid=False,
        landscape=LandscapeParams(
            world_size=world_size,
            include_audio=False,
            include_smell=False,
            include_wind=True,
            include_rain=False,
            include_temperature=True,
            include_light=True,
            include_water=True,
            include_energy=True,
            include_biomass=True,
        ),
        bugs=BugParams(
            world_size=world_size,
            initial_players=initial_players,
            max_players=max_players,
            include_audio=False,
            include_smell=False,
            include_wind=True,
            include_rain=False,
            include_temperature=True,
            include_light=True,
            include_water=True,
            include_energy=True,
            include_biomass=True,
        ),
    )
    env = make_tera_arium(params)
    return env, params


def run_simulation(args):
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=1024)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--output", type=str, default="./migration_test.mp4")
    parser.add_argument("--report-dir", type=str, default="./migration_reports")
    parser.add_argument("--world-size", type=int, nargs=2, default=(64, 64))
    parser.add_argument("--tile-rows", type=int, default=1)
    parser.add_argument("--tile-cols", type=int, default=2)
    parser.add_argument("--max-players", type=int, default=64)
    parser.add_argument("--initial-players", type=int, default=32)
    args = parser.parse_args(args)

    world_size = tuple(args.world_size)
    tile_dimensions = (args.tile_rows, args.tile_cols)
    ndev = jax.device_count()
    if ndev < tile_dimensions[0] * tile_dimensions[1]:
        raise RuntimeError(
            f"Need at least {tile_dimensions[0] * tile_dimensions[1]} devices, "
            f"found {ndev}"
        )

    env, params = build_env(
        tile_dimensions, world_size, args.max_players, args.initial_players
    )

    def init_fn(key):
        return env.init(key)[0]

    def step_fn(key, state):
        key, action_key, step_key = jrng.split(key, 3)
        actions = jrng.randint(
            action_key, (params.max_players,), 0, env.num_actions
        )
        next_state, _, _, _, _ = env.step(step_key, state, actions, state.bug_traits)
        return next_state, actions

    def render_fn(state):
        return env.make_video_report(state)

    def report_fn(state, actions):
        return env.make_visualizer_report(state, actions)

    init_pm = jax.pmap(init_fn, axis_name="mesh")
    step_pm = jax.pmap(step_fn, axis_name="mesh")
    render_pm = jax.pmap(render_fn, axis_name="mesh")
    report_pm = jax.pmap(report_fn, axis_name="mesh")

    key = jrng.key(0)
    keys = jrng.split(key, tile_dimensions[0] * tile_dimensions[1])
    state = init_pm(keys)

    report_tiles = [[] for _ in range(tile_dimensions[0] * tile_dimensions[1])]

    writer = imageio.get_writer(args.output, fps=args.fps, codec="libx264")
    try:
        for _ in range(args.steps):
            key, step_key = jrng.split(key)
            step_keys = jrng.split(step_key, tile_dimensions[0] * tile_dimensions[1])
            state, actions = step_pm(step_keys, state)

            sharded_frames = render_pm(state)
            host_frames = np.array(jax.device_get(sharded_frames))
            frame = _stitch_frames(host_frames, tile_dimensions)
            frame = (frame * 255).astype(np.uint8)
            frame = frame[::-1]
            writer.append_data(frame)

            sharded_reports = report_pm(state, actions)
            host_reports = jax.device_get(sharded_reports)
            ntiles = tile_dimensions[0] * tile_dimensions[1]
            def _slice_report(report, idx):
                def _slice_leaf(leaf):
                    if leaf is None or leaf is False:
                        return leaf
                    if hasattr(leaf, "shape") and leaf.shape and leaf.shape[0] == ntiles:
                        return leaf[idx]
                    return leaf
                return jax.tree_util.tree_map(_slice_leaf, report)
            for i in range(ntiles):
                report_tiles[i].append(_slice_report(host_reports, i))
    finally:
        writer.close()

    if report_tiles:
        os.makedirs(args.report_dir, exist_ok=True)
        for i, reports in enumerate(report_tiles):
            if not reports:
                continue
            report_block = jax.tree_util.tree_map(
                lambda *xs: jnp.stack(xs), *reports
            )
            path = f"{args.report_dir}/tile_{i:03d}"
            os.makedirs(path, exist_ok=True)
            serial.save_leaf_data(
                report_block, f"{path}/block_0000.msgpack", compress=False
            )


def run_viewer(args):
    parser = argparse.ArgumentParser()
    parser.add_argument("--report-dir", type=str, default="./migration_reports")
    parser.add_argument("--world-size", type=int, nargs=2, default=(64, 64))
    parser.add_argument("--tile-rows", type=int, default=1)
    parser.add_argument("--tile-cols", type=int, default=2)
    parser.add_argument("--window-size", type=int, nargs=2, default=(1024, 1024))
    parser.add_argument("--max-players", type=int, default=64)
    parser.add_argument("--initial-players", type=int, default=32)
    args = parser.parse_args(args)

    world_size = tuple(args.world_size)
    tile_dimensions = (args.tile_rows, args.tile_cols)
    env, _ = build_env(
        tile_dimensions, world_size, args.max_players, args.initial_players
    )

    report_files = []
    for tile_dir in sorted(glob.glob(f"{args.report_dir}/tile_*")):
        blocks = sorted(glob.glob(f"{tile_dir}/block_*.msgpack"))
        report_files.append(blocks)

    viewer_kwargs = dict(
        example_report=env.default_visualizer_report(),
        report_files=report_files,
        world_size=(world_size[0] * tile_dimensions[0], world_size[1] * tile_dimensions[1]),
        tile_dimensions=tile_dimensions,
        get_terrain_map=env.visualizer_terrain_map,
        get_terrain_texture=env.visualizer_terrain_texture,
        window_size=tuple(args.window_size),
    )
    if hasattr(env, "get_player_color"):
        viewer_kwargs["get_player_color"] = env.get_player_color
    if hasattr(env, "print_player_info"):
        viewer_kwargs["print_player_info"] = env.print_player_info

    viewer = Viewer(**viewer_kwargs)
    viewer.start()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="mode", required=True)
    subparsers.add_parser("run")
    subparsers.add_parser("view")
    args, rest = parser.parse_known_args()

    if args.mode == "run":
        run_simulation(rest)
    else:
        run_viewer(rest)
