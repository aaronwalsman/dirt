import argparse
from typing import Tuple

import imageio
import numpy as np

import jax
import jax.random as jrng

from dirt.envs.tera_arium import make_tera_arium, TeraAriumParams
from dirt.gridworld2d.landscape import LandscapeParams
from dirt.bug import BugParams


def _stitch_frames(frames: np.ndarray, tile_dimensions: Tuple[int, int]) -> np.ndarray:
    tr, tc = tile_dimensions
    assert frames.shape[0] == tr * tc
    frames = frames.reshape(tr, tc, *frames.shape[1:])
    rows = [np.concatenate(list(frames[r]), axis=1) for r in range(tr)]
    return np.concatenate(rows, axis=0)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=1024)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--output", type=str, default="./migration_test.mp4")
    parser.add_argument("--world-size", type=int, nargs=2, default=(64, 64))
    parser.add_argument("--tile-rows", type=int, default=1)
    parser.add_argument("--tile-cols", type=int, default=2)
    parser.add_argument("--max-players", type=int, default=64)
    parser.add_argument("--initial-players", type=int, default=32)
    args = parser.parse_args()

    world_size = tuple(args.world_size)
    tile_dimensions = (args.tile_rows, args.tile_cols)
    ndev = jax.device_count()
    if ndev < tile_dimensions[0] * tile_dimensions[1]:
        raise RuntimeError(
            f"Need at least {tile_dimensions[0] * tile_dimensions[1]} devices, "
            f"found {ndev}"
        )

    params = TeraAriumParams(
        distributed=True,
        tile_dimensions=tile_dimensions,
        world_size=world_size,
        initial_players=args.initial_players,
        max_players=args.max_players,
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
            initial_players=args.initial_players,
            max_players=args.max_players,
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

    def init_fn(key):
        return env.init(key)[0]

    def step_fn(key, state):
        key, action_key, step_key = jrng.split(key, 3)
        actions = jrng.randint(
            action_key, (params.max_players,), 0, env.num_actions
        )
        next_state, _, _, _, _ = env.step(step_key, state, actions, state.bug_traits)
        return next_state

    def render_fn(state):
        return env.make_video_report(state)

    init_pm = jax.pmap(init_fn, axis_name="mesh")
    step_pm = jax.pmap(step_fn, axis_name="mesh")
    render_pm = jax.pmap(render_fn, axis_name="mesh")

    key = jrng.key(0)
    keys = jrng.split(key, tile_dimensions[0] * tile_dimensions[1])
    state = init_pm(keys)

    writer = imageio.get_writer(args.output, fps=args.fps, codec="libx264")
    try:
        for _ in range(args.steps):
            key, step_key = jrng.split(key)
            step_keys = jrng.split(step_key, tile_dimensions[0] * tile_dimensions[1])
            state = step_pm(step_keys, state)

            sharded_frames = render_pm(state)
            host_frames = np.array(jax.device_get(sharded_frames))
            frame = _stitch_frames(host_frames, tile_dimensions)
            frame = (frame * 255).astype(np.uint8)
            frame = frame[::-1]
            writer.append_data(frame)
    finally:
        writer.close()


if __name__ == "__main__":
    main()
