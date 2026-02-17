import pytest


jax = pytest.importorskip("jax")

import jax.random as jrng

from dirt.envs.tera_arium import make_tera_arium, TeraAriumParams
from dirt.gridworld2d.landscape import LandscapeParams
from dirt.bug import BugParams


def test_tera_arium_smoke():
    world_size = (64, 64)
    params = TeraAriumParams(
        world_size=world_size,
        initial_players=16,
        max_players=32,
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
            initial_players=16,
            max_players=32,
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

    key = jrng.key(0)
    key, init_key = jrng.split(key)
    state, obs, players = env.init(init_key)

    key, action_key = jrng.split(key)
    actions = jrng.randint(action_key, (params.max_players,), 0, env.num_actions)

    key, step_key = jrng.split(key)
    next_state, next_obs, players, parents, children = env.step(
        step_key, state, actions, state.bug_traits
    )

    assert obs is not None
    assert next_obs is not None
    assert players.shape[0] == params.max_players


@pytest.mark.skipif(jax.device_count() < 2, reason="Requires at least 2 devices")
def test_tera_arium_distributed_smoke():
    world_size = (64, 64)
    params = TeraAriumParams(
        distributed=True,
        tile_dimensions=(1, 2),
        world_size=world_size,
        initial_players=16,
        max_players=32,
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
            initial_players=16,
            max_players=32,
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

    @jax.pmap(axis_name="mesh")
    def init_fn(key):
        state, obs, players = env.init(key)
        return state, obs, players

    @jax.pmap(axis_name="mesh")
    def step_fn(key, state):
        key, action_key, step_key = jrng.split(key, 3)
        actions = jrng.randint(action_key, (params.max_players,), 0, env.num_actions)
        next_state, next_obs, players, parents, children = env.step(
            step_key, state, actions, state.bug_traits
        )
        return next_state, next_obs, players

    key = jrng.key(0)
    keys = jrng.split(key, jax.device_count())
    state, obs, players = init_fn(keys)

    key, step_key = jrng.split(key)
    step_keys = jrng.split(step_key, jax.device_count())
    next_state, next_obs, players = step_fn(step_keys, state)

    assert obs is not None
    assert next_obs is not None
    assert players.shape[-1] == params.max_players
