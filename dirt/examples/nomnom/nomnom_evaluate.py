import jax
import jax.numpy as jnp
import jax.random as jrng
import os
import glob
import re
import json

import argparse

from mechagogue.pop.natural_selection import (
    natural_selection, NaturalSelectionParams)
from mechagogue.breed.normal import normal_mutate
from mechagogue.static_dataclass import static_dataclass
from mechagogue.player_list import birthday_player_list, player_family_tree

from mechagogue.tree import tree_getitem
from mechagogue.serial import load_example_data

from dirt.models.nomnom_model import nomnom_linear_model, nomnom_unconditional_model
from dirt.examples.nomnom.train_nomnom import NomNomTrainParams, NomNomModelParams, NomNomParams, make_report
from dirt.examples.nomnom.nomnom_env_evaluate import nomnom_no_reproduce, place_food_in_middle

def simulate_player_single_agent(n_steps, single_player_params, key):
    """
    Runs a single-agent simulation in a 5*5 no-reproduce environment using
    one player's parameters.
    """
    reset_env, step_env = nomnom_no_reproduce()
    # reset_env, step_env = nomnom()

    rng_key, step_key = jrng.split(key)
    state, obs, _ = reset_env(rng_key)
    state = place_food_in_middle(state)
    model_params = NomNomModelParams()
    init, model = nomnom_linear_model(model_params)
    initial_food = jnp.sum(state.food_grid)

    for t in range(n_steps):
        action = model(step_key, obs, single_player_params)
        next_state, next_obs, _, _, _ = step_env(step_key, state, action)
        
        # print(f"  Step {t}, action={action}, energy={next_obs.energy}")
        
        state, obs = next_state, next_obs
        # print("Agent View:\n", obs.view)

    # def step_fn(carry, _):
    #     state, obs, key = carry
    #     key, subkey = jax.random.split(key)
    #     action = model(subkey, obs, single_player_params)
    #     state, obs, _, _, _ = step_env(subkey, state, action)
    #     return (state, obs, key), None

    # carry_init = (state, obs, key)
    # (final_state, final_obs, final_key), _ = jax.lax.scan(step_fn, carry_init, jnp.arange(n_steps))
    final_food = jnp.sum(state.food_grid)
    all_food_eaten = float(initial_food - final_food)
    return all_food_eaten

def evaluate_state_file(
    params,
    state_path, 
    max_population, 
    trials_per_agent,
    steps_per_trial,
    key = 1234
):
    key = jrng.key(key)
    epoch = 0
    # reset_env, step_env = nomnom_no_reproduce(params.env_params)
    reset_env, step_env = nomnom_no_reproduce(params.env_params)

    # - build mutate function
    mutate = normal_mutate(learning_rate=3e-4)
    
    # - build the model functions
    model_params = NomNomModelParams(
        view_width=params.env_params.view_width,
        view_distance=params.env_params.view_distance,
    )
    init_model, model = nomnom_linear_model(model_params)
    
    # - build the training functions
    reset_train, step_train = natural_selection(
        params.train_params,
        reset_env,
        step_env,
        init_model,
        model,
        mutate,
        make_report
    )
    
    # get the initial state of the training function
    key, reset_key = jrng.split(key)
    train_state, _ = jax.jit(reset_train)(reset_key)

    print(f"Loading checkpoint from {state_path}...")
    key, epoch, train_state = load_example_data(
        (key, epoch, train_state),
        state_path
    )
    print(f"Loaded train_state from epoch {epoch}.")

    env_state = train_state.env_state
    model_state = train_state.model_state
    obs = train_state.obs

    # Use the same approach that the environment uses for active players
    _, _, active_players = birthday_player_list(max_population)
    active_mask = active_players(env_state.family_tree.player_list)
    active_indices = jnp.where(active_mask)[0]
    
    print(f"Active players: {active_indices}")

    all_food_eaten = []

    for i in active_indices:
        # print(f"\n=== Simulating single-agent run for active player {i} ===")
        
        single_player_params = tree_getitem(model_state, i)
        
        for _ in range(trials_per_agent):
            key, sim_key = jrng.split(key)
            food_eaten = simulate_player_single_agent(
                steps_per_trial, single_player_params, sim_key
            )
            # print(f"player {i}: {food_eaten}")
            all_food_eaten.append(food_eaten)

    # def simulate_agents_and_trials(n_steps, params_array, keys_array):
    #     def run_agent(params, keys_for_agent):
    #         return jax.vmap(simulate_player_single_agent,
    #                     in_axes=(None, None, 0),
    #                     out_axes=0)(n_steps, params, keys_for_agent)
        
    #     results = jax.vmap(run_agent, in_axes=(0, 0), out_axes=0)(params_array, keys_array)
    #     return results
    
    # batch_params = jax.vmap(lambda i: tree_getitem(model_state, i))(active_indices)

    # n_agents = active_indices.shape[0]
    # keys_for_agents = jrng.split(key, n_agents * trials_per_agent)
    # keys_for_agents = keys_for_agents.reshape(n_agents, trials_per_agent, 2)

    # food_eaten_arr = simulate_agents_and_trials(steps_per_trial, batch_params, keys_for_agents)
    # all_food_eaten = food_eaten_arr.reshape(-1)

    if len(all_food_eaten) == 0:
        return (epoch, 0.0, 0.0)

    arr = jnp.array(all_food_eaten)
    mean_val = float(jnp.mean(arr))
    var_val = float(jnp.var(arr))
    return (epoch, mean_val, var_val)

def main(params):
    parser = argparse.ArgumentParser()
    parser.add_argument("--folder", type=str, default=".")
    parser.add_argument("--max_population", type=int, default=1)
    parser.add_argument("--trials_per_agent", type=int, default=5)
    parser.add_argument("--steps_per_trial", type=int, default=5)
    args = parser.parse_args()
    folder = args.folder
    state_files = glob.glob(os.path.join(folder, "train_state_*.state"))
    if not state_files:
        print(f"No .state files found in {folder}.")
        return

    def extract_epoch_from_filename(fname):
        match = re.search(r"train_state_(\d+)\.state", fname)
        if match:
            return int(match.group(1))
        else:
            return -1
        
    state_files = sorted(
        state_files, 
        key=lambda f: extract_epoch_from_filename(os.path.basename(f))
    )

    results = []
    for sf in state_files:
        # Evaluate
        epoch, mean_food, var_food = evaluate_state_file(
            params,
            sf,
            max_population=args.max_population,
            trials_per_agent=args.trials_per_agent,
            steps_per_trial=args.steps_per_trial
        )
        # we store epoch, mean, var
        results.append((epoch, mean_food, var_food))
        with open("/Users/wangchengrui11/Desktop/SUPER/MARL_Scaled/dirt/dirt/linear.json", "w") as f:
            json.dump(results, f)
        print(f"File={sf}, epoch={epoch}, mean={mean_food:.3f}, var={var_food:.3f}")
    

if __name__ == "__main__":
    max_players = 1

    custom_5x5_params_fixed_food = NomNomParams(
        world_size=(5, 5),
        initial_players=1,          
        max_players=1,               
        mean_initial_food=0,         
        max_initial_food=0,
        mean_food_growth=0,
        max_food_growth=0,
        initial_energy=1.0,          
        max_energy=5.0,
        food_metabolism=1.0,
        move_metabolism=-0.05,
        wait_metabolism=-0.025,
        senescence=0.0,
    )

    train_params = NaturalSelectionParams(
        max_population=max_players,
    )

    params = NomNomTrainParams(
        env_params=custom_5x5_params_fixed_food,
        train_params=train_params,
        epochs=4,
        steps_per_epoch=8,
    )

    # main(params)

    with open('/Users/wangchengrui11/Desktop/SUPER/MARL_Scaled/dirt/dirt/linear.json', 'r') as f:
        results_linear = json.load(f)
    with open('/Users/wangchengrui11/Desktop/SUPER/MARL_Scaled/dirt/dirt/unconditional.json', 'r') as f:
        results_unconditional = json.load(f)

    import matplotlib.pyplot as plt
    import numpy as np
    epochs = [r[0] for r in results_linear]

    means_linear = [r[1] for r in results_linear]
    vars_linear = [r[2] for r in results_linear]

    means_unconditional = [r[1] for r in results_unconditional]
    vars_unconditional = [r[2] for r in results_unconditional]

    # print(np.mean(means_linear))
    # print(np.mean(means_unconditional))
    # breakpoint()

    plt.figure(figsize=(10, 6))

    plt.plot(epochs, means_linear, 'g--', label='Linear Model')
    plt.fill_between(epochs, means_linear - 2*np.sqrt(vars_linear), means_linear + 2*np.sqrt(vars_linear), color='green', alpha=0.3)

    plt.plot(epochs, means_unconditional, 'r--', label='Unconditional Model')
    plt.fill_between(epochs, means_unconditional - 2*np.sqrt(vars_unconditional), means_unconditional + 2*np.sqrt(vars_unconditional), color='red', alpha=0.3)

    plt.xlabel("Epochs")
    plt.ylabel("Food")
    plt.title("Food Get in 5*5 Test Bed for linear and unconditional model")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    # plt.show()
    png_path = os.path.join('/Users/wangchengrui11/Desktop/SUPER/MARL_Scaled/plots', "food_plot_comparison_sd.png")
    plt.savefig(png_path)
    print(f"Plot saved to {png_path}")