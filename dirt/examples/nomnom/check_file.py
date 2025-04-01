import jax
import jax.numpy as jnp
import jax.random as jrng
import os
import glob

from mechagogue.serial import load_example_data
from mechagogue.breed.normal import normal_mutate
from mechagogue.pop.natural_selection import (
    natural_selection, NaturalSelectionParams)

from dirt.examples.nomnom.nomnom_model import nomnom_model
from dirt.examples.nomnom.train_nomnom import NomNomTrainParams, NomNomModelParams, NomNomParams, make_report
from dirt.examples.nomnom.nomnom_env_evaluate import nomnom_no_reproduce, place_food_in_middle

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
    
    view_width=5,
    view_distance=5
)

train_params = NaturalSelectionParams(
    max_population=max_players,
)

params = NomNomTrainParams(
    env_params=custom_5x5_params_fixed_food,
    train_params=train_params,
    epochs=4,
    steps_per_epoch=256,
)

def evaluate_state_file(
    params,
    key = 12345
):
    key = jrng.key(key)
    epoch = 0
    reset_env, step_env = nomnom_no_reproduce(params.env_params)

    # - build mutate function
    mutate = normal_mutate(learning_rate=3e-4)
    
    # - build the model functions
    model_params = NomNomModelParams(
        view_width=params.env_params.view_width,
        view_distance=params.env_params.view_distance,
    )
    init_model, model = nomnom_model(model_params)
    
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

    return (key, epoch, train_state)

def extract_epoch(filename):
    import re
    match = re.search(r"train_state_(\d+)\.state", filename)
    return int(match.group(1)) if match else float("inf")

def check_folder_for_loadable_states(folder, params):
    """
    Attempts to load all 'train_state_*.state' files in the given folder.
    Logs success/failure for each file.
    """
    # Find all train_state_*.state files
    pattern = os.path.join(folder, "train_state_*.state")
    state_files = glob.glob(pattern)
    
    # Sort numerically by epoch
    state_files = sorted(state_files, key=extract_epoch)

    if not state_files:
        print("No train_state_*.state files found in", folder)
        return

    print(f"Checking {len(state_files)} train_state files...")

    # Create a dummy template for loading

    success_files = []
    failed_files = []

    for state_file in state_files:
        print(f"Trying to load: {state_file}")
        try:
            # Attempt to load the file using the correct template
            dummy_template = evaluate_state_file(params, 12345)
            key, epoch, train_state = load_example_data(dummy_template, state_file)
            
            # If no error, log success
            success_files.append(state_file)
            print(f"Successfully loaded: {state_file} (Epoch {epoch})")
        except Exception as e:
            # If an error occurs, log failure
            failed_files.append((state_file, str(e)))
            print(f"Failed to load: {state_file}")
            print(f"Error: {e}")

    # Summary
    print("\n====== Summary ======")
    print(f"Successfully loaded {len(success_files)}/{len(state_files)} files.")
    print(f"Failed to load {len(failed_files)} files.")
    
    if failed_files:
        print("\n The following files failed to load:")
        for f, err in failed_files:
            print(f"   {f} -> {err}")

    return success_files, failed_files

folder_path = "/Users/wangchengrui11/Desktop/SUPER/MARL_Scaled/run0"

g, b = check_folder_for_loadable_states(folder_path, params)