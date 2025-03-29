import os
from io import StringIO

import jax
import jax.numpy as jnp
import jax.random as jrng
import numpy as np

from mechagogue.serial import load_example_data
from dirt.experiments.nomnom_tree_extraction.train import TrainReport, TrainParams
from mechagogue.player_list import birthday_player_list, player_family_tree

import matplotlib.pyplot as plt
import networkx as nx
from Bio import Phylo

def family_info(i, family_tree, params):
        birthdays = family_tree.player_list.players[i, ..., 0]
        current_time = family_tree.player_list.current_time[i]
        child_locations, = jnp.nonzero(
            birthdays == current_time,
            size=params.max_players,
            fill_value=params.max_players,
        )
        parents = family_tree.parents[i]
        parent_info =  parents[child_locations]
        parent_locations = parent_info[...,1]
        
        return parent_locations, child_locations



def tree_to_newick(G, root=None):
    """Convert a NetworkX tree (Graph or DiGraph) to Newick format."""
    if not isinstance(G, nx.DiGraph):
        if root is None:
            root = next(iter(G.nodes))  # Pick an arbitrary node as root
        G = nx.bfs_tree(G, source=root)  # Convert to directed graph

    def build_newick(node):
        """Recursively build Newick string from the tree."""
        children = list(G.successors(node))  # Get children of the node
        if not children:
            return str(node)  # Leaf node
        return f"({','.join(build_newick(child) for child in children)}){node}"
    
    return build_newick(root) + ";"  # Add Newick termination

if __name__ == '__main__':
  num_steps = 256
  params = load_example_data(TrainParams(), './params.state')
  
  init_players, step_players, active_players = birthday_player_list(
        params.max_players)
  init_family_tree, step_family_tree, active_family_tree = player_family_tree(
      init_players, step_players, active_players, 1)
  
  family_tree = init_family_tree(params.env_params.initial_players)
  
  example_report = TrainReport(family_tree) # family_tree
  report_paths = sorted([
      f'./{file_path}'
      for file_path in os.listdir('./')
      if file_path.startswith('report') and file_path.endswith('.state')
  ])
    
  all_players = jnp.zeros((params.runner_params.steps_per_epoch * params.runner_params.epochs, params.max_players))
  all_parents = jnp.zeros((params.runner_params.steps_per_epoch * params.runner_params.epochs, params.max_players))
  for j, r in enumerate(report_paths):
    report = load_example_data(example_report, r)
    players = report.players
    family_tree = report.family_tree
    # print(family_tree.player_list.players) # 0 is birthdays, 1 is locations (num_steps, max_players, [birthdays, locations])
    # print(family_tree.parents[0, :16]) # (num_steps, max_players, [1], [2])
    # edges = []
    # past_children = set()
    all_players = all_players.at[j*params.runner_params.steps_per_epoch:(j+1)*params.runner_params.steps_per_epoch, :].set(players)
    
    
    steps_per_epoch = params.runner_params.steps_per_epoch
    max_players = params.max_players
    
    parent_locations, child_locations = jax.vmap(family_info, in_axes=(0, None, None))(
        jnp.arange(steps_per_epoch), family_tree, params
    )

    # Extract parent indices
    parent_indices = parent_locations[..., 0]  # Shape: (steps_per_epoch, num_parents)
    child_indices = child_locations[...]

    # Compute row indices
    row_indices = jnp.repeat(jnp.arange(steps_per_epoch), parent_indices.shape[1]) + j * steps_per_epoch
    col_indices = parent_indices.flatten()
    child_values = child_indices.flatten()

    # Filter out invalid parent indices (negative values)
    valid_mask = col_indices >= 0
    row_indices = row_indices[valid_mask]
    col_indices = col_indices[valid_mask]
    child_values = child_values[valid_mask]

    # Perform batched updates
    all_parents = all_parents.at[row_indices, col_indices].set(child_values)
    
    # for k in range(steps_per_epoch):
    #   parent_locations, child_locations = family_info(k, family_tree, params)
    #   for parents in parent_locations:
    #       parent_index = parents[0]
    #       all_parents = all_parents.at[k + j*steps_per_epoch, parent_index].set(1)

    # for i in range(params.runner_params.steps_per_epoch):
    #     for k in range(params.max_players):
    #         p_index = parents[i][k][0][0]
    #         if p_index < 0:
    #             break
    #         else:
    #             all_parents = all_parents.at[i + j*params.runner_params.steps_per_epoch, p_index].set(1)
  
np.savetxt("players.txt", np.array(all_players, dtype=int), delimiter=", ", fmt="%d")
np.savetxt("parents.txt", np.array(all_parents, dtype=int), delimiter=", ", fmt="%d")