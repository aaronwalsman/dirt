import os
from io import StringIO

import jax
import jax.numpy as jnp
import jax.random as jrng
import numpy as np

from mechagogue.serial import load_example_data
from dirt.experiments.nomnom_example_experiment.train import TrainReport, TrainParams
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
  
  example_report = TrainReport() # family_tree
  report_paths = sorted([
      f'./{file_path}'
      for file_path in os.listdir('./')
      if file_path.startswith('report') and file_path.endswith('.state')
  ])
    
  values = []
  s = 0
  
  all_players = jnp.zeros((params.runner_params.steps_per_epoch * params.runner_params.epochs, params.max_players))
  print(all_players.shape)
  for j, r in enumerate(report_paths):
    report = load_example_data(example_report, r)
    players = report.players
    # family_tree = report.family_tree
    # print(family_tree.player_list.players) # 0 is birthdays, 1 is locations (num_steps, max_players, [birthdays, locations])
    # print(family_tree.parents[0, :16]) # (num_steps, max_players, [1], [2])
    edges = []
    past_children = set()
    all_players = all_players.at[j*params.runner_params.steps_per_epoch:(j+1)*params.runner_params.steps_per_epoch, :].set(players)
    # for k in range(params.steps_per_epoch):
    #   players[k]
    #   players = players.at[k*steps_per_epoch:(k+1)*params.steps_per_epoch, :].set(arr1)
      # parent_locations, child_locations = family_info(k, family_tree, params)
      # for parents, children in zip(parent_locations, child_locations):
      #   parent_index = int(parents[0])
      #   child_index = int(children)
      #   if parent_index == -1 or child_index == -1:
      #     break;
      #   if child_index in past_children:
      #     child_index = max(past_children) + 1
      #   if child_index < 500:
      #     past_children.add(child_index)
      #     edges.append((parent_index, child_index))
  
np.savetxt("data.txt", np.array(all_players, dtype=int), delimiter=", ", fmt="%d")