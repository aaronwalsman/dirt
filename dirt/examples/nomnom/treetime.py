import os
from io import StringIO

import jax
import jax.numpy as jnp
import jax.random as jrng

from mechagogue.serial import load_example_data
from dirt.examples.nomnom.train_nomnom import NomNomReport, NomNomTrainParams
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
  params = load_example_data(NomNomTrainParams(), './train_params.state')
  
  init_players, step_players, active_players = birthday_player_list(
        params.max_players)
  init_family_tree, step_family_tree, active_family_tree = player_family_tree(
      init_players, step_players, active_players, 1)
  
  family_tree = init_family_tree(params.initial_players)
  
  example_report = NomNomReport(family_tree)
  report_paths = sorted([
      f'./{file_path}'
      for file_path in os.listdir('./')
      if file_path.startswith('report') and file_path.endswith('.state')
  ])
    
  values = []
  s = 0
  for j, r in enumerate(report_paths):
    report = load_example_data(example_report, r)
    family_tree = report.family_tree
    print(f'{j}: FT')
    # print(family_tree.player_list.players) # 0 is birthdays, 1 is locations (num_steps, max_players, [birthdays, locations])
    # print(family_tree.parents[0, :16]) # (num_steps, max_players, [1], [2])
    edges = []
    past_children = set()
    for k in range(params.steps_per_epoch):
      parent_locations, child_locations = family_info(k, family_tree, params)
      for parents, children in zip(parent_locations, child_locations):
        parent_index = int(parents[0])
        child_index = int(children)
        if parent_index == -1 or child_index == -1:
          break;
        if child_index in past_children:
          child_index = max(past_children) + 1
        if child_index < 500:
          past_children.add(child_index)
          edges.append((parent_index, child_index))
  G = nx.DiGraph()
  print("Adding edges!")
  G.add_edges_from(edges)
  nx.write_edgelist(G, "test.edgelist")
  print("Done!")
  # G = nx.read_edgelist("test.edgelist")
  plt.figure(figsize=(8, 6))
  pos = nx.nx_pydot.pydot_layout(G, prog="twopi")
  # pos = nx.fruchterman_reingold_layout(G, seed=74)  # Adjust layout
  nx.draw(G, pos, with_labels=False, node_color="black", edge_color='grey', node_size=10, font_size=10)
  # print("Checking version.")
  # print(pgv.__version__)
  # pos = nx.nx_agraph.graphviz_layout(G, prog="twopi")
  # nx.draw(G, pos)
  plt.show()
  newick_str = tree_to_newick(G)
  # tree = Phylo.read("example_tree.nwk", "newick")
  tree = Phylo.read(StringIO(newick_str), "newick")
  # # Plot the tree
  fig, ax = plt.subplots(figsize=(8, 6))
  Phylo.draw(tree, axes=ax)
  plt.show()
  
  
  

        # if a child has already been born, check against set and assign it a new value.
      # print(report.player_x[0, :32])
      # print(parent_locations[:16])
      # print(child_locations[:16])
    # num_players = jnp.sum(report.players, axis=1)
    # values.at[j*num_steps:(j+1)*num_steps].set(num_players)
    # values += list(num_players)
    # s += jnp.sum(num_players)
  # sums.append(s)
    
  # axs[i].plot(values, color='green')
  # axs[i].set_xlabel(run)
  # plt.tight_layout()
  # plt.show()
  
  # plt.bar([s[s.find('=')+1:] for s in runs], sums)
  # # plt.xticks(rotation=60)
  # plt.xlabel('Number of Initial Players')
  # plt.ylabel('Total Individuals (log2)')
  # plt.yscale('log', base=2)
  # plt.title('Total Individuals Across Runs')
  # plt.show()