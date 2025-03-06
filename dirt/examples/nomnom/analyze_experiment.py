import os

import jax
import jax.numpy as jnp
import jax.random as jrng

from mechagogue.serial import load_example_data
from dirt.examples.nomnom.experiment_nomnom import NomNomReport

import matplotlib.pyplot as plt


if __name__ == '__main__':
  exp_name = 'a'
  num_steps = 256
  example_report = NomNomReport()
  exp_path = f'./experiments/{exp_name}'
  runs = sorted(os.listdir(exp_path), key=lambda s: int(s[s.find('=')+1:]))
  fig, axs = plt.subplots(len(runs), 1, figsize=(6, 1*len(runs)))
  sums = []
  for i, run in enumerate(runs):
    run_path = os.path.join(exp_path, run)
    report_paths = sorted([
        f'{run_path}/{file_path}'
        for file_path in os.listdir(run_path)
        if file_path.startswith('report') and file_path.endswith('.state')
    ])
    
    values = []
    s = 0
    for j, r in enumerate(report_paths):
       report = load_example_data(example_report, r)
       num_players = jnp.sum(report.players, axis=1)
       # values.at[j*num_steps:(j+1)*num_steps].set(num_players)
       values += list(num_players)
       s += jnp.sum(num_players)
    sums.append(s)
    
    axs[i].plot(values, color='green')
    axs[i].set_xlabel(run)
  plt.tight_layout()
  plt.show()
  
  plt.bar([s[s.find('=')+1:] for s in runs], sums)
  # plt.xticks(rotation=60)
  plt.xlabel('Number of Initial Players')
  plt.ylabel('Total Individuals (log2)')
  plt.yscale('log', base=2)
  plt.title('Total Individuals Across Runs')
  plt.show()