import os

import jax
import jax.numpy as jnp
import jax.random as jrng

from mechagogue.serial import load_example_data
from dirt.examples.nomnom.experiment_nomnom import NomNomReport

import matplotlib.pyplot as plt


if __name__ == '__main__':
  exp_name = 'senescence_variations_mult'
  num_steps = 256
  example_report = NomNomReport()
  exp_path = f'./experiments/{exp_name}'
  paramsets = sorted(os.listdir(exp_path), key=lambda s: float(s[s.find('=')+1:]))
  
  # 1. Population Curves
  
  fig, axs = plt.subplots(len(paramsets), 1, figsize=(6, 1*len(paramsets)))
  means = []
  stds = []
  for i, paramset in enumerate(paramsets):
    sums = []
    set_path = os.path.join(exp_path, paramset)
    run_paths = os.listdir(set_path)
    for run in run_paths:
      run_path = os.path.join(set_path, run)
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
      sums.append(jnp.asarray(values))
    
    mean = jnp.mean(jnp.asarray(sums), axis=0)
    means.append(jnp.mean(jnp.sum(jnp.asarray(sums), axis=0)))
    stds.append(jnp.std(jnp.sum(jnp.asarray(sums), axis=0)))
    for s in sums:
      axs[i].plot(s, color='green', alpha=0.3)
    axs[i].plot(mean, color='blue', alpha=0.9, label='Mean')
    axs[i].set_xlabel(paramset[:16])
  plt.suptitle('Population')
  plt.tight_layout()
  plt.show()
  
  
  # 2. Population Averages
  
  plt.bar([(s[s.find('=')+1:])[:6] for s in paramsets], means, yerr=stds)
  # plt.xticks(rotation=60)
  plt.xlabel('Senescence')
  plt.ylabel('Total Individuals (log2)')
  plt.yscale('log', base=2)
  plt.title('Total Individuals Across Runs')
  plt.show()
  
  
  
  fig, axs = plt.subplots(len(paramsets), 1, figsize=(6, 1*len(paramsets)))
  means = []
  stds = []
  for i, paramset in enumerate(paramsets):
    sums = []
    set_path = os.path.join(exp_path, paramset)
    run_paths = os.listdir(set_path)
    for run in run_paths:
      run_path = os.path.join(set_path, run)
      report_paths = sorted([
          f'{run_path}/{file_path}'
          for file_path in os.listdir(run_path)
          if file_path.startswith('report') and file_path.endswith('.state')
      ])
    
      values = []
      s = 0
      for j, r in enumerate(report_paths):
        report = load_example_data(example_report, r)
        v = jnp.where(jnp.sum(report.players, axis=1) != 0, jnp.sum(report.player_energy, axis=1) / jnp.sum(report.players, axis=1), 0)
        # vf = jnp.nan_to_num(v, nan=0.0)
        values += list(v)
        # num_players = jnp.sum(report.players, axis=1)
        # values.at[j*num_steps:(j+1)*num_steps].set(num_players)
      #   values += list(num_players)
      #   s += jnp.sum(num_players)
      sums.append(jnp.asarray(values))
    
    mean = jnp.mean(jnp.asarray(sums), axis=0)
    means.append(jnp.mean(jnp.mean(jnp.asarray(sums), axis=0)))
    stds.append(jnp.std(jnp.mean(jnp.asarray(sums), axis=0)))
    for s in sums:
      axs[i].plot(s, color='green', alpha=0.3)
    axs[i].plot(mean, color='blue', alpha=0.9, label='Mean')
    axs[i].set_xlabel(paramset[:16])
  plt.suptitle('Energy per Individual')
  plt.tight_layout()
  plt.show()
  
  
  plt.bar([(s[s.find('=')+1:])[:6] for s in paramsets], means, yerr=stds)
  # plt.xticks(rotation=60)
  plt.xlabel('Senescence')
  plt.ylabel('Average Energy per Individual')
  plt.title('Average Energy per Individual Across Runs')
  plt.show()