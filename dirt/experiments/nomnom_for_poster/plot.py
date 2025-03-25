import os

import numpy as np

import jax.numpy as jnp

import matplotlib.pyplot as plt

from mechagogue.serial import load_example_data

def plot(directories):
    fig = plt.figure()
    y_data = []
    for directory in directories:
        reports = os.listdir(directory)
        reports = sorted(
            [report for report in reports if report.startswith('report')])
        example_report = (
            jnp.zeros(100, dtype=jnp.int32), jnp.zeros(100, dtype=jnp.int32))
        
        y0 = []
        y1 = []
        for report in reports:
            report_path = f'{directory}/{report}'
            print(report_path)
            report_data = load_example_data(example_report, report_path)
            y0.append(report_data[0])
            y1.append(report_data[1])
        y0 = jnp.concatenate(y0)
        y1 = jnp.concatenate(y1)
        
        plt.plot(np.arange(y0.shape[0]), np.array(y0), color='#FF000088')
        plt.plot(np.arange(y1.shape[0]), np.array(y1), color='#0000FF88')
    
    ax = plt.gca()
    #ax.set_facecolor((188./255., 225./255., 242./255.))
    #fig.patch.set_facecolor((188./255., 225./255., 242./255.))
    ax.set_facecolor('#BCE1F2')
    fig.patch.set_facecolor('#BCE1F2')
    
    plt.show()

plot([
    'exp_6_0',
    'exp_6_1',
    'exp_6_2',
    'exp_6_3',
    'exp_6_4',
    'exp_6_5',
    'exp_6_6',
    'exp_6_7',
])
