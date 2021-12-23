# -*- coding: utf-8 -*-
"""
Created on Fri Apr 16 04:50:15 2021

@author: kaibu
"""
import csv
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

def plot_csv_data(file_name, is_test, **kwargs):
    print("reading from", file_name)
    fig = plt.figure(figsize=(20, 10))
    ax = fig.add_subplot(2, 1, 1)
    ax2 = fig.add_subplot(2, 1, 2)
    csv_data = pd.read_csv(file_name, na_values='NaN')
    cols = list(pd.read_csv(file_name, nrows =1))
    epochs = csv_data.axes[0].values
    opt = kwargs['opt']
    evl = kwargs['evl']

    if is_test:
        for i in range(len(opt)):
            opt[i] = opt[i] + '_test'

    plot = opt
    for f in evl:
        for col in cols:
            if f in col:
                plot.append(col)

    print(plot)
    for col in plot:
        if not is_test:

            if col == 'true_ndcg':
                ax.plot(epochs, csv_data[col].values, label=col)
                ax.set_yticks(np.arange(0.8, 0.9, 0.025))


            elif not 'test' in col:
                vals = csv_data[col].values
                x = []
                y = []
                for i in range(len(csv_data[col].values)):
                    if not pd.isnull(vals[i]):
                        x.append(i)
                        y.append(vals[i])
                ax2.plot(x, y, label=col)

        else:
            if col == 'true_ndcg_test':
                vals = csv_data[col].values
                x = []
                y = []
                for i in range(len(csv_data[col].values)):
                    if not pd.isnull(vals[i]):
                        x.append(i)
                        y.append(vals[i])
                ax.plot(x, y, label=col)
                ax.set_yticks(np.arange(0.8, 0.9, 0.025))


            elif 'test' in col:
                vals = csv_data[col].values
                x = []
                y = []
                for i in range(len(csv_data[col].values)):
                    if not pd.isnull(vals[i]):
                        x.append(i)
                        y.append(vals[i])
                ax2.plot(x, y, label=col)

    ax.grid()
    ax2.grid()
    ax.legend(loc='best')
    ax2.legend(loc='best')
    plt.show()

prefix = './csv/'
prefix2 = './csv/test2/'
plot_csv_data(prefix + "ndcg_di_efpr_Chi.csv", True, opt=['true_ndcg'], evl=['di', 'efpr_loss', 'xndcg'])