# SPDX-FileCopyrightText: 2023 CERN
# SPDX-License-Identifier: Apache-2.0

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import sys
from cycler import cycler
import matplotlib.gridspec as gridspec

if len(sys.argv) < 6:
	print("Usage: python3 plot_bar_chart.py output_file x_label y_label data_1.csv data_2.csv")
	exit()

output_file = sys.argv[1]
x_label = sys.argv[2]
y_label = sys.argv[3]
data_files = sys.argv[4:]


prop_cycle = plt.rcParams['axes.prop_cycle']
colors = prop_cycle.by_key()['color']
#Uncomment this line for custom color cycle
#plt.rc('axes', prop_cycle=(cycler('color', ['c', 'orange', 'r', 'g', 'yellow', 'violet'])))

plt.rcParams.update({'font.size': 30})

# plt.figure(figsize=(30, 12))

# Create a figure with gridspec
fig = plt.figure(figsize=(20, 15))
gs = gridspec.GridSpec(2, 1, height_ratios=[2, 1])

ax1 = fig.add_subplot(gs[0])
ax2 = fig.add_subplot(gs[1])

#Adjust the width based on the number of bars we need to plot, leaving space on the sides
width=1.5/(len(sys.argv)-2)

data1 = 64/ pd.read_csv(data_files[0])
data2 = 64/ pd.read_csv(data_files[1])

dataframes = [data1, data2]

x=[int(i) for i in data1.columns]

for i in range(len(dataframes)):
	data = dataframes[i]
	
	#Get the mean of each timing
	means = data.mean()
	#Get the standard error for each timing.
	errors = [np.std(data[column]) for column in data.columns]
	
	#Draw grid below other figures
	ax1.set_axisbelow(True)
	ax1.grid(True, axis='y', color='black', linestyle='dotted')

	#Plot the data
	ax1.errorbar(x=x, y=means, yerr=errors, linewidth=2, marker="s", markersize=10, elinewidth=1, label=data_files[i])
	ax1.set_xticks(x, data.columns)
	ax1.set_ylabel(y_label)
	ax1.set_xlabel(x_label)
	#plt.yscale("log")
	ax1.legend()

# Now plot the ratio of both
ratio = data1/data2

ratio_mean = ratio.mean()

# Correct one:
if(len(data)>1):
	ratio_error = np.sqrt((data1.std()/data1.mean())**2 + (data2.std()/data2.mean())**2) * ratio_mean
else:
	ratio_error = [0 * len(data.columns)]
# ratio_error = ratio.std()

#Draw grid below other figures
ax2.set_axisbelow(True)
ax2.grid(True, axis='y', color='black', linestyle='dotted')
#Plot the data in a bar chart
ax2.errorbar(x=x, y=ratio_mean, yerr=ratio_error, linewidth=2, marker="s", markersize=10, elinewidth=1, label="Ratio")
ax2.set_xticks(x, dataframes[0].columns)
ax2.set_ylabel("Speedup")

ax2.set_ylim([0, 2.5])
# ax2.legend()

plt.tight_layout(rect=[0, 0, 1, 0.95])

#plt.show()
plt.suptitle("Comparison of asynchronous AdePT with Geant4")
plt.savefig(output_file, dpi=300, bbox_inches='tight', pad_inches=0.5)
