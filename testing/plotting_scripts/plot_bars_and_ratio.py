# SPDX-FileCopyrightText: 2023 CERN
# SPDX-License-Identifier: Apache-2.0

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import sys
from cycler import cycler
import matplotlib.gridspec as gridspec

if len(sys.argv) < 7:
	print("Usage: python3 plot_bar_chart.py output_file x_label y_label max_values data_1.csv data_2.csv")
	exit()

output_file = sys.argv[1]
x_label = sys.argv[2]
y_label = sys.argv[3]
# max_values limits the number of bars we print to the x largest values, 0 for no limit
max_values = int(sys.argv[4])
data_files = sys.argv[5:]


prop_cycle = plt.rcParams['axes.prop_cycle']
colors = prop_cycle.by_key()['color']
#Uncomment this line for custom color cycle
#plt.rc('axes', prop_cycle=(cycler('color', ['c', 'orange', 'r', 'g', 'yellow', 'violet'])))

plt.rcParams.update({'font.size': 30})

# plt.figure(figsize=(30, 12))

# Create a figure with gridspec
fig = plt.figure(figsize=(40, 18))
gs = gridspec.GridSpec(2, 1, height_ratios=[2, 1])

ax1 = fig.add_subplot(gs[0])
ax2 = fig.add_subplot(gs[1])

#Adjust the width based on the number of bars we need to plot, leaving space on the sides
width=1.5/(len(sys.argv)-2)

data1 = pd.read_csv(data_files[0])
data2 = pd.read_csv(data_files[1])

# We will sort the first dataset in descending order, and use the same reordering for the second one as well
sort_indices = data1.mean().sort_values(ascending=False).index
data1 = data1.reindex(sort_indices, axis=1)
data2 = data2.reindex(sort_indices, axis=1)

dataframes = [data1, data2]

# Number of values we will put on the x axis
if(max_values):
    x = np.arange(max_values)
else:
    x = np.arange(len(data.columns))

for i in range(len(dataframes)):
	data = dataframes[i]
	
	#Get the mean of each timing
	means = data.mean()
	#Get the standard error for each timing.
	errors = [np.std(data[column]) for column in data.columns]
	
	#Draw grid below other figures
	ax1.set_axisbelow(True)
	ax1.grid(True, axis='y', color='black', linestyle='dotted')
	
	#Plot the data in a bar chart
	ax1.bar(x=x+((i-len(data_files)//2)*width), height=means[:max_values], yerr=errors[:max_values], width=width, label=data_files[i])
	ax1.set_xticks(x, data.columns[:max_values])
	ax1.set_ylabel(y_label)
	ax1.set_xlabel(x_label)
	#plt.yscale("log")
	ax1.legend()

# Now plot the ratio of both
ratio = data1/data2

ratio_mean = ratio.mean()

# Correct one:
# ratio_error = np.sqrt((data1.std()/data1.mean())**2 + (data2.std()/data2.mean())**2) * ratio_mean
ratio_error = ratio.std()

print(data1.iloc[:, :max_values])
print(data2.iloc[:, :max_values])
print(ratio.iloc[:, :max_values])
print(data1.std()[:max_values])
print(data2.std()[:max_values])
print(ratio_mean[:max_values])
print(ratio_error[:max_values])

#Draw grid below other figures
ax2.set_axisbelow(True)
ax2.grid(True, axis='y', color='black', linestyle='dotted')
#Plot the data in a bar chart
ax2.errorbar(x=x, y=ratio_mean[:max_values], yerr=ratio_error[:max_values], linewidth=1, marker="s", elinewidth=1, label="Ratio")
ax2.set_xticks(x, dataframes[0].columns[:max_values])
ax2.set_ylabel("Ratio")
# ax2.legend()

plt.tight_layout(rect=[0, 0, 1, 0.95])

#plt.show()
plt.suptitle("Accumulated energy deposition per physical volume, per 100 events")
plt.savefig(output_file, dpi=300, bbox_inches='tight', pad_inches=0.5)
