# SPDX-FileCopyrightText: 2023 CERN
# SPDX-License-Identifier: Apache-2.0

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import sys
from cycler import cycler
from matplotlib.offsetbox import AnchoredText

custom_fontsize = 18
hardware_limit = 8

if len(sys.argv) < 5:
	print("Usage: python3 plot_points.py output_file x_label y_label data.csv [data_2.csv data_3.csv ...]")
	exit()

output_file = sys.argv[1]
x_label = sys.argv[2]
y_label = sys.argv[3]
data_files = sys.argv[4:]

prop_cycle = plt.rcParams['axes.prop_cycle']
colors = prop_cycle.by_key()['color']
#Uncomment this line for custom color cycle
#plt.rc('axes', prop_cycle=(cycler('color', ['c', 'orange', 'r', 'g', 'yellow', 'violet'])))

width=0.2

fig,ax = plt.subplots()
fig.set_size_inches(12, 10)

for i in range(len(data_files)):
	file = data_files[i]
	data = pd.read_csv(file)

	#Sort the columns in ascending order
	#data = data.reindex(data.mean().sort_values().index, axis=1)
	
	#Get the mean of each timing
	means = data.mean()
	#Get the standard error for each timing.
	errors = [np.std(data[column]) for column in data.columns]
	print(errors)

	x = [int(i) for i in data.columns]
	
	#Draw grid below other figures
	ax.set_axisbelow(True)
	ax.grid(True, axis='y', color='black', linestyle='dotted')
	
	#Plot the data
	ax.errorbar(x=x, y=means, yerr=errors, label=file[file.rindex("/")+1:], linewidth=1, marker="o", markersize=4, elinewidth=1)
	ax.set_xticks(x, data.columns)
	
ax.set_ylim(bottom=0)
ax.set_ylabel(y_label, fontsize=custom_fontsize)
ax.set_xlabel(x_label, fontsize=custom_fontsize)
#Plot a vertical dotted line to indicate the hardware limit (Hardcoded)
ax.axvline(x=8, color='black', linestyle="dashed", linewidth=0.4)
plt.text(hardware_limit + 0.2, 6,'Hardware Limit', rotation=90, fontsize=custom_fontsize-2)

#Text box with the machine characteristics, geometry and gun configuration
info_text = \
"Machine:             Ryzen 7 5800X3D (8-Core) + RTX 4080\n\
Gun parameters:      Theta 10-170°, Phi 0-360°\n\
Gun composition:     $10^3$ e-, 10GeV\n\
Test:                20 runs, 16 events/run\n\
Geometry:            cms2018.gdml"

plt.rc("font", **{"size" : custom_fontsize})

ax.add_artist(AnchoredText(info_text, loc="upper right", prop={'family' : 'monospace', 'fontsize' : 13}))#, bbox=dict(boxstyle="square", facecolor="beige"), transform=ax.transAxes, horizontalalignment="right", verticalalignment="bottom")
#ax.legend(loc = (0.67, 0.6))
ax.legend()

#plt.show()
plt.savefig(output_file, dpi=300, bbox_inches='tight', pad_inches=0.5)
