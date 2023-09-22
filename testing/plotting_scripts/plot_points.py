# SPDX-FileCopyrightText: 2023 CERN
# SPDX-License-Identifier: Apache-2.0

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import sys
from cycler import cycler
from matplotlib.offsetbox import AnchoredText

custom_fontsize = 14

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

plt.figure(figsize=(12, 10))

for i in range(len(data_files)):
	file = data_files[i]
	data = pd.read_csv(file)

	#Sort the columns in ascending order
	#data = data.reindex(data.mean().sort_values().index, axis=1)
	
	#Get the mean of each timing
	means = data.mean()
	#Get the standard error for each timing.
	errors = [np.std(data[column]) for column in data.columns]

	x = np.arange(len(data.columns))
	
	#Draw grid below other figures
	plt.gca().set_axisbelow(True)
	plt.grid(True, axis='y', color='black', linestyle='dotted')
	
	#Plot the data
	plt.errorbar(x=x+((i-len(sys.argv[1:])//2)*width), y=means, yerr=errors, label=file[file.rindex("/")+1:], linewidth=0, marker="s", elinewidth=1)
	plt.yscale('log')
	plt.xticks(x, data.columns)
	plt.xticks(rotation=90)
	plt.ylabel(y_label, fontsize=custom_fontsize)

#Text box with the machine characteristics, geometry and gun configuration
info_text = \
"Gun parameters:      Theta 10-170°, Phi 0-360°\n\
Gun composition:     $5*10^3$ e-, 10GeV\n\
Test:                64 Events\n\
Geometry:            cms2018.gdml\n\
Field:               (0, 0, 3.8) Tesla"

plt.gca().add_artist(AnchoredText(info_text, loc="lower left", prop={'family' : 'monospace', 'fontsize' : 13}))

plt.rc("font", **{"size" : custom_fontsize})
plt.legend()

#plt.show()
plt.savefig(output_file, dpi=300, bbox_inches='tight', pad_inches=0.5)
