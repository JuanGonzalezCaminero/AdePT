# SPDX-FileCopyrightText: 2023 CERN
# SPDX-License-Identifier: Apache-2.0

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import sys
from cycler import cycler

if len(sys.argv) < 4:
	print("Usage: python3 group_lines.py output_file group_size data.csv")
	exit()

output_file = sys.argv[1]
group_size = int(sys.argv[2])
data_file = sys.argv[3]

data = pd.read_csv(data_file)

# d = {
#     'A': [1, 2, 3, 4],
#     'B': [5, 6, 7, 8],
#     'C': [9, 10, 11, 12],
#     'D': [13, 14, 15, 16]
# }

# data = pd.DataFrame(d)

# Number of complete groups
num_groups = len(data) // group_size
    
# Group and calculate the mean per group
output_data = data.iloc[:num_groups * group_size].groupby(data.index[:num_groups * group_size] // group_size).sum()

# print(output_data)

output_data.to_csv(output_file)

# # Split columns in several groups
# groups = [data.iloc[i:i + group_size] for i in range(0, len(data) - group_size + 1, group_size)]

# # Now build an output dataframe with the per column average of each group
# output_data = pd.DataFrame(columns=data.columns)

# for g in groups:
#     pd.concat([output_data, g.mean()], ignore_index=True)

# output_data.to_csv(output_file)