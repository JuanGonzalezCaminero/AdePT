# SPDX-FileCopyrightText: 2023 CERN
# SPDX-License-Identifier: Apache-2.0

import pandas as pd
import numpy as np
import sys
import os

if len(sys.argv) < 6:
	print("Usage: python3 transform_univariate_test_data_throughput.py output_file x_labels_file write_index data_directory num_primaries [data.csv data_2.csv ...]")
	exit()
	
output_file = sys.argv[1]
x_labels_file = sys.argv[2]
write_index = bool(int(sys.argv[3]))
data_directory = sys.argv[4]
num_primaries = float(sys.argv[5])
data_files = sys.argv[6:]

output = pd.DataFrame()
#This has to follow the order in which the runs are defined in the configuration
x_labels = open(x_labels_file).read().strip().split(",")

for dir in os.listdir(data_directory):
    partial_output = pd.DataFrame()
    for i in range(len(data_files)):
        file = data_directory + "/" + dir + "/" + data_files[i]
        data = pd.read_csv(file, sep="\s*,\s*", engine='python')
        throughput = num_primaries/data["Total"]
        partial_output[x_labels[i]] = throughput
    output = pd.concat([output, partial_output])

print(output)
    
output.to_csv(output_file + ".csv", index=write_index)

