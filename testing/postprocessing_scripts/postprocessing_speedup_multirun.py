# SPDX-FileCopyrightText: 2023 CERN
# SPDX-License-Identifier: Apache-2.0

import pandas as pd
import numpy as np
import sys
import os
from string import Template

if len(sys.argv) < 6:
	print("Usage: python3 transform_univariate_test_data_throughput.py output_file x_labels_file write_index column data_directory [data.csv data_2.csv ...]")
	exit()
	
output_file = sys.argv[1]
x_labels_file = sys.argv[2]
write_index = bool(int(sys.argv[3]))
column = sys.argv[4]
data_directory = sys.argv[5]
data_files = sys.argv[6:]

output = pd.DataFrame()
#This has to follow the order in which the runs are defined in the configuration
x_labels = open(x_labels_file).read().strip().split(",")

models = ["geant4", "adept"]

for model in models:
    for dir in os.listdir(data_directory):
        partial_output = pd.DataFrame()
        for i in range(len(data_files)):
            file = data_directory + "/" + dir + "/" + Template(data_files[i]).substitute({"model": model})
            data = pd.read_csv(file, sep="\s*,\s*", engine='python')
            partial_output[x_labels[i]] = [np.average(data[column])]
        output = pd.concat([output, partial_output])

output = 1/(output[int(len(output)/2):]/output[:int(len(output)/2)])
    
output.to_csv(output_file + ".csv", index=write_index)

