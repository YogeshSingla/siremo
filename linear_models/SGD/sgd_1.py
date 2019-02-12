# Question 3 Part 2
# SGD
import numpy as np
from sklearn import linear_model
import csv

#dataset files
dataset_input = 'data_100k_300.tsv'
dataset_output = 'result_100k_300.tsv'

#read from tsv
with open(dataset_input, 'rt') as tsv:
	AoA = [line.strip().split('\t') for line in tsv]

#preprocessing
num_data_points = int(AoA[0][0])
AoA.pop(0)
num_features = int(AoA[0][0])
AoA.pop(0)
y_label = AoA[0][0]
x_label = AoA[0][1:]
AoA.pop(0)

#model parameters
X = []
y = []
for row in AoA:
	X.append(list(map(float,row[1:])))
	y.append(row[0])
n_samples = num_data_points
n_features = num_features
my_learning_rate = 0.0000001
epochs = 12

#train model
clf = linear_model.SGDRegressor(learning_rate='constant',eta0=my_learning_rate,max_iter=epochs)
clf.fit(X, y)

#obtain weights
coeff = clf.coef_
bias = clf.intercept_
string_coeff = []
string_bias = str(bias[0])
for val in coeff:
	string_coeff.append(str(val))
string_coeff.append(string_bias)

#save results in tsv
with open(dataset_output, 'w', newline='') as f_output:
    tsv_output = csv.writer(f_output, delimiter='\t')
    tsv_output.writerow(x_label)
    tsv_output.writerow(string_coeff)
