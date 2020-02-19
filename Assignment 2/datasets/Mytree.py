# import packages
import numpy as np
from scipy.sparse import csr_matrix
from math import log2

traing_data = np.loadtxt(fname="trainData.txt",dtype=int)
testing_data = np.loadtxt(fname="testData.txt",dtype=int)
traing_label=np.loadtxt(fname="trainLabel.txt",dtype=int)
testing_Label=np.loadtxt(fname="testLabel.txt",dtype=int)

print(traing_data.shape)
print(testing_data.shape)
print(traing_label.shape)
print(testing_Label.shape)

nWords = max(max(traing_data[:,1]),max(testing_data[:,1]))
print(nWords)
