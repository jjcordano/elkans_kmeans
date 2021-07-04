import numpy as np
import time
from sklearn import datasets
from kmeans import Kmeans

# Small dataset : load Iris dataset
iris_data, iris_labels = datasets.load_iris()['data'], datasets.load_iris()['target']

# Large dataset: artificial dataset with 6000 observations, 7 dimensions and 3 clusters
blob_data, blob_labels  = datasets.make_blobs(10000,10,6)

for data in [iris_data, blob_data]:

    time_classic = []
    time_Elkan = []

    data_size = ''
    k = 0
    iterations = 0

    if data.shape[0] == iris_data.shape[0]:
        data_size = 'SMALL'
        k = 2
        iterations = 20
    else:
        data_size = 'LARGE'
        k = 6
        iterations = 3

    for i in range(iterations):
        start_c = time.time()
        classic = Kmeans(k=k)
        classic.fit(data)
        end_c = time.time()
        time_classic.append(end_c - start_c)
        
        start_E = time.time()
        elkan = Kmeans(k=k,method='Elkan')
        elkan.fit(data)
        end_E = time.time()
        time_Elkan.append(end_E - start_E)

    avg_time_classic = sum(time_classic)/len(time_classic)
    avg_time_Elkan = sum(time_Elkan)/len(time_Elkan)

    #print(classic.labels)
    #print(elkan.labels)

    print("\n")
    print("*********************** {} DATASET PERFORMANCE *************************".format(data_size))
    print("Average execution time for the classic Kmeans method : {} seconds.".format(round(avg_time_classic,4)))
    print("Average execution time for Elkan's Kmeans method : {} seconds.".format(round(avg_time_Elkan,4)))
    print("***************************************************************************")