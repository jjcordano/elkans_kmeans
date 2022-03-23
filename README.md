# Elkan's Kmeans:
## A word on the algorithm :

This library implements the well-known Kmeans clustering algorithm from scratch using 2 methods : 
- the classic algorithm, and
- the accelerated version proposed by Charles Elkan in 2003.

### Classic Kmeans:
The original Kmeans algorithm was proposed in parallel by Hugo Steinhaus in 1956 and later by James MacQueen in 1967. 

This common unsupervised clustering algorithm works as follows: at first, _k_ random centroids are initialized, then at each iteration _e_, the distance between each point and each centroid is calculated, and the point is assigned to its closest centroid.  At the end of each iteration, the position of each centroid is updated to become the average of the coordinates of each point assigned to it. This is repeated until the algorithm converges.

### Elkan's accelerated Kmeans:
This variation of the Kmeans clustering method was presented by Charles Elkan in his 2003 paper titled [__*Using the Triangle Inequality to Accelerate k-Means*__](https://www.aaai.org/Papers/ICML/2003/ICML03-022.pdf). 

As presented in the article, Elkan's Kmeans method is based on 2 Lemmas, and relies mainly on one fundamental innovation compared to the classic method : the computation of distances between centroids at the beginning of each iteration. The proofs for the following lemmas are trivial and presented in the article.

#### Lemma 1:
The first Lemma states that if the distance between a point *x* and a centroid *b* is shorter or equal than *half* the distance between *b* and the closest centroid *c* to *b*, then *b* is the closest centroid to *x*.

This assertion allows the algorithm to avoid certain distance calculations, especially in later iterations when many points are close to the center assigned at the previous iteration. This improvement alone still requires the computation of distances between each point and its assigned centroid, and between the point and all possible centroids if another possible centroid does not fulfill *Lemma 1*. To accelerate these steps as well, Elkan introduces *Lemma 2* presented below.

#### Lemma 2:
In plain words, the second lemma states that the distance between a point *x* and a centroid *b* at time *t* will always be larger or equal than the distance between point *x* and centroid *b* at time *t-1* minus the change in centroid *b*'s postion between time *t-1* and time *t*.

With this, Elkan introduces the notion of 'lower bound' between each point and all centroids not assigned to it.  In addition, to avoid computing the distance between each point and its centroid designated at the previous iteration, the 'upper bound' for this distance is used instead: the last known upper bound plus the change in the centroid's postion between time t-1 and time t.

### Execution time:
#### Classic Kmeans:
For the classic Kmeans method, the number of distance computations is __nke__, where _n_ is the number of data points, _k_ is the number of clusters to be found, and _e_ is the number of iterations required (Elkan, 2003).

#### Elkan's Kmeans: 
For the accelerated Kmeans method proposed by C. Elkan in 2003, the time complexity remains __O(nke)__ in the worst case, even though the number of distance calculations is roughly __O(n)__ only (Elkan, 2003).

#### Experimental results:
We compare the execution time of both the classic Kmeans algorithm and Elkan's accelerated version on 2 datasets:
- the Iris flowers dataset (150 observations, 4 dimensions, 3 clusters), and 
- a large artificial dataset built using the sklearn.datasets.make_blobs() method (10000 observations, 10 dimensions, 6 clusters).

The table below gives the average execution time for the Kmeans.fit() method over 20 or 3 iterations for the small dataset and large dataset respectively.

| Performance        | Classic           | Elkan's  |
| ------------- |:-------------:| :-----:|
| Small Dataset | 0.0726s | 0.0419s |
| Large Dataset  | 16.5847s      |   9.0199s |

The test_performance.py file described below performs this comparison. The values in the table above are on an indicative basis and are expected to vary at each execution of the main method.

## A word on the Kmeans library :
### Installation:

The required dependencies to use the Kmeans library are listed
in `requirements.txt`. You can install the dependencies with the
following command-line code:

```bash
pip install -U -r requirements.txt
```

The Kmeans class is accessible in the `kmeans.py` file.

### Class arguments & methods:
#### Arguments of the Kmeans class:
To declare an instance of the Kmeans algorithm, the Kmeans class takes the following arguments:
- _k_: the desired number of centroids.
- _max_iter_: Maximum number of iterations to be performed before the Kmeans algorithm terminates.
- _tolerance_: Threshold distance change for each centroid to terminate algorithm.
- _method_: 'classic' or 'Elkan'. Determines whether the classic Kmeans or Elkan's accelerated Kmeans algorithm will be used. Defaults to 'Elkan'.

#### Methods:
The Kmeans class in this library presents 2 methods:
- _Kmeans.fit()_: takes a numerical dataset as input.  Finds k centroids and assigns each point to its closest centroid, thus creating k clusters.
- _Kmeans.predict()_: takes an array of points to be labeled as input. Assigns each point to its closest centroid.

#### Variables:
The following public variables allow one to access information of a fitted instance of the algorithm:
- _Kmeans.centroids_: dictionary of the corrdinates for each centroid.
- _Kmeans.labels_: list of centroids assigned to each point.

### Testing performance:
The following command-line code compares the performance of the classic Kmeans algorithm and 
Elkan's accelerated method on the small and large datasets presented above.

```bash
python3 test_performance.py
```

Please note that performance will vary each time the performance test code is run.
