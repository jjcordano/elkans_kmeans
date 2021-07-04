# Implementing Elkan's Kmeans:
## A word on the algortihms :

This library implements the well-known Kmeans clustering algorithm from scratch using 2 methods : 
- the classic algorithm, and
- the accelerated version proposed by Charles Elkan in 2003.

### Classic Kmeans:
The original Kmeans algorithm was proposed in parallel by Hugo Steinhaus in 1956 and later by James MacQueen in 1967.
It is generally regarded as a fast clustering method as it is not based in computing the distance between all _n_ points in the dataset, but rather the distance between each point and _k_ centroids.

The algorithm works as follows: at first, initialize _k_ random centroids, then iterate until the algorithm converges.  At each iteration _e_, compute the distance between each point and each centroid, then assign each point to its closest centroid.  At the end, update the position of each centroid as the mean position of each point assigned to it.

### Elkan's accelerated Kmeans:
This variation of the Kmeans clustering method was presented by Charles ELkan in his 2003 paper titled __"Using the Triangle Inequality to Accelerate k-Means"__[https://www.aaai.org/Papers/ICML/2003/ICML03-022.pdf]. 

As exposed in the aforementioned article, Elkan's Kmeans method is based on 2 Lemmas and a fundamental innovation compared to the classic method : the computation of distances between centroids at the beginning of each iteration. The proofs for the following lemmas are trivial and presented in the article.

**Lemma 1:** In plain words, the first Lemma states that if the distance between a point x and a centroid b is shorter or equal than *half* the distance between b and the closest centroid c to b, then b is the closest centroid to x.

This powerful assertion allows the algorithm to avoid distance computations, especially in later iterations when many points are close to the center assigned at the previous iteration.  This improvement alone still requires the computation of distances between each point and its assigned centroid, and between the point and all possible centroids if another possible centroid does not fulfill *Lemma 1*. To accelearte these steps, Elkan introduces the following lemma :

**Lemma 2:** In plain words, the second lemma states that the distance between a point x and a centroid b at time t will always be larger or equal than the distance between point x and centroid b at time t-1 minus the change in centroid b's postion between time t-1 and time t.

With this lemma, Elkan introduces the notion of 'lower bound' between each point and all centroids not assigned to it.  In addition, to avoid computing the distance between each point and its centroid designated at the previous iteration, the 'upper bound' for this distance is used instead : the last known upper bound plus the change in the centroid's postion between time t-1 and time t.

### Computation time - Time complexity and experimental results:
**Classic Kmeans:** For the classic Kmeans method, the number of distance computations is __nke__, where _n_ is the number of data points, _k_ is the number of clusters to be found, and _e_ is the number of iterations required (Elkan, 2003).

**Elkan's Kmeans:** For the accelerated Kmeans method proposed by C. Elkan in 2003, the time complexity remains __O(nke)__ in the worst case, even though the number of distance calculations is roughly __O(n)__ only (Elkan, 2003).

**Experimental results:** We compare the execution time of both the classic Kmeans algorithm and Elkan's accelerated version on 2 datasets:
- the Iris flowers dataset (150 observations, 4 dimensions, 3 clusters), and 
- a large artificial dataset built using the sklearn.datasets.make_blobs() method (10000 observations, 10 dimensions, 6 clusters).

The table below gives the average execution time for the Kmeans.fit() method over 20 or 3 iterations for the small dataset and large dataset respectively.

| Performance        | Classic           | Elkan's  |
| ------------- |:-------------:| :-----:|
| Small Dataset | 0.0726s | 0.0419s |
| Large Dataset  | 16.5847s      |   9.0199s |

The main.py described below performs this comparison. The values in the table above are on an indicative basis and are expected to vary at each execution of the main meothod.

## A word on the Kmeans library :
### Installation:


### Presentation:
The Kmeans class in this library presents 2 methods :
- _fit()_: takes a numerical dataset as input.  Finds k centroids and assigns each point to its closest centroid, thus creating k clusters.
- _predict()_: takes an array of points to be labeled as input. Assigns each point to its closest centroid.





