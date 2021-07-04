#!/usr/bin/env python
# coding: utf-8

import numpy as np

class Kmeans:
    def __init__(self, k=2, max_iter=250, tolerance=0, method='Elkan'):
        """
        Initialises a Kmeans clustering algorithm.
        
        Args:
        - k: Number of clusters. Defaults to 2.
        - max_iter: Maximum number of iterations to be performed before the Kmeans algorithm terminates. Defaults to 250.
        - tolerance: Threshold distance change for each centroid to terminate algorithm. Defaults to 0.
        When the distance rate of change for each centroid between 2 subsequent iterations is lower than the tolerance, the algorithm terminates.
        - method: 'classic' or 'Elkan'. Determines whether the classic Kmeans or Elkan's accelerated Kmeans algorithm will be used. Defaults to 'Elkan'.
        """
        
        assert method in ['classic','Elkan'], "Method argument not valid"
        
        self.k = k
        self.max_iter = max_iter
        self.tol = tolerance
        self.method = method
    
    def fit(self, data):
        '''
        Finds k centroids for a dataset of numeric points.
        
        Args:
        - data: Numpy array or pandas DataFrame of numerical values.
        '''
        pointsArray = np.array(data)

        ## Initializing k random centroids within the bounds of the data points
        self.centroids = {}
        self.labels = [0 for point in pointsArray]
        initCentroids = []

        for dim in range(pointsArray.shape[1]):
            dim_min = np.min(pointsArray, axis=0)[dim]
            dim_max = np.max(pointsArray, axis=0)[dim]
            newCentroid = (dim_max-dim_min)*np.random.random_sample([1,self.k])+dim_min
            initCentroids = np.append(initCentroids,newCentroid)

        initCentroids = initCentroids.reshape((pointsArray.shape[1],self.k)).T

        self.centroids = dict(zip(list(range(self.k)),initCentroids))
        
        ## Classic Kmeans
        if self.method == 'classic':
            for i in range(self.max_iter):
                self.classifications = {}
                self.pointsClassif = {}

                for i in range(self.k):
                    self.classifications[i] = []
                    self.pointsClassif[i] = []

                for point in pointsArray:
                    distances = [np.linalg.norm(point-self.centroids[centroid]) for centroid in self.centroids]
                    classification = distances.index(min(distances))
                    self.classifications[classification].append(point)

                prevCentroids = dict(self.centroids)

                for classification in self.classifications:
                    if len(self.classifications[classification]) == 0:
                        pass

                    else:
                        self.centroids[classification] = np.average(self.classifications[classification],axis=0)

                optimized = [True for centroid in self.centroids]

                for centroid in self.centroids:
                    original_centroid = prevCentroids[centroid]
                    current_centroid = self.centroids[centroid]
                    if abs(np.sum((current_centroid-original_centroid)/original_centroid*100.0)) > self.tol :
                        optimized[centroid] = False

                if False not in optimized:
                    for i in range(pointsArray.shape[0]):
                        point = pointsArray[i]
                        distances = [np.linalg.norm(point-self.centroids[centroid]) for centroid in self.centroids]
                        classification = distances.index(min(distances))
                        self.pointsClassif[classification].append(i)
                        
                    break
        
        ## Accelerated (Elkan) Kmeans
        elif self.method == 'Elkan':
            
            self.classifications = {} ## Points coordinates by centroid
            self.pointsClassif = {} ## Points indices by centroid
            
            lowerBounds = np.zeros((pointsArray.shape[0],self.k)) ## Lower bounds matrix. Dimensions : (nb_points, nb_centroids)
            upperBounds = np.zeros((pointsArray.shape[0])) ## Upper bounds vector : Dimension : (nb_points)


            for i in range(self.k):
                self.classifications[i] = []
                self.pointsClassif[i] = []
                
            i=0
            for point in pointsArray:
                distances = [np.linalg.norm(point-self.centroids[centroid]) for centroid in self.centroids]
                classification = distances.index(min(distances))
                    
                ## Centroid assigned to each point
                self.classifications[classification].append(point)
                self.pointsClassif[classification].append(i)
                    
                ## Lower bound distance between the point and each center
                ## Initialized as distance between the point and each initial centroid
                lowerBounds[i] = distances
                    
                ## Upper bound distance between the point and assigned centroid
                upperBounds[i] = min(distances)
                    
                i+=1
                
            prevCentroids = dict(self.centroids.copy())
            prevClassifications = dict(self.classifications.copy())
            prevPointsClassif = dict(self.pointsClassif.copy())

            for classification in self.classifications:
                if len(self.classifications[classification]) == 0:
                    pass

                else:
                    self.centroids[classification] = np.average(self.classifications[classification],axis=0)

            optimized = [True for centroid in self.centroids]
            
            centroidDistanceChange = {}

            for centroid in self.centroids:
                original_centroid = prevCentroids[centroid]
                current_centroid = self.centroids[centroid]
                centroidDistanceChange[centroid] = np.linalg.norm(original_centroid-current_centroid)
                
                if abs(np.sum((current_centroid-original_centroid)/original_centroid*100.0)) > self.tol :
                    optimized[centroid] = False

            if False not in optimized:
                
                for centroid in self.pointsClassif:
                    for point in self.pointsClassif[centroid]:
                        self.labels[point] = centroid
                return
            
            ## Update lower and upper bound distances
            for centroid in self.pointsClassif:
                for i in list(range(pointsArray.shape[0])):
                    lowerBounds[i][centroid] -= centroidDistanceChange[centroid]
                    
                for i in self.pointsClassif[centroid]:
                    upperBounds[i] += centroidDistanceChange[centroid]
                    
            ## Repeat until convergence        
            for it in range(self.max_iter):
                
                listCentroids = list(range(self.k))
                self.classifications = {} ## Points coordinates by centroid
                self.pointsClassif = {} ## Points indices by centroid
                centroidDistances = {} ## Distances to other centroids for each centroid
                closestCentroidDistances = {} ## Distance to closest centroid for each centroid
                
                for i in range(self.k):
                    self.classifications[i] = []
                    self.pointsClassif[i] = []
                    centroidDistances[i] = [np.linalg.norm(self.centroids[i]-self.centroids[c_prime]) for c_prime in self.centroids]
                    closestCentroidDistances[i] = min(centroidDistances[i][:i]+centroidDistances[i][i+1:])
                
                for centroid in prevPointsClassif:
                    for i in prevPointsClassif[centroid]:
                        
                        r = True
                        distToCurrentCentroid = upperBounds[i]
                        
                        ## Check if upper bound lower than 1/2 of distance with closest centroid
                        if upperBounds[i] <= 0.5*closestCentroidDistances[centroid]:
                            ## If condition is met : said point keeps its centroid with no further computation needed
                            self.classifications[centroid].append(pointsArray[i])
                            self.pointsClassif[centroid].append(i)
                        
                        
                        else:
                            assigned_centroid = centroid
                            for c_prime in (listCentroids[:centroid]+listCentroids[centroid+1:]):
                                ## Check if lower bound between point and c_prime < upper bound between point and its current centroid
                                ## AND if (0.5*distance between current centroid and c_prime) < upper bound between point and its current centroid
                                if ((distToCurrentCentroid > lowerBounds[i][c_prime]) and (distToCurrentCentroid > 0.5*centroidDistances[centroid][c_prime])): 
                                    if r:
                                        distToCurrentCentroid = np.linalg.norm(pointsArray[i] - self.centroids[centroid])
                                        r = False
                                        
                                    distToCPrime = np.linalg.norm(pointsArray[i]-self.centroids[c_prime])
                                        
                                    if distToCurrentCentroid > distToCPrime:
                                        assigned_centroid = c_prime 
                        
                            self.classifications[assigned_centroid].append(pointsArray[i])
                            self.pointsClassif[assigned_centroid].append(i)
                                            
                prevCentroids = dict(self.centroids.copy())
                prevClassifications = dict(self.classifications.copy())
                prevPointsClassif = dict(self.pointsClassif.copy())
                
                for classification in self.classifications:
                    if len(self.classifications[classification]) == 0:
                        pass

                    else:
                        self.centroids[classification] = np.average(self.classifications[classification],axis=0)

                optimized = [True for centroid in self.centroids]
                
                centroidDistanceChange = {}

                for centroid in self.centroids:
                    original_centroid = prevCentroids[centroid]
                    current_centroid = self.centroids[centroid]
                    centroidDistanceChange[centroid] = np.linalg.norm(original_centroid-current_centroid)

                    if abs(np.sum((current_centroid-original_centroid)/original_centroid*100.0)) > self.tol :
                        optimized[centroid] = False

                if False not in optimized:
                    break
                    
                ## Update of lower and upper bound distances
                for centroid in self.pointsClassif:
                    for i in list(range(pointsArray.shape[0])):
                        lowerBounds[i][centroid] -= centroidDistanceChange[centroid]
                    
                    for i in self.pointsClassif[centroid]:
                        upperBounds[i] += centroidDistanceChange[centroid]
        
        ## Update labels (cluster) for each point
        for centroid in self.pointsClassif:
            for point in self.pointsClassif[centroid]:
                self.labels[point] = centroid
                        
    def predict(self,data):
        """
        Assigns point(s) to cluster (or centroid).
        Clusters must have been previously computed using Kmeans.fit() method on dataset.
        
        Args:
        - data: 1-d or 2-d numpy array of points to be assigned to a cluster.
        """
        
        data = np.array(data)
        
        assert data.flatten().shape[0] % self.centroids[self.labels[0]].shape[0] == 0, "All inputs must be of shape ({},)".format(self.centroids[self.labels[0]])
        
        if (data.flatten().shape[0] / self.centroids[self.labels[0]].shape[0]) == 1:
            distances = [np.linalg.norm(data - self.centroids[centroid]) for centroid in self.centroids]
            classification = distances.index(min(distances))
            
            return classification
        
        else:
            
            assert data.shape[1] == data.shape[-1] == self.centroids[self.labels[0]].shape[0], "All inputs must be of shape ({},)".format(self.centroids[self.labels[0]])
            
            classifications = []
            
            for point in data:
                distances = [np.linalg.norm(point - self.centroids[centroid]) for centroid in self.centroids]
                classification = distances.index(min(distances))
                classifications.append(classification)
                
            return classifications

