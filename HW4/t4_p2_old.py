# CS 181, Spring 2022
# Homework 4

import numpy as np
import matplotlib.pyplot as plt
import random
np.random.seed(2)

# Loading datasets for K-Means and HAC
small_dataset = np.load("/Users/kayla/Documents/GitHub/cs181-s22-homeworks/HW4/data/small_dataset.npy")
large_dataset = np.load("/Users/kayla/Documents/GitHub/cs181-s22-homeworks/HW4/data/large_dataset.npy")
data = np.load("/Users/kayla/Documents/GitHub/cs181-s22-homeworks/HW4/P2_Autograder_Data.npy")

# NOTE: You may need to add more helper functions to these classes
class KMeans(object):
    # K is the K in KMeans
    def __init__(self, K):
        self.K = K
        self.centers = None
        self.cluster_sum = None
        self.cluster_no = np.array([1 for _ in range(self.K)])
        self.errors = []

    # X is a (N x 28 x 28) array where 28x28 is the dimensions of each of the N images.
    def fit(self, X):
        np.random.seed(2)
        N = len(X) # 5000

        # initializing random cluster centers
        # ----- do i have to worry about choosing the same point accidentally
        if self.centers is None:
                self.centers = np.array([X[random.randint(0,N-1)] for _ in range(self.K)])
                # self.centers = np.random.randn(10, 784)
                self.cluster_sum = self.centers
                # self.centers = np.zeros((10, 784))

        errors = []

        # for _ in range(10):
        while True:
            current_error = 0
            
            for image in X:
                ind = 0
                min = float('inf')
                for i, center in enumerate(self.centers):
                    dist = np.linalg.norm(image-center)
                    if dist < min:
                        min = dist
                        ind = i
                self.cluster_no[ind] += 1
                # self.cluster_sum[ind] += image
                self.cluster_sum[ind] = np.add(self.cluster_sum[ind], image, out=self.cluster_sum[ind], casting="unsafe")
                current_error += min ** 2

            new_centers = np.array([self.cluster_sum[h] / self.cluster_no[h] for h in range(self.K)])
            if np.array_equal(self.centers, new_centers):
                break
            else:
                self.centers = new_centers

                errors.append(current_error)

                print(self.cluster_no)
                self.cluster_no = np.array([1 for _ in range(self.K)])
                self.cluster_sum = np.array([np.array([0 for _ in range(784)]) for _ in range(self.K)])

        self.errors = errors

    # This should return the arrays for K images. Each image should represent the mean of each of the fitted clusters.
    def get_mean_images(self):
        return np.array(self.centers)

class HAC(object):
    def __init__(self, linkage):
        self.linkage = linkage
        self.clusters = {} # dictionary of arrays of numpy arrays
        self.means = {}
    
    def fit(self, X):
        print("hi fit")

        # X contains all 300 indiv data points, reference X by index instead of storing the whole thing

        N = len(X)
        self.clusters[0] = [[i] for i in range(N)] # 300 x 1 x 1
        self.means[0] = [image for image in X] # each dict entry is an array of centroids (which are np.arrays) , dict of arrays of np arrays
        print(type(self.clusters[0][0]))
        
        # print('shape of self clusters 0')
        # print(np.shape(self.clusters[0]))
        # print(self.clusters[0][0])

        def maxDist(array, one, two):
            # print('maxDist')
            current = -1 * float('inf')
            for a in array[one]:
                for b in array[two]:
                    dist = np.linalg.norm(X[a]-X[b])
                    current = max(dist, current)
            # print(current)
            return current


        def minDist(array, one, two):
            current = float('inf')
            for a in array[one]:
                for b in array[two]:
                    dist = np.linalg.norm(X[a]-X[b])
                    current = min(dist, current)
            return current

        def closest2(array): # array is a list of the current clusters
            print('hi, closest2')
            print('---level ' + str(N - len(array)) + '---')
            # print(np.shape(array))

            # takes in a NOT NUMPY ARRAY self.clusters[i] and returns the indices of the two closest clusters
            n = len(array)
            a = 0
            b = 0

            min = float("inf")
            for one in range(n-1):
                for two in range(one + 1, n):
                    if self.linkage == 'max':
                        dist = maxDist(array, one, two)
                    
                    elif self.linkage == 'min':
                        dist = minDist(array, one, two)
                    
                    elif self.linkage == 'centroid':
                        # this one tricky
                        temp1 = X[np.array(array[one])]
                        temp2 = X[np.array(array[two])]
                        dist = np.linalg.norm(np.average(temp1)-np.average(temp2))
                    
                    if dist < min: 
                        min = dist
                        a = one
                        b = two

            print('found closest 2!!!')
            return a, b
                            

        for i in range(1, N):
            first, second = closest2(self.clusters[i-1])
            print("first")
            print(first)
            print('second')
            print(second)

            
            self.clusters[i] = self.clusters[i-1]
            self.clusters[i].remove(self.clusters[i-1][second])
            
            print("first array")
            print(self.clusters[i-1][first])
            print(type(self.clusters[i-1][first]))
            print("second array")
            print(self.clusters[i-1][second])

            self.clusters[i][first] += (self.clusters[i-1][second]) # have to use 1s and 0s, fixed size arrays
            
            print('combined cluster (after combining)')
            print(self.clusters[i][first])

            print('new goddamn cluster')
            print(self.clusters[i])

            # self.means[i][first] = 
        
        # finding means

    # Returns the mean image when using n_clusters clusters
    def get_mean_images(self, n_clusters):
        # redo this part
        return np.array(self.centers[self.N - n_clusters])

# Plotting code for parts 2 and 3
def make_mean_image_plot(data, standardized=False):
    # Number of random restarts
    niters = 3
    K = 10
    # Will eventually store the pixel representation of all the mean images across restarts
    allmeans = np.zeros((K, niters, 784))
    for i in range(niters):
        KMeansClassifier = KMeans(K=K)
        KMeansClassifier.fit(data)
        part1_error = KMeansClassifier.errors
        allmeans[:,i] = KMeansClassifier.get_mean_images()

    # # ~~ Part 1 ~~ plotting errors
    plt.plot([i for i in range(len(part1_error))], part1_error)

    fig = plt.figure(figsize=(10,10))
    plt.suptitle('Class mean images across random restarts' + (' (standardized data)' if standardized else ''), fontsize=16)
    for k in range(K):
        for i in range(niters):
            ax = fig.add_subplot(K, niters, 1+niters*k+i)
            plt.setp(ax.get_xticklabels(), visible=False)
            plt.setp(ax.get_yticklabels(), visible=False)
            ax.tick_params(axis='both', which='both', length=0)
            if k == 0: plt.title('Iter '+str(i))
            if i == 0: ax.set_ylabel('Class '+str(k), rotation=90)
            plt.imshow(allmeans[k,i].reshape(28,28), cmap='Greys_r')
    plt.show()


#### this is all for k means
# ~~ Part 2 ~~
# make_mean_image_plot(large_dataset, False)
make_mean_image_plot(data, False)

# ~~ Part 3 ~~
# TODO: Change this line! standardize large_dataset and store the result in large_dataset_standardized

# # method 1
# large_dataset_standardized = large_dataset
# for i, image in enumerate(large_dataset_standardized):
#     # print(image)
#     mean = np.mean(image)
#     stdev = np.std(image)
#     large_dataset_standardized[i] = (image - mean) / stdev
    
# # method 2 (pixel)
# pixels = 784
# p3Means = np.zeros(pixels)
# p3Devs = np.zeros(pixels)
# for i in range(pixels):
#     pixelList = [image[i] for image in large_dataset]
#     p3Means[i] = np.mean(pixelList)
#     dev = np.std(pixelList)
#     if dev == 0:
#         p3Devs[i] = 1
#     else: p3Devs[i] = dev
    
# large_dataset_standardized = np.divide(np.subtract(large_dataset, p3Means), p3Devs)

# # sohams
# def standardize(dataset):
#     variance = np.var(dataset, axis = 0)
#     variance[variance == 0] = 1
#     return (dataset - np.mean(dataset, axis = 0))/np.sqrt(variance)

# large_dataset_standardized = standardize(large_dataset)

# make_mean_image_plot(large_dataset_standardized, True)

# # Plotting code for part 4
# LINKAGES = [ 'max', 'min', 'centroid' ]
# n_clusters = 10

# fig = plt.figure(figsize=(10,10))
# plt.suptitle("HAC mean images with max, min, and centroid linkages")
# for l_idx, l in enumerate(LINKAGES):
#     # Fit HAC
#     hac = HAC(l)
#     hac.fit(small_dataset)
#     mean_images = hac.get_mean_images(n_clusters)
#     # Make plot
#     for m_idx in range(mean_images.shape[0]):
#         m = mean_images[m_idx]
#         ax = fig.add_subplot(n_clusters, len(LINKAGES), l_idx + m_idx*len(LINKAGES) + 1)
#         plt.setp(ax.get_xticklabels(), visible=False)
#         plt.setp(ax.get_yticklabels(), visible=False)
#         ax.tick_params(axis='both', which='both', length=0)
#         if m_idx == 0: plt.title(l)
#         if l_idx == 0: ax.set_ylabel('Class '+str(m_idx), rotation=90)
#         plt.imshow(m.reshape(28,28), cmap='Greys_r')
# plt.show()

# TODO: Write plotting code for part 5

# TODO: Write plotting code for part 6