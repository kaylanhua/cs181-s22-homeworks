# CS 181, Spring 2022
# Homework 4

import numpy as np
import matplotlib.pyplot as plt
import random
import copy
from copy import deepcopy
from scipy.spatial.distance import cdist
import seaborn as sns

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
        self.centers = np.random.randn(10,784)
        self.clusters = [[] for _ in range(self.K)]
        self.errors = []

    # X is a (N x 28 x 28) array where 28x28 is the dimensions of each of the N images.
    def fit(self, X):
        errors = []

        for _ in range(10):
            current_error = 0
            self.clusters = [[] for _ in range(self.K)]
            for image in X:
                ind = 0
                min = float('inf')
                for i, center in enumerate(self.centers):
                    dist = np.linalg.norm(image-center)
                    if dist < min:
                        min = dist
                        ind = i
                self.clusters[ind].append(image)
                current_error += min ** 2

            for index in range(self.K):
                self.centers[index] = np.mean(self.clusters[index], axis = 0)
            
            errors.append(current_error)

        self.errors = errors

    # This should return the arrays for K images. Each image should represent the mean of each of the fitted clusters.
    def get_mean_images(self):
        return self.centers

    def part5k(self):
        print("k means part 5 called")
        res = np.array([len(cluster) for cluster in self.clusters])
        print(res)
        return res

    def predict(self, X, pad = None):
        diffs = np.stack([X] * self.K, axis=1) - self.centers
        preds = np.argmin(np.sum(np.square(diffs), axis=2), axis=1)  # (N,)
        return preds

    def findClusterSize(self):
        sizes = []
        for cluster in self.clusters:
            sizes.append(len(cluster))
        return np.array(sizes)




class HAC(object):
    def __init__(self, linkage):
        self.linkage = linkage
        self.clusters = {} # dictionary of arrays of numpy arrays
        self.means = {}
        self.final = {} # cluster final
    
    def fit(self, X):
        print("hi fit")
        N = len(X)
        self.clusters[0] = [[i] for i in range(N)] # 300 x 1 x 1
        self.final[0] = [[i] for i in range(N)]
        self.means[0] = [image for image in X] # each dict entry is an array of centroids (which are np.arrays) , dict of arrays of np arrays
    
        def maxDist(array, one, two):
            clusterOne = X[np.array(array[one])]
            clusterTwo = X[np.array(array[two])]
            return np.max(cdist(clusterOne, clusterTwo))

        def minDist(array, one, two):
            clusterOne = X[np.array(array[one])]
            clusterTwo = X[np.array(array[two])]
            return np.min(cdist(clusterOne, clusterTwo))

        def closest2(array): # array is a list of the current clusters
            print('---level ' + str(N - len(array)) + '---')

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
                        dist = np.linalg.norm(np.average(temp1, axis=0)-np.average(temp2, axis=0))
                    
                    if dist < min: 
                        min = dist
                        a = one
                        b = two

            return a, b
                            

        for i in range(1, N+1):
            first, second = closest2(self.clusters[i-1])

            new_array = self.clusters[i-1]
            new_array[first] += self.clusters[i-1][second]
            new_array.remove(self.clusters[i-1][second])

            self.clusters[i] = new_array.copy()
            self.final[i] = copy.deepcopy(new_array)




    # Returns the mean image when using n_clusters clusters
    def get_mean_images(self, n_clusters, X):

        N = len(X)
        print(N)

        final = self.final[N - n_clusters]
        print('start get_mean_images')
        print(final)
        print('counts')
        print([len(cluster) for cluster in final])

        res = np.array([np.average(X[cluster], axis=0) for cluster in self.final[N - n_clusters]])

        print(res)
        print("get_mean_images done")
        return res

    def part5(self, n_clusters, X):
        N = len(X)
        res = np.array([len(cluster) for cluster in self.final[N - n_clusters]])
        return res

    def predict(self, X, n_clusters):
        means = self.get_mean_images(n_clusters, X)
        preds = []
        for original in X:
            ind = -1
            min = float('inf')
            for i, image in enumerate(means):
                dist = np.sum(np.square(original - image))
                if dist < min:
                    min = dist
                    ind = i
            preds.append(ind)
        return np.array(preds)


# Plotting code for parts 2 and 3
part5_results = []

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

        print('do u even get here')
        part5_k = KMeansClassifier.part5k()
        print(part5_k)
        part5_results.append(part5_k)

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



# #### this is all for k means
# # ~~ Part 2 ~~
# make_mean_image_plot(large_dataset, False)
# make_mean_image_plot(data, False)

# # ~~ Part 3 ~~
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

# make_mean_image_plot(large_dataset_standardized, True)



#### HAC CODE
# Plotting code for part 4
LINKAGES = [ 'max', 'min', 'centroid' ]

n_clusters = 10
part6 = []


fig = plt.figure(figsize=(10,10))
plt.suptitle("HAC mean images with max, min, and centroid linkages")
for l_idx, l in enumerate(LINKAGES):
    # Fit HAC
    hac = HAC(l)
    hac.fit(small_dataset)
    part6.append(hac)
    mean_images = hac.get_mean_images(n_clusters, small_dataset)
    part5_res = hac.part5(n_clusters, small_dataset)

    part5_results.append(part5_res)
    print("hist homie")
    print(part5_res)
    print(part5_results)

    # Make plot
    for m_idx in range(mean_images.shape[0]):
        m = mean_images[m_idx]
        ax = fig.add_subplot(n_clusters, len(LINKAGES), l_idx + m_idx*len(LINKAGES) + 1)
        plt.setp(ax.get_xticklabels(), visible=False)
        plt.setp(ax.get_yticklabels(), visible=False)
        ax.tick_params(axis='both', which='both', length=0)
        if m_idx == 0: plt.title(l)
        if l_idx == 0: ax.set_ylabel('Class '+str(m_idx), rotation=90)

        plt.imshow(m.reshape(28,28), cmap='Greys_r')
plt.show()

# TODO: Write plotting code for part 5

# [array([300,   3, 280,  27,  56,  68,   7, 107,  29,  55]), array([300,   1,   1,   1,   1, 289,   1,   1,   1,   1]), array([300,   2,   1,   1,   1,   1,   1,   1,   1,   1])]

# KMeansClassifier = KMeans(K=10)
# KMeansClassifier.fit(data)
# part1_error = KMeansClassifier.errors

# print('do u even get here')
# part5_k = KMeansClassifier.part5k()
# print(part5_k)
# part5_results.append(part5_k)

# print(part5_results)
# indices = np.array([i for i in range(10)])

# for result in part5_results:
#     plt.bar(indices, height=result)
#     plt.xticks(indices, [str(num) for num in indices]);
#     plt.ylabel('Number of Images')
#     plt.xlabel('Cluster Index')
#     plt.show()

# # TODO: Write plotting code for part 6
kmeans = KMeans(K = 10)
kmeans.fit(large_dataset)
print(kmeans.findClusterSize())

models = [kmeans] + part6
names = ['k-means'] + ['{} (hac)'.format(l) for l in LINKAGES]
for i in range(len(models)):
    for j in range(i + 1, len(models)):
        a = np.identity(10)[models[i].predict(small_dataset, n_clusters)]
        b = np.identity(10)[models[j].predict(small_dataset, n_clusters)]
        ab = np.stack([a] * 10, axis=1) * np.stack([b] * 10, axis=2)
        confusion = np.sum(ab, axis=0)

        plt.figure(figsize=(10, 10))
        result = plt.axes()
        result.set_title('{} vs {}'.format(names[i], names[j]))
        sns.heatmap(confusion, ax=result, annot=True, cmap='Blues')
        plt.show()
        
        
