from posixpath import join
import numpy as np

# Please implement the predict() method of this class
# You can add additional private methods by beginning them with
# two underscores. It may look like the __dummyPrivateMethod below. You can feel
# free to change any of the class attributes, as long as you do not change any
# of the given function headers (they must take and return the same arguments).

class KNNModel:
    def __init__(self, k):
        self.X = None
        self.y = None
        self.K = k

    # Just to show how to make 'private' methods
    def __dummyPrivateMethod(self, input):
        return None

    # TODO: Implement this method!
    def predict(self, X_pred):
        preds = []
        def __dist(ind1, ind2):
            return ((self.X[ind1][0] - X_pred[ind2][0]) / 3) ** 2 + (self.X[ind1][1] - X_pred[ind2][1]) ** 2

        N = len(self.X)
        for j in range(len(X_pred)):
            classes = []
            dists = []

            for i in range(N):
                dists.append((self.X[i], __dist(i, j), self.y[i]))

            dists.sort(key=lambda x:x[1])
            
            for j in range(self.K):
                classes.append(dists[j][2])

            preds.append(max(set(classes), key=classes.count))

        print("results for k=" + str(self.K) + " done")
        return np.array(preds)

    # In KNN, "fitting" can be as simple as storing the data, so this has been written for you
    # If you'd like to add some preprocessing here without changing the inputs, feel free,
    # but it is completely optional.
    def fit(self, X, y):
        self.X = X
        self.y = y