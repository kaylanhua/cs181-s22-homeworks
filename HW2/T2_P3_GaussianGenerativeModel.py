import numpy as np
from scipy.stats import multivariate_normal as mvn  # you may find this useful


# Please implement the fit(), predict(), and negative_log_likelihood() methods
# of this class. You can add additional private methods by beginning them with
# two underscores. It may look like the __dummyPrivateMethod below. You can feel
# free to change any of the class attributes, as long as you do not change any
# of the given function headers (they must take and return the same arguments).

class GaussianGenerativeModel:
    def __init__(self, is_shared_covariance=False):
        self.is_shared_covariance = is_shared_covariance

    # Just to show how to make 'private' methods
    def __dummyPrivateMethod(self, input):
        return None

    # TODO: Implement this method!
    def fit(self, X, y):

        self.num_classes = len(set(y))
        self.pi = [np.sum(y == k) for k in range(self.num_classes)]
        f = len(X[0]) # should be 2
        N = len(y)

        # mu
        self.mu = np.zeros([self.num_classes, f])
        for k in range(self.num_classes):
            sum0 = 0
            sum1 = 0
            n_k = np.sum(y == k)
            print("this is n_k")
            print(n_k)
            for i in range(N):
                if y[i] == k:
                    sum0 += X[i][0]
                    sum1 += X[i][1]
            # mag
            self.mu[k][0] = sum0 / n_k
            # tempz
            self.mu[k][1] = sum1 / n_k

        # sigma
        if self.is_shared_covariance:
            self.sigma = np.zeros([f, f])
        else:
            self.sigma = np.zeros([self.num_classes, f, f])
        for k in range(self.num_classes):
            for i in range(N):
                if y[i] == k:
                    subtract = (X[i] - self.mu[k]).reshape([2,1])
                    if self.is_shared_covariance:
                        self.sigma += subtract @ subtract.T / N
                    else:
                        self.sigma[k] += subtract @ subtract.T / np.sum(y == k)
        
        print("this is X")
        print(X)
        print("this is Y")
        print(y)

        print("mu")
        print(self.mu)
        print("sigma")
        print(self.sigma)
        return None

    # TODO: Implement this method!
    def predict(self, X_pred):
        # use the mu and sigma to predict which y, given an x (i.e. which distribution is most likely to produce the x that you were given, which is where the neg log likelihood comes in)
        # shared covariance matrix
        preds = []
        for x in X_pred:
            int_pred = []
            for k in range(self.num_classes):
                subtract = (x - self.mu[k]).reshape([2,1])
                if self.is_shared_covariance:
                    int_pred.append(np.log(np.power(np.linalg.det(2 * np.pi * self.sigma),-1 * 1/2))-float((1/2) * subtract.T @ np.linalg.inv(self.sigma) @ subtract))
                else:
                    int_pred.append(np.log(np.power(np.linalg.det(2 * np.pi * self.sigma[k]),-1 * 1/2))-float((1/2) * subtract.T @ np.linalg.inv(self.sigma[k]) @ subtract))
            preds.append(np.argmax(np.array(int_pred)))
        return np.array(preds)

    # TODO: Implement this method!
    def negative_log_likelihood(self, X, y):
        loss = 0
        for i in range(len(X)):
            for k in range(self.num_classes):
                if self.is_shared_covariance:
                    sigma = self.sigma
                else:
                    sigma = self.sigma[k]
                class_mvn = mvn(mean=self.mu[k] , cov=sigma)
                loss += (y[i] == k) * (class_mvn.logpdf(X[i]) + np.log(self.pi[k] / len(X)))
        return -1 * loss

