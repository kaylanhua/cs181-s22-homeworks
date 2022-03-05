import numpy as np
import matplotlib.pyplot as plt


# Please implement the fit(), predict(), and visualize_loss() methods of this
# class. You can add additional private methods by beginning them with two
# underscores. It may look like the __dummyPrivateMethod below. You can feel
# free to change any of the class attributes, as long as you do not change any
# of the given function headers (they must take and return the same arguments).

class LogisticRegression:
    def __init__(self, eta, lam, runs=200000):
        self.eta = eta
        self.lam = lam
        self.runs = runs
        self.losses = []

    # Just to show how to make 'private' methods
    def __dummyPrivateMethod(self, input):
        return None

    def __softmax(self, vec):
        return np.exp(vec)/np.sum(np.exp(vec))

    # TODO: Implement this method!
    def fit(self, X, y):
        # np.random.seed(1738)
        self.num_classes = len(set(y)) # 3
        N = X.shape[0]
        Xnew = np.hstack((np.ones(N), X[:,0], X[:,1])).reshape((self.num_classes, N)).T
        self.W = np.random.rand(Xnew.shape[1], self.num_classes)

        for _ in range(self.runs):
            xw = np.dot(Xnew, self.W)
            grad = np.zeros(self.W.shape)
            for i in range(N):
                for j in range(self.num_classes):
                    grad[:,j] += (self.__softmax(xw[i])[j] - (y[i]==j)) * Xnew[i]
            self.W -= self.eta * (grad + 2 * self.lam * self.W)
            self.losses.append(self.__loss(Xnew, y))
            # print(self.losses)

    # TODO: Implement this method!
    def predict(self, X_pred):
        preds = []
        Xnew = np.hstack((np.ones(len(X_pred)), X_pred[:,0], X_pred[:,1])).reshape((self.num_classes, len(X_pred))).T
        xw = np.dot(Xnew, self.W)
        for x in xw:
            preds.append(np.argmax(x))
        return np.array(preds)

    def __loss(self, X, y):
        loss = 0
        xw = np.dot(X, self.W)
        for i in range(len(X)):
            for k in range(len(set(y))):
                loss += (y[i] == k) * np.log(self.__softmax(xw[i])[k])
        return -1 * loss

    # TODO: Implement this method!
    def visualize_loss(self, output_file, show_charts=False):
        print(self.losses)
        it_limit = self.runs
        iterations = [i for i in range(it_limit)]
        plt.plot(iterations, self.losses)
        plt.ylabel('Negative Log-Likelihood Loss')
        plt.xlabel('Number of Iterations')
        plt.title('LR Loss for eta='+str(self.eta)+', lam='+str(self.lam))
        plt.savefig(output_file)