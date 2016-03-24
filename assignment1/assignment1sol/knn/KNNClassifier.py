import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import mode

class KNN(object):
    '''
    a kNN classifier with L2 dist.
    '''

    def __init__(self):
        pass

    def train(self, X, y):
        '''
        Train the classifier. For kNN this is just memorizing the training data.

        Inputs:
            - X: A numpy array of shape (m, n) containing the training data
                consisting of m samples each of dimension n.
            - y: A numpy array of shape (m,) containing the training labels, 
                where y[i] is the label for X[i].
        '''
        self.X = X
        self.y = y

    def predict(self, X_test, hpara = 1, show_img = False):
        '''
        Predict labels for test data using this classifier.

        Inputs:
        - X: A numpy array of shape (m_test, n) containing test data 
                consisting of m_test samples each of dimension n.
        - k: The number of nearest neighbors that vote for the predicted 
                labels.
        '''
        k = hpara

        # First we must compute the distances between all test examples 
        # and all train examples.
        # This should result in a m_test x m matrix where each element 
        # (i,j) is the distance between the i-th test and j-th train example.
        m_test = X_test.shape[0]
        m = self.X.shape[0]
        n = X_test.shape[1]
        y_hat = np.zeros(m_test, dtype = self.y.dtype)
        D = np.zeros((m_test, m))
        D = (X_test ** 2).dot(np.ones((n, m))) + np.ones((m_test, n)).dot(self.X.T ** 2) - 2 * X_test.dot(self.X.T)
        if show_img:
            print 'D shape is:', D.shape
            # We can visualize the distance matrix: each row is a single test # example and its distances to training examples
            plt.imshow(D, interpolation='none')
            plt.show()
        
        # Given these distances, for each test example we find the 
        # k nearest examples and have them vote for the label
        # Use the distance matrix to find the k nearest neighbors of the 
        # ith testing point, and use self.y_train to find the labels of 
        # these neighbors. Store these labels in closest_y.                           
        # Break ties by choosing the smaller label.                
        # Hint: Look up the function numpy.argsort.    
        for i in xrange(m_test):
            nearest_y = self.y[np.argsort(D[i, :])[:k]]
            y_hat[i] = mode(nearest_y)[0][0]
        return y_hat