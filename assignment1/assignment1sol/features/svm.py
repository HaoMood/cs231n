import numpy as np
import matplotlib.pyplot as plt
from gradient_check import grad_check_sparse

class SVM:
    def __init__(self, n, K):
        np.random.seed(0)
        # generate a random SVM weight matrix of small numbers
        self.W = np.random.randn(K, n) * 0.0001
        self.b = np.random.randn(K) * 0.0001

    def _costFcn(self, W, b, X, y, lamda):
        '''
        Structured SVM loss function.
        Inputs have dimension n, there are K classes, and we operate on minibatches of m examples.

        Inputs:
            - W: A numpy array of shape (K, n) containing weights.
            - X: A numpy array of shape (m, n) containing a minibatch of data.
            - y: A numpy array of shape (m,) containing training labels; y[i] = k means that X[i] has label k, where 0 <= k < K.
            - lamda: (float) regularization strength

        Returns a tuple of:
            - J as single float
            - dW: grad. wrt. weights W; an array of same shape as W
        '''
        J = 0.0
        dW = np.zeros(W.shape, dtype = np.float32)
        db = np.zeros(b.shape, dtype = np.float32)
        m, n = X.shape
        K = W.shape[0]
        X = X.T

        S = W.dot(X) + np.reshape(b, (K, 1))
        margin = S - S[y, np.arange(m)] + 1
        margin[y, np.arange(m)] = 0
        J = np.sum(np.maximum(0, margin)) / m + 0.5 * lamda * np.sum(W ** 2)

        dW = lamda * W
        dS = (margin > 0).astype(np.float32) / m
        dS[y, np.arange(m)] = -np.sum(margin > 0, axis = 0).astype(np.float32) / m
        dW += dS.dot(X.T)
        
        db = np.sum(dS, axis = 1)

        assert dW.shape == W.shape
        assert db.shape == b.shape
        return J, dW, db

    def train_check(self, X, y, lamda = 1e-3):    
        # Gradient check.
        # Numerically compute the gradient along several randomly 
        # chosen dimensions, and  compare them with your analytically 
        # computed gradient. The numbers should match almost exactly # along all dimensions.
        J, dW, db = self._costFcn(self.W, self.b, X, y, lamda)

        print '\nGradient check on W'
        f = lambda W: self._costFcn(W, self.b, X, y, lamda)[0]
        grad_numerical = grad_check_sparse(f, self.W, dW)

        print '\nGradient check on b'
        f = lambda b: self._costFcn(self.W, b, X, y, lamda)[0]
        grad_numerical = grad_check_sparse(f, self.b, db)   

    def train(self, X, y, hpara = (1e-3, 1e-5, 100, 200), debug = True, show_img = True):
        (alpha, lamda, T, B) = hpara
        '''
        Train this linear classifier using stochastic gradient descent.

        Inputs:
            - X: A numpy array of shape (m, n) containing training data; there are m training samples each of dimension n.
            - y: A numpy array of shape (m,) containing training labels; y[i] = k means that X[i] has label 0 <= k < K for K classes.
            - alpha: (float) learning rate for optimization.
            - lamda: (float) regularization strength.
            - T: (integer) number of steps to take when optimizing
            - B: (integer) number of training examples to use at each step.
            - debug: (boolean) If true, print progress during optimization.

        Outputs:
            A list containing the value of the loss function at each training iteration.
        '''
        # We now have the gradient and our gradient matches the numerical gradient. We are therefore ready to do SGD to minimize the loss.
        m, n = X.shape
        J_hist = []
        for t in xrange(T):
            # Sample B elements from the training data and their          
            # corresponding labels to use in this round of SGD
            # Store the data in X_batch and their corresponding labels 
            # in y_batch
            # After sampling X_batch should have shape (B, n)   
            # and y_batch should have shape (B,)                           
            # Hint: Use np.random.choice to generate indices. 
            # Sampling with replacement is faster than sampling without 
            # replacement.           
            idx = np.random.choice(m, B, replace = False)
            X_batch = X[idx]
            y_batch = y[idx]

            # Compute the loss and its gradient at W.
            J, dW, db = self._costFcn(self.W, self.b, X_batch, y_batch, lamda)
            J_hist.append(J)

            # perform parameter update
            self.W += -alpha * dW            
            self.b += -alpha * db

            if debug and t % 100 == 0:
                print 'iteration %d / %d: loss %f' % (t, T, J)

        if show_img:
            # A useful debugging strategy is to plot the loss as a function of
            # iteration number:
            plt.plot(J_hist)
            plt.xlabel('t')
            plt.ylabel('J')
            plt.show()

    def predict(self, X):
        '''
        Use the trained weights of this linear classifier to predict labels for
        data points.

        Inputs:
        - X: m x n array of training data. Each column is a n-dimensional point.

        Returns:
        - y: Predicted labels for the data in X. y is a 1-dimensional array of length m, and each element is an integer giving the predicted class.
        '''
        m, n = X.shape
        y = np.zeros(m)
        S = self.W.dot(X.T) + np.reshape(self.b, (-1, 1))
        y = np.argmax(S, axis = 0)
        return y

    def visualize_W(self):
        W = self.W.reshape(32, 32, 3, 10)
        W_min, W_max = np.min(W), np.max(W)
        classes = ['plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
        for k in xrange(10):
            plt.subplot(2, 5, k + 1)

            # Rescale the weights to be between 0 and 255
            Wimg = (W[:, :, :, k].squeeze() - W_min) / (W_max - W_min) * 255.0
            plt.imshow(Wimg.astype('uint8'))
            plt.axis('off')
            plt.title(classes[k])
        plt.show()