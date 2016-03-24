import numpy as np
import matplotlib.pyplot as plt
from math import sqrt, ceil
from gradient_check import grad_check_sparse

class NNet:
    '''
    A two-layer fully-connected neural network. The net has an input dimension of n0, a hidden layer dimension of n1, and performs classification over K = n2 classes.
    We train the network with a softmax loss function and L2 regularization on the
    weight matrices. The network uses a ReLU nonlinearity after the first fully
    connected layer.

    In other words, the network has the following architecture:

    input - fully connected layer - ReLU - fully connected layer - softmax

    The outputs of the second fully-connected layer are the scores for each class.
    '''
    def __init__(self, n0, n1, n2, std = 1e-4):
        '''
        Initialize the model. Weights are initialized to small random values and
        biases are initialized to zero. Weights and biases are stored in the
        variable self.weights, which is a dictionary with the following keys:

        W1: First layer weights; has shape (n1, n0)
        b1: First layer biases; has shape (n1,)
        W2: Second layer weights; has shape (n2, n1)
        b2: Second layer biases; has shape (n2,)

        Inputs:
        - input_size: The dimension n0 of the input data.
        - hidden_size: The number of neurons n1 in the hidden layer.
        - output_size: The number of classes n2.
        '''
        np.random.seed(1126)
        self.W = {}
        self.b = {}
        self.W[1] = std * np.random.randn(n1, n0)
        self.W[2] = std * np.random.randn(n2, n1)
        self.b[1] = np.zeros((n1))
        self.b[2] = np.zeros((n2))

    def _costFcn(self, X, y = None, lamda = 0.0):
        """
        Compute the loss and gradients for a two layer fully connected neural network.

        Inputs:
        - X: Input data of shape (m, n). Each X[i] is a training sample.
        - y: Vector of training labels. y[i] is the label for X[i], and each y[i] is an integer in the range 0 <= y[i] < n2. This parameter is optional; if it is not passed then we only return scores, and if it is passed then we instead return the loss and gradients.
        - lamda: Regularization strength.

        Returns:
        If y is None, return a matrix scores of shape (n2, m) where scores[k, i] is the score for class k on input X[i].

        If y is not None, instead return a tuple of:
        - J: Loss (data loss and regularization loss) for this batch of training
            samples.
        - grads: Dictionary mapping parameter names to gradients of those parameters with respect to the loss function; has the same keys as self.params.
        """
        # Unpack variables from the params dictionary
        W = {}
        b = {}
        W[1], W[2] = self.W[1], self.W[2]
        b[1], b[2] = self.b[1], self.b[2]
        m, n = X.shape
        J = None
        S = {}
        A = {}
        A[0] = X.T

        # TODO: Perform the forward pass, computing the class scores for the input. #
        S[1] = W[1].dot(A[0]) + np.reshape(b[1], (-1, 1))
        A[1] = np.maximum(0, S[1])
        S[2] = W[2].dot(A[1]) + np.reshape(b[2], (-1, 1))
        S[2] -= np.sum(S[2], axis = 0)
        A[2] = np.exp(S[2]) / np.sum(np.exp(S[2]), axis = 0)
        # If the targets are not given then jump out, we're done
        if y is None:
            return A[2]

        # Compute the loss
        # TODO: Finish the forward pass, and compute the loss. This should include  
        # both the data loss and L2 regularization for W1 and W2. Store the result  #
        # in the variable loss, which should be a scalar. Use the Softmax           #
        # classifier loss. So that your results match ours, multiply the            #
        # regularization loss by 0.5                                                #
        J = -np.sum(np.log(A[2][y, np.arange(m)])) / m + lamda * 0.5 * np.sum(W[1] ** 2) + lamda * 0.5 * np.sum(W[2] ** 2)

        # Backward pass: compute gradients
        # TODO: Compute the backward pass, computing the derivatives of the weights 
        # and biases. Store the results in the grads dictionary.
        dW = {}
        db = {}
        dA = {}
        dS = {}

        dS[2] = A[2]
        dS[2][y, np.arange(m)] -= 1
        dW[2] = dS[2].dot(A[1].T) / m
        db[2] = np.sum(dS[2], axis = 1) / m

        dA[1] = W[2].T.dot(dS[2]) / m
        dS[1] = dA[1] * (S[1] > 0)
        dW[1] = dS[1].dot(A[0].T)
        db[1] = np.sum(dS[1], axis = 1)

        dW[1] += lamda * W[1]
        dW[2] += lamda * W[2]
        
        assert dW[1].shape == W[1].shape
        assert dW[2].shape == W[2].shape
        assert db[1].shape == b[1].shape
        assert db[2].shape == b[2].shape
        return J, dW, db

    def train_check(self, X, y, lamda = 1e-3):    
        # Gradient check.
        # Numerically compute the gradient along several randomly 
        # chosen dimensions, and  compare them with your analytically 
        # computed gradient. The numbers should match almost exactly # along all dimensions.
        J, dW, db = self._costFcn(X, y, lamda)
        print 'J =', J, 'sanity check =', np.log(10)

        for l in xrange(1, 3):
            print '\n grad. check on W', l 
            f = lambda W: self._costFcn(X, y, lamda)[0]
            grad_numerical = grad_check_sparse(f, self.W[l], dW[l])

            print '\n grad. check on b', l 
            f = lambda b: self._costFcn(X, y, lamda)[0]
            grad_numerical = grad_check_sparse(f, self.b[l], db[l])


    def train(self, X, y, X_val, y_val, hpara = (1e-3, 1e-5, 100, 200, 0.95), debug = True, show_img = True):
        (alpha, lamda, T, B, rho) = hpara
        '''
        Train this linear classifier using stochastic gradient descent.

        Inputs:
            - X: A numpy array of shape (m, n) containing training data; there are m training samples each of dimension n.
            - y: A numpy array of shape (m,) containing training labels; y[i] = k means that X[i] has label 0 <= k < K for K classes.
            - alpha: (float) learning rate for optimization.
            - lamda: (float) regularization strength.
            - T: (integer) number of steps to take when optimizing
            - B: (integer) number of training examples to use at each step.
            - rho: Scalar giving factor used to decay the learning rate after each epoch.
            - debug: (boolean) If true, print progress during optimization.

        Outputs:
            A list containing the value of the loss function at each training iteration.
        '''
        # We now have the gradient and our gradient matches the numerical gradient. We are therefore ready to do SGD to minimize the loss.
        m, n = X.shape
        J_hist = []
        train_acc_hist = []
        val_acc_hist = []
        iterations_per_epoch = max(m / B, 1)
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
            J, dW, db = self._costFcn(X_batch, y_batch, lamda)
            J_hist.append(J)

            # perform parameter update       
            self.W[1] -= alpha * dW[1]            
            self.W[2] -= alpha * dW[2]            
            self.b[1] -= alpha * db[1]
            self.b[2] -= alpha * db[2]

            if debug and t % iterations_per_epoch == 0:
                # Every epoch, check train and val accuracy and decay learning rate
                print 'iteration %d / %d: loss %f' % (t, T, J)

                train_acc = np.mean(self.predict(X_batch) == y_batch)
                val_acc = np.mean(self.predict(X_val) == y_val)
                train_acc_hist.append(train_acc)
                val_acc_hist.append(val_acc)

                # Decay alpha
                alpha *= rho

        if show_img:
            # A useful debugging strategy is to plot the loss as a function of
            # iteration number:
            # Plot the loss function and train / validation accuracies
            plt.subplot(2, 1, 1)
            plt.plot(J_hist)
            plt.title('Loss history')
            plt.xlabel('t')
            plt.ylabel('J')

            plt.subplot(2, 1, 2)
            plt.plot(train_acc_hist, 'b')
            plt.plot(val_acc_hist, 'r')
            plt.title('Classification accuracy history')
            plt.xlabel('t')
            plt.ylabel('Clasification accuracy')
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
        h = self._costFcn(X)
        y = np.argmax(h, axis = 0)
        return y

    def visualize_W(self, padding = 3):
        W = self.W[1].reshape(32, 32, 3, -1).transpose(3, 0, 1, 2)
        (m, h, w, d) = W.shape
        grid_size = int(ceil(sqrt(m)))
        grid_H = h * grid_size + padding * (grid_size - 1)
        grid_W = w * grid_size + padding * (grid_size - 1)
        grid = np.zeros((grid_H, grid_W, d))

        next_idx = 0
        y0, y1 = 0, h
        for y in xrange(grid_size):
            x0, x1 = 0, w
            for x in xrange(grid_size):
                if next_idx < m:
                    img = W[next_idx]
                    low, high = np.min(img), np.max(img)
                    grid[y0: y1, x0: x1] = 255.0 * (img - low) / (high - low)
                    next_idx += 1
                x0 += w + padding
                x1 += w + padding
            y0 += h + padding
            y1 += h + padding

        plt.imshow(grid.astype('uint8'))
        plt.gca().axis('off')        
        plt.show()