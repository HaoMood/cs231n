import os
import numpy as np
import cPickle as pickle
import matplotlib.pyplot as plt

def load_batch(filename):
    '''
    load single batch of cifar-10
    '''
    with open(filename, 'rb') as f:
        datadict = pickle.load(f)
        X = datadict['data']
        y = datadict['labels']
        X = X.reshape(10000, 3, 32, 32).transpose(0, 2, 3, 1).astype('float')
        y = np.array(y)
        return X, y

def load(root_path, debug = True):
    '''
    load cifar-10 dataset
    '''
    xs = []
    ys = []
    for b in xrange(1, 6):
        file = os.path.join(root_path, 'data_batch_%d' % (b, ))
        X, y = load_batch(file)
        xs.append(X)
        ys.append(y)
    X = np.concatenate(xs)
    y = np.concatenate(ys)
    file = os.path.join(root_path, 'test_batch')
    X_test, y_test = load_batch(file)
    
    if debug:
        # As a sanity check, we print out the size of the training and test data.
        print 'Cifar-10 dataset has been loaded'
        print 'X shape', X.shape
        print 'y shape', y.shape
        print 'X_test shape', X_test.shape
        print 'y_test shape', y_test.shape

    return X, y, X_test, y_test

def show(X, y, sample_per_class = 7):
    '''
    show a few examples of training images from each class of cifar-10
    '''
    classes = ['plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
    K = len(classes)
    for k, class_name in enumerate(classes):
        idx = np.flatnonzero(y == k)
        idx = np.random.choice(idx, sample_per_class, replace = False)
        for row_cnt, sample_idx in enumerate(idx):
            plt_idx = row_cnt * K + k + 1
            plt.subplot(sample_per_class, K, plt_idx)
            plt.imshow(X[sample_idx].astype('uint8'))
            plt.axis('off')
            if row_cnt == 0:
                plt.title(class_name)
    plt.show()

def subsample_vec(X, y, X_test, y_test, m, m_test, debug = True):
    X = X[range(0, m)]
    y = y[range(0, m)]

    X_test = X_test[range(0, m_test)]
    y_test = y_test[range(0, m_test)]

    # Reshape the image data into rows
    X = X.reshape((m, -1))
    X_test = X_test.reshape((m_test, -1))
    assert X.shape[1] == X_test.shape[1], 'train and test data must have same shape'
    if debug:
        print 'Cifar-10 data has been vectorized'
        print 'X shape', X.shape
        print 'y shape', y.shape
        print 'X_test shape', X_test.shape
        print 'y_test shape', y_test.shape
    return X, y, X_test, y_test
    