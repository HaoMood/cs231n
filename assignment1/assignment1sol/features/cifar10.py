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
    
def split_vec(X, y, X_test, y_test, m, m_test, m_val, m_dev, debug = True, show_img = True):
    '''
    Split the data into train, val, and test sets. In addition we will
    create a small development set as a subset of the training data;
    we can use this for development so our code runs faster.
    '''
    # Our validation set will be m_val points from the original training set
    X_val = X[range(m, m + m_val)]
    y_val = y[range(m, m + m_val)]

    # Our training set will be the first m points from the original training set.
    X = X[range(m)]
    y = y[range(m)]

    # We will also make a development set, which is a small subset of the training set.
    idx = np.random.choice(m, m_dev, replace = False)
    X_dev = X[idx]
    y_dev = y[idx]

    # We use the first num_test points of the original test set as our test set.
    X_test = X_test[range(m_test)]
    y_test = y_test[range(m_test)]

    if debug:
        print 'Data has been splited.'
        print 'X shape', X.shape
        print 'y shape', y.shape
        print 'X_val shape', X_val.shape
        print 'y_val shape', y_val.shape
        print 'X_test shape', X_test.shape
        print 'y_test shape', y_test.shape
        print 'X_dev shape', X_dev.shape
        print 'y_dev shape', y_dev.shape
        
    # return X, y, X_test, y_test, X_dev, y_dev, X_val, y_val    
    # Preprocessing: reshape the image data into rows
    X = np.reshape(X, (m, -1))
    X_val = np.reshape(X_val, (m_val, -1))
    X_test = np.reshape(X_test, (m_test, -1))
    X_dev = np.reshape(X_dev, (m_dev, -1))

    assert X.shape[1] == X_val.shape[1]
    assert X_val.shape[1] == X_test.shape[1]
    assert X_test.shape[1] == X_dev.shape[1]

    if debug:
        print 'Data has been reshaped.'
        print 'X shape', X.shape
        print 'X_val shape', X_val.shape
        print 'X_test shape', X_test.shape
        print 'X_dev shape', X_dev.shape

    # Preprocessing: subtract the mean image
    # first: compute the image mean based on the training data
    img_mean = np.mean(X, axis = 0)
    
    if show_img:
        plt.figure(figsize = (4, 4))
        # visualize the mean image
        plt.imshow(img_mean.reshape((32, 32, 3)).astype('uint8'))
        plt.show()

    # second: subtract the mean image from train and test data
    X -= img_mean
    X_val -= img_mean
    X_test -= img_mean
    X_dev -= img_mean

    return X, y, X_test, y_test, X_dev, y_dev, X_val, y_val    

def load_raw(root_path, m_spec, debug = True):
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

    m, m_val, m_dev, m_test = m_spec
    X_val = X[range(m, m + m_val)]
    y_val = y[range(m, m + m_val)]

    # Our training set will be the first m points from the original training set.
    X = X[range(m)]
    y = y[range(m)]

    # We will also make a development set, which is a small subset of the training set.
    idx = np.random.choice(m, m_dev, replace = False)
    X_dev = X[idx]
    y_dev = y[idx]

    # We use the first num_test points of the original test set as our test set.
    X_test = X_test[range(m_test)]
    y_test = y_test[range(m_test)]

    if debug:
        print 'Data has been splited.'
        print 'X shape', X.shape
        print 'y shape', y.shape
        print 'X_val shape', X_val.shape
        print 'y_val shape', y_val.shape
        print 'X_test shape', X_test.shape
        print 'y_test shape', y_test.shape
        print 'X_dev shape', X_dev.shape
        print 'y_dev shape', y_dev.shape

    data = (X, y, X_test, y_test, X_val, y_val, X_dev, y_dev)
    return data