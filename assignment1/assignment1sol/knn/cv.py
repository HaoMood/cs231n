import numpy as np

def cv(X, y, num_folds, Classifier, hpara):
    # Split up the training data into folds. After splitting, 
    # X_folds and y_folds should each be lists of length 
    # num_folds, where y_folds[i] is the label vector for the 
    # points in X_folds[i].     
    # Hint: Look up the numpy array_split function
    X_folds = np.array_split(X, num_folds)
    y_folds = np.array_split(y, num_folds)

    acc = []
    for folds in xrange(num_folds):
        X_train = np.concatenate(X_folds[: folds] + X_folds[folds + 1: ])
        y_train = np.concatenate(y_folds[: folds] + y_folds[folds + 1: ])
        X_val = X_folds[folds]
        y_val = y_folds[folds]

        model = Classifier()
        model.train(X_train, y_train)
        y_hat = model.predict(X_val, hpara = hpara, show_img = False)
        acc.append(np.mean(y_hat == y_val))
    return acc