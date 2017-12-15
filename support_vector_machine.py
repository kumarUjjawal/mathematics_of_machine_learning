import numpy as np
from matplotlib import pyplot as plt
#matplotlib inline

# Define data
# Input data
X = np.array([
    [-2,4,-1],
    [4,1,-1],
    [1, 6, -1],
    [2, 4, -1],
    [6, 2, -1],
])

# Output data
y = np.array([-1,-1,1,1,1])

def svm_sgd(X,Y):
    # weights
    w = np.zeros(len(X[0]))

    # Learning rate
    eta = 1

    # iterations to train for
    epochs = 100000

    # store miss-classifications
    errors = []

    # Training: Gradient descent
    for epoch in range(1,epochs):
        error = 0
        for i,x in enumerate(X):
            if (Y[i] * np.dot(X[i],w)) < 1:
                # miss-classified update for weights
                w = w + eta * ((X[i] * Y[i]) + (-2 * (1/epochs) * w))
                error = 1
            else:
                w = w + eta * (-2 * (1/epochs) * w)
        errors.append(error)

    # Plot the rate of classification errors during training
    plt.plot(errors,'|')
    plt.ylim(0.5,1.5)
    plt.axes().set_yticklabels([])
    plt.xlabel('Epochs')
    plt.ylabel('Missclassified')
    plt.show()

    return w

w = svm_sgd(X, y)
print(w)


