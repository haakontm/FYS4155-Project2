import numpy as np
from sklearn.linear_model import SGDRegressor

def SGD_linreg(X, y, n_epochs, batch_size, eta, eta_scaling='constant'):
    datapoints = X.shape[0]
    beta = np.random.randn(X.shape[1], 1)
    eta0 = eta
    for epoch in range(n_epochs):
        perm = np.random.permutation(datapoints)
        X_shuffled = X[perm, :]
        y_shuffled = y[perm, :]
        for i in range(0, datapoints, batch_size):
            x_i = X_shuffled[i:i+batch_size, :]
            y_i = y_shuffled[i:i+batch_size, :]
            if eta_scaling == 'optimal':
                t = epoch * (datapoints // batch_size) + i // batch_size
                eta = 1 / (t + (1 / eta0))
            # print("Current learning rate: ", eta)
            gradient = (2 / batch_size) * x_i.T @ ((x_i @ beta) - y_i)
            beta = beta - gradient * eta
    return beta

m = 100
x = 2*np.random.rand(m,1)
y = 4+3*x+np.random.randn(m,1)

X = np.c_[np.ones((m,1)), x]
theta_linreg = np.linalg.inv(X.T @ X) @ (X.T @ y)
print("Own inversion")
print(theta_linreg)
sgdreg = SGDRegressor()
sgdreg.fit(x,y.ravel())
print("sgdreg from scikit")
print(sgdreg.intercept_, sgdreg.coef_)

betas = SGD_linreg(X, y.reshape(-1, 1), 100, 10, 0.5, eta_scaling='optimal')
print(betas)
