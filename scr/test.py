import numpy as np
from sklearn.linear_model import SGDRegressor

def SGD_linreg(X, y, n_epochs, batch_size, learning_rate):
    beta = np.random.randn(X.shape[1])
    n_batches = len(y) // batch_size
    for epoch in range(n_epochs):
        for i in range(n_batches):
            batch_idx = np.random.randint(n_batches * batch_size - batch_size)
            x_i = X[batch_idx:batch_idx + batch_size]
            y_i = y[batch_idx:batch_idx + batch_size]
            grad = 2 * x_i.T @ ((x_i @ beta) - y_i)
            step_length = 1 / (epoch * n_batches + i + 1 / learning_rate)
            beta = beta - step_length * grad
    return beta

m = 100
x = 2*np.random.rand(m,1)
y = 4+3*x+np.random.randn(m,1)

X = np.c_[np.ones((m,1)), x]
theta_linreg = np.linalg.inv(X.T @ X) @ (X.T @ y)
print("Own inversion")
print(theta_linreg)
sgdreg = SGDRegressor(max_iter = 100, eta0=0.1)
sgdreg.fit(x,y.ravel())
print("sgdreg from scikit")
print(sgdreg.intercept_, sgdreg.coef_)

betas = SGD_linreg(X, y.ravel(), 100, 3, 0.1)
print(betas)
