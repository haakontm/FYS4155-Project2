import numpy as np

class NeuralNetwork:

    def __init__(self, 
                 layers, 
                 activation_func='sigmoid', 
                 epochs=100, 
                 mini_batch_size=1,
                 eta=0.01,
                 eta_scaling='constant'):
        self.layers = layers
        self.n_layers = len(layers)
        self.weights = [np.random.randn(y, x) for x, y in zip(layers[:-1], layers[1:])]
        self.biases = [np.random.randn(x, 1) for x in layers[1:]]
        self.epochs = epochs
        self.batch_size = mini_batch_size
        self.eta = eta
        self.eta_scaling = eta_scaling

        if activation_func == 'sigmoid':
            self.activation_func = lambda z : 1.0/(1.0+np.exp(-z))
            self.d_activation_func = lambda z : self.activation_func(z) * (1 - self.activation_func(z))

    
    def predict(self, X):
        "This function assumes that each column is an entry in the dataset(rows are features)"
        for b, w in zip(self.biases, self.weights):
            X = self.activation_func(w @ X - b)
        return X


    def fit(self, X, y):
        datapoints = X.shape[0]
        for epoch in range(self.epochs):
            perm = np.random.permutation(datapoints)
            X_shuffled = X[perm, :]
            y_shuffled = y[perm, :]
            for i in range(0, datapoints, self.batch_size):
                x_i = X_shuffled[i:i+self.batch_size, :]
                y_i = y_shuffled[i:i+self.batch_size, :]
                if self.eta_scaling == 'optimal':
                    t = epoch * (datapoints // self.batch_size) + i // self.batch_size
                    self.eta = 1 / (t + 10)
                self.step(x_i, y_i)


    def step(self, X, y):
        "Update the weights and biases in the network with computed gradients from a minibatch"
        nab_w = [np.zeros(w.shape) for w in self.weights]
        nab_b = [np.zeros(b.shape) for b in self.biases]
        for i in range(self.batch_size):
            delta_nab_b, delta_nab_w = self.backpropagation(X[i, :], y[i, :])
            nab_w = [nw + dnw for nw, dnw in zip(nab_w, delta_nab_w)]
            nab_b = [nb + dnb for nb, dnb in zip(nab_b, delta_nab_b)]
        self.weights = [w - (self.eta / self.batch_size) * nw for w, nw in zip(self.weights, nab_w)]
        self.biases = [b - (self.eta / self.batch_size) * nb for b, nb in zip(self.biases, nab_b)]

    def backpropagation(self, X, y):
        nab_w = [np.zeros(w.shape) for w in self.weights]
        nab_b = [np.zeros(b.shape) for b in self.biases]
        a = X
        a_values = [X]
        z_values = []

        for b, w, in zip(self.biases, self.weights):
            z = w @ a + b
            z_values.append(z)
            a = self.activation_func(z)
            a_values.append(a)
        
        delta = (a_values[-1] - y) * self.d_activation_func(z_values[-1])
        nab_w[-1] = delta @ a_values[-2].T
        nab_b[-1] = delta

        for l in range(2, self.n_layers):
            print(z_values[-l + 2].shape)
            z = z_values[-l]
            d_a = self.d_activation_func(z)
            delta = (self.weights[-l + 1].T @ delta) * d_a
            print(delta.shape, a_values[-l - 1].shape)
            nab_w[-l] = delta @ a_values[-l - 1].T
            nab_b[-l] = delta
        return (nab_b, nab_w)



def FrankeFunction(x,y):
    term1 = 0.75*np.exp(-(0.25*(9*x-2)**2) - 0.25*((9*y-2)**2))
    term2 = 0.75*np.exp(-((9*x+1)**2)/49.0 - 0.1*(9*y+1))
    term3 = 0.5*np.exp(-(9*x-7)**2/4.0 - 0.25*((9*y-3)**2)) 
    term4 = -0.2*np.exp(-(9*x-4)**2 - (9*y-7)**2)
    return term1 + term2 + term3 + term4

def create_X(x, y, n):
	if len(x.shape) > 1:
		x = np.ravel(x)
		y = np.ravel(y)

	N = len(x)
	l = int((n+1)*(n+2)/2)		# Number of elements in beta
	X = np.ones((N,l))

	for i in range(1,n+1):
		q = int((i)*(i+1)/2)
		for k in range(i+1):
			X[:,q+k] = (x**(i-k))*(y**k)

	return X


from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error

polygrad = 5
N = 50  # Number of points
noise = 0.05

x = np.linspace(0, 1, N)
y = np.linspace(0, 1, N)
x, y = np.meshgrid(x, y)

z = FrankeFunction(x, y)
z += noise * np.random.randn(z.shape[0], z.shape[1])

X = create_X(x, y, polygrad)

X_train, X_test, z_train, z_test = train_test_split(X, z.reshape(-1, 1), test_size=0.3)

network = NeuralNetwork([X_train.shape[1], 3, 1], mini_batch_size=15, eta=0.0001)
network.fit(X_train, z_train)
pred = network.predict(X_test.T)

#print(network.weights, network.biases)
print(r2_score(z_test, pred.ravel()))
