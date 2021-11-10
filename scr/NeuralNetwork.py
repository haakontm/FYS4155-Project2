import numpy as np

class NeuralNetwork:

    def __init__(self, layers, activation_func='sigmoid'):
        self.layers = layers
        self.n_layers = len(layers)
        self.weights = [np.random.randn(y, x) for x, y in zip(layers[:-1], layers[1:])]
        self.biases = [np.random.randn(x, 1) for x in layers[1:]]

        if activation_func == 'sigmoid':
            self.activation_func = lambda z : 1.0/(1.0+np.exp(-z))

    
    def predict(self, X):    
        for b, w in zip(self.biases, self.weights):
            X = self.activation_func(w @ X - b)
            #print(X)
        return X





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

network = NeuralNetwork([X_train.shape[1], 5, 10, 1])
pred = network.predict(X_test.T)

#print(network.weights, network.biases)
print(r2_score(z_test, pred.ravel()))
