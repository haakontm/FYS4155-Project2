import numpy as np

class NeuralNetwork:

    def __init__(self, 
                 features,
                 hidden_neurons,
                 output_layers,
                 activation_func='sigmoid',
                 output_activation='softmax', 
                 epochs=100, 
                 mini_batch_size=1,
                 eta=0.01,
                 eta_scaling='constant'):
        self.features = features
        self.hidden_neurons = hidden_neurons
        self.output_layers = output_layers
        self.epochs = epochs
        self.batch_size = mini_batch_size
        self.eta = eta
        self.eta_scaling = eta_scaling

        if activation_func == 'sigmoid':
            self.activation_func = lambda z : 1.0/(1.0+np.exp(-z))
        elif activation_func == 'relu':
            self.activation_func = lambda z : np.maximum(np.zeros(z.shape), z)
        elif activation_func == 'lrelu':
            pass

        if output_activation == 'none':
            self.output_activation = lambda z : z
        elif output_activation == 'sigmoid':
            self.output_activation = lambda z : 1.0/(1.0+np.exp(-z))
        elif output_activation == 'softmax':
            self.output_activation = lambda z : z / np.sum(z, axis=1, keepdims=True)
        elif output_activation == 'scaled':
            self.output_activation = lambda z : (z - self.mean) / self.std

        self.init_wb()


    def init_wb(self):
        self.hidden_biases = np.random.randn(self.hidden_neurons)
        self.hidden_weights = np.random.randn(self.features, self.hidden_neurons)

        self.output_biases = np.random.randn(self.output_layers)
        self.output_weights = np.random.randn(self.hidden_neurons, self.output_layers)

    
    def predict(self, X):
        z_hidden = X @ self.hidden_weights + self.hidden_biases
        self.a_hidden = self.activation_func(z_hidden)

        z_output = self.a_hidden @ self.output_weights + self.output_biases
        return self.output_activation(z_output)


    def fit(self, X, y):
        self.mean = np.mean(y)
        self.std = np.std(y)
        datapoints = X.shape[0]
        for epoch in range(self.epochs):
            perm = np.random.permutation(datapoints)
            X_shuffled = X[perm, :]
            y_shuffled = y[perm, :]
            for i in range(0, datapoints, self.batch_size):
                self.x_i = X_shuffled[i:i+self.batch_size, :]
                self.y_i = y_shuffled[i:i+self.batch_size, :]
                self.backprop()

    def backprop(self):
        out = self.predict(self.x_i)

        err_output = out - self.y_i
        err_hidden = (err_output @ self.output_weights.T) * self.a_hidden * (1 - self.a_hidden)

        output_w_grad = self.a_hidden.T @ err_output
        output_b_grad = np.sum(err_output, axis=0)

        hidden_w_grad = self.x_i.T @ err_hidden
        hidden_b_grad = np.sum(err_hidden, axis=0)

        self.output_weights -= self.eta * output_w_grad
        self.output_biases -= self.eta * output_b_grad
        self.hidden_weights -= self.eta * hidden_w_grad
        self.hidden_biases -= self.eta * hidden_b_grad
