import numpy as np
from numba import jit
import os
import sys
import sklearn
from sklearn import datasets
from sklearn.preprocessing import OneHotEncoder
import autograd.numpy as grad_np
from autograd import elementwise_grad as egrad

def normalized(a, axis=-1, order=2):
    l2 = np.atleast_1d(np.linalg.norm(a, order, axis))
    l2[l2==0] = 1
    return a / np.expand_dims(l2, axis)

class Activation:
    def __init__(self):
        pass

    @staticmethod
    def relu(x):
        return np.vectorize(lambda xi: max(xi, 0.))(x)

    @staticmethod
    def relu_dx(x):
        return np.vectorize(lambda xi: 1. if xi > 0. else 0.)(x)

    @staticmethod
    def leaky_relu(x):
        return np.vectorize(lambda xi: xi if xi > 0. else 0.1*xi)(x)

    @staticmethod
    def leaky_relu_dx(x):
        return np.vectorize(lambda xi: 1. if xi > 0. else 0.1)(x)

    @staticmethod
    def softmax(x, epslion=1e-10):
        """Compute softmax values for each sets of scores in x."""
        x = np.array(x)
        e_x = np.exp(x)
        return e_x / (e_x.sum() + epslion)

    @staticmethod
    def softmax_dx(x, epslion=1e-10):
        ''' e^xi * (e^x0 + e^x1 + ... ) / (e^x0 + e^x1 + ...)^2 '''
        x = np.array(x)
        sum_e_x = np.exp(x).sum()
        return np.vectorize(lambda xi: np.exp(xi)*(sum_e_x - np.exp(xi)) / (sum_e_x*sum_e_x + epslion) )(x)

class Initializer:
    def __init__(self):
        pass

    @staticmethod
    def orthogonal(shape, gain=1.):
        num_rows = 1
        for dim in shape[:-1]:
            num_rows *= dim
        num_cols = shape[-1]
        flat_shape = (num_rows, num_cols)

        a = np.random.normal(0.0, 1.0, flat_shape)
        u, _, v = np.linalg.svd(a, full_matrices=False)
        # Pick the one with the correct shape.
        q = u if u.shape == flat_shape else v
        q = q.reshape(shape)
        return gain * q[:shape[0], :shape[1]]

    @staticmethod
    def zeros(shape):
        return np.zeros(shape)

    @staticmethod
    def constant(shape, value=0.):
        return np.full(shape, value)

class Loss:
    def __init__(self):
        pass

    @staticmethod
    def cross_entropy(x, t, epslion=1e-10):
        ''' return cross entropy loss of input-x and target-t'''
        x = np.array(x)
        t = np.array(t)
        return np.vectorize(lambda xi, ti: -( ti * np.log(xi+epslion) + (1.-ti) * np.log(1.-xi-epslion) ))(x, t).sum()

    @staticmethod
    def cross_entropy_dx(x, t, epslion=1e-10):
        ''' return the differential of cross_entropy by autograd since it's not differentialable when x=0 '''
        x = grad_np.array(x)
        t = grad_np.array(t)
        def ce(x):
            return -(t*grad_np.log(x) + (1.-t)*grad_np.log(1.-x))
        return egrad(ce)(x)

    @staticmethod
    def mse(x, t):
        x = np.array(x)
        t = np.array(t)
        return 0.5 * np.sqrt(np.square(x-t).sum())

    @staticmethod
    def mse_dx(x, t):
        x = np.array(x)
        t = np.array(t)
        return x-t

class MLPClassifier:
    class __fc_layer:
        def __init__(
                self,
                input_layer,
                n_unit=1024,
                activation='relu',
                weight_initializer='constant',
                bias_initializer='zeros'
            ):
            self.input_dim = input_layer if type(input_layer) is int else input_layer.n_unit
            self.n_unit = n_unit
            self.activation = getattr(Activation, activation.lower())
            self.activation_dx = getattr(Activation, activation.lower()+'_dx')
            self.weight_initializer = getattr(Initializer, weight_initializer.lower())
            self.bias_initializer = getattr(Initializer, bias_initializer.lower())
            self.w = self.weight_initializer(shape=(n_unit, self.input_dim))
            self.b = self.bias_initializer(shape=(n_unit))

            self.__cache = dict()

            self.next_layer = None
            if type(input_layer) is type(self):
                self.parent_layer = input_layer
                input_layer.next_layer = self
            else:
                self.parent_layer = None

        @jit
        def __call__(self, x):
            self.__cache['x'] = x
            self.__cache['wx'] = np.matmul(self.w, x) + self.b
            return self.activation(self.__cache['wx'])

        @jit
        def __grad(self):
            delta = self.__cache['delta'].reshape(self.__cache['delta'].shape[0], 1)
            x = self.__cache['x'].reshape(1, self.__cache['x'].shape[0])
            grad_w = np.matmul(delta, x)
            grad_b = self.__cache['delta']
            return grad_w, grad_b

        @jit
        def backpropagation(self, loss_dx_delta):
            if self.next_layer is None: # output_layer
                self.__cache['delta'] = loss_dx_delta * self.activation_dx( self.__cache['wx'] )
            else:
                self.__cache['delta'] = np.matmul(np.transpose(self.next_layer.w), loss_dx_delta) * self.activation_dx( self.__cache['wx'] )

            grad_w, grad_b = self.__grad()

            if self.parent_layer is not None: # not the first layer
                rtn_grad_w, rtn_grad_b = self.parent_layer.backpropagation(self.__cache['delta'])
                rtn_grad_w.append(grad_w)
                rtn_grad_b.append(grad_b)
            else:
                rtn_grad_w, rtn_grad_b = [grad_w], [grad_b]

            if self.next_layer is None: # output_layer
                rtn_grad_w = np.array(rtn_grad_w)
                rtn_grad_b = np.array(rtn_grad_b)
            return rtn_grad_w, rtn_grad_b

        @jit
        def update(self, updates_w, updates_b):
            self.w = self.w - updates_w[-1]
            self.b = self.b - updates_b[-1]

            if not(self.parent_layer is None): self.parent_layer.update(updates_w[:-1], updates_b[:-1])

    def __init__(
            self,
            hidden_layer_sizes=128,
            n_hidden_layers=10,
            activation='relu', solver='sgd',
            loss='cross_entropy',
            weight_initializer='orthogonal',
            bias_initializer='zeros',
            batch_size=16,
            learning_rate=0.01,
            learning_rate_decay=0.8
        ):
        self.hidden_layer_sizes = hidden_layer_sizes
        self.batch_size = batch_size
        self.hidden_layers = None
        self.output_layer = None
        self.activation = activation
        self.loss = getattr(Loss, loss.lower())
        self.loss_dx = getattr(Loss, loss.lower()+'_dx')
        self.weight_initializer = weight_initializer
        self.bias_initializer = bias_initializer
        self.learning_rate = learning_rate
        self.n_hidden_layers = n_hidden_layers
        self.learning_rate_decay = learning_rate_decay

        self.input_dim = -1
        self.output_dim = -1

    def __forward_pass(self, x):
        rtn = x
        for h in self.hidden_layers:
            rtn = h(rtn)
        return self.output_layer(rtn)

    def __loss(self, predict_y, true_y):
        return self.loss(predict_y, true_y)

    def __backpropagation(self, predict_y, true_y):
        loss_dx = self.loss_dx(predict_y, true_y)
        # print('bp', predict_y, true_y, loss_dx)
        return self.output_layer.backpropagation(loss_dx) # all gradient

    def __update(self, updates_w, updates_b):
        self.output_layer.update(updates_w, updates_b)

    def __batchlize(self, X, y):
        for index_batch in range(y.shape[0] // self.batch_size):
            index_start = self.batch_size * index_batch
            index_end = index_start + self.batch_size
            xi = X[index_start: index_end]
            yi = y[index_start: index_end]
            yield xi, yi

    def __add_hidden_layer(self):
        input_layer = self.hidden_layers[-1] if len(self.hidden_layers) > 0 else self.input_dim
        self.hidden_layers.append(self.__fc_layer(
                                    input_layer=input_layer,
                                    n_unit=self.hidden_layer_sizes,
                                    activation=self.activation,
                                    weight_initializer=self.weight_initializer,
                                    bias_initializer=self.bias_initializer
                                ))

    def fit(self, X, y, epochs=100, shuffle=True):
        X = np.array(X) # (?, input_dim)
        y = np.array(y) # (?, output_dim)

        self.input_dim = X.shape[1]
        self.output_dim = y.shape[1]

        self.hidden_layers = list()

        for i in range(self.n_hidden_layers):
            self.__add_hidden_layer()

        self.output_layer = self.__fc_layer(
                                    input_layer=self.hidden_layers[-1] if len(self.hidden_layers) else self.input_dim,
                                    n_unit=self.output_dim,
                                    activation='softmax',
                                    weight_initializer=self.weight_initializer,
                                    bias_initializer=self.bias_initializer
                                )

        for e in range(epochs):
            if shuffle: X, y = sklearn.utils.shuffle(X, y)
            for batch_x, batch_y in self.__batchlize(X, y):
                # SGD
                grad_w_sum = None
                grad_b_sum = None
                loss_sum = 0.
                for xi, yi in zip(batch_x, batch_y):
                    predict_y = self.__forward_pass(xi)
                    grads_w, grads_b = self.__backpropagation(predict_y, yi)
                    grad_w_sum = grads_w if grad_w_sum is None else grad_w_sum + grads_w
                    grad_b_sum = grads_b if grad_b_sum is None else grad_b_sum + grads_b
                    loss_sum += self.__loss(predict_y, yi)
                # calculate updates
                grad_w_avg = grad_w_sum / self.batch_size * self.learning_rate
                grad_b_avg = grad_b_sum / self.batch_size * self.learning_rate
                loss_avg = loss_sum / self.batch_size
                self.__update(grad_w_avg, grad_b_avg)

            # self.learning_rate *= self.learning_rate_decay
            # result
            print(e, self.__forward_pass(X[0]), y[0], loss_avg)

        return None

    def predict(X):
        pass

if __name__ == '__main__':
    iris = datasets.load_iris()
    X = normalized(iris.data)
    y = OneHotEncoder().fit_transform(iris.target.reshape(iris.target.shape[0], 1)).toarray()

    X, y = sklearn.utils.shuffle(X, y)

    # X = X[0:1]
    # y = y[0:1]

    mlp_clf = MLPClassifier(batch_size=min(16, y.shape[0]))
    mlp_clf.fit(X, y, epochs=10000)
