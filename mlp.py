import numpy as np
from numba import jit
import os
import sys
import sklearn
from sklearn import datasets
from sklearn.preprocessing import OneHotEncoder

def normalized(a, axis=-1, order=2):
    l2 = np.atleast_1d(np.linalg.norm(a, order, axis))
    l2[l2==0] = 1
    return a / np.expand_dims(l2, axis)

class Activation:
    def __init__(self):
        pass

    @staticmethod
    def linear(x):
        return x

    @staticmethod
    def linear_dx(x):
        return np.vectorize(lambda _: 1.)(x)

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

    # cross_entropy_dx simply doesn't work alone

    @staticmethod
    def softmax_cross_entropy(x, t):
        return Loss.cross_entropy(Activation.softmax(x), t)

    @staticmethod
    def softmax_cross_entropy_dx(x, t):
        return Activation.softmax(x) - t

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

class Optimizer:
    def __init__(self): pass

    class sgd:
        def __init__(self, momentum=0., nesterov=False):
            self.last_updates_w = 0
            self.last_updates_b = 0
            self.last_grad_w = 0
            self.last_grad_b = 0
            self.momentum = momentum
            self.nesterov = nesterov

        def cal_updates(self, grad_w, grad_b, lr):
            if self.nesterov:
                updates_w = ( grad_w + self.momentum*(self.last_updates_w + grad_w - self.last_grad_w) ) * lr
                updates_b = ( grad_b + self.momentum*(self.last_updates_b + grad_b - self.last_grad_b) ) * lr
            else:
                updates_w = (grad_w + self.momentum * self.last_updates_w) * lr
                updates_b = (grad_b + self.momentum * self.last_updates_b) * lr

            # update last weight and bias
            self.last_grad_w = grad_w
            self.last_grad_b = grad_b
            self.last_updates_w = updates_w
            self.last_updates_b = updates_b

            return updates_w, updates_b

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
            activation='relu',
            solver='sgd',
            momentum=0.,
            nesterov=False,
            loss='cross_entropy',
            weight_initializer='orthogonal',
            bias_initializer='zeros',
            batch_size=16,
            learning_rate=0.01,
            learning_rate_decay=0.95
        ):
        loss_need_convert = ['cross_entropy']

        self.hidden_layer_sizes = hidden_layer_sizes
        self.batch_size = batch_size
        self.hidden_layers = None
        self.output_layer = None
        self.activation = activation
        self.weight_initializer = weight_initializer
        self.bias_initializer = bias_initializer
        self.learning_rate = learning_rate
        self.n_hidden_layers = n_hidden_layers
        self.learning_rate_decay = learning_rate_decay
        self.optimizer = getattr(Optimizer, solver.lower())(momentum=momentum, nesterov=nesterov)

        # softmax + cross_entropy -> softmax_cross_entropy
        self.output_layer_activation = 'linear' if loss.lower() in loss_need_convert else 'softmax'
        loss = 'softmax_' + loss.lower() if loss.lower() in loss_need_convert else loss
        self.loss = getattr(Loss, loss.lower())
        self.loss_dx = getattr(Loss, loss.lower()+'_dx')

        self.input_dim = -1
        self.output_dim = -1

    def __forward_pass(self, x):
        rtn = x
        for h in self.hidden_layers:
            rtn = h(rtn)
        return self.output_layer(rtn)

    def __backpropagation(self, predict_y, true_y):
        loss_dx = self.loss_dx(predict_y, true_y)
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

    def __accuracy_loss(self, X, y):
        ''' return total accuracy and average loss '''
        loss = 0.
        n_currect = 0
        X = np.array(X)
        predict_y = self.predict(X)
        for py, yi in zip(predict_y, y):
            if np.argmax(py) == np.argmax(yi): n_currect += 1
            loss += self.loss(py, yi)
        return n_currect / X.shape[0], loss / X.shape[0]

    def fit(self, X, y, valid_set=None, epochs=100, shuffle=True):
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
                                    activation=self.output_layer_activation,
                                    weight_initializer=self.weight_initializer,
                                    bias_initializer=self.bias_initializer
                                )

        for index_epoch in range(epochs):
            if shuffle: X, y = sklearn.utils.shuffle(X, y)
            for batch_x, batch_y in self.__batchlize(X, y):
                # SGD
                grad_w_sum = None
                grad_b_sum = None
                for xi, yi in zip(batch_x, batch_y):
                    predict_y = self.__forward_pass(xi)
                    grads_w, grads_b = self.__backpropagation(predict_y, yi)
                    grad_w_sum = grads_w if grad_w_sum is None else grad_w_sum + grads_w
                    grad_b_sum = grads_b if grad_b_sum is None else grad_b_sum + grads_b
                # calculate updates
                grad_w_avg = grad_w_sum / self.batch_size
                grad_b_avg = grad_b_sum / self.batch_size

                updates_w, updates_b = self.optimizer.cal_updates(grad_w_avg, grad_b_avg, self.learning_rate)
                self.__update(updates_w, updates_b)

            # update learning rate
            self.learning_rate *= self.learning_rate_decay

            # result
            train_acc, train_loss = self.__accuracy_loss(X[:self.batch_size], y[:self.batch_size])
            if valid_set is not None:
                valid_acc, valid_loss = self.__accuracy_loss(valid_set[0], valid_set[1])

            print('epochs: %d/%d lr: %.6f' % (index_epoch, epochs, self.learning_rate), end=' ')
            print('loss: %.4f acc: %.4f ' % (train_loss, train_acc), end='\n' if valid_set is None else '')
            if valid_set is not None:
                print('valid_loss: %.4f valid_acc: %.4f' % (valid_loss, valid_acc))

        return None

    def predict(self, X):
        rtn = list()
        for x in X:
            if self.output_layer_activation == 'linear':
                rtn.append(Activation.softmax(self.__forward_pass(x)))
        return np.array(rtn)

if __name__ == '__main__':
    iris = datasets.load_iris()
    X = normalized(iris.data)
    y = OneHotEncoder().fit_transform(iris.target.reshape(iris.target.shape[0], 1)).toarray()

    X, y = sklearn.utils.shuffle(X, y)

    # split to training and validation set
    train_ratio = 0.7
    train_x, train_y = X[:int(X.shape[0]*train_ratio)], y[:int(y.shape[0]*train_ratio)]
    valid_x, valid_y = X[int(X.shape[0]*train_ratio):], y[int(y.shape[0]*train_ratio):]

    mlp_clf = MLPClassifier(batch_size=min(16, y.shape[0]), momentum=0.4, nesterov=True)
    mlp_clf.fit(train_x, train_y, valid_set=(valid_x, valid_y), epochs=10000)
