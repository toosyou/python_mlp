import better_exceptions
import sklearn
from sklearn import datasets
from sklearn.preprocessing import OneHotEncoder
import pickle
import sys
sys.path.append('..')
import mlp
from mlp import normalized, split
import numpy as np

def number_layer_exp(X, y):
    # split to training and validation set
    train_x, train_y, valid_x, valid_y = split(X, y, 0.7)

    histories = list()
    for n_hidden_layers in range(1, 11, 2):
        mlp_clf = mlp.MLPClassifier(
                                    hidden_layer_sizes=128,
                                    n_hidden_layers=n_hidden_layers,
                                    activation='relu',
                                    solver='sgd',
                                    momentum=0.,
                                    nesterov=False,
                                    loss='cross_entropy',
                                    weight_initializer='orthogonal',
                                    bias_initializer='zeros',
                                    batch_size=32,
                                    learning_rate=0.01,
                                    lr_decay_on_plateau=0.5,
                                    lr_decay_patience=15,
                                    early_stop_patience=40
                )
        history = mlp_clf.fit(train_x, train_y, valid_set=(valid_x, valid_y), epochs=2000, verbose=True)
        print('n_hidden_layer:', n_hidden_layers)
        print('loss:', history['loss'][-1])
        print('acc:', history['acc'][-1])
        print('valid_loss:', history['valid_loss'][-1])
        print('valid_acc:', history['valid_acc'][-1])
        print('n_epoch:', len(history['loss']) )
        histories.append(history)

    pickle.dump(histories, open('n_hidden_layer_history.pl', 'wb'))

def activation_function_exp(X, y):
    # split to training and validation set
    train_x, train_y, valid_x, valid_y = split(X, y, 0.7)
    relu_clf = mlp.MLPClassifier(
                            hidden_layer_sizes=128,
                            n_hidden_layers=9,
                            activation='relu',
                            solver='sgd',
                            momentum=0.,
                            nesterov=False,
                            loss='cross_entropy',
                            weight_initializer='orthogonal',
                            bias_initializer='zeros',
                            batch_size=32,
                            learning_rate=0.01,
                            lr_decay_on_plateau=0.5,
                            lr_decay_patience=15,
                            early_stop_patience=40
                )
    leaky_relu_clf = mlp.MLPClassifier(
                                hidden_layer_sizes=128,
                                n_hidden_layers=9,
                                activation='leaky_relu',
                                solver='sgd',
                                momentum=0.,
                                nesterov=False,
                                loss='cross_entropy',
                                weight_initializer='orthogonal',
                                bias_initializer='zeros',
                                batch_size=32,
                                learning_rate=0.01,
                                lr_decay_on_plateau=0.5,
                                lr_decay_patience=15,
                                early_stop_patience=40
                    )
    selu_clf = mlp.MLPClassifier(
                                hidden_layer_sizes=128,
                                n_hidden_layers=9,
                                activation='selu',
                                solver='sgd',
                                momentum=0.,
                                nesterov=False,
                                loss='cross_entropy',
                                weight_initializer='orthogonal',
                                bias_initializer='zeros',
                                batch_size=32,
                                learning_rate=0.01,
                                lr_decay_on_plateau=0.5,
                                lr_decay_patience=15,
                                early_stop_patience=40
                    )
    print('ReLU Start!')
    relu_history = relu_clf.fit(train_x, train_y, valid_set=(valid_x, valid_y), epochs=2000, verbose=True)

    print('Leaky Start!')
    leaky_relu_history = leaky_relu_clf.fit(train_x, train_y, valid_set=(valid_x, valid_y), epochs=2000, verbose=True)

    print('SELU Start!')
    selu_history = selu_clf.fit(train_x, train_y, valid_set=(valid_x, valid_y), epochs=2000, verbose=True)

    pickle.dump(relu_history, open('relu_history.pl', 'wb'))
    pickle.dump(leaky_relu_history, open('leaky_relu_history.pl', 'wb'))
    pickle.dump(selu_history, open('selu_history.pl', 'wb'))

def momentum_exp(X, y):
    train_x, train_y, valid_x, valid_y = split(X, y, 0.7)

    no_momentum_clf = mlp.MLPClassifier(
                            hidden_layer_sizes=128,
                            n_hidden_layers=9,
                            activation='relu',
                            solver='sgd',
                            momentum=0.,
                            nesterov=False,
                            loss='cross_entropy',
                            weight_initializer='orthogonal',
                            bias_initializer='zeros',
                            batch_size=32,
                            learning_rate=0.01,
                            lr_decay_on_plateau=0.5,
                            lr_decay_patience=15,
                            early_stop_patience=40
                            )

    momentum_clf = mlp.MLPClassifier(
                            hidden_layer_sizes=128,
                            n_hidden_layers=9,
                            activation='relu',
                            solver='sgd',
                            momentum=0.4,
                            nesterov=False,
                            loss='cross_entropy',
                            weight_initializer='orthogonal',
                            bias_initializer='zeros',
                            batch_size=32,
                            learning_rate=0.01,
                            lr_decay_on_plateau=0.5,
                            lr_decay_patience=20,
                            early_stop_patience=60
                            )

    nesterov_clf = mlp.MLPClassifier(
                            hidden_layer_sizes=128,
                            n_hidden_layers=9,
                            activation='relu',
                            solver='sgd',
                            momentum=0.9,
                            nesterov=True,
                            loss='cross_entropy',
                            weight_initializer='orthogonal',
                            bias_initializer='zeros',
                            batch_size=32,
                            learning_rate=0.01,
                            lr_decay_on_plateau=0.5,
                            lr_decay_patience=20,
                            early_stop_patience=80
                            )
    print('no momentum start!')
    # no_momentum_history = no_momentum_clf.fit(train_x, train_y, valid_set=(valid_x, valid_y), epochs=2000, verbose=True)

    print('momentum start!')
    momentum_history = momentum_clf.fit(train_x, train_y, valid_set=(valid_x, valid_y), epochs=2000, verbose=True)

    print('nesterov start!')
    nesterov_history = nesterov_clf.fit(train_x, train_y, valid_set=(valid_x, valid_y), epochs=2000, verbose=True)

    # pickle.dump(no_momentum_history, open('no_momentum_history.pl', 'wb'))
    pickle.dump(momentum_history, open('momentum_history.pl', 'wb'))
    pickle.dump(nesterov_history, open('nesterov_history.pl', 'wb'))

def get_iris():
    iris = datasets.load_iris()
    X = normalized(iris.data)
    y = OneHotEncoder().fit_transform(iris.target.reshape(iris.target.shape[0], 1)).toarray()

    X, y = sklearn.utils.shuffle(X, y)
    return X, y

def get_data(filename='elliptic200'):
    data = []
    rep = {}
    if filename in ["cross200", "elliptic200"]:
        ATTR_LEN = 2
        fdata = []
        with open("./"+filename+".txt", "r") as f:
            for l in f:
                cl = [ x for x in l[:-1].split(" ") if x ]
                x = [ float(n) for n in cl[:-1] ]
                assert(len(x) == ATTR_LEN)
                y = cl[-1]
                if y not in rep.keys():
                    rep[y] = len(rep.keys())
                fdata.append([x, y])
        max_rep = len(rep.keys())
        for d in fdata:
            oh = np.zeros((1, max_rep))
            oh[0][rep[d[1]]] = 1
            d = [np.array([d[0]]).transpose(), oh.transpose(), d[1]]
            data.append(d)
        # return data, ATTR_LEN, rep

    # reshape
    X = list()
    y = list()
    for d in data:
        X.append(d[0])
        y.append(d[1])
    X = np.array(X)
    X = X.reshape(X.shape[0], X.shape[1])
    y = np.array(y)
    y = y.reshape(y.shape[0], y.shape[1])

    return X, y

if __name__ == '__main__':

    X, y = get_data(filename='cross200')
    print(X.shape, y.shape)

    train_x, train_y, valid_x, valid_y = split(X, y, 0.7)
    mlp_clf = mlp.MLPClassifier(
                                hidden_layer_sizes=32,
                                n_hidden_layers=8,
                                activation='selu',
                                solver='sgd',
                                momentum=0.4,
                                nesterov=False,
                                loss='cross_entropy',
                                weight_initializer='orthogonal',
                                bias_initializer='zeros',
                                batch_size=16,
                                learning_rate=0.01,
                                lr_decay_on_plateau=0.5,
                                lr_decay_patience=40,
                                early_stop_patience=100
                            )
    mlp_clf.fit(train_x, train_y, valid_set=(valid_x, valid_y), epochs=2000, verbose=True)

    predict_y = mlp_clf.predict(X).argmax(axis=1)
    pickle.dump(X, open('cross200_x.pl', 'wb'))
    pickle.dump(predict_y, open('cross200_predict.pl', 'wb'))
    pickle.dump(y, open('cross200_y.pl', 'wb'))
