# Simple MLP Written in Python3
```python
import mlp
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
```
