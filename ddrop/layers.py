from keras.layers import Dense
from keras.wrappers import Wrapper
import keras.backend as K


class DropConnectDense(Dense):
    def __init__(self, *args, **kwargs):
        self.prob = kwargs.pop('prob', 0.5)
        if 0. < self.prob < 1.:
            self.uses_learning_phase = True
        super(DropConnectDense, self).__init__(*args, **kwargs)

    def call(self, x, mask=None):
        if 0. < self.prob < 1.:
            self.W = K.in_train_phase(K.dropout(self.W, self.prob), self.W)
            self.b = K.in_train_phase(K.dropout(self.b, self.prob), self.b)

        # Same as original
        output = K.dot(x, self.W)
        if self.bias:
            output += self.b
        return self.activation(output)


class DropConnect(Wrapper):
    def __init__(self, layer, prob=1., **kwargs):
        self.prob = prob
        self.layer = layer
        if 0. < self.prob < 1.:
            self.uses_learning_phase = True
        super(DropConnect, self).__init__(**kwargs)

    def call(self, x, mask=None):
        if 0. < self.prob < 1.:
            self.W = K.in_train_phase(K.dropout(self.W, self.prob), self.W)
            self.b = K.in_train_phase(K.dropout(self.b, self.prob), self.b)
        return self.layer.call(x, mask=mask)
