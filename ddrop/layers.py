from keras.layers import Dense, Wrapper
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
        super(DropConnect, self).__init__(layer, **kwargs)
        if 0. < self.prob < 1.:
            self.uses_learning_phase = True

    def build(self, input_shape):
        if not self.layer.built:
            self.layer.build(input_shape)
            self.layer.built = True
        super(DropConnect, self).build()

    def get_output_shape_for(self, input_shape):
        return self.layer.get_output_shape_for(input_shape)

    def call(self, x, mask=None):
        if 0. < self.prob < 1.:
            self.layer.W = K.in_train_phase(K.dropout(self.layer.W, self.prob), self.layer.W)
            self.layer.b = K.in_train_phase(K.dropout(self.layer.b, self.prob), self.layer.b)
        return self.layer.call(x, mask=mask)
