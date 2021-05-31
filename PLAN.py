import tensorflow as tf
from keras import backend as K
import numpy as np
import pandas as pd
from keras.models import Model
from keras import activations
from keras.engine.topology import Layer
from keras.utils import conv_utils
from keras.layers import LSTM, InputLayer, Dense, Input, Flatten, concatenate, Reshape
from keras.layers.normalization import BatchNormalization
from keras.callbacks import EarlyStopping
from keras.optimizers import Adam
from keras import metrics

batch_size = 64
max_epoch = 200
seq_len = 7 # Both LSTM networks in Modules Aand B of PLAN’s architecture take the data of the past 7days as input.
static_factor_len = 4 # The data of four environmental factors are fed into module C
local_image_size = 7
cnn_hidden_dim_first = 16

sess = tf.Session()
K.set_session(sess)


class ModuleA_Seq_Conv(Layer):

    def __init__(self, output_dim, seq_len, kernel_size, activation=None,
                 kernel_initializer='glorot_uniform',
                 bias_initializer='zeros', padding='same', strides=(1, 1), **kwargs):
        super(ModuleA_Seq_Conv, self).__init__(**kwargs)
        self.output_dim = output_dim
        self.seq_len = seq_len
        self.bias_initializer = bias_initializer
        self.kernel_size = kernel_size
        self.kernel_initializer = kernel_initializer
        self.padding = padding
        self.strides = strides
        self.activation = activations.get(activation)

    def build(self, input_shape):
        batch_size = input_shape[0]
        self.kernel = []
        self.bias = []
        for eachlen in range(self.seq_len):
            self.kernel += [self.add_weight(shape=self.kernel_size,
                                            initializer=self.kernel_initializer,
                                            trainable=True, name='kernel_{0}'.format(eachlen))]

            self.bias += [self.add_weight(shape=(self.kernel_size[-1],),
                                          initializer=self.bias_initializer,
                                          trainable=True, name='bias_{0}'.format(eachlen))]
        self.build = True

    def call(self, inputs):
        output = []
        for eachlen in range(self.seq_len):

            tmp = K.bias_add(K.conv2d(inputs[:, eachlen, :, :, :], self.kernel[eachlen],
                                      strides=self.strides, padding=self.padding), self.bias[eachlen])

            if self.activation is not None:
                output += [self.activation(tmp)]

        output = tf.stack(output, axis=1)
        return output

    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[1], input_shape[2], input_shape[3], self.output_dim)


def build_model(train_LocustObservationData, train_MeteoData, train_TopoData, train_Label, validSet):

    ############ Module A ############
    moduleA_input = Input(shape=(seq_len, local_image_size, local_image_size, 2), name='moduleA_input') # Image representation of the locust survey data

    moduleA_cnn = ModuleA_Seq_Conv(output_dim=cnn_hidden_dim_first, seq_len=seq_len,
                             kernel_size=(3, 3, 2, cnn_hidden_dim_first), activation='sigmoid',
                             kernel_initializer='glorot_uniform', bias_initializer='zeros', padding='same',
                             strides=(1, 1))(moduleA_input)
    moduleA_cnn = BatchNormalization()(moduleA_cnn)
    moduleA_cnn = ModuleA_Seq_Conv(output_dim=cnn_hidden_dim_first, seq_len=seq_len,
                             kernel_size=(3, 3, cnn_hidden_dim_first, cnn_hidden_dim_first), activation='sigmoid',
                             kernel_initializer='glorot_uniform', bias_initializer='zeros', padding='same',
                             strides=(1, 1))(moduleA_cnn)
    moduleA_cnn = BatchNormalization()(moduleA_cnn)
    moduleA_cnn = ModuleA_Seq_Conv(output_dim=cnn_hidden_dim_first, seq_len=seq_len,
                             kernel_size=(3, 3, cnn_hidden_dim_first, cnn_hidden_dim_first), activation='sigmoid',
                             kernel_initializer='glorot_uniform', bias_initializer='zeros', padding='same',
                             strides=(1, 1))(moduleA_cnn)

    moduleA_cnn = Flatten()(moduleA_cnn)
    moduleA_cnn = Reshape(target_shape=(seq_len, -1))(moduleA_cnn)

    moduleA_lstm_input = Dense(units=64, activation='sigmoid')(moduleA_cnn)
    moduleA_lstm = LSTM(units=256, return_sequences=False, dropout=0)(moduleA_lstm_input)

    ############ Module B ############
    moduleB_input = Input(shape=(seq_len, 6), dtype='float32', name='moduleB_input') # six time-series data of length 7, each of which corresponds to the historical pattern of an environmental factor,
    moduleB_lstm_output = LSTM(units=64, return_sequences=False, dropout=0)(moduleB_input)
    
    concat_moduleAandB = concatenate([moduleA_lstm, moduleB_lstm_output], axis=-1)

    ############ Module C ############
    moduleC_input = Input(shape=(static_factor_len,), dtype='float32', name='moduleC_input') # vector of four elements which corresponds to values of four static variables
    moduleC_output = Dense(units=2, activation='sigmoid')(moduleC_input)

    concat_moduleAandBandC = concatenate([concat_moduleAandB, moduleC_output], axis=-1)
    #res = Dense(units=1, activation='sigmoid')(concat_moduleAandBandC)
    res = Dense(units=2, activation='softmax')(concat_moduleAandBandC) #3 Output layer

    model = Model(inputs=[moduleA_input, moduleB_input, moduleC_input], outputs=res)

    adam_opt = Adam(lr=0.001)
    #model.compile(loss='binary_crossentropy', optimizer=adam_opt, metrics=['accuracy'])
    model.compile(loss='categorical_crossentropy', optimizer=adam_opt, metrics=['accuracy'])

    earlyStopping = EarlyStopping(monitor='val_loss', patience=10, verbose=1, mode='min')
    model.fit([train_LocustObservationData, train_MeteoData, train_TopoData], train_Label, batch_size=batch_size, epochs=max_epoch,
              validation_data=validSet, callbacks=[earlyStopping])

    return model

#### The code is adapted from https://github.com/huaxiuyao/DMVST-Net ###
