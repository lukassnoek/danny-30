from tensorflow.keras import Model
from tensorflow.keras.layers import Conv2DTranspose
from tensorflow.keras.layers import Input, Reshape, Dense, Conv2D
from tensorflow.keras.callbacks import Callback


def DannyNet(input_shape=(128,), output_shape=(128, 128)):
    """ Creates DannyNet instance. 
    
    Parameters
    ----------
    input_shape : tuple
        Shape of the input data (w/o batch dim)
    output_shape : tuple
        Shape of the output data (w/o batch dim)
        
    Returns
    -------
    model : tf.keras.Model
        DannyNet instance
    """

    code = Input(input_shape, name='input')
    # Some FC layers
    x = Dense(units=256, activation='relu')(code)
    x = Dense(units=256, activation='relu')(x)

    # Start upsampling
    x = Reshape((16, 16, 1))(x)
    x = Conv2DTranspose(filters=64, kernel_size=1, activation='relu')(x)
    x = Conv2DTranspose(filters=64, kernel_size=1, strides=2, activation='relu', padding='same')(x)
    x = Conv2DTranspose(filters=128, kernel_size=3, strides=2, activation='relu', padding='same')(x)
    x = Conv2DTranspose(filters=128, kernel_size=3, strides=2, activation='relu', padding='same')(x)
    
    # Some regular conv layers (keep increasing RF)
    x = Conv2D(filters=256, kernel_size=5, strides=1, activation='relu', padding='same')(x)
    x = Conv2D(filters=256, kernel_size=5, strides=1, activation='relu', padding='same')(x)
    x = Conv2D(filters=512, kernel_size=7, strides=1, activation='relu', padding='same')(x)
    x = Conv2D(filters=512, kernel_size=7, strides=1, activation='relu', padding='same')(x)

    # Reduce channels
    x = Conv2D(filters=1, kernel_size=1, strides=1, activation='relu', padding='same')(x)
    x = Reshape(output_shape)(x)

    model = Model(inputs=code, outputs=x, name='DannyNet')
    return model


class EarlyStoppingByLossVal(Callback):
    def __init__(self, monitor='loss', value=0.01):
        super(Callback, self).__init__()
        self.monitor = monitor
        self.value = value

    def on_epoch_end(self, epoch, logs={}):
        current = logs.get(self.monitor)

        if current < self.value:
            self.model.stop_training = True