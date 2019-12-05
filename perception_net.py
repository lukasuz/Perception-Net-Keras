from keras.layers import Conv1D, Conv2D, Dropout, Activation, Dense, GlobalAveragePooling2D, MaxPool1D, Input, Reshape, DepthwiseConv2D, MaxPool2D, SeparableConv2D
from keras.activations import relu
from keras.models import Model
from keras.backend import int_shape

# Padding not specified, assume padding 'same'

def perception_net(input_dim, num_classes, filters=(48, 96, 96), dilation=False, separate_modalities=False):
    """ PerceptionNet Model.
    See paper: https://arxiv.org/abs/1811.00170

    Arguments:
        input_dim: 2D input array, where the first value is the number of signals
            and the second the time steps.
        num_classes: Number of classes to predicts.
        filters: Optional, amount of feature maps employed by the convolutions.
        dilation: Optional, true whether dilation should be employed in the
            1D Conv blocks.
    """

    if dilation:
        kernel_width = 8
        dilation_rate = 2
    else:
        kernel_width = 15
        dilation_rate = 1

    input_tensor = Input(shape=input_dim)
    x = Reshape(target_shape=(input_dim[0], input_dim[1], 1))(input_tensor)

    # First "Conv1D" block
    x = Conv2D(filters=filters[0],
               kernel_size=(1,kernel_width), # height, width
               dilation_rate=dilation_rate,
               kernel_initializer='random_uniform',
               padding="same")(x)
    x = Activation('relu')(x)
    x = MaxPool2D(pool_size=(1,2),
                  strides=(1,2))(x)
    x = Dropout(rate=0.4)(x)

    # Second "Conv1D" block
    x = Conv2D(filters=filters[1],
               kernel_size=(1,kernel_width), #height, width
               dilation_rate=dilation_rate,
               kernel_initializer='random_uniform',
               padding="same")(x)
    x = Activation('relu')(x)
    x = MaxPool2D(pool_size=(1,2),
                  strides=(1,2))(x)
    x = Dropout(rate=0.4)(x)

    # Late sensor fusion
    x = Conv2D(filters=filters[2],
               kernel_size=(3,15),
               strides=(3,1),
               kernel_initializer='random_uniform',
               padding="same")(x)
    x = Activation('relu')(x)
    x = GlobalAveragePooling2D()(x)
    x = Dropout(rate=0.4)(x)

    # Classification
    x = Dense(units=num_classes, activation='softmax')(x)

    return Model(input_tensor, x, name='PerceptionNet')


if __name__ == "__main__":
    model = perception_net(input_dim=(6, 128), num_classes=6, filters=(12,24,24), dilation=False)
    print(model.summary())

