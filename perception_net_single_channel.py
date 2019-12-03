from keras.layers import Conv1D, Conv2D, Dropout, Activation, Dense, GlobalAveragePooling2D, MaxPool1D, Input, Reshape, concatenate, Lambda
from keras.activations import relu
from keras.models import Model
from keras.backend import int_shape

def perception_net(input_dim, num_classes, filters=(36, 72, 72), dilation=False):
    """ Single channel PerceptionNet model.
    Modificated version which uses separate convolutions for each channels.
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

    feature_maps = []
    for i in range(input_dim[0]):
        branch = Lambda(lambda x: x[:,i,:])(x) # Channel extraction has to wrapped in Lambda layer

        # First "Conv1D" block
        branch = Conv1D(filters=filters[0],
                        kernel_size=kernel_width, #height, width
                        kernel_initializer='random_uniform',
                        dilation_rate=dilation_rate,
                        padding="same")(branch)
        branch = Activation('relu')(branch)
        branch = MaxPool1D(pool_size=2,
                      strides=2)(branch)
        branch = Dropout(rate=0.4)(branch)

        # Second "Conv1D" block
        branch = Conv1D(filters=filters[1],
                        kernel_size=kernel_width, #height, width
                        kernel_initializer='random_uniform',
                        dilation_rate=dilation_rate,
                        padding="same")(branch)
        branch = Activation('relu')(branch)
        branch = MaxPool1D(pool_size=2,
                           strides=2)(branch)
        branch = Dropout(rate=0.4)(branch)
        branch = Reshape(target_shape=(1, int_shape(branch)[1], int_shape(branch)[2]))(branch)
        
        feature_maps.append(branch)

    x = Lambda(lambda x: concatenate(x, axis=1))(feature_maps)

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

    return Model(input_tensor, x, name='PerceptionNetSingleChannel')


if __name__ == "__main__":
    model = perception_net(input_dim=(6, 128), num_classes=6)
    print(model.summary())

