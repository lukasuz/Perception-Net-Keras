from keras.layers import Conv1D, Conv2D, Dropout, Activation, Dense, GlobalAveragePooling2D, MaxPool1D, Input, Reshape, DepthwiseConv2D, MaxPool2D
from keras.activations import relu
from keras.models import Model
from keras.backend import int_shape

def perception_net(input_dim, num_classes):
    # Padding not specified
    input_tensor = Input(shape=input_dim)
    x = Reshape(target_shape=(input_dim[0], input_dim[1], 1))(input_tensor)
    x = Conv2D(filters=48,
               kernel_size=(1,15), #height, width
               kernel_initializer='random_uniform',
               padding="same")(x)
    x = Activation('relu')(x)
    x = MaxPool2D(pool_size=(1,2),
                  strides=(1,2))(x)
    x = Dropout(rate=0.4)(x)

    x = Conv2D(filters=96,
               kernel_size=(1,15), #height, width
               kernel_initializer='random_uniform',
               padding="same")(x)
    x = Activation('relu')(x)
    x = MaxPool2D(pool_size=(1,2),
                  strides=(1,2))(x)
    x = Dropout(rate=0.4)(x)

    # Fusion 
    x = Conv2D(filters=96,
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
    model = perception_net(input_dim=(6, 128), num_classes=6)
    print(model.summary())

