from keras.layers import Conv1D, Conv2D, Dropout, Activation, Dense, GlobalAveragePooling2D, MaxPool1D, Input, Reshape
from keras.activations import relu
from keras.models import Model
from keras.backend import int_shape

def perception_net(input_dim, num_classes):
    # What is the max pool size?
    # Padding not specified
    input_tensor = Input(shape=input_dim)

    x = Conv1D(filters=48, 
               kernel_size=15,
               kernel_initializer='random_uniform',
               padding="same")(input_tensor)
    x = Activation('relu')(x)
    x = MaxPool1D(pool_size=2,
                  strides=2)(x)
    x = Dropout(rate=0.4)(x)

    x = Conv1D(filters=96,
               kernel_size=15,
               kernel_initializer='random_uniform',
               padding="same")(x)
    x = Activation('relu')(x)
    x = MaxPool1D(pool_size=2, strides=2)(x)
    x = Dropout(rate=0.4)(x)

    x = Reshape(target_shape=(int_shape(x)[1], int_shape(x)[2], 1))(x)

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
    model = perception_net(input_dim=(128, 6), num_classes=6)
    print(model.summary())

