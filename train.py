from perception_net import perception_net
from prep_data import get_data
from keras.optimizers import Adadelta
from keras.callbacks import EarlyStopping

input_dim = (128, 6)
num_classes = 6

data, labels = get_data()
model = perception_net(input_dim, num_classes)
optimizer = Adadelta(lr=1.0,
                     rho=0.95,
                     epsilon=1e-08)

model.compile(loss="categorical_crossentropy",
              optimizer=optimizer,
              metrics=['accuracy'])

model.fit(data,
          labels,
          batch_size=64,
          epochs=2000,
          validation_split=0.15,
          shuffle=True,
          callbacks=[EarlyStopping(patience=100)])