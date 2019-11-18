from perception_net import perception_net
from prep_data import get_train_data, get_test_data
from keras.optimizers import Adadelta
from keras.callbacks import EarlyStopping, ModelCheckpoint
import pathlib
import os
import datetime as D

# PerceptionNet parameters (channels last format)
input_dim = (128, 6)
num_classes = 6

# Hyper parameters
batch_size = 64
epochs = 2
validation_split = 0.15
stopping_patience = 100
model_save_frequency = 1

# AdaDelta parameters
learning_rate = 1.0
rho = 0.95
epsilon = 1e-08

save_path = './run_data/'

data, labels = get_train_data()
test_data, test_labels = get_test_data()
model = perception_net(input_dim, num_classes)

session_path = os.path.join(save_path,  model.name + '_' + str(D.datetime.now()))
weight_path = os.path.join(session_path, 'weights.{epoch:02d}-{val_loss:.4f}.hdf5')
print(weight_path)
pathlib.Path(session_path).mkdir(parents=True, exist_ok=True)


optimizer = Adadelta(lr=learning_rate,
                     rho=rho,
                     epsilon=epsilon)
callbacks = [
    EarlyStopping(patience=stopping_patience),
    ModelCheckpoint(filepath=weight_path, period=model_save_frequency)
]

model.compile(loss="categorical_crossentropy",
              optimizer=optimizer,
              metrics=['accuracy'])

model.fit(data,
          labels,
          batch_size=batch_size,
          epochs=epochs,
          validation_split=validation_split,
          shuffle=True,
          callbacks=callbacks)

results = model.evaluate(test_data,
                         test_labels,
                         batch_size=batch_size)

print('test loss, test acc:', results)

model.save_weights(filepath=os.path.join(session_path, 'weights.FINAL-{0:.4f}.hdf5'.format(results[0])))