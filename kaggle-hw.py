from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.optimizers import SGD
import pandas as pd

df = pd.read_csv("/Users/docam/data/kaggle-hw/train.csv")
train = df.values
X_train = train[:, 1:]
Y_train = train[:, 0]

model = Sequential()

model.add(Dense(output_dim=1, input_dim=784))
model.add(Activation("relu"))
# model.add(Dense(output_dim=1))
# model.add(Activation("softmax"))

model.compile(loss='categorical_crossentropy', optimizer='sgd', metrics=['accuracy'])
# model.compile(loss='categorical_crossentropy', optimizer=SGD(lr=0.01, momentum=0.9, nesterov=True))

model.fit(X_train, Y_train, nb_epoch=5, batch_size=32)