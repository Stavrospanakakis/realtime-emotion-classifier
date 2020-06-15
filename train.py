from helpers import Get_dataset, Show_plots
from keras.models import Sequential, load_model
from keras.layers import Conv2D, MaxPooling2D, Dense, Dropout, Flatten, Activation
from keras.optimizers import Adam

# get the dataset
X_train, X_test, y_train, y_test = Get_dataset()

epochs = 20

# create and train the model
model = Sequential()

model.add(Conv2D(64, (3, 3), input_shape=X_train.shape[1:]))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(64, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(64, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Dense(128))
model.add(Activation('relu'))
model.add(Dropout(0.5))

model.add(Flatten())
model.add(Dense(6))
model.add(Activation('softmax'))

optimizer = Adam(learning_rate=0.001)

model.compile(
    loss='categorical_crossentropy',
    optimizer=optimizer,
    metrics=['accuracy'],
)

history = model.fit(
    X_train,
    y_train,
    batch_size=128,
    epochs=epochs,
    validation_data = (X_test, y_test)
)

# save the model
model.save("./models/model.h5")

# show the plots
Show_plots(history, epochs)