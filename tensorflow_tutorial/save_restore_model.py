#%%
from __future__ import absolute_import, print_function, division

import tensorflow as tf
from tensorflow import keras

print (tf.__version__)


#%%
(train_images, train_labels), (test_images, test_labels) = keras.datasets.mnist.load_data()

train_labels = train_labels[:1000]
test_labels = test_labels[:1000]

train_images = train_images[:1000].reshape(-1, 28*28) / 255.0
test_images  = test_images[:1000].reshape(-1, 28*28) / 255.0

#%%
def create_model():
    model = keras.Sequential([
        keras.layers.Dense(512, activation=tf.nn.relu, input_shape=(784, )),
        keras.layers.Dropout(0.2),
        keras.layers.Dense(10, activation=tf.keras.activations.softmax)
    ])

    model.compile(
        optimizer='adam',
        loss=tf.keras.losses.sparse_categorical_crossentropy,
        metrics=['accuracy']
    )

    return model

model = create_model()
model.summary()

#%%
import os
checkpoint_path = 'training_1/cp.ckpt'
checkpoint_dir = os.path.dirname(checkpoint_path)

cp_callback = tf.keras.callbacks.ModelCheckpoint(checkpoint_path,
                                                 save_weights_only=True,
                                                 verbose=1)

model = create_model()

model.fit(train_images, train_labels,
          epochs=10, batch_size=8, 
          validation_data=(test_images, test_labels),
          callbacks=[cp_callback])

#%%
model = create_model()

loss, acc = model.evaluate(test_images, test_labels)
print ("Untrained model, accuracy: {:5.2f}%".format(100*acc))

#%%
model.load_weights(checkpoint_path)

loss, acc = model.evaluate(test_images, test_labels)
print ("Trained model, accuracy: {:5.2f}%".format(100*acc))

#%%
checkpoint_path = "training_2/cp-{epoch:04d}.ckpt"
checkpoint_dir = os.path.dirname(checkpoint_path)

cp_callback = tf.keras.callbacks.ModelCheckpoint(checkpoint_path,
                                                 save_weights_only=True,
                                                 period=5,
                                                 verbose=1)

model = create_model()

model.fit(train_images, train_labels,
          epochs=50,
          validation_data=(test_images, test_labels),
          callbacks=[cp_callback])

#%%
latest = tf.train.latest_checkpoint(checkpoint_dir)
latest

#%%
model = create_model()
model.load_weights(latest)
loss, acc = model.evaluate(test_images, test_labels)
print ("Restored model, accuracy: {:5.2f}%".format(acc*100))

#%%
model.save_weights("./checkpoint/my_checkpoint")

model = create_model()
model.load_weights("./checkpoint/my_checkpoint")
loss, acc = model.evaluate(test_images, test_labels)
print ("Restored model, accuracy: {:5.2f}%".format(acc*100))

#%%
model = create_model()

model.fit(train_images, train_labels, epochs=5)

model.save('my_model.h5')

#%%
new_model = keras.models.load_model('my_model.h5')
new_model.summary()

#%%
loss, acc = new_model.evaluate(test_images, test_labels)
print ("Restored model, accuracy: {:5.2f}".format(acc*100))

#%%
model = create_model()

model.fit(train_images, train_labels, epochs=5)

saved_model_path = tf.contrib.saved_model.save_keras_model(model, "./saved_model")

#%%
new_model = tf.contrib.saved_model.load_keras_model(saved_model_path)
new_model

#%%
new_model.compile(
    optimizer='adam',
    loss=tf.keras.losses.sparse_categorical_crossentropy,
    metrics=['accuracy']
)

loss, acc = new_model.evaluate(test_images, test_labels)
print ("Restored model, accuracy: {:5.2f}".format(acc*100))

#%%
