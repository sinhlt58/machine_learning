#%%
import tensorflow as tf

tf.enable_eager_execution()

#%%
v = tf.Variable(2)

v.assign(tf.square(3))

print (tf.square(3))

#%%
class Model(tf.keras.layers.Layer):
    def __init__(self):
        super(Model, self).__init__()
        self.W = tf.Variable(5.0)
        self.b = tf.Variable(0.0)

    def build(self, input_shape):
        print ("sss")
        print (input_shape)
        

    def call(self, input):
        return self.W*input + self.b

model = Model()

print (model(3))

#%%
def loss(predicted_y, desired_y):
    return tf.reduce_mean(tf.square(predicted_y - desired_y))

#%%
TRUE_W = 3
TRUE_B = 2
NUM_SAMPLE = 1000

inputs = tf.random_normal(shape=[NUM_SAMPLE])
noise  = tf.random_normal(shape=[NUM_SAMPLE])
outputs = inputs * TRUE_W + TRUE_B + noise
#%%
import matplotlib.pyplot as plt

plt.scatter(inputs, outputs, c = 'b')
plt.scatter(inputs, model(inputs), c = 'r')
plt.show()

print ("Current loss")
print (loss(model(inputs), outputs).numpy())

#%%
def train(model, inputs, ouputs, learning_rate):
    with tf.GradientTape() as t:
        current_loss = loss(model(inputs), outputs)
    dW, db = t.gradient(current_loss, [model.W, model.b])
    model.W.assign_sub(learning_rate*dW)
    model.b.assign_sub(learning_rate*db)

#%%
model = Model()

Ws, bs = [], []
epochs = range(10)
for epoch in epochs:
    Ws.append(model.W.numpy())
    bs.append(model.b.numpy())
    current_loss = loss(model(inputs), outputs)

    train(model, inputs, outputs, learning_rate=0.1)
    print('Epoch %2d: W=%1.2f b=%1.2f, loss=%2.5f' %
        (epoch, Ws[-1], bs[-1], current_loss))

#%%
plt.plot(epochs, Ws, 'r',
         epochs, bs, 'b')
plt.plot([TRUE_W] * len(epochs), 'r--')
plt.plot([TRUE_B] * len(epochs), 'b--')
plt.legend(['W', 'b', 'true W', 'true b'])
plt.show()
#%%
