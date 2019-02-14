#%%
import tensorflow as tf
tf.enable_eager_execution()

layer = tf.keras.layers.Dense(10, input_shape=(None, 5))

layer(tf.zeros([10, 5]))
layer.variables

#%%
layer.kernel, layer.bias

#%%
class MyDenseLayer(tf.keras.layers.Layer):
    def __init__(self, num_outputs):
        print ("before parent init")
        super(MyDenseLayer, self).__init__()
        self.num_outputs = num_outputs
        print ("after parent init")

    def build(self, input_shape):
        self.kernel = self.add_variable("kernel",
                                        shape=[int(input_shape[-1]),
                                               self.num_outputs])
        print ("inside build")

    def call(self, input):
        print ("inside call")
        return tf.matmul(input, self.kernel)

layer = MyDenseLayer(10)
print (layer(tf.zeros([10, 5])))
print (layer.trainable_variables)

#%%
class ResnetIdentityBlock(tf.keras.Model):
    def __init__(self, kernel_size, filters):
        super(ResnetIdentityBlock, self).__init__(name='')
        filters1, filters2, filters3 = filters

        self.conv2a = tf.keras.layers.Conv2D(filters1, (1, 1))
        self.bn2a = tf.keras.layers.BatchNormalization()

        self.conv2b = tf.keras.layers.Conv2D(filters2, kernel_size, padding='same')
        self.bn2b = tf.keras.layers.BatchNormalization()

        self.conv2c = tf.keras.layers.Conv2D(filters3, (1, 1))
        self.bn2c = tf.keras.layers.BatchNormalization()

    def call(self, input_tensor, training=False):
        x = self.conv2a(input_tensor)
        x = self.bn2a(x, training=training)
        x = tf.nn.relu(x)

        x = self.conv2b(x)
        x = self.bn2b(x, training=training)
        x = tf.nn.relu(x)

        x = self.conv2c(x)
        x = self.bn2c(x, training=training)

        x += input_tensor
        return tf.nn.relu(x)

block = ResnetIdentityBlock(1, [1, 2, 3])
print (block(tf.zeros([1, 2, 3, 3])))
print ([x.name for x in block.trainable_variables])

#%%
my_seq = tf.keras.Sequential([
    tf.keras.layers.Conv2D(1, (1, 1)),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Conv2D(2, 1, padding='same'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Conv2D(3, (1, 1)),
    tf.keras.layers.BatchNormalization()
])
my_seq(tf.zeros([1, 2, 3, 3]))

#%%
