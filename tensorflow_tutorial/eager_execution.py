#%%
import tensorflow as tf

tf.enable_eager_execution()

#%%
print(tf.add(2, 3))
print(tf.add([1,2], [1,2]))
print(tf.square(2))
print(tf.reduce_sum([1,2,3]))
print(tf.encode_base64('Hi there'))

print(tf.square(1) + tf.square(1))

#%%
x = tf.matmul([[1]], [[2,3]])
print(x.shape)
print(x.dtype)

#%%
import numpy as np

ndarray = np.ones([3, 3])

tensor = tf.multiply(ndarray, 2)
print (tensor)

#%%
print (np.add(tensor, 1))


#%%
print (tensor.numpy())

#%%
x = tf.random_uniform([3, 3])

print (tf.test.is_gpu_available())

print (x.device)

#%%
import time

def time_matmul(x):
    start = time.time()
    for i in range (10):
        tf.matmul(x, x)
    result = time.time() - start
    print ("10 loops: {:0.2f}ms".format(result*1000))

print ("On CPU:")
with tf.device("CPU:0"):
    x = tf.random_uniform([1000, 1000])
    assert x.device.endswith("CPU:0")
    time_matmul(x)

print ("On GPU:")
if tf.test.is_gpu_available():
    with tf.device("GPU:0"):
        x = tf.random_uniform([1000, 1000])
        assert x.device.endswith("GPU:0")
        time_matmul(x)

#%%
ds_tensors = tf.data.Dataset.from_tensor_slices([1, 2, 3, 4, 5])

import tempfile
_, filename = tempfile.mkstemp()

with open(filename, 'w') as f:
    f.write("""Line 1
    Line 2
    Line 3
    """)

ds_file = tf.data.TextLineDataset(filename)

ds_tensors = ds_tensors.map(tf.square).shuffle(2).batch(2)
ds_file = ds_file.batch(2)

for x in ds_tensors:
    print(x)

for x in ds_file:
    print(x)

#%%
