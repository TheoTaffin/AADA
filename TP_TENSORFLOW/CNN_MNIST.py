# Convolutional neural network for handwritten digits recognition
import ctypes

cudart64_100 = ctypes.WinDLL("C:\\Program Files\\NVIDIA GPU Computing "
                       "Toolkit\\CUDA\\v10.0\\bin\\cudart64_100.dll")
cublas64_100 = ctypes.WinDLL("C:\\Program Files\\NVIDIA GPU Computing "
                       "Toolkit\\CUDA\\v10.0\\bin\\cublas64_100.dll")
cufft64_100 = ctypes.WinDLL("C:\\Program Files\\NVIDIA GPU Computing "
                             "Toolkit\\CUDA\\v10.0\\bin\\cufft64_100.dll")
curand64_100 = ctypes.WinDLL("C:\\Program Files\\NVIDIA GPU Computing "
                             "Toolkit\\CUDA\\v10.0\\bin\\curand64_100.dll")
cusolver64_100 = ctypes.WinDLL("C:\\Program Files\\NVIDIA GPU Computing "
                             "Toolkit\\CUDA\\v10.0\\bin\\cusolver64_100.dll")
cusparse64_100 = ctypes.WinDLL("C:\\Program Files\\NVIDIA GPU Computing "
                             "Toolkit\\CUDA\\v10.0\\bin\\cusparse64_100.dll")
cudnn64_7 = ctypes.WinDLL("C:\\Program Files\\NVIDIA GPU Computing "
                             "Toolkit\\CUDA\\v10.0\\bin\\cudnn64_7.dll")
# necessary on a local machine bcs of tensorflow version
########################### IGNORE PREVIOUS PART IF YOU ARE ON COLAB OR DON'T CARE ABOUT GPU USAGE
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.examples.tutorials.mnist import input_data

batch_size = 100
epochs = 5
learning_rate = 1e-3


# Loading mnist image and normalizing the values (pixels grayscaled 0 - 255)
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

mnist_train_images = mnist.train.images.reshape(-1, 28, 28, 1)/255
mnist_train_labels = mnist.train.labels
mnist_test_images = mnist.test.images.reshape(-1, 28, 28, 1)/255
mnist_test_labels = mnist.test.labels

# Tensorflow 1.15 does not easily support variable batch size. Since we define it before the
# beginning of this script, we'll set shape of the batch size directly in our placeholders
ph_input = tf.placeholder(shape=(batch_size, 28, 28, 1), dtype=tf.float32)
ph_output = tf.placeholder(shape=(batch_size, 10), dtype=tf.float32)

# network architecture :
"""
None is there actually equal to our batch size.
:conv2d_1: input_shape = (None, 28, 28 ,1), filter=16, kernel = (5, 5), padding = same, 
            output_shape = (None, 16, 28, 28, 1), activation='Relu'
:conv2d_2: input_shape = (None, 28, 28 , 16), filter=16, kernel = (5, 5), padding = same, 
            output_shape = (None, 28, 28, 16), activation='Relu'
:maxpool2d_1: input_shape = (None, 28, 28, 16), kernel=(2, 2), padding = same, strides = [1, 2, 2, 1]
              output_shape = (None, 14, 14, 16)

:conv2d_3: input_shape = (None, 14, 14 ,16), filter=32, filter_shape = (5, 5), padding = same, 
            output_shape = (None, 14, 14, 32), activation='Relu'
:conv2d_4: input_shape = (None, 14, 14 ,32), filter=32, filter_shape = (5, 5), padding = same, 
            output_shape = (None, 14, 14, 32), activation='Relu'
:maxpool2d_2: input_shape = (None, 14, 14, 32), kernel=(2, 2), padding = same, strides = [1, 2, 2, 1]
              output_shape = (None, 7, 7, 32)
:flatten_1: intput_shape = (None, 7, 7, 32), output_shape = (None, 1568)
:dense_1: input_shape = (None, 1568), output_shape = (None, 10)
"""


def compute_normalization(data_input):
    # computes the mean and variance of x across axes (here batch, height, width)
    mean, variance = tf.nn.moments(data_input, axes=[0, 1, 2])

    # we initialize scale and beta to respectively 1 and 0 so they don't influence the
    # normalization
    scale = tf.Variable(np.ones(shape=data_input.shape[-1], dtype=np.float32))
    beta = tf.Variable(np.zeros(shape=data_input.shape[-1], dtype=np.float32))

    # factor to induce variance (noise) to the normalization
    variance_epsilon = 1e-3
    normalization = tf.nn.batch_normalization(data_input, mean, variance, scale, beta, variance_epsilon)
    return normalization


def build_conv_layer(data_input, filters=16, kernel=(5, 5), padding='SAME'):
    # conv layer wrapper
    input_channel = int(data_input.shape[3])

    # This initializes the weights following a normal distribution. The shape is defines by :
    # kernel height, kernel width, input_channel, ouput_channel (which is actually equivalent in
    # our cases to the number of filter present in the previous layer, and the number of filter
    # of the conv2d layer that is being built
    filters_weights_init = tf.Variable(tf.truncated_normal(shape=(kernel[0],
                                                                  kernel[1],
                                                                  input_channel,
                                                                  filters)),
                                       dtype=tf.float32)
    # biases initializer
    filters_biases_init = tf.Variable(np.zeros(shape=filters), dtype=tf.float32)

    conv_layer = tf.nn.conv2d(input=data_input, filter=filters_weights_init, padding=padding)
    conv_layer = tf.nn.bias_add(conv_layer, filters_biases_init)
    conv_layer_norm = compute_normalization(conv_layer)
    activation = tf.nn.relu(conv_layer_norm)
    return activation


def build_maxpool_layer(data_input, strides=[1, 2, 2, 1], kernel=[2, 2],
                        padding='SAME'):
    # max pooling layer wrapper
    maxpool_layer = tf.nn.max_pool2d(input=data_input, strides=strides, ksize=kernel,
                                     padding=padding)
    return maxpool_layer


def build_output_layer(previous_layer, number_of_classes=10):
    layer_weights_output = tf.Variable(tf.truncated_normal(shape=(previous_layer.shape[1].value,
                                                                  number_of_classes)),
                                       dtype=tf.float32)
    layer_biases_output = tf.Variable(np.ones(10), dtype=tf.float32)
    layer_values = tf.matmul(previous_layer, layer_weights_output) + layer_biases_output
    return layer_values


conv2d_1 = build_conv_layer(ph_input, filters=16,  kernel=(5, 5))
conv2d_2 = build_conv_layer(conv2d_1, filters=16,  kernel=(5, 5))
maxpool2d_1 = build_maxpool_layer(conv2d_2)
conv2d_3 = build_conv_layer(maxpool2d_1, filters=32,  kernel=(5, 5))
conv2d_4 = build_conv_layer(conv2d_3, filters=32,  kernel=(5, 5))
maxpool2d_2 = build_maxpool_layer(conv2d_4)
# shape = (batch_size, 32, 7, 7)
flatten_1 = tf.reshape(maxpool2d_2, [maxpool2d_2.shape[0],
                                     maxpool2d_2.shape[1]*maxpool2d_2.shape[2]*maxpool2d_2.shape[3]])
output_layer_values = build_output_layer(flatten_1, number_of_classes=10)
output_layer = tf.nn.softmax(output_layer_values)

# loss, train and accuracy metrics declaration
loss = tf.nn.softmax_cross_entropy_with_logits_v2(labels=ph_output, logits=output_layer_values)
train = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)
accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(output_layer, 1), tf.argmax(ph_output, 1)),
                                  dtype=tf.float32))

# We now use tf.session to handle the training process :
with tf.Session() as s:

    # Variable initialization
    s.run(tf.global_variables_initializer())

    tab_acc_train = []
    tab_acc_test = []

    for training_id in range(epochs):
        print(f"Training ID : {training_id}")

        for batch in range(0, len(mnist_train_images), batch_size):
            # we pass the command "train" to the s.run method to train the model. We give the
            # placeholders previously defined and their input values (the images and labels to the
            # feed_dict argument
            s.run(train, feed_dict={
                ph_input: mnist_train_images[batch:batch+batch_size],
                ph_output: mnist_train_labels[batch:batch+batch_size]
            })

        # After each epochs, meaning after a full iteration over the mnist dataset, we make
        # predictions and use the accuracy command to do so. We can with that evaluate how the
        # model progresses after each iteration over the dataset
        tab_acc = []
        for batch in range(0,  len(mnist_train_images), batch_size):
            acc = s.run(accuracy, feed_dict={
                ph_input: mnist_train_images[batch:batch+batch_size],
                ph_output: mnist_train_labels[batch:batch+batch_size]
            })
            # We store those be able to track the progress
            tab_acc.append(acc)

        # Computation of the mean of the accuracy of the predicted values
        print(f"train accuracy: {np.mean(tab_acc)}")
        """
        In the original subject, we store 1-mean to represent the error percentage. I just find
        accuracy percentage more readable in a graph, so I swapped it
        """
        tab_acc_train.append(np.mean(tab_acc))

        # We now make the same evaluation but on the testing set. The idea is to compare results
        # between data that the model has never seen. This is a good metric for tracking
        # overfitting
        tab_acc = []
        for batch in range(0,  len(mnist_test_images), batch_size):
            acc = s.run(accuracy, feed_dict={
                ph_input: mnist_test_images[batch:batch+batch_size],
                ph_output: mnist_test_labels[batch:batch+batch_size]
            })
            # We store those be able to track the progress
            tab_acc.append(acc)

        # Computation of the mean of the accuracy of the predicted values
        print(f"test accuracy: {np.mean(tab_acc)}")
        tab_acc_test.append(np.mean(tab_acc))
        results = s.run(output_layer, feed_dict={ph_input: mnist_test_images[0:batch_size]})

plt.ylim(ymin=0, ymax=1)
plt.grid()
plt.plot(tab_acc_train, label="Train accuracy")
plt.plot(tab_acc_test, label="Test accuracy")
plt.legend(loc="upper right")
plt.show()

print(f"best train accuracy: {tab_acc_train[-1]}")
print(f"best test accuracy: {tab_acc_test[-1]}")

for image in range(batch_size//10):
    real_label = np.argmax(mnist_test_labels[image])
    predicted_label = np.argmax(results[image])
    plt.imshow(mnist_test_images[image])
    plt.title(f"real label:{real_label} , predicted label:{predicted_label}")
    plt.show()
