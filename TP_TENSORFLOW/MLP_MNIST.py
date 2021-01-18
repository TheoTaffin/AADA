import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import cv2

from tensorflow.examples.tutorials.mnist import input_data

# Loading mnist image and normalizing the values (pixels grayscaled 0 - 255)
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
mnist_train_images = mnist.train.images/255
mnist_train_labels = mnist.train.labels
mnist_test_images = mnist.test.images/255
mnist_test_labels = mnist.test.labels


# visualizing a bunch of images
nb_img = 10
data = mnist.train.next_batch(nb_img)

images = data[0]
labels = data[1]

# import matplotlib for visualization
for index, image in enumerate(images):
    print("Label:", labels[index])
    plt.imshow(image.reshape(28, 28), cmap='gray')
    plt.title(f"Digit in the image {np.argmax(labels[index])}")
    plt.show()

# Neural network implementation
# the first placeholder represents the input of the network, and the seconds the output
# The shape of the first is 784 bcs we have images of 28 by 28 pixels, which once flattened
# gives a 784 vector. The shape of the second is 10 bcs we have 10 possible class
ph_input = tf.placeholder(shape=(None, 784), dtype=tf.float32)
ph_output = tf.placeholder(shape=(None, 10), dtype=tf.float32)

# training parameters
dense_layer_size = 100
"""
changed that to 1e-3 instead of 1e-4 to converge faster
"""
learning_rate = 1e-3
batch_size = 100
epochs = 200

### Disclaimer : I'm not exactly sure yet how to name the following variables. The idea is that
# they represent a layer of 100 neurons, and allow for the different computation needed to
# predict a the class of our images

# weight and biases initialization. We use tf.truncated_normal to initialize the weights with
# random values following a normal distribution
layer_weights_init = tf.Variable(tf.truncated_normal(shape=(784, dense_layer_size)),
                                 dtype=tf.float32)
layer_biases_init = tf.Variable(np.zeros(shape=dense_layer_size), dtype=tf.float32)

# we compute with the following variables the results of matmult (matrix multiplication) between
# the placeholder images and the weights, to which we add the biases. Then we use the sigmoid
# function to activate or not the neuron. intermediate bcs they represent the 100 neurons layer
# in the middle of our model
intermediate_layer_values = tf.matmul(ph_input, layer_weights_init) + layer_biases_init
intermediate_layer_state = tf.nn.sigmoid(intermediate_layer_values)

# Same but for the output
layer_weights_output = tf.Variable(tf.truncated_normal(shape=(dense_layer_size, 10)),
                                   dtype=tf.float32)
layer_biases_output = tf.Variable(np.zeros(10), dtype=tf.float32)
output_layer_values = tf.matmul(intermediate_layer_state, layer_weights_output) + \
                                layer_biases_output
output_layer = tf.nn.softmax(output_layer_values)

# For the training phase, we need to chose a loss function to optimize and an optimization
# method. Loss function will be cross entropy and the optimizer GradientDescentOptimizer
loss = tf.nn.softmax_cross_entropy_with_logits_v2(labels=ph_output, logits=output_layer_values)
train = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(loss)
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

# Doesn't work for me, need some adjustment (probably coming from imshow)
# np.set_printoptions(formatter={'float': '{:0.3f}'.format})
# for image in range(batch_size):
#     print("image", image)
#     print("network output:", results[image], np.argmax(results[image]))
#     print("expected output :", mnist_test_labels[image], np.argmax(mnist_test_labels[image]))
#     cv2.imshow('image', mnist_test_images[image].reshape(28, 28))
#     if cv2.waitKey()&0xFF == ord('q'):
#         break
