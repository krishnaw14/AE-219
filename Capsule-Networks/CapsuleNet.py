import numpy as np 
import tensorflow as tf 

#Load the mnist dataset
from tensorflow.examples.tutorials.mnist import input_data

tf.reset_default_graph()

mnist = input_data.read_data_sets('tmp/data')

#Defining the parameters of convolution layers in the 1st two layer operations
conv1_params = {
    "filters": 256,
    "kernel_size": 9,
    "strides": 1,
    "padding": "valid",
    "activation": tf.nn.relu,
}

conv2_params = {
    "filters": 32*8, #(primaryCaps_n_blocks*primaryCaps_dim) 
    "kernel_size": 9,
    "strides": 2,
    "padding": "valid",
    "activation": tf.nn.relu
}


#Input layer for the images
X = tf.placeholder(shape=[None, 28,28,1], dtype=tf.float32, name="X")
#None is for batchsize that will be determined at training, 28x28x1 is image dimension

#First layer is a convolution layer
conv1 = tf.layers.conv2d(X,name='conv1', **conv1_params)
#Shape of conv1: (None, 20, 20,256)

#Primary Capsule Layer
#Primary Capsule layer consists of 32 capsules with each capsule a 6x6 units
primaryCaps_blocks = 32
primaryCaps_units = primaryCaps_blocks*6*6 
primaryCaps_dim = 8

#Another Convolution is applied and the output is reshaped for the primary capsule layer
conv2 = tf.layers.conv2d(conv1, name='conv2', **conv2_params)
#Shape of conv2: (None,6,6,256) 
#We want to split the last dimension 256 into 32 vectors of 8 dimensions which can be seen as 32 capsules with 6x6x8 units
#We can do the split by reshaping conv2 to (None,6,6,32,8)
# But as this capsule layer is fully connected to the next capsule layer, I am simply flatten the 6×6 grids
#So I am reshping conv2 to the shape(None,6x6x32,8)

#This capsule layer is fully connected to the next capsule layer,
primaryCaps_raw = tf.reshape(conv2, [-1, primaryCaps_units, primaryCaps_dim], name = "primaryCaps_raw")

#squash function scales down the vector norm between 0 and 1
#Implementation detail: add epsilon to the norm of s to avoid dividing by zero which will blow up training
def squash(s, axis=-1, epsilon=1e-7, name=None):
	with tf.name_scope(name, default_name = 'squash'):
		squared_norm = tf.reduce_sum(tf.square(s), axis=-1, keep_dims=True)
		safe_norm = tf.sqrt(squared_norm+epsilon)
		squash_factor = squared_norm/(1.+squared_norm)
		unit_vector = s/safe_norm
		return squash_factor*unit_vector

primaryCaps_output = squash(primaryCaps_raw, name = "primaryCaps_output")

#Digit Capsule Layer

digitCaps_dim = 16
digitCaps_blocks = 10

Wij = tf.random_normal(shape = (1, primaryCaps_units, digitCaps_blocks, digitCaps_dim, primaryCaps_dim))
W = tf.Variable(Wij, name="W")

batch_size = tf.shape(X)[0]
W_tiled = tf.tile(W, [batch_size,1,1,1,1], name="W_tiled")

primaryCaps_output_expanded = tf.expand_dims(primaryCaps_output, -1, name = "primaryCaps_output_expanded")
primaryCaps_output_tile = tf.expand_dims(primaryCaps_output_expanded, 2, name="primaryCaps_output_tile")
primaryCaps_output_tiled = tf.tile(primaryCaps_output_tile, [1,1,digitCaps_blocks, 1,1], name = "primaryCaps_output_tiled")

digitCaps_predicted_vectors = tf.matmul(W_tiled, primaryCaps_output_tiled, name = "digitCaps_predicted_vectors")

#Dynamic Routing

raw_weights = tf.zeros([batch_size, primaryCaps_units, digitCaps_blocks,1,1])

routing_weights = tf.nn.softmax(raw_weights, axis=2, name = "routing_weights")

weighted_predictions = tf.multiply(routing_weights, digitCaps_predicted_vectors, name = "weighted_predictions")
weighted_sum = tf.reduce_sum(weighted_predictions, axis=1, keep_dims = True, name="weighted_sum")

digitCaps_output_1 = squash(weighted_sum, axis=-1, name="digitCaps_output_1")
digitCaps_output_1_tiled = tf.tile(digitCaps_output_1, [1, primaryCaps_units, 1,1,1], name="digitCaps_output_1_tiled")

agreement = tf.matmul(digitCaps_predicted_vectors, digitCaps_output_1_tiled, transpose_a=True, name="agreement")

raw_weights_2 = tf.add(raw_weights, agreement, name="raw_weights_2")


routing_weights_2 = tf.nn.softmax(raw_weights_2,
                                        dim=2,
                                        name="routing_weights_2")
weighted_predictions_2 = tf.multiply(routing_weights_2,
                                           digitCaps_predicted_vectors,
                                           name="weighted_predictions_2")
weighted_sum_2 = tf.reduce_sum(weighted_predictions_2,
                                     axis=1, keep_dims=True,
                                     name="weighted_sum_2")
digitCaps_output_2 = squash(weighted_sum_2,
                              axis=-2,
                              name="caps2_output_2")

digitCaps_output = digitCaps_output_2

def safe_norm(s, axis=-1, epsilon=1e-7, keep_dims=False, name=None):
    with tf.name_scope(name, default_name="safe_norm"):
        squared_norm = tf.reduce_sum(tf.square(s), axis=axis,
                                     keep_dims=keep_dims)
        return tf.sqrt(squared_norm + epsilon)

y_proba = safe_norm(digitCaps_output, axis=-2, name="y_proba")

y_proba_argmax = tf.argmax(y_proba, axis=2, name="y_proba")

y_pred = tf.squeeze(y_proba_argmax, axis=[1,2], name="y_pred")

y = tf.placeholder(shape=[None], dtype=tf.int64, name="y")

#Loss function
m_plus = 0.9
m_minus = 0.1
lambda_ = 0.5

T = tf.one_hot(y, depth = digitCaps_blocks, name = "T")

digitCaps_output_norm = safe_norm(digitCaps_output, axis=-2, keep_dims=True, name = "digitCaps_output_norm")
present_error_raw = tf.square(tf.maximum(0., m_plus - digitCaps_output_norm),
                              name="present_error_raw")
present_error = tf.reshape(present_error_raw, shape=(-1, 10),
                           name="present_error")

absent_error_raw = tf.square(tf.maximum(0., digitCaps_output_norm - m_minus),
                             name="absent_error_raw")
absent_error = tf.reshape(absent_error_raw, shape=(-1, 10),
                          name="absent_error")

L = tf.add(T*present_error, lambda_*(1-T)*absent_error, name="L")

margin_loss = tf.reduce_mean(tf.reduce_sum(L, axis=1), name="margin_loss")

#Reconstruction
mask_with_labels = tf.placeholder_with_default(False, shape=(),
                                               name="mask_with_labels")
reconstruction_targets = tf.cond(mask_with_labels, # condition
                                 lambda: y,        # if True
                                 lambda: y_pred,   # if False
                                 name="reconstruction_targets")

reconstruction_mask = tf.one_hot(reconstruction_targets,
                                 depth=digitCaps_blocks,
                                 name="reconstruction_mask")
reconstruction_mask_reshaped = tf.reshape(
    reconstruction_mask, [-1, 1, digitCaps_blocks, 1, 1],
    name="reconstruction_mask_reshaped")

digitCaps_output_masked = tf.multiply(
    digitCaps_output, reconstruction_mask_reshaped,
    name="caps2_output_masked")

decoder_input = tf.reshape(digitCaps_output_masked,
                           [-1, digitCaps_blocks * digitCaps_dim],
                           name="decoder_input")

#Decoder
n_hidden1 = 512
n_hidden2 = 1024
n_output = 28*28

with tf.name_scope("decoder"):
    hidden1 = tf.layers.dense(decoder_input, n_hidden1,
                              activation=tf.nn.relu,
                              name="hidden1")
    hidden2 = tf.layers.dense(hidden1, n_hidden2,
                              activation=tf.nn.relu,
                              name="hidden2")
    decoder_output = tf.layers.dense(hidden2, n_output,
                                     activation=tf.nn.sigmoid,
                                     name="decoder_output")

#Reconstrustion loss
X_flat = tf.reshape(X, [-1, n_output], name="X_flat")
squared_difference = tf.square(X_flat - decoder_output,
                               name="squared_difference")
reconstruction_loss = tf.reduce_mean(squared_difference,
                                    name="reconstruction_loss")

alpha = 0.0005
loss = tf.add(margin_loss, alpha * reconstruction_loss, name="loss")

correct = tf.equal(y, y_pred, name="correct")
accuracy = tf.reduce_mean(tf.cast(correct, tf.float32), name="accuracy")

optimizer = tf.train.AdamOptimizer()
training_op = optimizer.minimize(loss, name = 'training_op')

init = tf.global_variables_initializer()
saver = tf.train.Saver()

#Training
n_epochs = 10
batch_size = 50
restore_checkpoint = True

n_iterations_per_epoch = mnist.train.num_examples // batch_size
n_iterations_validation = mnist.validation.num_examples // batch_size
best_loss_val = np.infty
checkpoint_path = "./my_capsule_network"

with tf.Session() as sess:
    if restore_checkpoint and tf.train.checkpoint_exists(checkpoint_path):
        saver.restore(sess, checkpoint_path)
    else:
        init.run()

    for epoch in range(n_epochs):
        for iteration in range(1, n_iterations_per_epoch + 1):
            X_batch, y_batch = mnist.train.next_batch(batch_size)
            # Run the training operation and measure the loss:
            _, loss_train = sess.run(
                [training_op, loss],
                feed_dict={X: X_batch.reshape([-1, 28, 28, 1]),
                           y: y_batch,
                           mask_with_labels: True})
            print("\rIteration: {}/{} ({:.1f}%)  Loss: {:.5f}".format(
                      iteration, n_iterations_per_epoch,
                      iteration * 100 / n_iterations_per_epoch,
                      loss_train),
                  end="")

        # At the end of each epoch,
        # measure the validation loss and accuracy:
        loss_vals = []
        acc_vals = []
        for iteration in range(1, n_iterations_validation + 1):
            X_batch, y_batch = mnist.validation.next_batch(batch_size)
            loss_val, acc_val = sess.run(
                    [loss, accuracy],
                    feed_dict={X: X_batch.reshape([-1, 28, 28, 1]),
                               y: y_batch})
            loss_vals.append(loss_val)
            acc_vals.append(acc_val)
            print("\rEvaluating the model: {}/{} ({:.1f}%)".format(
                      iteration, n_iterations_validation,
                      iteration * 100 / n_iterations_validation),
                  end=" " * 10)
        loss_val = np.mean(loss_vals)
        acc_val = np.mean(acc_vals)
        print("\rEpoch: {}  Val accuracy: {:.4f}%  Loss: {:.6f}{}".format(
            epoch + 1, acc_val * 100, loss_val,
            " (improved)" if loss_val < best_loss_val else ""))

        # And save the model if it improved:
        if loss_val < best_loss_val:
            save_path = saver.save(sess, checkpoint_path)
            best_loss_val = loss_val
            
n_iterations_test = mnist.test.num_examples // batch_size

with tf.Session() as sess:
    saver.restore(sess, checkpoint_path)

    loss_tests = []
    acc_tests = []
    for iteration in range(1, n_iterations_test + 1):
        X_batch, y_batch = mnist.test.next_batch(batch_size)
        loss_test, acc_test = sess.run(
                [loss, accuracy],
                feed_dict={X: X_batch.reshape([-1, 28, 28, 1]),
                           y: y_batch})
        loss_tests.append(loss_test)
        acc_tests.append(acc_test)
        print("\rEvaluating the model: {}/{} ({:.1f}%)".format(
                  iteration, n_iterations_test,
                  iteration * 100 / n_iterations_test),
              end=" " * 10)
    loss_test = np.mean(loss_tests)
    acc_test = np.mean(acc_tests)
    print("\rFinal test accuracy: {:.4f}%  Loss: {:.6f}".format(
        acc_test * 100, loss_test))


