import scipy.signal as signal

import numpy as np              
import tensorflow as tf
from tensorflow.keras.datasets import mnist


if tf.config.experimental.list_physical_devices('GPU'):
    device = '/GPU:0'
    print(f"Using device: {device}")
else:
    device = '/CPU:0'
    print(f"Using device: {device}")

class Layer:
    def __init__(self):
        self.name = None
        self.output = None

    def forward(self,input1):

        pass
    def backward(self,output_gradient,learning_rate):

        pass

class DenseLayer(Layer):        

    def __init__(self,input_size,output_size):           # no of neuron in input and output layer is this
        super(DenseLayer, self).__init__()  
        self.weights = tf.Variable(tf.random.normal((input_size, output_size), dtype=tf.float32) * 0.01)
        self.biases = tf.Variable(tf.random.normal((output_size, ), dtype=tf.float32) * 0.01)


    def forward(self, input1):
        input1 = tf.convert_to_tensor(input1, dtype=tf.float32)

        # Ensure input is 2D: (batch_size, input_size)
        if len(input1.shape) == 1:
            input1 = tf.reshape(input1, (1, -1))  # Convert (input_size,) to (1, input_size)

        self.input1 = input1  # Store for backpropagation

        # Perform matrix multiplication and add bias (broadcasting applied automatically)
        return tf.matmul(self.input1, self.weights) + tf.reshape(self.biases, (1, -1))


    def backward(self, output_gradient, learning_rate):
        # Compute weight and input gradients using TensorFlow
        weights_gradient = tf.matmul(tf.transpose(self.input1), output_gradient)  # (input_size, output_size)
        input_gradient = tf.matmul(output_gradient, tf.transpose(self.weights))  # (batch_size, input_size)

        # Update weights and biases
        self.weights.assign_sub(learning_rate * weights_gradient)
        self.biases.assign_sub(learning_rate * tf.reduce_mean(output_gradient, axis=0))  # Keep shape consistent

        return input_gradient

class Convolutional(Layer): # very much counter-intuitively this 2 class thing works actually this clears all the dimension mismatch
    def __init__(self, input_shape, kernel_size, depth):     # almost all msimatch occur due to batch size = 32 and
        super().__init__()                                   # that TensorFlow requires that the input depth must be evenly divisible by filter depth. and here 32 becomes input depth and hence many issues comeup
        input_depth, input_height, input_width = input_shape

        self.depth = depth
        self.input_shape = input_shape
        self.output_shape = (depth, input_height - kernel_size + 1, input_width - kernel_size + 1)

        # Kernel shape: (height, width, input_channels, output_channels)
        self.kernels = tf.Variable(tf.random.normal((kernel_size, kernel_size, input_depth, depth), dtype=tf.float32))
        self.biases = tf.Variable(tf.zeros((depth,), dtype=tf.float32))  # Bias shape: (depth,)
        # print(f"Kernel shape: {self.kernels.shape}")

    def forward(self, input1):
        input1 = tf.convert_to_tensor(input1, dtype=tf.float32)
        # Ensure batch dimension exists
        if tf.rank(input1) == 3:  # (C, H, W)
            input1 = tf.expand_dims(input1, axis=0)  # (1, C, H, W)

        self.input1 = input1  # Store for backpropagation

        # Convert to NHWC format (batch, height, width, channels)
        input_nhwc = tf.transpose(self.input1, [0, 2, 3, 1])

        # Perform convolution
        self.output = tf.nn.conv2d(
            input=input_nhwc,       # here this should be input not from user but because tensorflow here want input that to be convolved
            filters=self.kernels,
            strides=1,
            padding="VALID"
        ) + tf.reshape(self.biases, (1, 1, 1, -1))  # Ensure correct broadcasting

        # Convert back to NCHW format (batch, channels, height, width)
        self.output = tf.transpose(self.output, [0, 3, 1, 2])

        return self.output


    class Convolutional(Layer):

        def __init__(self, input_shape, kernel_size, depth):
            super().__init__()
            input_depth, input_height, input_width = input_shape

            self.depth = depth
            self.input_shape = input_shape
            self.output_shape = (depth, input_height - kernel_size + 1, input_width - kernel_size + 1)

            # Kernel shape: (height, width, input_channels, output_channels)
            self.kernels = tf.Variable(tf.random.normal((kernel_size, kernel_size, input_depth, depth), dtype=tf.float32))
            self.biases = tf.Variable(tf.zeros((depth,), dtype=tf.float32))  # Bias shape: (depth,)

        def forward(self, input1):

            input1 = tf.convert_to_tensor(input1, dtype=tf.float32)

            # Ensure batch dimension exists
            if tf.rank(input1) == 3:  # (C, H, W)
                input1 = tf.expand_dims(input1, axis=0)  # (1, C, H, W)


            self.input1 = input1  # Store for backpropagation

            # Convert to NHWC format (batch, height, width, channels)
            input_nhwc = tf.transpose(self.input1, [0, 2, 3, 1])


            # Perform convolution
            self.output = tf.nn.conv2d(
                input=input_nhwc,
                filters=self.kernels,
                strides=[1, 1, 1, 1],
                padding="VALID"
            ) + self.biases  # Bias broadcasting

            # Convert back to NCHW format (batch, channels, height, width)
            self.output = tf.transpose(self.output, [0, 3, 1, 2])

            return self.output

        def backward(self, output_gradient, learning_rate):       

            # Transpose input and output gradient to NHWC format
            input_nhwc = tf.transpose(self.input1, perm=[0, 2, 3, 1])  # NCHW -> NHWC
            output_gradient_nhwc = tf.transpose(output_gradient, perm=[0, 2, 3, 1])  # NCHW -> NHWC

            # Transpose kernel for conv2d_transpose
            kernel_transposed = tf.transpose(self.kernels, perm=[0, 1, 3, 2])  # (3, 3, 1, 5) -> (3, 3, 5, 1)

            # Expected output shape in NHWC format
            expected_output_shape = tf.shape(input_nhwc)  # Should be (32, 28, 28, 1)

            # Compute input gradient
            input_gradient_nhwc = tf.nn.conv2d_transpose(
                output_gradient_nhwc,  # Output gradient in NHWC format
                filters=kernel_transposed,  # Transposed kernel in NHWC format
                output_shape=expected_output_shape,  # Shape of the input to the forward convolution
                strides=[1, 1, 1, 1],  # Strides used in the forward convolution
                padding='VALID'  # Padding used in the forward convolution
            )

            # Convert the result back to NCHW format
            input_gradient = tf.transpose(input_gradient_nhwc, perm=[0, 3, 1, 2])  # NHWC -> NCHW

            # Compute kernel gradient
            kernel_gradient = tf.nn.conv2d(
                input=input_nhwc,  # Input in NHWC format
                filters=output_gradient_nhwc,  # Output gradient in NHWC format
                strides=[1, 1, 1, 1],  # Strides used in the forward convolution
                padding="VALID"  # Padding used in the forward convolution
            )

            # Compute biases gradient
            biases_gradient = tf.reduce_mean(output_gradient_nhwc, axis=[0, 1, 2], keepdims=True)

            # Update weights and biases
            self.kernels.assign_sub(learning_rate * kernel_gradient)
            self.biases.assign_sub(learning_rate * biases_gradient)

            return input_gradient


class Reshape(Layer):
    def __init__(self, input_shape, output_shape):
        self.input_shape = input_shape
        self.output_shape = output_shape

    def forward(self, input1):
        self.input1 = input1  # Store for backward pass
        return tf.reshape(input1, (tf.shape(input1)[0], -1))  # Preserve batch size

    def backward(self, output_gradient, learning_rate):
        return tf.reshape(output_gradient, tf.shape(self.input1))  # Ensures correct reshaping


def categorical_cross_entropy(y_true, y_pred):
    return -tf.reduce_mean(tf.reduce_sum(y_true * tf.math.log(y_pred + 1e-9), axis=-1))


def categorical_cross_entropy_prime(y_true, y_pred):
    return y_pred - y_true  # Correct derivative



class ActivationLayer(Layer):          
    def __init__(self,activation,activation_prime):              # activation prime is derivative of activation
        super(ActivationLayer, self).__init__()                  # same reason as done in dense layer
        self.activation = activation
        self.activation_prime = activation_prime

    def forward(self,input1):
        self.input1 = input1
        return self.activation(self.input1)         # returning value of activation function applied on input

    def backward(self,output_gradient,learning_rate):

        return tf.multiply(output_gradient,self.activation_prime(self.input1))       # returning del_E/del_X = del_E/del_Y multiply f dash x


class Softmax(ActivationLayer):       # using softmax as we have 10 classes 0 1 2 ... 9 all the digits
        def __init__(self):
            def softmax(x):
                return tf.nn.softmax(x, axis=-1)

            def softmax_prime(x):
                s = softmax(x)
                return s * (1 - s)

            super(Softmax, self).__init__(softmax, softmax_prime)


def categorical_one_hot(y, num_classes=10):  # if using inuilt one hot do not write this def
    y_one_hot = tf.one_hot(y, depth=num_classes, dtype=tf.float32)  # Convert to one-hot
    return y_one_hot

def preprocess_data(x, y):
    x = tf.convert_to_tensor(x, dtype=tf.float32) / 255.0
    x = tf.reshape(x, (-1, 1, 28, 28))  # Keep CNN format (batch, channels, height, width)

    # One-hot encoding: (batch_size, 10)
    y = tf.one_hot(y, depth=10, dtype=tf.float32)
    return x, y


# Load dataset
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, y_train = preprocess_data(x_train, y_train)            # using utility from keras
x_test, y_test = preprocess_data(x_test,y_test)


#######################

network= [
    Convolutional((1,28,28),3,5),       # 5 kernel each 3*3 making of 1st layer
    ActivationLayer(tf.nn.relu, lambda x: tf.where(x > 0, tf.ones_like(x), tf.zeros_like(x))),  # ReLU activation
    Reshape((5,5,26),(5*26*26)),                         # reshaping into column vector
    DenseLayer(5*26*26,100),
    ActivationLayer(tf.nn.relu, lambda x: tf.where(x > 0, tf.ones_like(x), tf.zeros_like(x))),  # ReLU
    DenseLayer(100,10),         # now output will have 10 neurons
    Softmax()

    # no of nerurons  3380 + 100 + 10
]
epochs = 1000
learning_rate = 0.1

### train

for e in range(epochs):
    error = 0
    batch_size = 32
    for i in range(0, len(x_train), batch_size):
        batch_x = x_train[i:i + batch_size]
        batch_y = y_train[i:i + batch_size]

        # Ensure batch_x and batch_y always have batch dimension
        batch_x = tf.convert_to_tensor(batch_x, dtype=tf.float32)
        batch_y = tf.convert_to_tensor(batch_y, dtype=tf.float32)

        output = batch_x
        for layer in network:
            output = layer.forward(output)

        # Compute error
        error += tf.reduce_mean(categorical_cross_entropy(batch_y, output))

        grad = categorical_cross_entropy_prime(batch_y, output)
        for layer in reversed(network):
            grad = layer.backward(grad, learning_rate)

    error /= (len(x_train) // batch_size)
    print(f"{e + 1}/{epochs}, error={error}")

print("Training loop finished!")  # Debugging line

# Testing Loop
print("Starting accuracy test...")

correct = 0
total = len(x_test)

with tf.device('/GPU:0'):
    for i in range(total):
        x = x_test[i:i+1]  # Keep batch dimension
        y = y_test[i:i+1]  # Keep batch dimension

        output = x
        for layer in network:
            output = layer.forward(output)

        # Compare prediction with true label
        if tf.argmax(output, axis=1) == tf.argmax(y, axis=1):
            correct += 1

# Print accuracy
print(f"Test Accuracy: {correct / total * 100:.2f}%")


