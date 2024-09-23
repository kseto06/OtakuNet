# Class to create a Deep Learning Neural Network model framework, from scratch

import numpy as np #Only np allowed, to speed up computational time for linalg computations
import math

class Layer:
    def __init__(self):
        self.input = None
        self.output = None

# Linear activation
def linear(z: np.ndarray) -> np.ndarray:
    return z

# Relu activation
def relu(z: np.ndarray) -> np.ndarray:
    return np.maximum(0, z)

# Relu prime (derivative for backprop)
def relu_prime(z: np.ndarray) -> np.ndarray:
    return np.where(z > 0, 1, 0) # If z > 0, element is 1, else element is 0
    
# Dense (forward prop layer)
def Dense(a_in, W, b, g):
    """
    Computes dense layer
    Args:
    a_in (ndarray (n, )) : Data, 1 example 
    W    (ndarray (n,j)) : Weight matrix, n features per unit, j units
    b    (ndarray (j, )) : bias vector, j units  
    Returns
    a_out (ndarray (j,))  : j units
    """
    num_units = W.shape[1]
    a_out = np.zeros(num_units)
    for j in range(num_units):
        # Activation Function g(z), z = wx + b
        z = np.dot(a_in * W[:, j]) + b[j]
        a_out[j] = g(z)
    return a_out

# Forward prop
def ForwardProp(a_in: np.ndarray, params: list, activation: callable, layer: int, cache = dict()):
    # Retrieve the parameters
    W, b = params

    # Compute forward prop
    z = np.dot(W, a_in) + b
    a_out = activation(z)

    cache[f'Z{layer}'] = z
    cache[f'A{layer}'] = a_out
    cache[f'W{layer}'] = W
    cache[f'b{layer}'] = b

    return a_out, cache

# MSE Cost Function
def MSECost(a_out, Y):
    '''
    Compute mean squared error cost function

    Inputs:
    a_out = output of forward prop (Dense)
    Y = 'true' labels vector, same shape as output of forward prop

    Output:
    Cost = loss
    '''
    m = a_out.shape[0] #num of training examples
    cost = 0.
    for i in range(m):
        cost += (a_out[i] - Y[i])**2

    return (cost / (2*m))

# Back propagation implementation to get gradients
def Backprop(a_in, Y, cache):
    '''
    Arguments:
    a_in -- input dataset, of shape (input size, number of examples)
    Y -- true "label" vector 
    cache -- cache output from Forward Prop. Dictionary containing [Z, a, w, b]
    
    Returns:
    gradients -- A dictionary with the gradients with respect to each parameter, activation and pre-activation variables
    """
    '''
    gradients = {}
    L = len(cache) // 4 # num of layers
    m = a_in.shape[1] #num of exs

    # Initialize the gradient for the last layer
    dA_prev = (1./m * cache[f'A{L}'] - Y)

    # Based on backprop computation steps/formulae from Andrew Ng's DL Specialization:
    for layer in reversed(range(1, L+1)):
        #dA_prev represent dA[l-1]
        dZ = dA_prev * relu_prime(cache[f'A{layer}'])
        if (layer > 1):
            dW = np.dot(dZ, cache[f'A{layer - 1}'].T)
        else: 
            dW = np.dot(dZ, a_in.T)
        dB = np.sum(dZ, axis=1, keepdims=True)
        dA_prev = np.dot(cache[f'W{layer}'].T, dZ)

        #Storing the gradients:
        gradients[f'dZ{layer}'] = dZ
        gradients[f'dW{layer}'] = dW
        gradients[f'dB{layer}'] = dB

    return gradients

def generate_minibatches(X, Y, mini_batch_size = 128, seed = 0):
    '''
    Generate a list of random minibatches from (X, Y)

    Inputs:
    X -- input data, shape[input size, num of exs]
    Y -- true 'label' vector

    m/mini_batch_size mini-batches with full 128 exs
    final minibatch if there isn't 128 is (m - mini_batch_size * m/mini_batch_size)
    mini_batch_X = shuffled_X[:, i:j]
    '''
    np.random.seed(seed)
    m = X.shape[1]
    mini_batches = []

    # Shuffle X, Y
    permutation = list(np.random.permutation(m))
    shuffled_X = X[:, permutation]
    shuffled_Y = Y[:, permutation].reshape((1,m))

    inc = mini_batch_size

    # Creating the mini-batch
    num_minibatches = math.floor(m / mini_batch_size)
    for k in range(0, num_minibatches): #Create a counter to count the minibatches without repeating
        # By formula given above
        mini_batch_X = shuffled_X[:, k*mini_batch_size : (k+1)*mini_batch_size]
        mini_batch_Y = shuffled_Y[:, k*mini_batch_size : (k+1)*mini_batch_size]
        mini_batch = (mini_batch_X, mini_batch_Y) #Grouping
        mini_batches.append(mini_batch)

    #Handling the case where the last mini-batch may < mini_batch_size
    if m % mini_batch_size != 0:
        # Apply the "last batch" formula
        mini_batch_X = shuffled_X[:, int(m/mini_batch_size)*mini_batch_size:]
        mini_batch_Y = shuffled_Y[:, int(m/mini_batch_size)*mini_batch_size:]
        mini_batch = (mini_batch_X, mini_batch_Y) #Grouping
        mini_batches.append(mini_batch)

    return mini_batches
    
def gradient_descent(params, gradients, learning_rate = 0.001, num_layers = 3):
    L = len(params) // num_layers #num of layers in NN's

    # Update rule for each parameter
    for l in range(1, L+1): #Start from 1 because w and b counts start from 1 (w1,w2...)
        params[f'W{l}'] = params[f'W{l}'] - learning_rate*gradients[f'dW{l}']
        params[f'b{l}'] = params[f'b{l}'] - learning_rate*gradients[f'db{l}']
        
    return params