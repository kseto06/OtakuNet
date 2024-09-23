# Class to create a Deep Learning Neural Network model framework, from scratch

import numpy as np #Only np allowed, to speed up computational time for linalg computations

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
def MSE():
    pass

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
    
def Adam():
    pass