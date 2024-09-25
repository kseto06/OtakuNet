# Class to create a Deep Learning Neural Network model framework, from scratch

import numpy as np #Only np allowed, to speed up computational time for linalg computations
import math

# Linear activation
def linear(z: np.ndarray) -> np.ndarray:
    return z

# Relu activation
def relu(z: np.ndarray) -> np.ndarray:
    return np.maximum(0, z)

# Relu prime (derivative for backprop)
def relu_prime(z: np.ndarray) -> np.ndarray:
    return np.where(z > 0, 1, 0) # If z > 0, element is 1, else element is 0

def init_params(layer_dimensions: list) -> dict:
    '''
    Inputs:
    layer_dimensions -- array containing the dimensions of each layer in the NN

    Outputs:
    params -- dictionary of params (W1, b1, W2, b2, ...)
    '''
    np.random.seed(3)
    params = {}
    L = len(layer_dimensions) # Number of layers in NN

    for layer in range(1, L):
        # Weight (strength of the connection from neuron j to i) matrix. Generates a matrix of shape[num units in current layer, num units in prev layer]
        # Scaled by He initialization for ReLU, maintaining variance of activations across layers
        # NOTE: using np.random because np.zeros initialization will cause symmetry - gradients by backprop all = 0
        params[f'W{layer}'] = np.random.randn(int(layer_dimensions[layer]), int(layer_dimensions[layer-1])) * np.sqrt(2. / int(layer_dimensions[layer-1])) 

        # Bias matrix. Generates a matrix of shape[num units in current layer, 1 value]
        params[f'b{layer}'] = np.zeros((int(layer_dimensions[layer]), 1)) 

        assert params['W' + str(layer)].shape[0] == int(layer_dimensions[layer]), int(layer_dimensions[layer-1])
        assert params['W' + str(layer)].shape[0] == int(layer_dimensions[layer]), 1
        
    return params
    
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
def ForwardProp(a_in: np.ndarray, params: list, activation: callable, layer: int, cache = dict()) -> tuple[np.ndarray, dict]:
    # Retrieve the parameters
    W = params[f'W{layer}']
    b = params[f'b{layer}']

    # print(f"Layer {layer}: W shape = {W.shape}, b shape = {b.shape}")

    # Compute forward prop
    if layer == 1: #Transpose the first layer
        a_in = a_in.transpose()
    
    # print(f"After transposing: a_in shape = {a_in.shape}")
    z = np.dot(W, a_in) + b
    a_out = activation(z)

    # Store the cache values to use in backprop
    cache[f'Z{layer}'] = z
    cache[f'A{layer}'] = a_out
    cache[f'W{layer}'] = W
    cache[f'b{layer}'] = b

    return a_out, cache

# MSE Cost Function
def MSECost(a_out: np.ndarray, Y: np.ndarray) -> float:
    '''
    Compute mean squared error cost function

    Inputs:
    a_out = output of forward prop (Dense)
    Y = 'true' labels vector, same shape as output of forward prop

    Output:
    Cost = loss
    '''
    # Transpose back a_out
    a_out = a_out.transpose()

    if a_out.shape != Y.shape:
        print("a_out shape: ",a_out.shape)
        print("Y shape: ", Y.shape)

    m = a_out.shape[0] #num of training examples
    cost = 0.
    for i in range(m):
        cost += (a_out[i] - Y[i])**2

    return (cost / (2*m))

# Back propagation implementation to get gradients
def Backprop(a_in: np.ndarray, Y: np.ndarray, cache: dict) -> dict:
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
    m = a_in.shape[0] #num of exs

    # Transpose Y for consistent shapes
    Y = Y.transpose()

    # Initialize the gradient for the last layer
    dA_prev = (1./m * cache[f'A{L}'] - Y)

    # Based on backprop computation steps/formulae from Andrew Ng's DL Specialization:
    for layer in reversed(range(1, L+1)):
        #dA_prev represent dA[l-1]
        dZ = dA_prev * relu_prime(cache[f'A{layer}'])
        if (layer > 1):
            dW = np.dot(dZ, cache[f'A{layer - 1}'].T)
        else: 
            dW = np.dot(dZ, a_in)
        dB = np.sum(dZ, axis=1, keepdims=True)
        dA_prev = np.dot(cache[f'W{layer}'].T, dZ)

        #Storing the gradients:
        gradients[f'dZ{layer}'] = dZ
        gradients[f'dW{layer}'] = dW
        gradients[f'db{layer}'] = dB

    return gradients

def generate_minibatches(X: np.ndarray, Y: np.ndarray, mini_batch_size = 64, seed = 0) -> list:
    '''
    Generate a list of random minibatches from (X, Y)

    Inputs:
    X -- input data, shape[num of exs, input size]
    Y -- true 'label' vector (ratings), of shape (num of exs, 1)

    m/mini_batch_size mini-batches with full 64 exs
    final minibatch if there isn't 64 is (m - mini_batch_size * m/mini_batch_size)
    mini_batch_X = shuffled_X[:, i:j]
    '''
    np.random.seed(seed)
    m = X.shape[0]
    mini_batches = []

    if (X.shape[0] != Y.shape[0]):
        print(f'X: {X.shape} != Y: {Y.shape}')

    # Shuffle X, Y
    # permutation = list(np.random.permutation(m))
    try:
        permutation = list(np.random.permutation(m))
        shuffled_X = X[permutation, :]
        shuffled_Y = Y[permutation, :].reshape((m, 1)) #Y is 1D
    except IndexError as e:
        print("Error: ", e)

    inc = mini_batch_size

    # Creating the mini-batch
    num_minibatches = math.floor(m / mini_batch_size)
    for k in range(0, num_minibatches): #Create a counter to count the minibatches without repeating
        # By formula given above
        mini_batch_X = shuffled_X[k*mini_batch_size : (k+1)*mini_batch_size, :]
        mini_batch_Y = shuffled_Y[k*mini_batch_size : (k+1)*mini_batch_size, :]
        mini_batch = (mini_batch_X, mini_batch_Y) #Grouping
        mini_batches.append(mini_batch)

    #Handling the case where the last mini-batch may < mini_batch_size
    if m % mini_batch_size != 0:
        # Apply the "last batch" formula
        mini_batch_X = shuffled_X[int(m/mini_batch_size)*mini_batch_size:, :]
        mini_batch_Y = shuffled_Y[int(m/mini_batch_size)*mini_batch_size:, :]
        mini_batch = (mini_batch_X, mini_batch_Y) #Grouping
        mini_batches.append(mini_batch)

    return mini_batches
    
# Function to perform one update step (epoch) of gradient descent
def gradient_descent(params: dict, gradients: dict, learning_rate = 0.001, num_layers = 3) -> dict:
    L = len(params) // num_layers #num of layers in NN's

    # Update rule for each parameter
    for l in range(1, L+1): #Start from 1 because w and b counts start from 1 (w1,w2...)
        params[f'W{l}'] = params[f'W{l}'] - learning_rate*gradients[f'dW{l}']
        params[f'b{l}'] = params[f'b{l}'] - learning_rate*gradients[f'db{l}']
        
    return params

# Function to initialize the parameters for the gradient and squared gradient
def init_Adam(params: dict) -> tuple[dict, dict]:
    L = len(params) // 2
    v = {}
    s = {}

    for l in range(L):
        v[f'dW{l+1}'] = np.zeros((params[f'W{l+1}'].shape[0], params[f'W{l+1}'].shape[1]))
        v[f'db{l+1}'] = np.zeros((params[f'b{l+1}'].shape[0], params[f'b{l+1}'].shape[1]))
        s[f'dW{l+1}'] = np.zeros((params[f'W{l+1}'].shape[0], params[f'W{l+1}'].shape[1]))
        s[f'db{l+1}'] = np.zeros((params[f'b{l+1}'].shape[0], params[f'b{l+1}'].shape[1]))

    return v, s

def Adam(params: dict, gradients: dict, v: dict, s: dict, t: float, learning_rate: float, beta1: float, beta2: float, epsilon: float) -> tuple[dict, dict, dict, dict, dict]:
    '''
    Function inspired by the formulae/procedure described in Andrew Ng's DL Course

    v: moving avg of first grad
    s: moving avg of squared grad
    '''
    if t < 0:
        print(f't = {t}. t must be greater than 0')

    L = len(params) // 2
    v_corrected = {}
    s_corrected = {}
    
    # Perform Adam update
    for l in range(L):
        # print(beta1)
        # print(beta2)
        # print(1-np.power(beta1, t))

        # Calculate momentums with beta1
        v[f'dW{l+1}'] = beta1 * v[f'dW{l+1}'] + (1-beta1) * gradients[f'dW{l+1}']
        v[f'db{l+1}'] = beta1 * v[f'db{l+1}'] + (1-beta1) * gradients[f'db{l+1}']

        # Calculate corrected v values:
        v_corrected[f'dW{l+1}'] = v[f'dW{l+1}'] / (1-np.power(beta1, t))
        v_corrected[f'db{l+1}'] = v[f'db{l+1}'] / (1-np.power(beta1, t))

        # Calculate the RMSProps with beta2
        s[f'dW{l+1}'] = beta2 * s[f'dW{l+1}'] + (1-beta2) * np.square(gradients[f'dW{l+1}'])
        s[f'db{l+1}'] = beta2 * s[f'db{l+1}'] + (1-beta2) * np.square(gradients[f'db{l+1}'])

        # Calculate corrected s values:
        s_corrected[f'dW{l+1}'] = s[f'dW{l+1}'] / (1-np.power(beta2, t))
        s_corrected[f'db{l+1}'] = s[f'db{l+1}'] / (1-np.power(beta2, t))

        params[f'W{l+1}'] = params[f'W{l+1}'] - learning_rate * v_corrected[f'dW{l+1}'] / (np.sqrt(s_corrected[f'dW{l+1}'])+epsilon)
        params[f'b{l+1}'] = params[f'b{l+1}'] - learning_rate * v_corrected[f'db{l+1}'] / (np.sqrt(s_corrected[f'db{l+1}'])+epsilon)

    return params, v, s, v_corrected, s_corrected