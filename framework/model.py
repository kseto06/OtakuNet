# Class to create a Deep Learning Neural Network model framework, from scratch

import numpy as np #Only np allowed, to speed up computational time for linalg computations
import math
from framework import l2_normalize

# Linear activation
def linear(z: np.ndarray) -> np.ndarray:
    return z

def linear_prime(z: np.ndarray) -> np.ndarray:
    # Derivative of a linear function f(x) = x is just one
    dZ = np.ones_like(z)
    return dZ

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

    # print(cache)

    return a_out, cache

# MSE Cost Function
def MSECost(y_pred: np.ndarray, Y: np.ndarray) -> float:
    '''
    Compute mean squared error cost function.
    For content-based, make sure y_pred already has their dot product computed

    Inputs:
    y_pred = prediction of forward prop (Dense)
    Y = 'true' labels vector, same shape as output of forward prop

    Output:
    Cost = loss
    '''
    # Transpose back y_pred
    y_pred = y_pred.transpose()

    if y_pred.shape[0] != Y.shape[0]:
        print("y_pred shape: ",y_pred.shape)
        print("Y shape: ", Y.shape)

    m = y_pred.shape[0] #num of training examples
    cost = 0.
    for i in range(m):
        cost += (y_pred[i] - Y[i])**2

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
    # print(f'Backprop a_in shape: {a_in.shape[0]}. Transposed: {a_in.transpose().shape[0]}')

    gradients = {}
    L = len(cache) // 4 # num of layers based on paramater types (Z, A, W, b)
    m = a_in.shape[0] #num of exs
    # print(m)

    # Initialize the gradient for the last layer ()
    dA_prev = (1./m * cache[f'A{L}'] - Y.T)

    # Based on backprop computation steps/formulae from Andrew Ng's DL Specialization:
    for layer in reversed(range(1, L+1)):
        #dA_prev represent dA[l-1]

        # Depending on the activation function. First layer is linear
        if (layer == L):
            dZ = dA_prev * linear_prime(cache[f'Z{layer}'])
        else:
            dZ = dA_prev * relu_prime(cache[f'Z{layer}'])
        
        # Exception for first layer in NN (last layer in backprop)
        if (layer > 1):
            dW = np.dot(dZ, cache[f'A{layer - 1}'].T)
        else: 
            dW = np.dot(dZ, a_in)

        # print(dZ.shape) --> (num of neurons, 64 exs)
        dB = np.sum(dZ, axis=1, keepdims=True)
        dA_prev = np.dot(cache[f'W{layer}'].T, dZ)

        #Storing the gradients:
        gradients[f'dZ{layer}'] = dZ
        gradients[f'dW{layer}'] = dW
        gradients[f'db{layer}'] = dB

    return gradients

# Back propagation implementation to get gradients
def Backprop_Revised(a_in: np.ndarray, Y: np.ndarray, cache: dict, inputs: np.ndarray, error_function = MSECost) -> dict:
    '''
    Arguments:
    a_in -- input dataset, of shape (input size, number of examples)
    Y -- true "label" vector 
    cache -- cache output from Forward Prop. Dictionary containing [Z, a, w, b]
    
    Returns:
    gradients -- A dictionary with the gradients with respect to each parameter, activation and pre-activation variables
    """
    '''
    # print(f'Backprop a_in shape: {a_in.shape[0]}. Transposed: {a_in.transpose().shape[0]}')

    gradients = {}
    L = len(cache) // 4 # num of paramater types (Z, A, W, b)
    m = a_in.shape[0] #num of exs
    derivative_activation = None

    # Get the pre-activations from the cache
    z = []
    for layer in range(1, L+1):
        z.append(cache[f'Z{layer}'])

    # Compute derivative loss
    derivative_loss = (1./m * cache[f'A{L}'] - Y.T) #

    # Main function
    for layer in reversed(range(1, L+1)):

        # Compute derivative of activation function with respect to the pre-activation
        if layer == L:
            derivative_activation = linear_prime(z[layer-1])
        else:
            derivative_activation = relu_prime(z[layer-1])

        # Compute delta (hadamard of d_loss & d_activation)
        delta = derivative_loss * derivative_activation

        # Compute gradients
        gradients[f'dW{layer}'] = np.multiply(inputs[layer].T, delta).T
        gradients[f'db{layer}'] = np.sum(delta, axis=0, keepdims=True).T

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

def create_minibatches(X_user: np.ndarray, X_item: np.ndarray, Y: np.ndarray, mini_batch_size = 64, seed = 0) -> list:
    '''
    Generate a list of random minibatches from (X_user, X_item, Y)

    Inputs:
    X -- input data, shape[num of exs, input size]
    Y -- true 'label' vector (ratings), of shape (num of exs, 1)

    m/mini_batch_size mini-batches with full 64 exs
    final minibatch if there isn't 64 is (m - mini_batch_size * m/mini_batch_size)
    mini_batch_X = shuffled_X[:, i:j]
    '''
    np.random.seed(seed)
    m = X_user.shape[0]
    minibatches = []

    if X_user.shape[0] != X_item.shape[0] or X_user.shape[0] != Y.shape[0] or X_user.shape[0] != X_item.shape[0]:
        raise ValueError(f'X_user: {X_user.shape} != X_item: {X_item.shape} != Y: {Y.shape}')

    # Shuffle X, Y
    # permutation = list(np.random.permutation(m))
    try:
        permutation = list(np.random.permutation(m))
        shuffled_X_user = X_user[permutation, :]
        shuffled_X_item = X_item[permutation, :]
        shuffled_Y = Y[permutation, :].reshape((m, 1)) #Y is 1D
    except IndexError as e:
        print("Error: ", e)

    # Creating the mini-batch
    num_minibatches = math.floor(m / mini_batch_size)
    for k in range(0, num_minibatches): #Create a counter to count the minibatches without repeating
        # By formula given above
        minibatch_X_user = shuffled_X_user[k*mini_batch_size : (k+1)*mini_batch_size, :]
        minibatch_X_item = shuffled_X_item[k*mini_batch_size : (k+1)*mini_batch_size, :]
        minibatch_Y      = shuffled_Y[k*mini_batch_size : (k+1)*mini_batch_size, :]
        minibatch = (minibatch_X_user, minibatch_X_item, minibatch_Y) #Grouping
        minibatches.append(minibatch)

    #Handling the case where the last mini-batch may < mini_batch_size
    if m % mini_batch_size != 0:
        # Apply the "last batch" formula
        minibatch_X_user = shuffled_X_user[int(m/mini_batch_size)*mini_batch_size:, :]
        minibatch_X_item = shuffled_X_item[int(m/mini_batch_size)*mini_batch_size:, :]
        minibatch_Y      = shuffled_Y[int(m/mini_batch_size)*mini_batch_size:, :]
        minibatch = (minibatch_X_user, minibatch_X_item, minibatch_Y) #Grouping
        minibatches.append(minibatch)

    return minibatches

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
    L = len(params) // 2 # num of param types (dW, db)
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

    L = len(params) // 2 # num of param types (dW, db)
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

# Scheduled learning rate decay function. Allows us to start with high learning rates and then gradually decrease it so GD doesn't overshoot near the minimum
def schedule_lr_decay(learning_rate_initial: float, epoch_num: int, decay_rate: float, time_interval = 1000) -> float:
    # Calculates updated learning rate with exponential weight decay:
    learning_rate = learning_rate_initial / (1 + (decay_rate * (epoch_num / time_interval)))
    return learning_rate    

# Predict function (to get y_pred) -- basically a ForwardProp
def predict(item_test: np.ndarray, user_test: np.ndarray, params_u: dict, params_i: dict, cache_u = dict(), cache_i = dict()) -> np.ndarray:
    # 1a. Forward prop with user 
    a1_u, cache_u = ForwardProp(user_test, params_u, relu, 1, cache_u) #cache_u empty on the 1st layer
    a2_u, cache_u = ForwardProp(a1_u, params_u, relu, 2, cache_u) 
    a3_u, cache_u = ForwardProp(a2_u, params_u, relu, 3, cache_u)
    a4_u, cache_u = ForwardProp(a3_u, params_u, linear, 4, cache_u)

    # 1b. Forward prop with items
    a1_i, cache_i = ForwardProp(item_test, params_i, relu, 1, cache_i) #cache_i empty on the 1st layer
    a2_i, cache_i = ForwardProp(a1_i, params_i, relu, 2, cache_i) 
    a3_i, cache_i = ForwardProp(a2_i, params_i, relu, 3, cache_i)
    a4_i, cache_i = ForwardProp(a3_i, params_i, linear, 4, cache_i)

    # 1c. Transpose back the vectors to correct shape
    a4_u = a4_u.T
    a4_i = a4_i.T

    # 2. L2 Normalization of vectors
    a4_u = l2_normalize(vector=a4_u, axis=1)
    a4_i = l2_normalize(vector=a4_i, axis=1)

    # 3. Current prediction (dot product):
    y_pred = np.sum(np.dot(a4_u, a4_i.T), axis=1)

    # Final rating prediction
    return y_pred

#Evaluate function -- return the loss and accuracy of the model on test sets
def evaluate(y_pred: np.ndarray, y_test: np.ndarray) -> None:
    # Calculate loss
    loss = MSECost(y_pred, y_test)

    # Calculate % accuracy
    mae = np.mean(np.abs(y_pred - y_test))
    accuracy = (1 - (mae / (np.max(y_test) - np.min(y_test))))*100

    # Print results
    print(f'Test Loss: {loss} | Test Accuracy: {accuracy}')
    
    return None