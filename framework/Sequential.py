from model import *
import numpy as np

class Sequential:
    def __init__(self):
        pass

    # Sequential model that puts calculations together
    @staticmethod
    def Sequential(X: np.ndarray, Y: np.ndarray, params: dict, learning_rate = None, cache = None) -> tuple[np.ndarray, dict, float, dict]:
        '''
        Sequential (sequence) that puts all the calculations together

        Inputs:
        X: minibatch_x
        Y: minibatch_y 
        params: Dictionary of initialized parameters
        learning_rate: Grad Des learning rate alpha
        cache: List of weight and bias values stored in forward prop
        '''
        num_layers = 4

        # 1. fwd prop with the current minibatch. Add multiple layers
        # Constructing the neural network:
        a1, cache = ForwardProp(X, params, relu, 1, cache) #Cache empty on the 1st layer
        a2, cache = ForwardProp(a1, params, relu, 2, cache) 
        a3, cache = ForwardProp(a2, params, relu, 3, cache)
        a4, cache = ForwardProp(a3, params, linear, 4, cache)

        # 2. Using the result of the NN (a3) to compute loss with MSE:
        loss = MSECost(a4, Y)

        # 3. Apply backprop to find derivative/gradients
        gradients = Backprop(X, Y, cache)

        # 4. Use gradients to apply gradient descent
        params = gradient_descent(params, gradients, learning_rate, num_layers)

        return a4, cache, loss, params
            