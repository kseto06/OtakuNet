from model import *
import numpy as np

class Sequential:
    def __init__(self):
        pass

    # Sequential model that puts calculations together
    @staticmethod
    def Sequential(X: np.ndarray, Y: np.ndarray, params: dict, v: dict, s: dict, t: int, learning_rate = None, cache = None,
                    beta1 = float, beta2 = float,  epsilon = float) -> tuple[np.ndarray, dict, float, dict, dict, dict, float]:
        '''
        Sequential (sequence) that puts all the calculations together

        Inputs:
        X: minibatch_x
        Y: minibatch_y 
        params: Dictionary of initialized parameters
        learning_rate: Grad Des learning rate alpha
        cache: List of weight and bias values stored in forward prop
        '''
        num_layers = 5

        # 1. fwd prop with the current minibatch. Add multiple layers
        # Constructing the neural network:
        a1, cache = ForwardProp(X, params, relu, 1, cache) #Cache empty on the 1st layer
        a2, cache = ForwardProp(a1, params, relu, 2, cache) 
        a3, cache = ForwardProp(a2, params, relu, 3, cache)
        a4, cache = ForwardProp(a3, params, relu, 4, cache)
        a5, cache = ForwardProp(a4, params, linear, 5, cache)

        # 2. Using the result of the NN (a3) to compute loss with MSE:
        loss = MSECost(a5, Y)

        # 3. Apply backprop to find derivative/gradients
        gradients = Backprop(a5, Y, cache)

        # 4. Update parameters using Adam
        t = t + 1.
        params, v, s, _, _= Adam(params, gradients, v, s, t, learning_rate, beta1, beta2, epsilon)

        return a5, cache, loss, params, v, s, t
            