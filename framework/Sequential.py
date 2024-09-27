from model import *
from framework import l2_normalize
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
            
    # Sequential model that puts calculations together
    @staticmethod
    def RecSys_Sequential(X_u: np.ndarray, X_i: np.ndarray, Y: np.ndarray, params_u: dict, params_i: dict, v_u: dict, v_i: dict, 
                          s_u: dict, s_i: dict, t: float, learning_rate = None, cache_i = dict(), cache_u = dict(),
                          beta1 = 0.9, beta2 = 0.999, epsilon = 1e-8) -> tuple[np.ndarray, dict, dict, dict, 
                                                                                  np.ndarray, dict, dict, dict,
                                                                                  float, float]:
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

        # 1a. Forward prop with user 
        a1_u, cache_u = ForwardProp(X_u, params_u, relu, 1, cache_u) #cache_u empty on the 1st layer
        a2_u, cache_u = ForwardProp(a1_u, params_u, relu, 2, cache_u) 
        a3_u, cache_u = ForwardProp(a2_u, params_u, relu, 3, cache_u)
        a4_u, cache_u = ForwardProp(a3_u, params_u, linear, 4, cache_u)

        # 1b. Forward prop with items
        a1_i, cache_i = ForwardProp(X_i, params_i, relu, 1, cache_i) #cache_i empty on the 1st layer
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

        # print(f'a4_u shape: {a4_u.shape}, a4_i shape: {a4_i.shape}, y_pred shape: {y_pred.shape}')

        # 2. Using the result of the NN (a3) to compute loss with MSE:
        loss = MSECost(y_pred, Y)

        # 3. Apply backprop to find derivative/gradients
        gradients_u = Backprop(X_u, Y, cache_u)
        gradients_i = Backprop(X_i, Y, cache_i)

        # 4. Update parameters using Adam for user and item
        t = t + 1.
        params_u, v_u, s_u, _, _ = Adam(params_u, gradients_u, v_u, s_u, t, learning_rate, beta1, beta2, epsilon)
        params_i, v_i, s_i, _, _ = Adam(params_i, gradients_i, v_i, s_i, t, learning_rate, beta1, beta2, epsilon)

        return (params_u, v_u, s_u, 
                params_i, v_i, s_i,
                y_pred, loss, t)
        