import numpy as np
import random 
import torch
import get_imagees as img
import database as db
import identify as idt
from torch import tensor
import numpy as np
import torch.nn as nn
from torch.nn import Conv2d
from torch.nn.functional import cross_entropy
from pathlib import Path
import pickle


relu = nn.functional.relu
softmax = nn.functional.softmax

class Pill_Model:
    def __init__(self, input_shape):
        """ 
        Initializes layers in  model, and sets them 
        as attributes of model.
        """

        # weight initializer is up for change
        self.conv1 = Conv2d(in_channels=input_shape[0], out_channels=, kernel_size=(7, 7), stride=1)
    
    def __call__(self, x):
        '''Passes data as input to our model, performing a "forward-pass".
        
        This allows us to conveniently initialize a model `m` and then send data through it
        to be classified by calling `m(x)`.

        Model 
        
        Parameters
        ----------
        x : ?
            ?
        Returns
        -------
        ?
            ?
        '''

        
        return 

        
    @property
    def parameters(self):
        """ A convenience function for getting all the parameters of our model.
        
        This can be accessed as an attribute, via `model.parameters` 
        
        Returns
        -------
        Tuple[Tensor, ...]
            A tuple containing all of the learnable parameters for our model """
        # STUDENT CODE HERE
        return self.w_embed.parameters
