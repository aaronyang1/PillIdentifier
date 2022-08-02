import numpy as np
import random 
import torch
import images.get_images as img
import identify as idt
from torch import tensor
import numpy as np
import torch.nn as nn
from torch.nn import Conv2d
from torch.nn.functional import cross_entropy
from torch.nn import ReLU
from torch.optim import SGD
from pathlib import Path
import pickle

class Pill_Model:

    def __init__(self, input_shape): #note the first dim will be 3, for 3 color channels
        """ 
        Initializes layers in  model, and sets them 
        as attributes of model.
        """

        self.conv1 = Conv2d(in_channels=input_shape[0], out_channels=8, kernel_size=(7, 7), stride=1)
        self.conv2 = Conv2d(in_channels=8, out_channels=16, kernel_size=(7, 7), stride=1)
        self.conv3 = Conv2d(in_channels=16, out_channels=32, kernel_size=(7, 7), stride=1)
        self.batchnorm1 = nn.BatchNorm2d(num_features=8)
        self.batchnorm2 = nn.BatchNorm2d(num_features=16)
        self.batchnorm3 = nn.BatchNorm2d(num_features=32)
        self.maxpool = nn.MaxPool2d()
        self.dense = nn.Linear(in_features=, out_features=6, bias=, )
 
    def __call__(self, x):
        '''Passes data as input to our model, performing a "forward-pass".
        
        This allows us to conveniently initialize a model `m` and then send data through it
        to be classified by calling `m(x)`.

        Model 
        
        Parameters
        ----------
        x : An array of image descriptor vectors?
        Returns
        -------
        list of probabilities?
        pass1 = self.batchnorm1(self.maxpool(ReLU(self.conv1)))
        pass2 = self.batchnorm2(self.maxpool(ReLU(self.conv2(pass1))))
        pass3 = self.batchnorm3(self.maxpool(ReLU(self.conv3(pass3))))
        return self.dense(pass3)
        '''
        # TODO : incorporate more layers if needed

        pass1 = self.batchnorm1(self.maxpool(ReLU(self.conv1)))
        pass2 = self.batchnorm2(self.maxpool(ReLU(self.conv2(pass1))))
        pass3 = self.batchnorm3(self.maxpool(ReLU(self.conv3(pass3))))
        return self.dense(pass3)

        
    @property
    def parameters(self):
        """ A convenience function for getting all the parameters of our model.
        
        This can be accessed as an attribute, via `model.parameters` 
        conv1 + self.conv2..parameters + self.conv3 +.parameters
        Returns
        -------
        Tuple[Tensor, ...]
            A tuple containing all of the learnable parameters for our model """
        # STUDENT CODE HERE
        return self.conv1.parameters + self.conv2.parameters + self.conv3.parameters

def train_model() :

    model=Pill_Model()
    optim = SGD(model.parameters, learning_rate = 1e-3, momentum = 0.9)
    losses = []
    loss_cnt = 0    
    num_epochs = 6
    batch_size = 32

       
    for epoch_cnt in range(num_epochs):
        train_idxs = np.array(5 * 4)# create array of N indices

        test_idxs = np.arange( // 5 * 4) # TODO : ?
           # np.random.shuffle()
               p.random.shuffle(train_idxs) # TODO : oh wait is this copied code
   #           
    for batch_cnt in range(0, len(train_idxs)//batch_size):
        batch = [coco_data.data["images"][i]["id"] for i in ids] 


         





