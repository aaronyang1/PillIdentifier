import numpy as np
import random 
import torch
import get_images as img
import identify as idt
from torch import tensor
import numpy as np
import torch.nn as nn
from torch.nn import Conv2d
from torch.nn.functional import cross_entropy
from torch.nn import ReLU
from torch.optim import S
softmax = nn.functional.softmax
f#rom torch.nn import SoftmaxGD
from pathlib import Path
import pickle

class Pill_Model:

    def __init__(self, input_shape): #note the first dim will be 3, for 3 color channels
        """ 
        Initializes layers in  model, and sets them 
        as attributes of model.
        """

        # data in shape of N x 416 x 416 x 3 (N = # of photos)
        self.conv1 = Conv2d(in_channels=input_shape[0], out_channels=8, kernel_size=(7, 7), stride=1)
        self.conv2 = Conv2d(in_channels=8, out_channels=16, kernel_size=(7, 7), stride=1)
        self.conv3 = Conv2d(in_channels=16, out_channels=32, kernel_size=(7, 7), stride=1)
        self.batchnorm1 = nn.BatchNorm2d(num_features=8)
        self.batchnorm2 = nn.BatchNorm2d(num_features=16)
        self.batchnorm3 = nn.BatchNorm2d(num_features=32)
        self.maxpool = nn.MaxPool2d()

        # TODO : 
        self.dense = nn.Linear(in_features=, out_features=6, bias=False)
 
    def __c
        all__(self, x):
        '''Passes data as input to our model, performing a "forward-pass".
        
        This allows us to conveniently initialize a model `m` and then send data through it
        to be classified by calling `m(x)`.

        Model 
        
        Parameters
        ----------
        x : An array of image descriptor vectors?
        Returns
        -------
        self.maxpool(ReLU(self.conv3(pass3))))
        return self.dense(pass3)
        '''
        # TODO : incorporate more layers if needed

        pass1 = self.batchnorm1(self.maxpool(ReLU(self.conv1(x))))
        pass2 = self.batchnorm2(self.maxpool(ReLU(self.conv2(pass1))))
        #do sofsoftmax(tmax on the final) result to get the probability distribution
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

#accuracy function
def accuracy():

#one hot encodings for x classes
        return self.conv1.parameters + self.conv2.parameters + self.conv3.parameters

def train_model():

    model=Pill_Model()

    optim = SGD(model.parameters, learning_rate = 1e-3, momentum = 0.9)
    losses = []
    loss_cnt = 0    
    num_epochs = 6
    batch_size = 32

    #N images, 416 *416*3
    length_input=N #represents number of photos
    input_data= 0
    for epoch_cnt in range(num_epochs):
        train_idxs = np.array((5, 4)) # create array of N indices, change size of array
        np.random.shuffle(train_idxs) 
            for batch_cnt in range(0, len(train_idxs)//batch_si
            ids = train_idxs[batch_cnt * batch_size:(batch_cnt + 1) * batch_size]ze):
                batchinput_imagesdata["images"][i]["id"] for i in i#change name of variable, it should not be input_images
            conf_ids = [input_images["images"][random.randint(0, )]["id"] for i in range(batch_size)] #change input_images
            
            true_desc = []
            conf_desc = []
            captions = []
            
            for i in range(batch_size):
                if batch[i] in resnet18_features and conf_ids[i] in resnet18_features and batch[i] in coco_data.image_to_caps:
                    true_desc.append(resnet18_features[batch[i]])
                    conf_desc.append(resnet18_features[conf_ids[i]])

                    # all_captions = [coco_data.I_To_C(i) for i in batch]
                    captions.append(coco_data.rand_cap(batch[i]))

            # true_desc = true_desc[~np.all(true_desc == 0, axis=0)]

            true_embed = model(np.array(true_desc))
            conf_embed = model(np.array(conf_desc))
            
            
            caption_embed = np.array([ec.embed(c) for c in captions])
        ds] 


         





