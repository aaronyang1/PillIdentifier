import numpy as np
import model as md
import get_images as img

#first convert input image to an array, using the function to convert input image to array, from get_images.py
def pill_predict(image: np.array):
    """ 
    Function that takes in an image of a pill and returns the model's prediction
    
    Input: np.array
    
    input_model(image.shape[0])
    
    Output: string or list of strings of most likely pill
    """
    input_model=md.model
    input_model(image.shape[0])



    

    
