import numpy as np
import model as md
import get_images as img
#firt convert input image to an array, using the function to convert input image to array, from get_images.py
def pill_predict(image: np.array):
    """ Function that takes in an image of a pill and returns the model's prediction
        Input: np.array
        Output: string or list of strings of most likely pill
    """
    input_model=md.Pill_Model() #is this how this works? the goal is to convert the input array 

    
