import xlrd
from collections import Counter
import urllib.request
from PIL import Image
import os
from collections import Counter
import numpy as np
import matplotlib.pyplot as plt
import random

# Give the location of the file
imgs = []
def get_images():
    global imgs
    loc = ("/Users/aaronyang/Desktop/better_pills.xls")

    wb = xlrd.open_workbook(loc)
    sheet = wb.sheet_by_index(0)
    sheet.cell_value(0, 0)

    # Extracting number of rows
    num_rows = sheet.nrows

    for i in range(1, num_rows):
        url = "https://data.lhncbc.nlm.nih.gov/public/Pills/" + sheet.row_values(i)[2]
        urllib.request.urlretrieve(url, str(i))
        img = Image.open(str(i))
        x_crop_len = min(img.size[0] // 8, img.size[1] // 6) * 4
        y_crop_len = min(img.size[0] // 8, img.size[1] // 6) * 3
        l = (img.size[0] - x_crop_len) // 2
        r = l + x_crop_len
        t = (img.size[1] - y_crop_len) // 2
        b = t + y_crop_len
        img = img.crop((l, t, r, b))
        img = img.resize((200, 150))
        imgs.append(np.array([np.array(img), sheet.row_values(i)[4]]))
        os.remove(str(i))
    imgs = np.array(imgs)
def save_images():
    global imgs
    with open('imgs.npy', 'wb') as f:
        np.save(f, imgs)
def load_images():
    global imgs
    with open('imgs.npy', 'rb') as f:
        imgs = np.load(f)
'''
for i in range(5):
    url = "https://data.lhncbc.nlm.nih.gov/public/Pills/" + sheet.row_values(i)[2]
    urllib.request.urlretrieve(url, str(i))
    img = Image.open(str(i))
    print(img.size)
    x_crop_len = min(img.size[0] // 4, img.size[1] // 3) * 4
    y_crop_len = min(img.size[0] // 4, img.size[1] // 3) * 3

    l = (img.size[0] - x_crop_len) // 2
    r = l + x_crop_len
    b = (img.size[1] - y_crop_len) // 2
    t = b + y_crop_len
    img = img.crop((l, t, r, b))
    img = img.resize((2000, 1500))
    os.remove(str(i))
'''


# End goal is to return a array of tuples where each tuple is in the form (image (numpy array), pill name (string))