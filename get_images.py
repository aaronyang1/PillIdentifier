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
loc = ("/Users/aaronyang/Desktop/better_pills.xls")

wb = xlrd.open_workbook(loc)
sheet = wb.sheet_by_index(0)
sheet.cell_value(0, 0)

# Extracting number of rows
num_rows = sheet.nrows
print(sheet.row_values(0))

for i in range(1, num_rows):
    info = sheet.row_values(i)
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
    img.show()
    os.remove(str(i))

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