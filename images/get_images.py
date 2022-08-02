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
loc = ("/Users/aaronyang/Desktop/5_pills.xls")

wb = xlrd.open_workbook(loc)
sheet = wb.sheet_by_index(0)
sheet.cell_value(0, 0)

# Extracting number of rows
num_rows = sheet.nrows
print(sheet.row_values(0))



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
# Creating dataset
x = np.array(x)
y = np.array(y)
fig = plt.subplots(figsize =(10, 7))
# Creating plot
plt.hist2d(x, y) 
plt.title("Simple 2D Histogram")
  
# show plot
plt.show()

url = "https://data.lhncbc.nlm.nih.gov/public/Pills/" + sheet.row_values(1)[2]
urllib.request.urlretrieve(url, "1")
img = Image.open("1")
print(img.size)
img.show()


img = img.resize((2000, 1500))
print(img.size)
img.show()
os.remove("1")
'''
# End goal is to return a array of tuples where each tuple is in the form (image (numpy array), pill name (string))