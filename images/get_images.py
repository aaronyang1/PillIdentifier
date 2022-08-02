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


x = []
y = []
for i in range(num_rows):
    url = "https://data.lhncbc.nlm.nih.gov/public/Pills/" + sheet.row_values(i)[2]
    urllib.request.urlretrieve(url, str(i))
    img = Image.open(sheet.row_values(i)[2])
    x.append(img.size[0])
    y.append(img.size[1])
    os.remove(str(i))
    
# Import libraries
  
# Creating dataset
x = np.array(x)
y = np.array(y)
fig = plt.subplots(figsize =(10, 7))
# Creating plot
plt.hist2d(x, y)
plt.title("Simple 2D Histogram")
  
# show plot
plt.show()


# End goal is to return a array of tuples where each tuple is in the form (image (numpy array), pill name (string))