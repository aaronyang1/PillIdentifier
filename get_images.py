import xlrd
from collections import Counter
drugs = Counter([])

 
# Give the location of the file
loc = ("/Users/aaronyang/Desktop")
 
wb = xlrd.open_workbook(loc)
sheet = wb.sheet_by_index(0)
sheet.cell_value(0, 0)

# Extracting number of rows
num_rows = sheet.nrows

for (i in range(num_rows)):
    info = sheet.row_values(i)
    if info[3] == "C3PI_Test":
        drugs.update(info[4].split()[1])
print(drugs.most_common(5))