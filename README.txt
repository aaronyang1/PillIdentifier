Pill Identifier

get_images.py
- Use the excel code and download image stuff that bhargav found to:
  - Filter the excel spreadsheet by type and label to get the ones we need
  - Download them from the NIH website using file path
  - Normalize/clean data
    - eg. turn them all into PNGs or smth idk man
  - Turn them into a numpy array
    - Yoink code from week 2?
  - train/test split

model.py
- make some 1-hot encoding for classes 
- Create & train CNN :))))

identify.py
- Function that takes in an image of a pill and returns the model's prediction 
- Function that takes a picture and returns the model's prediction (optional?) 
