# CV project

### Intro
This repository is dedicated to rectification of scanned images of printed documents. We apply Hough transform to find the best rotation angle to define the skew, then rotate and use morphological operations to detect tables on scans. 

### Structure
1. Folders **images** and **output_images** contain scanned documents before and after CV aalgorithms application, output contains all phases of algorithm.
2. Notebook **rectification.ipynb** is a convinient way to repsresent the code.
3. Script **rotate_and_remove_table.py** takes input image and returns 4 image compilation, bottom-right is final one.
