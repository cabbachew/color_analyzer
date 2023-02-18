from collections import Counter
from sklearn.cluster import KMeans
from matplotlib import colors
import matplotlib.pyplot as plt
import numpy as np
import cv2
import datetime

# Read the image using OpenCV
image = cv2.imread('test_image.jpg')
# Must convert to RGB because OpenCV reads in BGR
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) 
# Uncomment to see the image
# plt.imshow(image) 

# Function to convert RGB to Hex
def rgb_to_hex(rgb_color):
    hex_color = "#"
    for i in rgb_color:
        i = int(i)
        # 02x means 2 digits in hex
        hex_color += ("{:02x}".format(i)) 
        # Alternative: hex_color += hex(i)[2:]
    return hex_color

# Preprocess the image
def prep_image(raw_img):
    # Resize the image
    modified_img = cv2.resize(raw_img, (900, 600), interpolation = cv2.INTER_AREA)
    # Convert 2D image to 1D array with 3 columns that represent RGB
    modified_img = modified_img.reshape(modified_img.shape[0]*modified_img.shape[1], 3)
    return modified_img
  
def color_analysis(img):
  # clf means classifier
  clf = KMeans(n_clusters = 5) # Group the colors into 5 clusters
  color_labels = clf.fit_predict(img) # Fit the model and predict the labels.
  counts = Counter(color_labels) # Count the number of pixels that belong to each cluster
  center_colors = clf.cluster_centers_ # Get the colors that are in the center of each cluster
  ordered_colors = [center_colors[i] for i in counts.keys()] # Get the colors in the order of the clusters
  hex_colors = [rgb_to_hex(ordered_colors[i]) for i in counts.keys()]
  
  # Plot the pie chart
  plt.figure(figsize = (12, 8))
  plt.pie(counts.values(), labels = hex_colors, colors = hex_colors)
  plt.savefig("color_analysis_report.png")
  # Dynamic file save name
  # plt.savefig("color_analysis_report_{}.png".format(datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")))
  print(hex_colors)

prepped_image = prep_image(image)
color_analysis(prepped_image)