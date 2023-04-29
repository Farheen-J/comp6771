## Imports
from tqdm import tqdm
import cv2
import os

## Load Source Images
source_directory = '../input/sharp'
images = os.listdir(source_directory)

## Define Destination for Blurred Images
destination_directory = '../input/gaussian_blurred'

## Looping through each source image
for x, image in tqdm(enumerate(images), total=len(images)):
    ## Reading each image
    image = cv2.imread(f"{source_directory}/{images[x]}")

    ## Applying Gaussian Blur
    blur = cv2.GaussianBlur(image, (51, 51), 0)

    ## Writing Blurred Image to destination
    cv2.imwrite(f"{destination_directory}/{images[x]}", blur)

print('Blurring Finished')