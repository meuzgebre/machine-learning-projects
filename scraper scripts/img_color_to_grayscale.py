''' 
    Authore: Meuz G
    Small script to convert all the images in a given directory in to grayscale
    It relays on `opencv` module
    Usage: pyhton --dir [dir_name]
    Return Folder in the base directory with directory name `gray`
'''

import os, math
import cv2

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
dir_path = ''
dest_dir = ''

def to_grayscale(filename):

    # Read the image
    image = cv2.imread(os.path.join(dir_path, filename))
    
    # Convert the image
    gray_img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Save the image
    cv2.imwrite(os.path.join(dest_dir, filename), gray_img)

def convert_files(dir_path):

    # Get All files in a dir

    all_files = os.listdir(dir_path)

    percentage = 100/len(all_files)
    rate = 0

    for file in all_files:    
        # Check if the file is an image
        to_grayscale(file)

        # Loading progrees bar      
        rate += percentage
        print(math.ceil(rate) * '#', f'{math.floor(rate)}%')


if __name__ == '__main__':

    dir_path = input('Enter directory/ location: ')
    dest_dir = os.path.join(BASE_DIR, f'{dir_path}_gray')
    os.mkdir(dest_dir)
    convert_files(os.path.join(BASE_DIR, dir_path))