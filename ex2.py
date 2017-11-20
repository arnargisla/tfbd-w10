import sys
from PIL import Image
import numpy as np
import hashlib
import binascii


def convert_to_grayscale(image):
    width, height = image.size

    gray_image = Image.new("RGB", (width, height), "white")
    pixels = gray_image.load()

    for i in range(width):
        for j in range(height):
            pixel = image.getpixel((i, j))
            red, green, blue = pixel[0], pixel[1], pixel[2]
            gray = (red * 0.299) + (green * 0.587) + (blue * 0.114)
            pixels[i, j] = (int(gray), int(gray), int(gray))

    return gray_image

def compare_adjacent(pixels):
    rows, columns = pixels.shape
    comparison_array = np.empty([rows,columns - 1], dtype=bool)
    for i in range(rows):
        for j in range(columns - 1):
            if pixels[i][j] > pixels[i][j+1]:
                comparison_array[i][j] = True
            else:
                comparison_array[i][j] = False

    return comparison_array

def image_hash():
    file = sys.argv[1]
    image = Image.open(file)

    #Grayscale the image
    grayscale_image = convert_to_grayscale(image)

    #Resize to fx. 9 x 8 pixels
    resized_image = grayscale_image.resize((9,8))

    #Compare adjacent values (x > y)
    pixels = [int((t[0] + t[1] + t[2])/len(t)) for t in list(resized_image.getdata())]
    adjacent_values_comparison_array = compare_adjacent(np.array(pixels).reshape(8,9))

    #Convert to hash
    h = hashlib.md5(adjacent_values_comparison_array).hexdigest()

    return h

if __name__=="__main__":
    print(image_hash())

