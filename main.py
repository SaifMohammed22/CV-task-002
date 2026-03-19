"""
Main file
"""
import os
from src import PrewittFilter, GaussianFilter, CanyFilter
from src import read_image, save_image


def main():
    OUTPUT_PATH = "output"
    image_path = "images/The Scream.jpg"

    if os.path.exists(OUTPUT_PATH):
        pass
    else:
        os.mkdir(OUTPUT_PATH)
    
    image = read_image(image_path)
    prewitt = PrewittFilter()
    gaussian = GaussianFilter()
    cany = CanyFilter()
    X, Y = prewitt.apply(image)
    smoothed = gaussian.apply(image)
    GX, GY = cany.getGradients(image)
    mag = cany.getMagnitude(GX, GY)
    angle = cany.getAngle(GX, GY)

    save_image(OUTPUT_PATH + "/smoothed.png", smoothed)
    save_image(OUTPUT_PATH + "/X.png", X)
    save_image(OUTPUT_PATH + "/Y.png", Y)
    save_image(OUTPUT_PATH + "/GX.png", GX)
    save_image(OUTPUT_PATH + "/GY.png", GY)
    save_image(OUTPUT_PATH + "/angle.png", angle)
    save_image(OUTPUT_PATH + "/magnitude.png", mag)

if __name__ == "__main__":
    main()
    
