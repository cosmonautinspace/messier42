from PIL import Image

'''This script is for the final adjustment of pixel values of a given unstrectched image'''

image = Image.open('data/images/input/inputCropped.TIFF')


#just for testing purposes. real value will be calculated using the ann model
factor=30
for r in range(image.height):
    for c in range(image.width):
        pixelValue = image.getpixel((c,r))
        image.putpixel((c,r),(pixelValue[0]*factor,pixelValue[1]*int(factor/2),pixelValue[2]*factor))

image.save('data/images/output/test.png')