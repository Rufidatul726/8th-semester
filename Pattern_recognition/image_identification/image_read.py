from PIL import Image
import numpy

#load image
with Image.open("image-1.jpg") as im:
    im_toarray = numpy.array(im)

im.close()


#convert it to grayscale
grayscale_image = []

for row in im_toarray:
    grayscale_image_row = []
    for pixel in row:
        r, g, b = pixel

        grayscale_image_value = int(0.2989 * r + 0.5870 * g + 0.1140 * b)
        grayscale_image_row.append(grayscale_image_value)

    grayscale_image.append(grayscale_image_row)


grayscale_image_np = numpy.array(grayscale_image, dtype=numpy.uint8)

grayscale_image_pil = Image.fromarray(grayscale_image_np)
grayscale_image_pil.save("grayscale_image.jpg")

print("Grayscale image saved as grayscale_image.jpg")


#write the image in rgb value
print(im_toarray)