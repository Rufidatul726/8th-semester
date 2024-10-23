from PIL import Image
import numpy

#load image
with Image.open("image-1.jpg") as im:
    im_toarray = numpy.array(im)
    # cv2.imshow('image', im_toarray)

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


#write the image in rgb value
with open("image_array.txt", "w+") as new_file:
    # new_file.write(str(im_toarray))
    for data in grayscale_image_np:
        for pixel in data:
            new_file.write(str(pixel.tolist()))


#operation in the image file
with Image.open("grayscale_image.jpg", "r") as grayscale_file:
    data = numpy.array(grayscale_file)

new_constant = 86

new_data_array=[]

for rows in data:
    new_data_array_row=[]
    print(rows)
    new_pixel = int(rows) + 86 % 255

    new_data_array.append(new_pixel)

new_data_image_np = numpy.array(new_data_array, dtype=numpy.uint8)

new_data_image_pil = Image.fromarray(new_data_image_np)
new_data_image_pil.save("new_data_image.jpg")

#make it negative
neg_array = []
for rows in data:
    neg_array_row=[]
    print(rows)
    for intensity_val in rows:
        new_pixel = 255 - int(intensity_val)
        neg_array_row.append(new_pixel)

    neg_array.append(neg_array_row)

inverted_image_np = numpy.array(neg_array, dtype=numpy.uint8)
inverted_image = Image.fromarray(inverted_image_np)
inverted_image.save("inverted_image.jpg")




new_data_image_pil = Image.fromarray(new_data_image_np)
new_data_image_pil.save("new_data_image.jpg")