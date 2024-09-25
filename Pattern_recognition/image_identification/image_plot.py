bite_array = bytearray()

with open('image-array.txt', 'r') as f:
    array_2d = f.read()

f.close()

for row in array_2d:
    for pixel in row:
        bite_array.append(int(pixel))

with open('image_plotting.jpg', 'wb') as file:
    file.write(bite_array)