# import os
# import numpy as np
# from PIL import Image

# # Get a list of image file paths
# image_paths = [os.path.join(images_folder_path, f) for f in os.listdir(images_folder_path) if f.endswith(('.png', '.jpg', '.jpeg', '.bmp'))]

# # Read images into an array
# images = [Image.open(path) for path in image_paths]

# # Convert images to grayscale and then to 8-bit binary codes
# binary_images = []
# for img in images:
#     # Convert to grayscale
#     gray_img = img.convert('L')
#     # Convert to numpy array
#     gray_array = np.array(gray_img)
#     # Convert to 8-bit binary using numpy's binary_repr function
#     binary_image = np.vectorize(np.binary_repr)(gray_array, width=8)
#     binary_images.append(binary_image)

# # Convert the list of 2D arrays into a 3D numpy array with 0's and 1's
# binary_3d_array = np.array([[[int(bit) for bit in row] for row in img] for img in binary_images])

# print(binary_3d_array)

# import os
# import numpy as np
# from PIL import Image

# # Assuming we're using the paths from the uploaded images in this environment
# #images_folder_path = '/mnt/data/'
# images_folder_path = 'C:\\Users\\rupak\\OneDrive\\Desktop\\M.I.E\\images'
# image_files = [f for f in os.listdir(images_folder_path) if f.endswith(('.png', '.jpg', '.jpeg', '.bmp'))]

# # Read images into an array
# images = [Image.open(os.path.join(images_folder_path, file)) for file in image_files]

# # Find the maximum width and height to pad the images to the same size
# max_width = max(img.size[0] for img in images)
# max_height = max(img.size[1] for img in images)

# # Pad images to the same size and convert to bit planes
# bit_planes_list = []
# for img in images:
#     # Create a new image with the maximum size and a black background
#     new_img = Image.new('L', (max_width, max_height))
#     # Paste the original image into the center of the new image
#     new_img.paste(img, ((max_width - img.size[0]) // 2, (max_height - img.size[1]) // 2))
#     # Convert the padded image to grayscale
#     gray_img = new_img.convert('L')
#     # Convert to numpy array and then to 8-bit binary
#     gray_array = np.array(gray_img)
#     # Create bit planes
#     bit_planes = np.unpackbits(np.expand_dims(gray_array, axis=-1), axis=-1)
#     bit_planes_reshaped = bit_planes.reshape(gray_array.shape + (8,))
#     bit_planes_list.append(bit_planes_reshaped)

# # Combine into a 4D array
# binary_4d_array = np.stack(bit_planes_list)

# # Print the shape of the array and a small part of the array
# print("Shape of the final array:", binary_4d_array.shape)
# print("Small section of the array:")
# print(binary_4d_array[0, :5, :5, :])  # Print the first 5x5 section of the first image


import os
import numpy as np
from PIL import Image

# Path to your 'images' folder
images_folder_path = 'C:\\Users\\rupak\\OneDrive\\Desktop\\M.I.E\\images'

# Get a list of image file paths
image_files = os.listdir(images_folder_path)

# Assuming we take the first image from the list
image_path = os.path.join(images_folder_path, image_files[0])
image = Image.open(image_path)

# Convert the image to grayscale
gray_image = image.convert('L')
# Convert to numpy array and then to 8-bit binary
gray_array = np.array(gray_image)
# Create bit planes
bit_planes = np.unpackbits(np.expand_dims(gray_array, axis=-1), axis=-1)
# Reshape the bit planes to form a 3D array
bit_planes_3d = bit_planes.reshape(gray_array.shape + (8,))

# Print the shape of the 3D array and a small part of the array
print("Shape of the 3D array:", bit_planes_3d.shape)
print("Small section of the array:")
print(bit_planes_3d[:5, :5, :])  # Print the first 5x5 section of the bit planes