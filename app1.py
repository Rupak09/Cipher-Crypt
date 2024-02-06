import streamlit as st
import numpy as np
from PIL import Image
import hashlib
import matplotlib.pyplot as plt

st.set_page_config(page_title="Cipher Crypt", page_icon=":lock_with_ink_pen:", layout="wide")

def main():
    st.title("Cipher Crypt")
    st.subheader("A place where we secure your images.")

    st.title("Multiple Image Encryption and Decryption")

    # Upload multiple images
    uploaded_images = st.file_uploader("Upload Images", type=["jpg", "jpeg", "png"], accept_multiple_files=True)

    if uploaded_images:
        st.write(f"Number of Images Uploaded: {len(uploaded_images)}")

        # Canvas initialization
        canvas_size = (512, 512)
        canvas_list = []

        # Separate images based on dimensions
        small_images = []
        large_images = []

        for img_file in uploaded_images:
            img = Image.open(img_file)
            if img.size == (256, 256):
                small_images.append(img)
            elif img.size == (512, 512):
                large_images.append(img)
            else:
              st.error("Please upload images of the right dimensions (256x256 or 512x512 pixels)!")
              break     

        # Place large images on separate canvases
        for i, large_img in enumerate(large_images):
            canvas = Image.new("RGB", canvas_size, "white")
            canvas.paste(large_img, (0, 0))
            canvas_list.append(canvas)

        # Place small images on canvases (4 small images per canvas)
        for i in range(0, len(small_images), 4):
            canvas = Image.new("RGB", canvas_size, "white")
            for j in range(4):
                if i + j < len(small_images):
                    x_offset = (j % 2) * 256
                    y_offset = (j // 2) * 256
                    canvas.paste(small_images[i + j], (x_offset, y_offset))
            canvas_list.append(canvas)

        # Display the canvases
        for i, canvas in enumerate(canvas_list):
            st.image(canvas, caption=f"Canvas {i + 1}")

        # Encrypt button
        if st.button("Encrypt and Decrypt Images"):
            st.info("Encrypting...")

            encrypted_images = []
            decrypted_images = []

            # Perform encryption and decryption for each image
            for i, img_file in enumerate(canvas_list):
                #st.write(f"\nEncrypting Image {i + 1}")
                img = img_file  # Keep the image in RGB

                # Encryption
                #image_hash, _ = sha_256(img)  # Calculate SHA-256
                #st.info(f"SHA-256 for Image {i + 1}: {image_hash.hexdigest()}")

                encrypted_matrix,final_encrypted_matrix,rna_translation_matrix= encryption_function(img)
                encrypted_images.append(final_encrypted_matrix)

                # Decryption
                decrypted_matrix = decryption_function(img ,encrypted_matrix, rna_translation_matrix)
                decrypted_images.append(decrypted_matrix)

            st.success("Encryption and Decryption complete!")
            
            # Set the size of the figure based on the image size
            image_size = encrypted_images[0].shape[:2]  # Assuming all images have the same size
            fig_width = len(uploaded_images) * 5  # Adjust the multiplier as needed
            fig_height = 5  # Adjust the height as needed

            plt.figure(figsize=(fig_width, fig_height))
            # Display the encrypted and decrypted images in RGB
            for i, (encrypted_img, decrypted_img) in enumerate(zip(encrypted_images, decrypted_images)):
                plt.subplot(2, len(uploaded_images), i + 1)
                plt.imshow(encrypted_img.astype(np.uint8))
                plt.title(f'Encrypted Image {i + 1}')
                plt.axis("off")

                plt.subplot(2, len(uploaded_images), len(uploaded_images) + i + 1)
                plt.imshow(decrypted_img.astype(np.uint8))
                plt.title(f'Decrypted Image {i + 1}')
                plt.axis("off")
            plt.tight_layout()
            st.pyplot()

def sha_256(image):
    # Convert the RGB image to grayscale image
    gray_image = image.convert('L')

    # Convert to numpy array and then to 8-bit binary
    gray_array = np.array(gray_image)


    # Generating SHA-256
    hash_sha256 = hashlib.sha256()
    hash_sha256.update(gray_array.tobytes())
    hash_hexdigest = hash_sha256.hexdigest()
    hash_values = [int(hash_hexdigest[i:i+2], 16) for i in range(0, len(hash_hexdigest), 2)]

    return hash_sha256, hash_values
external_keys = [1,2,3,4,5,6]
def calculate_intermediate_params(c_values, k_values):
    # Convert hexadecimal strings to integers
   # k_values = [int(k, 16) for k in k_values]
    h1 = (c_values[0] + (k_values[0] ^ k_values[1] ^ k_values[2] ^ k_values[3] ^ k_values[4])) / 256
    h2 = (c_values[1] + (k_values[5] ^ k_values[6] ^ k_values[7] ^ k_values[8] ^ k_values[9])) / 256
    h3 = (c_values[2] + (k_values[10] ^ k_values[11] ^ k_values[12] ^ k_values[13] ^ k_values[14])) / 256
    h4 = (c_values[3] + (k_values[15] ^ k_values[16] ^ k_values[17] ^ k_values[18] ^ k_values[19])) / 256
    h5 = (c_values[4] + (k_values[20] ^ k_values[21] ^ k_values[22] ^ k_values[23] ^ k_values[24] ^ k_values[25])) / 256
    h6 = (c_values[5] + (k_values[26] ^ k_values[27] ^ k_values[28] ^ k_values[29] ^ k_values[30] ^ k_values[31])) / 256

    return h1, h2, h3, h4, h5, h6
def calculate_initial_values(h1, h2, h3, h4, h5, h6):
    x0 = ((h1 + h2 + h5) * 10**8) % 256 / 255
    y0 = ((h3 + h4 + h6) * 10**8) % 256 / 255
    z0 = ((h1 + h2 + h3 + h4) * 10**8) % 256 / 255
    p0 = ((h1 + h2 + h3) * 10**8) % 256 / 255
    q0 = ((h4 + h6 + h5) * 10**8) % 256 / 255

    return x0, y0, z0, p0, q0
def calculate_chaotic_system_parameters(h1, h2, h3, h4, h5, h6):
    
    a = (h1 + h2 / (h1 + h2 + h3 + h4 + h5 + h6)) * 100 % 3 + 1
    b = (h3 + h4 / (h1 + h2 + h3 + h4 + h5 + h6)) * 100 % 3 + 1
    c = (h5 + h6 / (h1 + h2 + h3 + h4 + h5 + h6)) * 100 % 3 + 1
    d = (h1 + h2 + h3 / (h1 + h2 + h3 + h4 + h5 + h6)) % 1

    return a, b, c, d
def generate_sequences_3d_sine_chaos(x0, y0, z0, a, b, c, u):  
  iterations = 1000 + u  # Total number of iterations
  
  X1 = []
  X2 = []
  X3 = []
  
  # Initialize x, y, z with the initial values x0, y0, z0
  x = x0
  y = y0
  z = z0
  
  # Iterate the 3D Sine chaos system
  for i in range(iterations):
      # Add a check to avoid division by zero
      den_x = np.sin(np.pi * y * (1 - z))
      den_y = np.sin(np.pi * z * (1 - x))
      den_z = np.sin(np.pi * x * (1 - y))
      if den_y == 0 or den_z == 0:
        continue
      xi = (a**3 * np.sin(np.pi * x * (1 - y)) / den_x) % 1
      yi = (b**3 * np.sin(np.pi * y * (1 - z)) / den_y) % 1
      zi = (c**3 * np.sin(np.pi * z * (1 - x)) / den_z) % 1
  
      # Discard the first 1000 iterations
      if i >= 1000:
          X1.append(xi)
          X2.append(yi)
          X3.append(zi)
  
      # Update x, y, z for the next iteration
      x, y, z = xi, yi, zi
  return X1,X2,X3
def dna_encoding_rules(rule_index):
       encoding_rules = {
           1: {'00': 'A', '11': 'T', '01': 'C', '10': 'G'},
           2: {'00': 'A', '11': 'T', '10': 'C', '01': 'G'},
           3: {'01': 'A', '10': 'T', '00': 'C', '11': 'G'},
           4: {'01': 'A', '10': 'T', '11': 'C', '00': 'G'},
           5: {'10': 'A', '01': 'T', '00': 'C', '11': 'G'},
           6: {'10': 'A', '01': 'T', '11': 'C', '00': 'G'},
           7: {'11': 'A', '00': 'T', '01': 'C', '10': 'G'},
           8: {'11': 'A', '00': 'T', '10': 'C', '01': 'G'},
        } 
       return encoding_rules[rule_index]
      
def dna_encoding(p6):
    u, v, channels, w = p6.shape

    encoded_dna = np.empty((u, v, channels, w//2), dtype='U1')
    for i in range(u):
        #l_i = np.mod(np.sum(p6[i]), 8) + 1  # Calculate L(i)
        l_i=1
        encoding_rule = dna_encoding_rules(l_i)
        for j in range(v):
          for c in range(channels):
            for k in range(0, w, 2):
                 value1 = p6[i, j, c, k]
                 value2 = p6[i, j, c, k + 1]
                 pair = f"{value1}{value2}"  # Combine two values into one pair
                 encoded_dna[i, j, c, k // 2] = encoding_rule[pair]

    return encoded_dna
transcription_table = {
    'A': 'U',
    'T': 'A',
    'C': 'G',
    'G': 'C'
}
def dna_transcription(p7):
    # Apply the transcription rules using vectorization
    p8 = np.vectorize(transcription_table.get)(p7)

    return p8
def generate_sequences_Y(p0, q0, d, u):
    Y = []
    Z = []
    p=p0
    q=q0
    for i in range(u*4):
        p1 = np.sin(np.pi * d * (q + 3) * p * (1 - p))
        q1 = np.sin(np.pi * d * (p + 3) * q * (1 - q))
        Y.append(p1)
        p = p1
        
        Z.append(q1)
        q = q1
       
        
    return Y
def mutation_rules():
    return {
        0: {'A': 'A', 'U': 'U', 'G': 'G', 'C': 'C'},
        1: {'A': 'U', 'U': 'A', 'G': 'C', 'C': 'G'},
        2: {'A': 'G', 'U': 'C', 'G': 'A', 'C': 'U'},
        3: {'A': 'C', 'U': 'G', 'G': 'U', 'C': 'A'}
    }

def rna_mutation(p8, Y):
    u, v, channels, w = p8.shape
    # Reshape Y to match the shape of p8
    Y_reshaped = Y.reshape((u, v,channels, w))
    
    p9 = np.empty_like(p8, dtype='U1')

    # Convert Y2 to integer Y1
    Y1 = np.floor(np.mod(Y_reshaped * 10**5, 4)).astype(int)

    mutation_rules_dict = mutation_rules()

    # Use vectorized operations for efficient mutation
    mutation_func = np.vectorize(lambda mode, base: mutation_rules_dict[mode].get(base, base))
    p9 = mutation_func(Y1, p8)

    return p9 
def translation_rules():
    return {
        'A': 'U',
        'U': 'A',
        'C': 'G',
        'G': 'C'
    }
def rna_translation(p9):
    translation_rules_dict = translation_rules()

    # Use vectorized operations for efficient translation
    translation_func = np.vectorize(lambda base: translation_rules_dict[base])
    p10 = translation_func(p9)

    return p10  
def generate_sequences_Z(p0, q0, d, u):
    Y = []
    Z1 = []
    Z=[]
    p=p0
    q=q0
    for i in range(u*4):
        p1 = np.sin(np.pi * d * (q + 3) * p * (1 - p))
        q1 = np.sin(np.pi * d * (p + 3) * q * (1 - q))
        Y.append(p1)
        p = p1
        Z1.append(q1)
        q = q1
       
    for i in range(u):
        Z.append(Z1[i]) 
    return Z 
def rna_encoding(Z, p10, u):
    # Transform Z into the range of 0-255
    Z_prime = np.floor(np.mod(Z * 10**5, 256)).astype(np.uint8)

    # Reshape Z_prime into a 2D array
    Z_reshaped = Z_prime.reshape(p10.shape[0], p10.shape[1],p10.shape[2])

    # Create bit planes
    rna_bit_planes = np.unpackbits(np.expand_dims(Z_reshaped, axis=-1), axis=-1)
    # Reshape the bit planes to form a 3D array
    rna_bit_planes_4d = rna_bit_planes.reshape(Z_reshaped.shape + (8,))

    # Apply encoding rules
    encoding_rules = {
        '00': 'A',
        '11': 'U',
        '01': 'C',
        '10': 'G'
    }
   # Initialize a 3D array to store the RNA sequence with the desired shape
    rna_sequence_4d = np.empty((rna_bit_planes_4d.shape[0], rna_bit_planes_4d.shape[1],rna_bit_planes_4d.shape[2], 4), dtype='U1')
    # Iterate through the 3D array
    for i in range(rna_bit_planes_4d.shape[0]):
        for j in range(rna_bit_planes_4d.shape[1]):
             for c in range(rna_bit_planes_4d.shape[2]):
              # Pair 2-bits and apply encoding rules
               for k in range(0, rna_bit_planes_4d.shape[3], 2):
                bit_pair = ''.join(map(str, rna_bit_planes_4d[i, j, c, k:k+2]))
                rna_sequence_4d[i, j, c, k:k+2] = list(encoding_rules[bit_pair])

    return rna_sequence_4d
xor_truth_table = {
    'A': {'A': 'A', 'U': 'U', 'C': 'C', 'G': 'G'},
    'U': {'A': 'U', 'U': 'A', 'C': 'G', 'G': 'C'},
    'C': {'A': 'C', 'U': 'G', 'C': 'A', 'G': 'U'},
    'G': {'A': 'G', 'U': 'C', 'C': 'U', 'G': 'A'}
}
def rna_computing(encoded_array, p10):
    # Assuming the first value of P11 is the same as the encoded array
    P11 = np.zeros_like(p10, dtype=np.object_)
    #P11[0] = encoded_array[0]

    u = P11.shape[0]

    for j in range(0, u):
        # Vectorized XOR operation using the truth table
        P11[j] = np.vectorize(lambda x, y: xor_truth_table[x][y])(encoded_array[j], p10[j])

    return P11
def binary_matrix_to_decimal(matrix):
    # Reshape the matrix to a 2D array
    flattened_array = matrix.reshape(-1, 8)

    # Convert each 8-bit binary value to decimal
    decimal_matrix = np.zeros((len(flattened_array),), dtype=int)
    for i, binary_value in enumerate(flattened_array):
        binary_string = ''.join(map(str, binary_value))
        decimal_matrix[i] = int(binary_string, 2)

    # Reshape the result back to the original 2D shape
    decimal_matrix = decimal_matrix.reshape(matrix.shape[:-1])

    return decimal_matrix
#Decryption part 
def rna_computing_reverse(encoded_array, p11):
# Assuming the first value of P10 is the same as the encoded array
    P10 = np.zeros_like(p11, dtype=np.object_)
    P10[0] = encoded_array[0]

    u = P10.shape[0]

    for j in range(0, u):
        # Vectorized XOR operation using the truth table
        P10[j] = np.vectorize(lambda x, y: xor_truth_table[x][y])(encoded_array[j], p11[j])

    return P10
def reverse_rna_translation(p11):
    reverse_translation_rules = {
        'U': 'A',
        'A': 'U',
        'G': 'C',
        'C': 'G'
    }

    # Use vectorized operations for efficient reverse translation
    reverse_translation_func = np.vectorize(lambda base: reverse_translation_rules[base])
    p10 = reverse_translation_func(p11)

    return p10
def reverse_mutation_rules():
    return {
        0: {'A': 'A', 'U': 'U', 'G': 'G', 'C': 'C'},
        1: {'A': 'U', 'U': 'A', 'G': 'C', 'C': 'G'},
        2: {'A': 'G', 'U': 'C', 'G': 'A', 'C': 'U'},
        3: {'A': 'C', 'U': 'G', 'G': 'U', 'C': 'A'}
    }

def reverse_rna_mutation(p10, Y):
    u, v, channels, w = p10.shape
    # Reshape Y to match the shape of p10
    Y_reshaped = Y.reshape((u, v, channels, w))

    # Convert Y2 to integer Y1
    Y1 = np.floor(np.mod(Y_reshaped * 10**5, 4)).astype(int)

    reverse_mutation_rules_dict = reverse_mutation_rules()

    # Use vectorized operations for efficient reverse mutation
    reverse_mutation_func = np.vectorize(lambda base, mode: reverse_mutation_rules_dict[mode].get(base, base))
    p9 = reverse_mutation_func(p10, Y1)

    return p9
reverse_transcription_table_rna_to_dna = {
    'U': 'A',
    'A': 'T',
    'G': 'C',
    'C': 'G'
}

def reverse_rna_transcription(p9):
    # Apply the reverse transcription rules using vectorization
    p8 = np.vectorize(reverse_transcription_table_rna_to_dna.get)(p9)

    return p8
def dna_decoding(encoded_dna, rule_index):
    decoding_rules = {
        1: {'A': '00', 'T': '11', 'C': '01', 'G': '10'},
        2: {'A': '00', 'T': '11', 'C': '10', 'G': '01'},
        3: {'A': '01', 'T': '10', 'C': '00', 'G': '11'},
        4: {'A': '01', 'T': '10', 'C': '11', 'G': '00'},
        5: {'A': '10', 'T': '01', 'C': '00', 'G': '11'},
        6: {'A': '10', 'T': '01', 'C': '11', 'G': '00'},
        7: {'A': '11', 'T': '00', 'C': '01', 'G': '10'},
        8: {'A': '11', 'T': '00', 'C': '10', 'G': '01'},
    }
    u, v, channels,  w_half = encoded_dna.shape

    decoded_dna = np.empty((u, v , channels, w_half * 2), dtype=int)
    for i in range(u):
        for j in range(v):
          for c in range(channels):
            for k in range(w_half):
                pair = encoded_dna[i, j, c, k]
                values = decoding_rules[rule_index][pair]
                decoded_dna[i, j, c, 2 * k:2 * k + 2] = [int(bit) for bit in values]

    return decoded_dna
#Code for encryption
def encryption_function(image):
    image_hash, hash_blocks = sha_256(image)
    h1, h2, h3, h4, h5, h6 = calculate_intermediate_params(external_keys, hash_blocks)
    x0,y0,z0,p0,q0=calculate_initial_values(h1,h2,h3,h4,h5,h6)
    a,b,c,d=calculate_chaotic_system_parameters(h1,h2,h3,h4,h5,h6)
    rgb_array = np.array(image)
    # Create bit planes for each channel
    bit_planes = np.unpackbits(rgb_array, axis=-1)
    # Reshape the bit planes to form a 4D array with dimensions (height, width, channels=3, 8)
    bit_planes_4d = bit_planes.reshape(rgb_array.shape + (8,))
    u=bit_planes_4d.shape[0] * bit_planes_4d.shape[1] *3
    X1, X2, X3= generate_sequences_3d_sine_chaos(x0, y0, z0, a, b, c, u)
    # Shuffle the binary values in the 3D matrix
    shuffled_bit_planes_4d = 1 - bit_planes_4d
    encoded_matrix = dna_encoding(shuffled_bit_planes_4d)
    dna_transcription_matrix = dna_transcription(encoded_matrix)
    Y= generate_sequences_Y(p0,q0,d,u)
    rna_mutation_matrix = rna_mutation(dna_transcription_matrix, np.array(Y))
    rna_translation_matrix = rna_translation(rna_mutation_matrix)
    Z= generate_sequences_Z(p0,q0,d,u)
    encoded_array = rna_encoding(np.array(Z),rna_translation_matrix,u)
    rna_computing_matrix = rna_computing(encoded_array,rna_translation_matrix)
    decoding_rules = {
    'A': '00',
    'U': '11',
    'C': '01',
    'G': '10'
    }
    # Initialize output 4D array for RGB
    encrypted_matrix_shape = (rna_computing_matrix.shape[0], rna_computing_matrix.shape[1], rna_computing_matrix.shape[2], 8)
    encrypted_matrix = np.zeros(encrypted_matrix_shape, dtype=int)
    
    # Map each base to binary for each channel
    for i in range(rna_computing_matrix.shape[0]):
        for j in range(rna_computing_matrix.shape[1]):
            for c in range(rna_computing_matrix.shape[2]):
                for k, base in enumerate(rna_computing_matrix[i, j, c]):
                    encrypted_matrix[i, j, c, k*2:k*2+2] = [int(x) for x in decoding_rules[base]]
    final_encrypted_matrix = binary_matrix_to_decimal(encrypted_matrix)                
    return encrypted_matrix,final_encrypted_matrix,rna_translation_matrix
def decryption_function(image,encrypted_matrix,rna_translation_matrix):
    image_hash, hash_blocks = sha_256(image)
    h1, h2, h3, h4, h5, h6 = calculate_intermediate_params(external_keys, hash_blocks)
    x0,y0,z0,p0,q0=calculate_initial_values(h1,h2,h3,h4,h5,h6)
    a,b,c,d=calculate_chaotic_system_parameters(h1,h2,h3,h4,h5,h6)
    rgb_array = np.array(image)
    # Create bit planes for each channel
    bit_planes = np.unpackbits(rgb_array, axis=-1)
    # Reshape the bit planes to form a 4D array with dimensions (height, width, channels=3, 8)
    bit_planes_4d = bit_planes.reshape(rgb_array.shape + (8,))
    u=bit_planes_4d.shape[0] * bit_planes_4d.shape[1] *3
    X1, X2, X3= generate_sequences_3d_sine_chaos(x0, y0, z0, a, b, c, u)
    Y= generate_sequences_Y(p0,q0,d,u)
    Z= generate_sequences_Z(p0,q0,d,u)
    encoding_rules = {
    '00': 'A',
    '11': 'U',
    '01': 'C',
    '10': 'G'
    }

    # Assuming 'encrypted_matrix' is your matrix with shape (163, 310, 8)
    
    # Initialize a 3D array to store the decoded RNA sequence
    encoded_rna_matrix = np.empty((encrypted_matrix.shape[0], encrypted_matrix.shape[1], encrypted_matrix.shape[2], encrypted_matrix.shape[3]//2), dtype='U1')
    
    # Iterate through the 3D array
    for i in range(encrypted_matrix.shape[0]):
        for j in range(encrypted_matrix.shape[1]):
             for c in range(encrypted_matrix.shape[2]):
              # Pair 2-bits and apply decoding rules
               bit_pairs = ''.join(map(str, encrypted_matrix[i, j, c, :]))
               encoded_rna_matrix[i, j, c, :] = [encoding_rules[bit_pairs[k:k+2]] for k in range(0, len(bit_pairs), 2)]
    
    encoded_array = rna_encoding(np.array(Z),rna_translation_matrix,u)
    decoded_rna_computing_matrix = rna_computing_reverse(encoded_array,encoded_rna_matrix)
    reversed_rna_translated_matrix = reverse_rna_translation(decoded_rna_computing_matrix)
    reverse_rna_mutation_matrix = reverse_rna_mutation(reversed_rna_translated_matrix,np.array(Y))
    reverse_transcription_matrix = reverse_rna_transcription(reverse_rna_mutation_matrix)
    rule_index=1
    decoded_dna_matrix = dna_decoding(reverse_transcription_matrix,rule_index)
    reshuffled_bit_planes_3d = 1 - decoded_dna_matrix
    final_decrypted_matrix = binary_matrix_to_decimal(reshuffled_bit_planes_3d)
    return final_decrypted_matrix


if __name__ == "__main__":
    main()