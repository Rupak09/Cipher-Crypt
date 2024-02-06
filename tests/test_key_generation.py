# import numpy as np

# def generate_key_parameters(hash_value, c1, c2, c3, c4, c5, c6):
#     """
#     Generate the key parameters for the encryption process.
    
#     :param hash_value: A 256-bit hash value of the plain image divided into 8-bit blocks (32 blocks).
#     :param c1, c2, c3, c4, c5, c6: The external keys provided by the user.
#     :return: Initial values and control parameters for the chaotic maps.
#     """


    
#     # Split the hash into 32 blocks of 8-bits each
#     k = [int(hash_value[i:i+2], 16) for i in range(0, len(hash_value), 2)]

#     # Calculate h1 to h6 using the provided formulae
#     h1 = (c1 + (k[0] ^ k[1] ^ k[2] ^ k[3] ^ k[4])) / 256
#     h2 = (c2 + (k[5] ^ k[6] ^ k[7] ^ k[8] ^ k[9])) / 256
#     h3 = (c3 + (k[10] ^ k[11] ^ k[12] ^ k[13] ^ k[14])) / 256
#     h4 = (c4 + (k[15] ^ k[16] ^ k[17] ^ k[18] ^ k[19])) / 256
#     h5 = (c5 + (k[20] ^ k[21] ^ k[22] ^ k[23] ^ k[24] ^ k[25])) / 256
#     h6 = (c6 + (k[26] ^ k[27] ^ k[28] ^ k[29] ^ k[30] ^ k[31])) / 256

#     # Compute the initial values for the chaotic maps
#     x0 = np.mod((h1 + h2 + h5) * 10**8, 256) / 255.0
#     y0 = np.mod((h3 + h4 + h6) * 10**8, 256) / 255.0
#     z0 = np.mod((h1 + h2 + h3 + h4) * 10**8, 256) / 255.0
#     p0 = np.mod((h1 + h2 + h3) * 10**8, 256) / 255.0
#     q0 = np.mod((h4 + h5 + h6) * 10**8, 256) / 255.0

#     # Calculate the control parameters for the chaotic maps
#     sum_h = h1 + h2 + h3 + h4 + h5 + h6
#     a = np.mod(((h1 + h2) / sum_h) * 100, 3) + 1
#     b = np.mod(((h3 + h4) / sum_h) * 100, 3) + 1
#     c = np.mod(((h5 + h6) / sum_h) * 100, 3) + 1
#     mu = np.mod((h1 + h2 + h3) / sum_h, 1)

#     return x0, y0, z0, p0, q0, a, b, c, mu

