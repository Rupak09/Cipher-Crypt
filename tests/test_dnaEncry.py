









































































































































































# from test_hash import compute_image_hash
# from tests.test_key_generation import generate_key_parameters
# from test_chaoticEn import generate_chaotic_sequences
# #import numpy as np

# # Path to your image file
# image_path = 'path_to_your_image.jpg'

# # Compute the SHA-256 hash of the image
# image_hash = compute_image_hash(image_path)

# # External keys provided by the user
# c_values = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]

# # Generate initial conditions and control parameters for the chaotic maps
# x0, y0, z0, p0, q0, a, b, c, mu = generate_key_parameters(image_hash, *c_values)

# # Generate the chaotic sequences
# u = 256  # The 'u' parameter should be determined based on your specific needs, like image size
# X1, Y1, Z1, P, Q = generate_chaotic_sequences(x0, y0, z0, p0, q0, a, b, c, mu, u)

# # Use the chaotic sequences for the encryption process
# # This part will depend on how you want to apply the sequences
# # For example, you may scramble the bit planes, encode them to DNA/RNA, etc.