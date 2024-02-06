# import hashlib

# def compute_image_hash(image_path):
#     """
#     Compute the SHA-256 hash of an image file.

#     :param image_path: Path to the image file.
#     :return: A hexadecimal string of the SHA-256 hash.
#     """
#     sha256 = hashlib.sha256()
#     with open(image_path, 'rb') as image_file:
#         while chunk := image_file.read(8192):
#             sha256.update(chunk)
#     return sha256.hexdigest()
