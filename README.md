
# Multiple Image Encryption Using Central Dogma, Hash SHA256, and Chaotic Systems

The project aims to create a secure encryption algorithm based on the central dogma, catering to the encryption and decryption of multiple images. 
Additionally, the integration of chaotic systems enhances the genomic coding process, fortifying the encryption with an added layer of complexity for heightened security. This unique fusion of central dogma principles and chaotic dynamics strives to achieve innovative and robust image protection mechanisms.


## Introduction

The central dogma of molecular biology is a concept that describes the flow of genetic information within a biological system, particularly in living cells. The central dogma was first proposed by Francis Crick in 1957 and later elaborated in 1970. It outlines the three main processes involved in the expression of genetic information:

1.	Replication: The process by which a cell duplicates its DNA to pass on genetic information to its offspring. During replication, the DNA molecule unwinds, and each strand serves as a template for the synthesis of a new complementary strand, resulting in two identical DNA molecules.
2.	Transcription: The synthesis of RNA from a DNA template. In this process, a specific segment of DNA (a gene) serves as a template for the synthesis of a complementary RNA molecule. The RNA molecule, known as messenger RNA (mRNA), carries the genetic information from the DNA to the ribosomes, where protein synthesis will occur.
3.	Translation: The process in which the information carried by mRNA is used to build a corresponding protein. During translation, the mRNA is read by ribosomes, and transfer RNA (tRNA) molecules bring amino acids to the ribosome in a specific order dictated by the mRNA sequence. This sequence of amino acids forms a protein

## Features of the algorithm

Image Upload:
The code allows users to upload multiple images with specific dimensions (256x256 or 512x512 pixels). 

Key Points about the Image Upload : 

Validation:
Images are validated to ensure they meet the required dimensions. An error message is displayed if the dimensions are incorrect.

Canvas Preparation:
Images are organized onto canvases based on their dimensions: large images (512x512) and small images (256x256).
Canvases are created to display multiple images side by side in the Streamlit app.

Encryption and Decryption Process:
The encryption and decryption processes involve several steps, combining cryptographic techniques and chaotic systems. 

Here's an overview:

Hashing (SHA-256):
Images are hashed using the SHA-256 algorithm to generate a unique identifier for each image.

Chaotic System:
Chaotic systems with parameters derived from the SHA-256 hash are used to generate sequences (X1, X2, X3) for encryption and decryption.
Parameters like h1, h2, h3, h4, h5, and h6 are calculated from the hash.

DNA Encoding:
The binary representation of image pixel values is encoded using DNA encoding rules.
DNA encoding converts binary values (00, 01, 10, 11) into DNA base pairs (A, C, G, T).


RNA Mutation and Translation:
The encoded DNA sequences undergo RNA mutation based on chaotic system-generated sequences (Y).
RNA translation transforms the mutated RNA sequences into a different set of base pairs.


RNA Computing:
XOR operations are performed between the translated RNA sequence and chaotic system-generated sequences.

Decoding to Image:
The final RNA sequence is decoded back to binary values and then to pixel values, reconstructing the encrypted image.
The decrypted image is visualized alongside the original image for comparison.

Chaotic System:
The chaotic system contributes to the pseudo-randomness in the encryption and decryption processes.

 Key points about the chaotic system:

Parameter Calculation:
Intermediate parameters (h1, h2, h3, h4, h5, h6) for the chaotic system are calculated based on SHA-256 hash values.

Initial Values:
Initial values (x0, y0, z0, p0, q0) for the chaotic system are derived from the intermediate parameters.

Chaotic System Equations:
The chaotic system equations involve iterations using sine functions and parameters (a, b, c, d) derived from the hash.

Sequences Generation:
Sequences (X1, X2, X3) are generated from the chaotic system, contributing to the encryption and decryption processes.

## Features of UI

- Light/dark mode toggle
- Live previews
- Fullscreen mode
- Cross-platform


## Conclusion

To improve security and encryption efficiency, this project proposes a Multiple Image Encryption algorithm based on genetic central dogma and 3D bit planes, which are related to plain images. Meanwhile, the proposed algorithm is based on the complexity of the genetic central dogma and 3D bit planes, so our algorithm is very secure in theory. 

Experimental results and algorithm analyses demonstrated that the proposed algorithm is efficient and sufficiently secure against most common attacks, such as brute-force attacks and statistical analysis attacks. Therefore, our algorithm is meant to be an excellent candidate to ensure the network security of multiple images in the fields of military, medical, educational, etc. The precision of data can affect the quality of chaotic sequences, even the algorithm's performance. We will pay more attention to the chaotic degradation problem in the future. Meanwhile, we will further optimize our algorithm and improve the speed through hardware.




# For Open-source : 
### 1. Browse or Create Issues

- **Browse Existing Issues:** Take a look at the [Existing Issues](https://github.com/Rupak09/Cipher-Crypt/issues) to find tasks, bugs, or features that you can contribute to.
- **Create New Issues:** If you encounter a bug, have a feature request, or want to suggest an enhancement, please create a new issue using the [Issue Tracker](https://github.com/Rupak09/Cipher-Crypt/issues).

### 2. Fork the Repository

Click on the uppermost "Fork" button on the [Cipher Crypt repository](https://github.com/Rupak09/Cipher-Crypt.git).

### 3. Clone your Forked Copy

Clone your forked copy of the project to your local machine.


git clone https://github.com/<your_user_name>/Cipher-Crypt.git

### 4. Install project dependencies:
```bash  
pip install -r requirements.txt
```

### 5. Run locally:
```bash  
streamlit run app1.py
```

for further queries contact: rupakr31@gmail.com

