from encoder import generate_indicies_zigzag, __basis_generator
import numpy as np
from huffman import decode as h_decode


# Thuật toán giải mã huffman
def huffman_decode(huffcoded, code_dict):
    return h_decode(huffcoded, code_dict)


# Thuật toán giải mã run length code (entropy)
def run_length_decode(rlcoded):
    serialized = []
    i = 0
    while i < len(rlcoded):
        if rlcoded[i] == 0:
            serialized.extend([0]*(rlcoded[i+1]+1))
            i += 2
        else:
            serialized.append(rlcoded[i])
            i += 1
    return np.asarray(serialized)


# Tái tạo lại cấu trúc
def deserialize(serialized, n_blocks, n_rows=8, n_cols=8):
    output = np.zeros((n_blocks, n_rows, n_cols), dtype=np.int16)
    step = 0
    for matrix in output:
        for i, j in generate_indicies_zigzag(n_rows, n_cols):
            matrix[i, j] = serialized[step]
            step += 1
    return output


# Giải lượng tử hóa
def dequantize(quantized, quantization_table):
    return np.array([block * quantization_table for block in quantized])


# Đảo ngược DCT
def idct(dct_values, basis):
    b = dct_values.shape[0]
    outblock = np.zeros((b, b))
    for x in range(b):
        for y in range(b):
            outblock = outblock + dct_values[x, y] * basis(x, y)
    return outblock


# Ứng dụng DCT
def apply_idct_to_all(subdivded_dct_values):
    basis = __basis_generator(subdivded_dct_values.shape[1])
    divided_image = np.array([idct(sub_image, basis) for
                              sub_image in subdivded_dct_values])
    return divided_image.clip(min=0, max=255).round().astype(np.uint8)


# Tái cấu trúc ảnh
def get_reconstructed_image(divided_image, n_rows, n_cols, box_size=8):
    image_reconstructed = np.zeros(
        (n_rows*box_size, n_cols*box_size), dtype=np.uint8)
    c = 0
    for i in range(n_rows):
        for j in range(n_cols):
            image_reconstructed[i*box_size: i*box_size+box_size,
                                j*box_size:j*box_size+box_size] = divided_image[c]
            c += 1
    return image_reconstructed
