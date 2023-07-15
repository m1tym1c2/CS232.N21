import numpy as np
from PIL import Image
import encoder as e
import decoder as d


# Ma trận lượng tử
table_8_low = [[1,  1,  1,  1,  1,  2,  2,  4],
               [1,  1,  1,  1,  1,  2,  2,  4],
               [1,  1,  1,  1,  2,  2,  2,  4],
               [1,  1,  1,  1,  2,  2,  4,  8],
               [1,  1,  2,  2,  2,  2,  4,  8],
               [2,  2,  2,  2,  2,  4,  8,  8],
               [2,  2,  2,  4,  4,  8,  8,  16],
               [4,  4,  4,  4,  8,  8, 16,  16]]
table_8_high = [[1,    2,    4,    8,    16,   32,   64,   128],
                [2,    4,    4,    8,    16,   32,   64,   128],
                [4,    4,    8,    16,   32,   64,   128,  128],
                [8,    8,    16,   32,   64,   128,  128,  256],
                [16,   16,   32,   64,   128,  128,  256,  256],
                [32,   32,   64,   128,  128,  256,  256,  256],
                [64,   64,   128,  128,  256,  256,  256,  256],
                [128,  128,  128,  256,  256,  256,  256,  256]]

table_16_low = np.repeat(np.repeat(table_8_low, 2, axis=0), 2, axis=1)
table_16_high = np.repeat(np.repeat(table_8_high, 2, axis=0), 2, axis=1)


# Quá trình nén
def encode(image, box_size, quantization_table):
    image_array = e.reshape_image(image, box_size)
    sub_images, n_rows, n_cols = e.get_sub_images(image_array, box_size)

    dct_values = e.apply_dct_to_all(sub_images)

    quantized = e.quantize(dct_values, quantization_table)

    serialized = e.serialize(quantized)

    rlcoded = e.run_length_code(serialized)

    huffcoded, code_dict = e.huffman_encode(rlcoded)

    return huffcoded, code_dict, n_rows, n_cols


# Quá trình giải nén
def decode(huffcoded, code_dict, n_rows, n_cols, box_size, quantization_table):
    rlcoded = d.huffman_decode(huffcoded, code_dict)

    serialized = d.run_length_decode(rlcoded)

    quantized = d.deserialize(serialized, n_rows*n_cols, box_size, box_size)

    subdivded_dct_values = d.dequantize(quantized, quantization_table)

    sub_images = d.apply_idct_to_all(subdivded_dct_values)

    reconstructed_image = d.get_reconstructed_image(sub_images, n_rows, n_cols,
                                                    box_size)
    return reconstructed_image
