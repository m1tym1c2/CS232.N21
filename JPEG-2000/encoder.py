# Khai báo thư viện
import numpy as np
import pandas as pd
from huffman import encode as h_encode


# reshape ảnh đầu vào
def reshape_image(image, box_size=8):
    n_rows = np.int(np.floor(image.size[0]/box_size))
    n_cols = np.int(np.floor(image.size[1]/box_size))
    image = image.resize((n_rows*box_size, n_cols*box_size))
    image_array = np.asarray(image)
    return image_array


# Lấy các vùng ảnh
def get_sub_images(image_array, box_size=8):
    n_rows = np.int(image_array.shape[0]/box_size)
    n_cols = np.int(image_array.shape[1]/box_size)
    image_blocks = np.asarray([np.zeros((box_size, box_size), dtype='uint8')
                               for i in range(n_rows*n_cols)], dtype='uint8')
    c = 0
    for i in range(n_rows):
        for j in range(n_cols):
            image_blocks[c] = image_array[i*box_size: i*box_size+box_size,
                                          j*box_size:j*box_size+box_size]
            c += 1
    return image_blocks, n_rows, n_cols


# Tổng hợp dct basic
def __basis_generator(b=8):
    i = j = np.arange(b)
    basis_cache = {}

    def helper(u, v):
        base = basis_cache.get((u, v), None)
        if base is None:
            base = np.dot(np.cos((2*i + 1) * u * np.pi / (2*b)).reshape(-1, 1),
                          np.cos((2*j + 1) * v * np.pi / (2*b)).reshape(1, -1))
            basis_cache[(u, v)] = base
        return base
    return lambda u, v: helper(u, v)


# Phép biến đổi Discrete cosine transform
def dct(sub_image, basis):
    b = sub_image.shape[0]

    def scale(idx):
        return 2 if idx == 0 else 1
    outblock = np.zeros((b, b))
    for u in range(b):
        for v in range(b):
            outblock[u, v] =\
                np.sum(basis(u, v) * sub_image) / \
                (b**2/4) / scale(u) / scale(v)
    return outblock


# Áp dụng Phép biến đổi Discrete cosine transform
def apply_dct_to_all(subdivded_image):
    basis = __basis_generator(subdivded_image.shape[1])
    dct_divided_image = np.array([dct(sub_image, basis)
                                  for sub_image in subdivded_image])
    return dct_divided_image


# Lượng tử hóa
def quantize(dct_divided_image, quantization_table):
    return np.array([(sub_image / quantization_table).round().astype(int)
                     for sub_image in dct_divided_image])


# Phương pháp zigzag
def generate_indicies_zigzag(rows=8, cols=8):
    i = j = 0
    going_up = True
    forReturn = [[0, 0] for i in range(rows*cols)]
    for step in range(rows*cols):
        i_new, j_new = (i-1, j+1) if going_up else (i+1, j-1)
        forReturn[step] = [i, j]
        if i_new >= rows:
            j += 1
            going_up = not going_up
        elif j_new >= cols:
            i += 1
            going_up = not going_up
        elif i_new < 0:
            j += 1
            going_up = not going_up
        elif j_new < 0:
            i += 1
            going_up = not going_up
        elif i_new == rows and j_new == cols:
            assert step == (rows*cols - 1)
        else:
            i, j = i_new, j_new
    return forReturn


def serialize(quantized_dct_image, jpeg2000=False):
    if not jpeg2000:
        rows, columns = quantized_dct_image[0].shape
        output = np.zeros(len(quantized_dct_image)*rows*columns, dtype='int')
        step = 0
        for matrix in quantized_dct_image:
            for i, j in generate_indicies_zigzag(rows, columns):
                output[step] = matrix[i, j]
                step += 1
    else:
        rows, columns = quantized_dct_image.shape
        output = np.zeros(rows*columns, dtype='int')
        step = 0
        for i, j in generate_indicies_zigzag(rows, columns):
            output[step] = quantized_dct_image[i, j]
            step += 1

    return output


# Thuật toán RLC
def run_length_code(serialized):
    max_len = 256
    rlcoded = []
    zero_count = 0
    for number in serialized:
        if number == 0:
            zero_count += 1
            if zero_count == max_len:
                rlcoded.append(0)
                rlcoded.append(zero_count-1)
                zero_count = 0
        else:
            if zero_count > 0:
                rlcoded.append(0)
                rlcoded.append(zero_count-1)
                zero_count = 0
            rlcoded.append(number)
    if zero_count > 0:
        rlcoded.append(0)
        rlcoded.append(zero_count-1)
    return np.asarray(rlcoded)


# Mã hóa huffman
def huffman_encode(rlcoded):
    counts_dict = dict(pd.Series(rlcoded).value_counts())
    code_dict = h_encode(counts_dict)
    huffcoded = ''.join([code_dict[i] for i in rlcoded])
    return huffcoded, code_dict
