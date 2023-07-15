import numpy as np
from scipy.fftpack import dct, idct
from bitarray import bitarray
from PIL import Image
import json

def horizontal_prediction(matrix, x, y, macro_size):
    height, width = macro_size, macro_size
    result = np.zeros((height, width))

    for i in range(x, x + macro_size):
        for j in range(y, y + macro_size):
            if j == 0:
                result[i - x, j - y] = matrix[i, j]
            elif j == y:
                result[i - x, j - y] = matrix[i, j - 1]
            else:
                result[i-x, j-y] = result[i-x, j-y-1]

    return result

def vertical_prediction(matrix, x, y, macro_size):
    height, width = macro_size, macro_size
    result = np.zeros((height, width))

    for i in range(x, x + macro_size):
        for j in range(y, y + macro_size):
            if i == 0:
                result[i-x, j-y] = matrix[i, j]
            elif i == x:
                result[i - x, j - y] = matrix[i - 1, j]
            else:
                result[i-x, j-y] = result[i-x-1, j-y]

    return result


def dc_prediction(matrix, x, y, macro_size):
    if x == 0 and y == 0:
        return np.mean(matrix[:macro_size, :macro_size]) * np.ones((macro_size, macro_size))

    top_row = matrix[x - 1, y:y + macro_size] if x > 0 else matrix[x, y:y + macro_size]
    left_column = matrix[x:x + macro_size, y - 1] if y > 0 else matrix[x:x + macro_size, y]
    mean_value = np.mean(np.concatenate((top_row, left_column)))
    return mean_value * np.ones((macro_size, macro_size))

def true_motion_prediction(matrix, x, y, macro_size):
    height, width = matrix.shape
    result = np.zeros((macro_size, macro_size))

    for i in range(x, x + macro_size):
        for j in range(y, y + macro_size):
            if i == x and j == y:
                result[i-x, j-y] = matrix[i, j]
            elif i == x:
                result[i-x, j-y] = np.mean(matrix[i, y:j])
            elif j == y:
                result[i-x, j-y] = np.mean(matrix[x:i, j])
            else:
                result[i-x, j-y] = np.mean(matrix[x:i, j]) + np.mean(matrix[i, y:j]) - np.mean(matrix[x:i, y:j])

    return result

def ssd(block1, block2):
    return np.sum((block1 - block2)**2)

def best_intra_prediction_mode(matrix, x, y, macro_size):
    best_ssd = float('inf')
    best_mode = None
    modes = ['H_PRED', 'V_PRED', 'DC_PRED', 'TM_PRED']
    for mode in modes:
        prediction = None

        if mode == 'H_PRED':
            prediction = horizontal_prediction(matrix, x, y, macro_size)
        elif mode == 'V_PRED':
            prediction = vertical_prediction(matrix, x, y, macro_size)
        elif mode == 'DC_PRED':
            prediction = dc_prediction(matrix, x, y, macro_size)
        elif mode == 'TM_PRED':
            prediction = true_motion_prediction(matrix, x, y, macro_size)

        block = matrix[x:x + macro_size, y:y + macro_size]
        mode_ssd = ssd(block, prediction)

        if mode_ssd < best_ssd:
            best_ssd = mode_ssd
            best_mode = mode

    return best_mode


def block_setting(img, block_size):
    blocks = []
    for i in range(0, img.shape[0], block_size):
        for j in range(0, img.shape[1], block_size):
            blocks.append(img[i:i + block_size, j:j + block_size])
    return blocks


def prediction(img, block_size):
    residuals = []
    for i in range(0, img.shape[0], block_size):
        for j in range(0, img.shape[1], block_size):
            if i + block_size - 1 > img.shape[0] or j + block_size - 1 > img.shape[1]:
                break
            else:
                best_mode = best_intra_prediction_mode(img, i, j, block_size)
                if best_mode == 'H_PRED':
                    predicted_block = horizontal_prediction(img, i, j, block_size)
                elif best_mode == 'V_PRED':
                    predicted_block = vertical_prediction(img, i, j, block_size)
                elif best_mode == 'DC_PRED':
                    predicted_block = dc_prediction(img, i, j, block_size)
                elif best_mode == 'TM_PRED':
                    predicted_block = true_motion_prediction(img, i, j, block_size)

                residuals.append(img[i:i + block_size, j:j + block_size] - predicted_block)

    return residuals


def transformation(residuals):
    transformed = []
    for block in residuals:
        transformed.append(dct(dct(block.T, norm='ortho').T, norm='ortho'))
    return transformed

def quantization(transformed_blocks, quality):
    quantized_blocks = []
    for block in transformed_blocks:
        quantization_table = create_quantization_table(quality)  # Tạo bảng lượng tử hóa
        quantized_block = np.round(block / quantization_table)  # Áp dụng lượng tử hóa
        quantized_blocks.append(quantized_block)
    return quantized_blocks

def create_quantization_table(quality):
    # Tạo bảng lượng tử hóa dựa trên chất lượng
    if quality < 1:
        quality = 1
    elif quality > 100:
        quality = 100

    # Bảng lượng tử hóa mặc định (chất lượng 50)
    default_quantization_table = np.array(
        [[16, 11, 10, 16, 24, 40, 51, 61],
         [12, 12, 14, 19, 26, 58, 60, 55],
         [14, 13, 16, 24, 40, 57, 69, 56],
         [14, 17, 22, 29, 51, 87, 80, 62],
         [18, 22, 37, 56, 68, 109, 103, 77],
         [24, 35, 55, 64, 81, 104, 113, 92],
         [49, 64, 78, 87, 103, 121, 120, 101],
         [72, 92, 95, 98, 112, 100, 103, 99]])

    if quality < 50:
        scale_factor = 5000 / quality
    else:
        scale_factor = 200 - 2 * quality

    quantization_table = np.floor((default_quantization_table * scale_factor + 50) / 100)

    # Đảm bảo các giá trị trong bảng không vượt quá giới hạn (1 đến 255)
    quantization_table = np.clip(quantization_table, 1, 255)

    return quantization_table.astype(int)


def zigzag_scanning(quantized_blocks):
    zigzag_blocks = []
    for block in quantized_blocks:
        zigzag_block = []
        size = block.shape[0]
        i, j = 0, 0
        for _ in range(size**2):
            if i < size and j < size:  # Kiểm tra phạm vi hợp lệ
                zigzag_block.append(block[i, j])
            if (i + j) % 2 == 0:  # Di chuyển lên
                if j < size - 1:
                    j += 1
                else:
                    i += 1
            else:  # Di chuyển xuống
                if i < size - 1:
                    i += 1
                else:
                    j += 1
        zigzag_blocks.append(np.array(zigzag_block))
    return zigzag_blocks

def run_length_encoding(zigzag_blocks):
    encoded_data = []
    for block in zigzag_blocks:
        encoded_block = []
        count = 0
        for value in block:
            if value != 0:
                encoded_block.extend([count, value])
                count = 0
            else:
                count += 1
        encoded_block.append(0)  # Kết thúc khối bằng 0
        encoded_data.append(encoded_block)
    return encoded_data

# Đọc ảnh và chuyển đổi thành ma trận numpy
image = Image.open('messi.jpg')
# image_array = np.array(image)
image_array = np.array(image.convert('L'))
# Tham số nén
block_size = 8
quality = 50  # Chất lượng nén (thay đổi theo nhu cầu)

# Chia ảnh thành các khối
blocks = block_setting(image_array, block_size)

# Dự đoán và tính toán các khối dư thừa
residuals = prediction(image_array, block_size)

# Biến đổi các khối dư thừa
transformed_blocks = transformation(residuals)

# Lượng tử hóa các khối biến đổi
quantized_blocks = quantization(transformed_blocks, quality)

# Quét Zigzag các khối lượng tử hóa
zigzag_blocks = zigzag_scanning(quantized_blocks)

# Mã hóa chạy động các khối Zigzag
encoded_data = run_length_encoding(zigzag_blocks)

print(encoded_data)
