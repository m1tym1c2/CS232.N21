# Khai báo thư viện
import numpy as np
import pandas as pd
from encoder import serialize
from fractions import Fraction


# Phương pháp lifting trong biến đổi riêng thành phần
def lfilter(taps, array, filter_centre):
    arr = array.copy()
    left_pad_len = len(taps) - filter_centre - 1
    right_pad_len = filter_centre
    arr = np.concatenate(
        (array[1:1+left_pad_len][::-1], array,
         array[-right_pad_len-1:-1][::-1]))
    return np.convolve(arr, taps[::-1], 'valid')


# Checking
def check_image(image, depth):
    cols, rows = image.size
    divisor = 2**depth
    n_rows = round(rows/divisor) * divisor
    n_cols = round(cols/divisor) * divisor
    image = image.resize((n_cols, n_rows))
    image_array = np.asarray(image)
    return image_array, Fraction(n_rows, n_cols)


# Phép biến đổi Discrete wavelet transform
def dwt(image_array, quantization_Array):
    LPF = [-0.125, 0.25, 0.75, 0.25, -0.125]
    LPF_center = 2
    HPF = [-0.5, 1, -0.5]
    HPF_center = 2
    nrow, ncol = image_array.shape
    LL = np.zeros((nrow, ncol))
    LH = np.zeros((nrow, ncol))
    HL = np.zeros((nrow, ncol))
    HH = np.zeros((nrow, ncol))
    filtered_image = [LL, LH, HL, HH]
    LowPass_rows = np.zeros((nrow, ncol))
    HighPass_rows = np.zeros((nrow, ncol))
    for i in range(0, nrow):
        LowPass_rows[i, :] = lfilter(LPF, image_array[i, :], LPF_center)
        HighPass_rows[i, :] = lfilter(HPF, image_array[i, :], HPF_center)
    for i in range(0, len(filtered_image)):
        filtered_image[i] = filtered_image[i][:, ::2]
    for i in range(0, ncol):
        LL[:, i] = lfilter(LPF, LowPass_rows[:, i], LPF_center)
        LH[:, i] = lfilter(HPF, LowPass_rows[:, i], HPF_center)
        HL[:, i] = lfilter(LPF, HighPass_rows[:, i], LPF_center)
        HH[:, i] = lfilter(HPF, HighPass_rows[:, i], HPF_center)
    for i in range(0, len(filtered_image)):
        filtered_image[i] = filtered_image[i][::2, :]
        filtered_image[i] = np.round(
            filtered_image[i]/quantization_Array[i]).astype(int)
    return filtered_image


# Phép biến đổi Discrete wavelet transform trong phân mức cây tứ phân
def dwt_levels(filtered_image, levels, quantization_Array):
    assert len(levels) <= 4
    for level in levels:
        filtered_image[level[0]] = dwt(
            filtered_image[level[0]], quantization_Array)
        try:
            dwt_levels(filtered_image[level[0]],
                       level[1], quantization_Array)
        except IndexError:
            continue


# Áp dụng Phép biến đổi Discrete wavelet transform
def dwt_serialize(filtered_image, output, length):
    for i in filtered_image:
        if isinstance(i, list):
            output_temp, length_temp = dwt_serialize(i, [], [])
            output = output + output_temp
            length.append(length_temp)
        else:
            new_output = (serialize(i, True).tolist())
            output = output+new_output
            length = length+[len(new_output)]
    return output, length
