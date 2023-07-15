# Khai báo thư viện
import numpy as np
from encoder_2000 import lfilter
from decoder import deserialize
from fractions import Fraction


# Phép biến đổi Discrete Wavelet Transform
def idwt(filtered_image, quantization_Array):
    for i in range(0, len(filtered_image)):
        filtered_image[i] = filtered_image[i]*quantization_Array[i]
    LPF = [0.5, 1, 0.5]
    LPF_center = 1
    HPF = [-0.125, -0.25, 0.75, -0.25, -0.125]
    HPF_center = 1
    LowPass1_rows = np.zeros(
        (filtered_image[0].shape[0]*2, filtered_image[0].shape[1]))
    LowPass1_rows[::2, :] = filtered_image[0]
    LowPass2_rows = np.zeros(
        (filtered_image[0].shape[0]*2, filtered_image[0].shape[1]))
    LowPass2_rows[::2, :] = filtered_image[1]
    HighPass1_rows = np.zeros(
        (filtered_image[0].shape[0]*2, filtered_image[0].shape[1]))
    HighPass1_rows[::2, :] = filtered_image[2]
    HighPass2_rows = np.zeros(
        (filtered_image[0].shape[0]*2, filtered_image[0].shape[1]))
    HighPass2_rows[::2, :] = filtered_image[3]
    for i in range(0, LowPass1_rows.shape[1]):
        LowPass1_rows[:, i] = lfilter(LPF, LowPass1_rows[:, i], LPF_center)
        LowPass2_rows[:, i] = lfilter(HPF, LowPass2_rows[:, i], HPF_center)
        HighPass1_rows[:, i] = lfilter(LPF, HighPass1_rows[:, i], LPF_center)
        HighPass2_rows[:, i] = lfilter(HPF, HighPass2_rows[:, i], HPF_center)
    LowPass_temp = LowPass1_rows+LowPass2_rows
    LowPass_rows = np.zeros(
        (filtered_image[0].shape[0]*2, filtered_image[0].shape[1]*2))
    LowPass_rows[:, ::2] = LowPass_temp
    HighPass_temp = HighPass1_rows+HighPass2_rows
    HighPass_rows = np.zeros(
        (filtered_image[0].shape[0]*2, filtered_image[0].shape[1]*2))
    HighPass_rows[:, ::2] = HighPass_temp
    for i in range(0, LowPass_rows.shape[0]):
        HighPass_rows[i, :] = lfilter(HPF, HighPass_rows[i, :], HPF_center)
        LowPass_rows[i, :] = lfilter(LPF, LowPass_rows[i, :], LPF_center)
    image_array = HighPass_rows + LowPass_rows
    image_array = image_array.clip(0, 255).astype(np.uint8)
    return image_array


# Ngược phép biến đổi Discrete Wavelet Transform
def dwt_deserialize(serialized, length, quantization_Array, aspect_ratio):
    quarter_len = int(len(serialized)/4)
    images = []
    for i in range(4):
        if isinstance(length[i], list):
            images.append(dwt_deserialize(serialized[quarter_len*i:
                                                     quarter_len*i +
                                                     quarter_len],
                                          length[i], quantization_Array,
                                          aspect_ratio))
        else:
            rows = int(np.sqrt((quarter_len * aspect_ratio).numerator))
            columns = int(np.sqrt((quarter_len / aspect_ratio).numerator))
            images.append(deserialize(serialized[quarter_len*i:
                                                 quarter_len*i + quarter_len],
                                      1, rows,
                                      columns).squeeze())
    return idwt(images, quantization_Array)
