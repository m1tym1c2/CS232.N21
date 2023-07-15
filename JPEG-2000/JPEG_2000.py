# Khai bóa thư viện
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import timeit
from PIL import Image
from encoder import reshape_image
from encoder_2000 import check_image, dwt
import main_2000 as m2


# Phương pháp so sánh
def rmse(im, reconstructed_image):
    error = im - reconstructed_image
    mse = np.sum(np.square(error)) / (im.shape[0] * im.shape[1])
    rmse = np.sqrt(mse)
    return rmse


im = Image.open('1.jpg')
im = im.convert("L")
im

levels = [
    [0,
     [[0]]
     ]
]


quantization_Array = [1, 2, 3, 4]


encoded = []
reconstructed = []
execution_time = []

start = timeit.default_timer()
huffcoded, code_dict, length, aspect_ratio = m2.encode(
    im, levels, quantization_Array)
encoded.append(huffcoded)
reconstructed.append(m2.decode(huffcoded, code_dict,
                     length, quantization_Array, aspect_ratio))
execution_time.append(timeit.default_timer() - start)

print("Execution time: ", execution_time[0])


quantization_Array = [1, 64, 128, 256]

start = timeit.default_timer()

huffcoded, code_dict, length, aspect_ratio = m2.encode(
    im, levels, quantization_Array)
encoded.append(huffcoded)
reconstructed.append(m2.decode(huffcoded, code_dict,
                     length, quantization_Array, aspect_ratio))
execution_time.append(timeit.default_timer() - start)

print("Execution time: ", execution_time[1])


levels = [[1, [[0]]]]

quantization_Array = [1, 2, 3, 4]

start = timeit.default_timer()

huffcoded, code_dict, length, aspect_ratio = m2.encode(
    im, levels, quantization_Array)
encoded.append(huffcoded)
reconstructed.append(m2.decode(huffcoded, code_dict,
                     length, quantization_Array, aspect_ratio))
execution_time.append(timeit.default_timer() - start)

print("Execution time: ", execution_time[2])


quantization_Array = [1, 64, 128, 256]

start = timeit.default_timer()

huffcoded, code_dict, length, aspect_ratio = m2.encode(
    im, levels, quantization_Array)
encoded.append(huffcoded)
reconstructed.append(m2.decode(huffcoded, code_dict,
                     length, quantization_Array, aspect_ratio))
execution_time.append(timeit.default_timer() - start)

print("Execution time: ", execution_time[3])


f, axarr = plt.subplots(2, 2, figsize=(48, 48))
axarr[0, 0].imshow(Image.fromarray(reconstructed[0]).resize(
    (im.size[0], im.size[1])), cmap="gray")
axarr[0, 0].set_title("LL,LL with quantization [1,2,3,4]", fontsize=40)
axarr[0, 1].imshow(Image.fromarray(reconstructed[1]).resize(
    (im.size[0], im.size[1])), cmap="gray")
axarr[0, 1].set_title("LL,LL with quantization [1, 64, 128, 256]", fontsize=40)
axarr[1, 0].imshow(Image.fromarray(reconstructed[2]).resize(
    (im.size[0], im.size[1])), cmap="gray")
axarr[1, 0].set_title("HH,LL with quantization [1,2,3,4]", fontsize=40)
axarr[1, 1].imshow(Image.fromarray(reconstructed[3]).resize(
    (im.size[0], im.size[1])), cmap="gray")
axarr[1, 1].set_title("HH,LL with quantization [1, 64, 128, 256]", fontsize=40)


imarr, _ = check_image(im, 3)

size_before = imarr.size * imarr.itemsize * 8
size_after = []
for i in range(len(encoded)):
    size_after.append(len(encoded[i]))

start = timeit.default_timer()
x = 2837*3847
stop = timeit.default_timer()
single_FLOP = stop - start
rms_error = []
for i in range(len(reconstructed)):
    rms_error.append(rmse(imarr, reconstructed[i]))


comp_type = ["LL,LL [1,2,3,4]", "LL,LL [2, 64, 128, 256]",
             "HH,LL [1,2,3,4]", "HH,LL [2, 64, 128, 256]",]
data = {'compression type': comp_type,
        'size in bits': size_after,
        'compression ratio': [size_before/x for x in size_after],
        "# of flops": [t/single_FLOP for t in execution_time],
        'RMSE': rms_error}
df = pd.DataFrame(data)
pd.set_option('expand_frame_repr', False)
print(df)
