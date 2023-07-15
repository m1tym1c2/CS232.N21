import numpy as np
import matplotlib.pyplot as plt
import timeit
from PIL import Image
import main as m
from encoder import reshape_image
import pandas as pd


# Áp dụng PSNR để so sánh
def rmse(im, reconstructed_image):
    error = im - reconstructed_image
    mse = np.sum(np.square(error)) / (im.shape[0] * im.shape[1])
    rmse = np.sqrt(mse)
    return rmse


# Loading ảnh input
im = Image.open('1.jpg')
im = im.convert('L')


# Khai báo biến
box_size = 8
encoded = []
reconstructed = []
execution_time = []

start = timeit.default_timer()
huffcoded, code_dict, n_rows, n_columns = m.encode(im, box_size, m.table_8_low)
encoded.append(huffcoded)
reconstructed.append(m.decode(huffcoded, code_dict,  n_rows,
                     n_columns, box_size, m.table_8_low))
execution_time.append(timeit.default_timer() - start)
print("Execution time: ", execution_time[0])
box_size = 8
start = timeit.default_timer()


huffcoded, code_dict, n_rows, n_columns = m.encode(
    im, box_size, m.table_8_high)
encoded.append(huffcoded)
reconstructed.append(m.decode(huffcoded, code_dict,  n_rows,
                     n_columns, box_size, m.table_8_high))
execution_time.append(timeit.default_timer() - start)

print("Execution time: ", execution_time[1])


imarr = reshape_image(im)

size_before = imarr.size * imarr.itemsize * 8
print("Size in bits of image before compression: ", size_before)


size_after = []
for i in range(len(encoded)):
    size_after.append(len(encoded[i]))
start = timeit.default_timer()
x = 2837*3847
stop = timeit.default_timer()
single_FLOP = stop - start


rms_error = []
for i in range(0, 2):
    if (i > 1):
        imarr = reshape_image(im, 16)
    rms_error.append(rmse(imarr, reconstructed[i]))


comp_type = ["8x8 low", "8x8 high"]
data = {'compression type': comp_type,
        'size in bits': size_after,
        'compression ratio': [size_before/x for x in size_after],
        "# of flops": [t/single_FLOP for t in execution_time],
        'RMSE': rms_error}

df = pd.DataFrame(data)

print(df)


f, axarr = plt.subplots(1, 2, figsize=(15, 15))
axarr[0].imshow(reconstructed[0], cmap='gray')
axarr[0].set_title("8x8 low compression", fontsize=10)
axarr[1].imshow(reconstructed[1], cmap='gray')
axarr[1].set_title("8x8 high compression", fontsize=10)
plt.show()
