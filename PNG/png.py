from PIL import Image
import numpy as np
import zlib
import io
import matplotlib.pyplot as plt
import time

def show_image_grid(images, titles, rows, cols):
    fig, axes = plt.subplots(rows, cols, figsize=(12, 10))

    for i, ax in enumerate(axes.flat):
        ax.imshow(images[i])
        ax.set_title(titles[i])
        ax.axis('off')

    plt.tight_layout()
    plt.show()

def paeth_predict(a, b, c):
    p = a + b - c
    pa = np.abs(p - a)
    pb = np.abs(p - b)
    pc = np.abs(p - c)
    mask = (pa <= pb) & (pa <= pc)
    return np.where(mask, a, np.where(pb <= pc, b, c))

def filter_sub(image):
    filtered_image = np.zeros_like(image)
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            if j == 0:
                filtered_image[i][j] = image[i][j]
            else:
                filtered_image[i][j] = (image[i][j] - image[i][j-1]) % 256
    return filtered_image

def filter_up(image):
    filtered_image = np.zeros_like(image)
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            if i == 0:
                filtered_image[i][j] = image[i][j]
            else:
                filtered_image[i][j] = (image[i][j] - image[i - 1][j]) % 256
    return filtered_image

def filter_average(image):
    filtered_image = np.zeros_like(image)
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            if i == 0 and j == 0:
                filtered_image[i][j] = image[i][j]
            elif i == 0:
                filtered_image[i][j] = (image[i][j] - (image[i][j-1] + image[i-1][j]) // 2) % 256
            elif j == 0:
                filtered_image[i][j] = (image[i][j] - (image[i-1][j] + image[i][j-1]) // 2) % 256
            else:
                filtered_image[i][j] = (image[i][j] - (image[i][j-1] + image[i-1][j]) // 2) % 256
    return filtered_image

def reverse_filter_sub(image):
    defiltered_image = np.zeros_like(image)
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            if j == 0:
                defiltered_image[i][j] = image[i][j]
            else:
                defiltered_image[i][j] = (image[i][j] + defiltered_image[i][j-1]) % 256
    return defiltered_image

def filter_paeth(image):
    filtered_image = np.zeros_like(image)
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            if i == 0 and j == 0:
                filtered_image[i][j] = image[i][j]
            elif i == 0:
                a = 0
                b = image[i][j-1]
                c = 0
                filtered_image[i][j] = (image[i][j] - paeth_predict(a, b, c)) % 256
            elif j == 0:
                a = image[i-1][j]
                b = 0
                c = 0
                filtered_image[i][j] = (image[i][j] - paeth_predict(a, b, c)) % 256
            else:
                a = image[i-1][j]
                b = image[i][j-1]
                c = image[i-1][j-1]
                filtered_image[i][j] = (image[i][j] - paeth_predict(a, b, c)) % 256
    return filtered_image

def filter_none(image):
    return image

def reverse_filter_up(image):
    defiltered_image = np.zeros_like(image)
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            if i == 0:
                defiltered_image[i][j] = image[i][j]
            else:
                defiltered_image[i][j] = (image[i][j] + defiltered_image[i - 1][j]) % 256
    return defiltered_image

def reverse_filter_average(image):
    defiltered_image = np.zeros_like(image)
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            if i == 0 and j == 0:
                defiltered_image[i][j] = image[i][j]
            elif i == 0:
                defiltered_image[i][j] = (image[i][j] + (defiltered_image[i][j-1] // 2)) % 256
            elif j == 0:
                defiltered_image[i][j] = (image[i][j] + (defiltered_image[i-1][j] // 2)) % 256
            else:
                defiltered_image[i][j] = (image[i][j] + ((defiltered_image[i][j-1] + defiltered_image[i-1][j]) // 2)) % 256
    return defiltered_image

def reverse_filter_paeth(image):
    defiltered_image = np.zeros_like(image)
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            if i == 0 and j == 0:
                defiltered_image[i][j] = image[i][j]
            elif i == 0:
                a = 0
                b = defiltered_image[i][j-1]
                c = 0
                defiltered_image[i][j] = (image[i][j] + paeth_predict(a, b, c)) % 256
            elif j == 0:
                a = defiltered_image[i-1][j]
                b = 0
                c = 0
                defiltered_image[i][j] = (image[i][j] + paeth_predict(a, b, c)) % 256
            else:
                a = defiltered_image[i-1][j]
                b = defiltered_image[i][j-1]
                c = defiltered_image[i-1][j-1]
                defiltered_image[i][j] = (image[i][j] + paeth_predict(a, b, c)) % 256
    return defiltered_image

def reverse_filter_none(image):
    return image


# Đọc ảnh đầu vào
input_image = Image.open("xucxac.png")
input_image_np = np.array(input_image)

# Chuyển ảnh thành dạng bytes
image_bytes = io.BytesIO()
input_image.save(image_bytes, format='PNG')
image_bytes = image_bytes.getvalue()

# Lấy độ dài của dữ liệu ảnh dưới dạng bytes
image_length = len(image_bytes)

data = image_length * 0.05
# Apply filter
start_time = time.time()
filtered_image_sub_np = filter_sub(input_image_np)
filtered_image_up_np = filter_up(input_image_np)
filtered_image_average_np = filter_average(input_image_np)
filtered_image_paeth_np = filter_paeth(input_image_np)
filtered_image_none_np = filter_none(input_image_np)


filters = [filtered_image_sub_np, filtered_image_up_np, filtered_image_average_np, filtered_image_paeth_np, filtered_image_none_np]
titles = ['Filter Sub', 'Filter Up', 'Filter Average', 'Filter Paeth', 'Filter None']
show_image_grid(filters, titles, 1, 5)

compressed_sizes = []
compressed_datas = []

for filter in filters:
    # Chuyển ảnh đã filter thành dạng bytes
    image_bytes = io.BytesIO()
    filtered_image = Image.fromarray(np.uint8(filter))
    filtered_image.save(image_bytes, format='PNG')
    image_bytes = image_bytes.getvalue()
    # Nén ảnh bằng kỹ thuật deflate
    compressed_datas.append(zlib.compress(image_bytes, 9))
    compressed_sizes.append(len(compressed_datas[-1]))

end_time = time.time()
best_filter_index = np.argmin(compressed_sizes)
best_filter_name = ["Sub", "Up", "Average", "Paeth", "None"][best_filter_index]


compressed_image_filename = "compressed_image.png"
with open(compressed_image_filename, "wb") as file:
    file.write(compressed_datas[best_filter_index])

l = len(compressed_datas[best_filter_index]) - data
# Giải nén ảnh
decompressed_data = zlib.decompress(compressed_datas[best_filter_index])

# Chuyển data giải nén về dạng ảnh
decompressed_image = Image.open(io.BytesIO(decompressed_data))
decompressed_image_np = np.array(decompressed_image)

# Apply reverse filter
if best_filter_name == "Sub":
    defiltered_image_np = reverse_filter_sub(decompressed_image_np)
elif best_filter_name == "Up":
    defiltered_image_np = reverse_filter_up(decompressed_image_np)
elif best_filter_name == "Average":
    defiltered_image_np = reverse_filter_average(decompressed_image_np)
elif best_filter_name == "Paeth":
    defiltered_image_np = reverse_filter_paeth(decompressed_image_np)
elif best_filter_name == "None":
    defiltered_image_np = reverse_filter_none(decompressed_image_np)

# Tạo ảnh từ dữ liệu sau khi reverse filter
output_image = Image.fromarray(np.uint8(defiltered_image_np))
# output_image.show()
# Lưu ảnh output với tên là "defiltered_image.png"
output_image.save("defiltered_image.png")

# In kích thước ảnh trước và sau khi nén
print("Filter cho kết quả nén tốt nhất là:", best_filter_name)
print("Kích thước ảnh trước khi nén:", image_length)
print("Đã lưu ảnh đã nén vào tệp tin", compressed_image_filename)
print("Kích thước ảnh sau khi nén:", l)
print("Tỷ lệ nén: {:.3f}".format(image_length / l))
print("Thời gian nén: {:.3f}".format(end_time-start_time))