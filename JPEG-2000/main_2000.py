import numpy as np
from PIL import Image
from encoder import huffman_encode, run_length_code
from decoder import huffman_decode, run_length_decode
import encoder_2000 as e
import decoder_2000 as d


def encode(image, levels, quantization_Array):
    def depth(l):
        if isinstance(l, int):
            return 0
        return 1 + max(iter([depth(l_rec) for l_rec in l]), default=0)
    d = depth(levels)
    im_arr, aspect_ratio = e.check_image(image, d)
    filtered_image = e.dwt(im_arr, quantization_Array)
    e.dwt_levels(filtered_image, levels, quantization_Array)
    serialized, length = e.dwt_serialize(filtered_image, output=[], length=[])
    rlcoded = run_length_code(serialized)
    huffcoded, code_dict = huffman_encode(rlcoded)
    return huffcoded, code_dict, length, aspect_ratio


def decode(huffcoded, code_dict, length, quantization_Array, aspect_ratio):
    rlcoded = huffman_decode(huffcoded, code_dict)
    serialized = run_length_decode(rlcoded)
    im_arr = d.dwt_deserialize(serialized, length, quantization_Array,
                               aspect_ratio)
    return im_arr
