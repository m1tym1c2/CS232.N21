class TreeNode:
    # Khởi tạo
    def __init__(self, name, prob, lchild=None, rchild=None):
        self.name = name
        self.prob = prob
        self.lchild = lchild
        self.rchild = rchild


# Mã hóa
def encode(counts_dict):
    tree_nodes = [TreeNode(name, count, None, None)
                  for name, count in counts_dict.items()]
    return assign_codes(huffman_partition(tree_nodes))


# Tái tạo cây huffman
def huffman_partition(tree_nodes):
    sorted_xs = sorted(tree_nodes, reverse=True, key=lambda x: x.prob)

    def helper(xs):
        rchild = xs.pop(-1)
        lchild = xs.pop(-1)
        insort_wkey(xs, TreeNode(None,
                                 rchild.prob + lchild.prob, lchild, rchild),
                    key=lambda x: x.prob)
        if len(xs) == 1:
            return xs[0]
        return huffman_partition(xs)
    return helper(sorted_xs)


# Sử dụng tiền tố từ cây huffman
def assign_codes(tree):
    def assign_codes_helper(tree, code_lists, code=""):
        if tree.lchild is None and tree.rchild is None:
            code_dict[tree.name] = code
            return
        assign_codes_helper(tree.lchild, code_lists, code+'0')
        assign_codes_helper(tree.rchild, code_lists, code+'1')
    code_dict = {}
    assign_codes_helper(tree, code_dict)
    return code_dict


# Đảo ngược dictionary
def reverse_dict(d):
    return dict(map(reversed, d.items()))


# Giải nén
def decode(s, code_dict):
    invcodemap = reverse_dict(code_dict)
    result = []
    incoming = ""
    while len(s) != 0:
        code = invcodemap.get(incoming+s[0], None)
        if code is not None:
            result.append(code)
            s, incoming = s[1:], ""
        else:
            s, incoming = s[1:], incoming+s[0]
    return result

# Chèn nhị phân sử dụng một chứ năng được áp dụng trước khi so sánh và giữ nó được sắp xếp giả sử nếu a được sắp xếp.


def insort_wkey(a, x, key=lambda x: x, lo=0, hi=None):
    if lo < 0:
        raise ValueError('lo must be non-negative')
    if hi is None:
        hi = len(a)
    while lo < hi:
        mid = (lo+hi)//2
        if key(x) < key(a[mid]):
            hi = mid
        else:
            lo = mid+1
    a.insert(lo, x)
