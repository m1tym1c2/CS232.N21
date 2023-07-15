class TreeNode:
    def __init__(self, name, prob, lchild=None, rchild=None):
        self.name = name
        self.prob = prob
        self.lchild = lchild
        self.rchild = rchild


def encode(counts_dict):
    tree_nodes = [TreeNode(name, count, None, None)
                  for name, count in counts_dict.items()]
    return assign_codes(huffman_partition(tree_nodes))


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


def reverse_dict(d):
    return dict(map(reversed, d.items()))


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
