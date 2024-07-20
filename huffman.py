import heapq
from collections import defaultdict


class HuffmanNode:
    def __init__(self, value, freq):
        self.value = value
        self.freq = freq
        self.left = None
        self.right = None

    def __lt__(self, other):
        return self.freq < other.freq


def build_huffman_tree(frequencies):
    heap = [HuffmanNode(key, freq) for key, freq in frequencies.items()]
    heapq.heapify(heap)
    
    while len(heap) > 1:
        left = heapq.heappop(heap)
        right = heapq.heappop(heap)
        
        merged_node = HuffmanNode(None, left.freq + right.freq)
        merged_node.left = left
        merged_node.right = right
        
        heapq.heappush(heap, merged_node)
        
    return heap[0]


def build_huffman_codes(node, prefix='', codes={}):
    if node:
        if node.value is not None:
            codes[node.value] = prefix
        build_huffman_codes(node.left, prefix + '0', codes)
        build_huffman_codes(node.right, prefix + '1', codes)
    return codes


def generate_huffman_codes(frequencies):
    root = build_huffman_tree(frequencies)
    codes = build_huffman_codes(root)
    return codes


def huffman_encode(data, huffman_codes):
    encoded_data = ''
    for symbol in data:
        encoded_data += huffman_codes[symbol]
    return encoded_data



def huffman_decode(encoded_data, huffman_codes):
    inverse_huffman_codes = {v: k for k, v in huffman_codes.items()}
    decoded_data = []
    code = ''
    for bit in encoded_data:
        code += bit
        if code in inverse_huffman_codes:
            decoded_data.append(int(inverse_huffman_codes[code]))
            code = ''
    return decoded_data



def save_huffman_codes(huffman_codes, filename):
    with open(filename, 'w') as file:
        for symbol, code in huffman_codes.items():
            file.write(f"{symbol}:{code}\n")


def load_huffman_codes(filename):
    huffman_codes = {}
    with open(filename, 'r') as file:
        for line in file:
            symbol, code = line.strip().split(':')
            huffman_codes[symbol] = code
    return huffman_codes
