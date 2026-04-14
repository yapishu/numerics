"""
Export GPT-2 BPE tokenizer as a jammed Hoon noun.

The noun structure matches the $tokenizer type in /lib/tokenizer.hoon:
  [vocab=(map @t @ud) inverse=(map @ud @t) merges=(map [@t @t] @ud) byte-map=(map @ @t) inverse-byte-map=(map @t @)]

vocab: BPE token string -> token ID
inverse: token ID -> BPE token string
merges: (pair-of-strings) -> merge rank (lower = higher priority)
byte-map: raw byte (0-255) -> unicode string representation (GPT-2's bytes_to_unicode mapping)
inverse-byte-map: unicode string -> raw byte

Usage:
    python3 tokenizer_to_noun.py --output gpt2-tokenizer.jam
"""

import argparse
import sys
import json

sys.setrecursionlimit(200_000)

sys.path.insert(0, '/home/reid/gits/np/tools/pkg/pynoun')
from noun import Cell

# Reuse fast jam from the weights converter
sys.path.insert(0, '/home/reid/gits/np/numerics/saloon/tools')
exec(open('/home/reid/gits/np/numerics/saloon/tools/weights_to_noun.py').read().split('def main')[0])


def str_to_atom(s):
    """Encode a Python string to a Hoon @t atom (bytes as little-endian int)."""
    if not s:
        return 0
    return int.from_bytes(s.encode('utf-8'), 'little')


def build_hoon_list_as_tree(items):
    """Build a noun as a balanced binary tree of [key value] cells.

    The Hoon side converts this to a proper `map` on load using `malt`.
    A balanced binary tree keeps noun depth at O(log n) so jam doesn't
    blow the stack on 50K-entry vocabs.

    Tree format: [[k v] [left right]] where an empty subtree is 0.
    Leaf: [[k v] 0].
    """
    if not items:
        return 0
    # Flatten: we want the items as a binary tree noun that, when walked,
    # yields the list. We use the structure [head tail-tree] where tail-tree
    # is either 0 (end) or another Cell.
    # But to keep noun depth low, we use the balanced cell tree trick:
    # build a perfectly balanced tree whose in-order traversal gives items.
    # Then the Hoon side uses a helper to re-build the list from the tree.
    def build(lo, hi):
        if lo >= hi:
            return 0
        if lo + 1 == hi:
            k, v = items[lo]
            return Cell(Cell(k, v), 0)
        mid = (lo + hi) // 2
        k, v = items[mid]
        return Cell(Cell(k, v), Cell(build(lo, mid), build(mid + 1, hi)))
    return build(0, len(items))


def bytes_to_unicode():
    """GPT-2's bytes_to_unicode mapping.
    Returns dict mapping byte (int) -> unicode character.
    """
    bs = (
        list(range(ord("!"), ord("~") + 1))
        + list(range(ord("\xa1"), ord("\xac") + 1))
        + list(range(ord("\xae"), ord("\xff") + 1))
    )
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return dict(zip(bs, [chr(c) for c in cs]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--output', default='gpt2-tokenizer.jam')
    ap.add_argument('--model', default='gpt2')
    args = ap.parse_args()

    from transformers import GPT2Tokenizer
    tok = GPT2Tokenizer.from_pretrained(args.model)

    print(f"loading {args.model}...", flush=True)
    print(f"  vocab size: {tok.vocab_size}", flush=True)

    vocab_dict = tok.get_vocab()
    # vocab: str -> int (we'll emit a list of pairs; Hoon side builds the map)
    vocab_items = [(str_to_atom(t), i) for t, i in vocab_dict.items()]
    print(f"  vocab entries: {len(vocab_items)}", flush=True)
    vocab_noun = build_hoon_list_as_tree(vocab_items)

    # inverse vocab
    inverse_items = [(i, str_to_atom(t)) for t, i in vocab_dict.items()]
    inverse_noun = build_hoon_list_as_tree(inverse_items)

    # merges: each (a,b) pair with rank = position
    # Try different attribute names across transformers versions
    bpe_ranks = getattr(tok, 'bpe_ranks', None)
    if bpe_ranks is None:
        bpe_ranks = getattr(tok, '_bpe_ranks', None)
    if bpe_ranks is None:
        # Try to load merges.txt from the cache directly
        from huggingface_hub import hf_hub_download
        merges_file = hf_hub_download(args.model, 'merges.txt')
        bpe_ranks = {}
        with open(merges_file) as f:
            lines = f.readlines()
            start = 1 if lines[0].startswith('#') else 0
            for i, line in enumerate(lines[start:]):
                parts = line.strip().split(' ')
                if len(parts) == 2:
                    bpe_ranks[(parts[0], parts[1])] = i

    merge_items = []
    for rank, (a, b) in enumerate(bpe_ranks.keys()):
        merge_items.append((Cell(str_to_atom(a), str_to_atom(b)), rank))
    print(f"  merges: {len(merge_items)}", flush=True)
    merges_noun = build_hoon_list_as_tree(merge_items)

    # byte-to-unicode map
    b2u = bytes_to_unicode()
    byte_map_items = [(b, str_to_atom(c)) for b, c in b2u.items()]
    byte_map_noun = build_hoon_list_as_tree(byte_map_items)

    # inverse byte map
    inverse_byte_map_items = [(str_to_atom(c), b) for b, c in b2u.items()]
    inverse_byte_map_noun = build_hoon_list_as_tree(inverse_byte_map_items)

    # tokenizer = [vocab inverse merges byte-map inverse-byte-map]
    tokenizer = Cell(vocab_noun,
               Cell(inverse_noun,
               Cell(merges_noun,
               Cell(byte_map_noun, inverse_byte_map_noun))))

    print("jamming...", flush=True)
    jammed = fast_jam(tokenizer)

    data = jammed.to_bytes((jammed.bit_length() + 7) // 8, 'little')
    with open(args.output, 'wb') as f:
        f.write(data)
    print(f"wrote {len(data)} bytes to {args.output}")


if __name__ == '__main__':
    main()
