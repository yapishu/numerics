"""
Convert PyTorch model weights to a jammed Hoon noun for Maroon.

Usage:
  python weights_to_noun.py --model gpt2 --output weights.jam

The output is a jammed noun matching the $model-weights type in /lib/maroon.hoon:
  [tok-emb pos-emb blocks=[blk1 blk2 ...] ln-f-g ln-f-b out-proj]

Each ray is [meta data=@ux] where meta is [shape bloq kind tail].
"""

import sys
import struct
import argparse
import numpy as np

sys.path.insert(0, '/home/reid/gits/np/tools/pkg/pynoun')
from noun import Cell, deep


class BitWriter:
    """Fast bitstream writer using a bytearray. ~1000x faster than BitArray
    for large writes because byte-aligned writes go straight to memory."""
    __slots__ = ('buf', 'cur', 'bits')

    def __init__(self):
        self.buf = bytearray()
        self.cur = 0
        self.bits = 0

    def pos(self):
        return len(self.buf) * 8 + self.bits

    def write(self, val, count):
        if count == 0:
            return
        val &= (1 << count) - 1
        # Fill partial byte
        if self.bits > 0:
            can_take = 8 - self.bits
            n = min(can_take, count)
            self.cur |= (val & ((1 << n) - 1)) << self.bits
            self.bits += n
            val >>= n
            count -= n
            if self.bits == 8:
                self.buf.append(self.cur)
                self.cur = 0
                self.bits = 0
        # Whole bytes fast path — uses to_bytes, very fast for big ints
        if count >= 8:
            whole_bits = count & ~7
            byte_count = whole_bits >> 3
            low_bits = val & ((1 << whole_bits) - 1)
            self.buf.extend(low_bits.to_bytes(byte_count, 'little'))
            val >>= whole_bits
            count -= whole_bits
        # Tail bits
        if count > 0:
            self.cur = val
            self.bits = count

    def finish(self):
        """Return the bitstream as a Python int."""
        if self.bits > 0:
            self.buf.append(self.cur)
        return int.from_bytes(self.buf, 'little')


def fast_jam(n):
    """Fast jam using BitWriter. Same algorithm as pynoun.jam but ~1000x
    faster because byte-aligned writes don't go through a per-bit loop."""
    w = BitWriter()
    refs = {}

    def mat(i):
        if i == 0:
            w.write(1, 1)
            return
        a = i.bit_length()
        b = a.bit_length()
        above = b + 1
        below = b - 1
        w.write(1 << b, above)
        w.write(a & ((1 << below) - 1), below)
        w.write(i, a)

    def back(ref):
        w.write(0b11, 2)
        mat(ref)

    def r(a):
        if deep(a):
            dupe = refs.get(a, None)
            if dupe is not None:
                back(dupe)
            else:
                refs[a] = w.pos()
                w.write(0b01, 2)  # LSB-first: bit 0 = 1 (cell), bit 1 = 0
                r(a.head)
                r(a.tail)
        else:
            dupe = refs.get(a, None)
            if dupe is not None:
                isize = a.bit_length()
                dsize = dupe.bit_length()
                if isize < dsize:
                    w.write(0, 1)
                    mat(a)
                else:
                    back(dupe)
            else:
                refs[a] = w.pos()
                w.write(0, 1)
                mat(a)

    r(n)
    return w.finish()


def jam(n):
    return fast_jam(n)

# Lagoon ray encoding

def float32_to_uint(f):
    """Convert a float32 to its IEEE 754 bit representation as uint32."""
    return struct.unpack('<I', struct.pack('<f', f))[0]

def float16_to_uint(f):
    """Convert a float16 to its IEEE 754 bit representation as uint16."""
    return struct.unpack('<H', struct.pack('<e', np.float16(f)))[0]

def make_ray(array, bloq=5):
    """Convert a numpy array to a Lagoon ray noun [meta data=@ux].

    bloq=5 -> @rs (float32), bloq=4 -> @rh (float16)

    Fast path: use numpy to get raw IEEE 754 bytes, then one int.from_bytes.
    The ray data format is: elements in row-major order at LSB, MSB pin on top.
    """
    shape = list(array.shape)
    flat = array.flatten()
    n = len(flat)

    if bloq == 5:  # float32
        # Get raw IEEE 754 bytes for each element, little-endian
        raw_bytes = flat.astype(np.float32).tobytes()  # element 0 at byte 0
        # int.from_bytes little-endian places byte 0 at LSB — correct!
        element_bits = int.from_bytes(raw_bytes, 'little')
        # Add MSB pin at position n*32
        data = (1 << (n * 32)) | element_bits
    elif bloq == 4:  # float16
        raw_bytes = flat.astype(np.float16).tobytes()
        element_bits = int.from_bytes(raw_bytes, 'little')
        data = (1 << (n * 16)) | element_bits
    else:
        raise ValueError(f"Unsupported bloq: {bloq}")

    # meta = [shape bloq kind tail]
    # shape is a Hoon list: [a [b [c 0]]]
    shape_noun = 0  # null
    for s in reversed(shape):
        shape_noun = Cell(s, shape_noun)

    kind = 0x3435_3769  # %i754 as @tas atom
    meta = Cell(shape_noun, Cell(bloq, Cell(kind, 0)))

    return Cell(meta, data)

def make_linear_weights(weight, bias, bloq=5):
    """Convert weight matrix and bias vector to linear-weights noun."""
    w_ray = make_ray(weight, bloq)
    b_ray = make_ray(bias.reshape(1, -1), bloq)  # reshape bias to [1 d_out]
    return Cell(w_ray, b_ray)

def hf_gpt2_to_noun(model_name='gpt2', bloq=5):
    """Load a HuggingFace GPT-2 model and convert to model-weights noun."""
    try:
        from transformers import GPT2LMHeadModel
    except ImportError:
        print("pip install transformers torch")
        sys.exit(1)

    import time
    print(f"Loading {model_name}...", flush=True)
    model = GPT2LMHeadModel.from_pretrained(model_name)
    sd = model.state_dict()
    config = model.config

    print(f"  d_model={config.n_embd}, n_heads={config.n_head}, "
          f"n_layers={config.n_layer}, vocab={config.vocab_size}")

    # Token embeddings: [vocab_size, d_model]
    tok_emb = make_ray(sd['transformer.wte.weight'].numpy(), bloq)
    print("  tok_emb done", flush=True)

    # Positional embeddings: [max_seq, d_model]
    pos_emb = make_ray(sd['transformer.wpe.weight'].numpy(), bloq)
    print("  pos_emb done", flush=True)

    # Transformer blocks
    blocks = 0  # null (end of list)
    for i in range(config.n_layer - 1, -1, -1):
        prefix = f'transformer.h.{i}'

        # Attention weights: GPT-2 uses a single [d_model, 3*d_model] projection
        # Split into Q, K, V
        c_attn_w = sd[f'{prefix}.attn.c_attn.weight'].numpy()  # [d_model, 3*d_model]
        c_attn_b = sd[f'{prefix}.attn.c_attn.bias'].numpy()    # [3*d_model]

        d = config.n_embd
        wq = make_linear_weights(c_attn_w[:, :d], c_attn_b[:d], bloq)
        wk = make_linear_weights(c_attn_w[:, d:2*d], c_attn_b[d:2*d], bloq)
        wv = make_linear_weights(c_attn_w[:, 2*d:], c_attn_b[2*d:], bloq)

        # Output projection
        wo = make_linear_weights(
            sd[f'{prefix}.attn.c_proj.weight'].numpy(),
            sd[f'{prefix}.attn.c_proj.bias'].numpy(), bloq)

        # Layer norms
        ln1_g = make_ray(sd[f'{prefix}.ln_1.weight'].numpy(), bloq)
        ln1_b = make_ray(sd[f'{prefix}.ln_1.bias'].numpy(), bloq)
        ln2_g = make_ray(sd[f'{prefix}.ln_2.weight'].numpy(), bloq)
        ln2_b = make_ray(sd[f'{prefix}.ln_2.bias'].numpy(), bloq)

        # Feed-forward
        ff1 = make_linear_weights(
            sd[f'{prefix}.mlp.c_fc.weight'].numpy(),
            sd[f'{prefix}.mlp.c_fc.bias'].numpy(), bloq)
        ff2 = make_linear_weights(
            sd[f'{prefix}.mlp.c_proj.weight'].numpy(),
            sd[f'{prefix}.mlp.c_proj.bias'].numpy(), bloq)

        # block-weights = [wq wk wv wo ln1-g ln1-b ln2-g ln2-b ff1 ff2]
        block = Cell(wq, Cell(wk, Cell(wv, Cell(wo,
                Cell(ln1_g, Cell(ln1_b, Cell(ln2_g, Cell(ln2_b,
                Cell(ff1, ff2)))))))))

        blocks = Cell(block, blocks)
        print(f"  block {i} done", flush=True)

    # Final layer norm
    ln_f_g = make_ray(sd['transformer.ln_f.weight'].numpy(), bloq)
    ln_f_b = make_ray(sd['transformer.ln_f.bias'].numpy(), bloq)

    # Output projection (GPT-2 ties weights with token embeddings)
    out_proj = make_ray(sd['transformer.wte.weight'].numpy().T, bloq)  # transpose for [d_model, vocab]

    print("  final layers done", flush=True)

    # model-weights = [tok-emb pos-emb blocks ln-f-g ln-f-b out-proj]
    weights = Cell(tok_emb, Cell(pos_emb, Cell(blocks,
              Cell(ln_f_g, Cell(ln_f_b, out_proj)))))

    return weights, config

def main():
    parser = argparse.ArgumentParser(description='Convert model weights to Hoon noun')
    parser.add_argument('--model', default='gpt2', help='HuggingFace model name')
    parser.add_argument('--output', default='weights.jam', help='Output file')
    parser.add_argument('--bloq', type=int, default=5, help='Precision: 5=float32, 4=float16')
    parser.add_argument('--test', action='store_true', help='Create tiny test model instead')
    args = parser.parse_args()

    if args.test:
        # Create a tiny model for testing: d=4, heads=1, layers=1, vocab=8
        print("Creating tiny test model...")
        d, ff_d, vocab, max_seq = 4, 8, 8, 4

        tok_emb = make_ray(np.random.randn(vocab, d).astype(np.float32) * 0.02, args.bloq)
        pos_emb = make_ray(np.random.randn(max_seq, d).astype(np.float32) * 0.02, args.bloq)

        def rand_linear(di, do):
            return make_linear_weights(
                np.random.randn(di, do).astype(np.float32) * 0.02,
                np.zeros(do, dtype=np.float32), args.bloq)

        wq = rand_linear(d, d)
        wk = rand_linear(d, d)
        wv = rand_linear(d, d)
        wo = rand_linear(d, d)
        ln1_g = make_ray(np.ones(d, dtype=np.float32), args.bloq)
        ln1_b = make_ray(np.zeros(d, dtype=np.float32), args.bloq)
        ln2_g = make_ray(np.ones(d, dtype=np.float32), args.bloq)
        ln2_b = make_ray(np.zeros(d, dtype=np.float32), args.bloq)
        ff1 = rand_linear(d, ff_d)
        ff2 = rand_linear(ff_d, d)

        block = Cell(wq, Cell(wk, Cell(wv, Cell(wo,
                Cell(ln1_g, Cell(ln1_b, Cell(ln2_g, Cell(ln2_b,
                Cell(ff1, ff2)))))))))

        ln_f_g = make_ray(np.ones(d, dtype=np.float32), args.bloq)
        ln_f_b = make_ray(np.zeros(d, dtype=np.float32), args.bloq)
        out_proj = make_ray(np.random.randn(d, vocab).astype(np.float32) * 0.02, args.bloq)

        weights = Cell(tok_emb, Cell(pos_emb, Cell(Cell(block, 0),
                  Cell(ln_f_g, Cell(ln_f_b, out_proj)))))
        config = type('Config', (), {'n_embd': d, 'n_head': 1, 'n_layer': 1,
                                      'n_inner': ff_d, 'vocab_size': vocab,
                                      'n_positions': max_seq})()
    else:
        weights, config = hf_gpt2_to_noun(args.model, args.bloq)

    print("Jamming noun...")
    jammed = jam(weights)

    # Write as bytes
    data = jammed.to_bytes((jammed.bit_length() + 7) // 8, 'little')
    with open(args.output, 'wb') as f:
        f.write(data)

    print(f"Written {len(data)} bytes to {args.output}")
    print(f"Model: d={config.n_embd}, heads={config.n_head}, "
          f"layers={config.n_layer}, vocab={config.vocab_size}")

if __name__ == '__main__':
    main()
