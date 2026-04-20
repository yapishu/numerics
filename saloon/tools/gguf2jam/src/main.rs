//! gguf2jam: convert model weights to jammed Hoon nouns for Maroon.
//!
//! Currently supports safetensors (HuggingFace native format).
//! GGUF support planned.
//!
//! Output format matches $model-weights in /lib/maroon.hoon:
//!   [tok-emb pos-emb blocks ln-f-g ln-f-b out-proj]
//!
//! Each tensor is encoded as a Lagoon $ray:
//!   [[shape bloq kind tail] data=@ux]

use std::fs::File;
use std::path::PathBuf;

use anyhow::{Context, Result, bail};
use axsys_noun::atom::Atom;
use axsys_noun::cell::Cell;
use axsys_noun::noun::Noun;
use axsys_noun::serdes::Jam;
use clap::Parser;
use memmap2::Mmap;
use safetensors::SafeTensors;
use safetensors::tensor::Dtype;

#[derive(Parser, Debug)]
#[command(version, about = "Convert model weights to Maroon-compatible jammed nouns")]
struct Args {
    /// Path to the input file (safetensors)
    input: PathBuf,

    /// Path to write the jammed noun output
    #[arg(short, long)]
    output: PathBuf,

    /// Model architecture: "gpt2" | "qwen3" | "tokenizer"
    /// (use "tokenizer" with input pointing at a HF tokenizer.json)
    #[arg(long, default_value = "gpt2")]
    arch: String,

    /// Output precision: 5=float32 (@rs), 4=float16 (@rh)
    #[arg(long, default_value_t = 5)]
    bloq: u8,

    /// Int8-quantize block weights for ~4x size reduction.
    #[arg(long)]
    quantize: bool,

    /// MLX quantization group size (qwen3 only, defaults to 128).
    #[arg(long, default_value_t = 128)]
    group_size: usize,

    /// Emit one jam per layer into a directory (one file per block, plus
    /// tok-emb / ln-f / manifest). Useful for inspecting individual layers.
    #[arg(long)]
    split: bool,
}

/// Build the `%i754` atom used as `kind` in the ray meta.
/// In Urbit @tas atoms are little-endian ASCII.
fn kind_i754() -> Atom {
    // "i754" as bytes, little-endian
    Atom::from(vec![b'i', b'7', b'5', b'4'])
}

fn kind_uint() -> Atom {
    Atom::from(vec![b'u', b'i', b'n', b't'])
}

/// Build a `%fp` or `%q8` tag atom.
fn tag_fp() -> Atom { Atom::from(vec![b'f', b'p']) }
fn tag_q8() -> Atom { Atom::from(vec![b'q', b'8']) }
fn tag_mlx2() -> Atom { Atom::from(vec![b'm', b'l', b'x', b'2']) }

/// Build a uint8 ray (bloq=3, kind=%uint) from raw bytes.
fn build_ray_u8(bytes: &[u8], shape: &[usize]) -> Noun {
    let n_elements = bytes.len();
    let mut data_bytes = Vec::with_capacity(n_elements + 1);
    data_bytes.extend_from_slice(bytes);
    data_bytes.push(0x01);  // MSB pin

    let data = Atom::from(data_bytes);

    let mut shape_noun = Noun::Atom(Atom::from(0u8));
    for &dim in shape.iter().rev() {
        shape_noun = Noun::Cell(Cell::from([
            Noun::Atom(Atom::from(dim as u64)),
            shape_noun,
        ]));
    }

    let meta = Noun::Cell(Cell::from([
        shape_noun,
        Noun::Atom(Atom::from(3u64)),
        Noun::Atom(Noun::from(kind_uint()).into_atom().unwrap()),
        Noun::Atom(Atom::from(0u8)),
    ]));

    Noun::Cell(Cell::from([meta, Noun::Atom(data)]))
}

/// Encode a Hoon @rs (float32) atom for the scale.
fn f32_to_atom(f: f32) -> Atom {
    let bits = f.to_bits();
    let bytes = bits.to_le_bytes().to_vec();
    Atom::from(bytes)
}

/// Quantize an fp32 tensor to int8 + scale; emit a [%q8 ray scale] noun.
/// Symmetric per-tensor: scale = max(abs(W)) / 127.
fn make_q8_weight(f32_bytes: &[u8], shape: &[usize]) -> Noun {
    assert_eq!(f32_bytes.len() % 4, 0);
    let n = f32_bytes.len() / 4;

    // find max abs
    let mut max_abs: f32 = 0.0;
    for chunk in f32_bytes.chunks_exact(4) {
        let v = f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]);
        let a = v.abs();
        if a > max_abs { max_abs = a; }
    }
    let scale = if max_abs == 0.0 { 1.0 } else { max_abs / 127.0 };

    // quantize each element to int8 (stored as u8 two's complement)
    let mut q_bytes = Vec::with_capacity(n);
    for chunk in f32_bytes.chunks_exact(4) {
        let v = f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]);
        let q = (v / scale).round().clamp(-128.0, 127.0) as i8;
        q_bytes.push(q as u8);
    }

    let ray = build_ray_u8(&q_bytes, shape);
    Noun::Cell(Cell::from([
        Noun::Atom(tag_q8()),
        ray,
        Noun::Atom(f32_to_atom(scale)),
    ]))
}

/// Wrap a weight ray as [%fp r=ray].
fn wrap_fp(ray: Noun) -> Noun {
    Noun::Cell(Cell::from([Noun::Atom(tag_fp()), ray]))
}

/// Convert a flat slice of float32 bytes + shape to a Lagoon ray noun.
///
/// A ray is `[meta data=@ux]` where:
///   meta = `[shape bloq kind tail]`
///   data = elements in row-major order at LSB, MSB pin on top
///
/// We build the atom as bytes directly: elements in row-major order (byte 0
/// of element 0 at LSB), then a single 1 bit on top to pin the size.
fn build_ray_f32(bytes: &[u8], shape: &[usize], bloq: u8) -> Noun {
    assert_eq!(bloq, 5, "only bloq=5 (float32) currently supported in this path");
    assert_eq!(bytes.len() % 4, 0);
    let n_elements = bytes.len() / 4;
    let element_bits: usize = n_elements * 32;

    // Build data: elements in little-endian bytes, followed by a 1 at bit N*32
    let mut data_bytes = Vec::with_capacity(bytes.len() + 1);
    data_bytes.extend_from_slice(bytes);
    // Add a 1 bit at position `element_bits` (the MSB pin).
    // Since element_bits is always a multiple of 8, append a single 0x01 byte.
    // (element_bits / 8 == bytes.len(), so we append at the end.)
    data_bytes.push(0x01);

    let data = Atom::from(data_bytes);

    // shape noun: build as a Hoon list [a [b [c 0]]]
    let mut shape_noun = Noun::Atom(Atom::from(0u8));
    for &dim in shape.iter().rev() {
        shape_noun = Noun::Cell(Cell::from([
            Noun::Atom(Atom::from(dim as u64)),
            shape_noun,
        ]));
    }

    // meta = [shape bloq kind tail]
    //   where tail = 0
    let meta = Noun::Cell(Cell::from([
        shape_noun,
        Noun::Atom(Atom::from(bloq as u64)),
        Noun::Atom(Noun::from(kind_i754()).into_atom().unwrap()),
        Noun::Atom(Atom::from(0u8)),
    ]));

    Noun::Cell(Cell::from([meta, Noun::Atom(data)]))
}

/// Build a ray from a safetensors view. Handles dtype conversion.
fn tensor_to_ray(
    tensors: &SafeTensors,
    name: &str,
    bloq: u8,
) -> Result<Noun> {
    let tensor = tensors
        .tensor(name)
        .with_context(|| format!("missing tensor: {}", name))?;
    let shape: Vec<usize> = tensor.shape().to_vec();
    let bytes = tensor.data();

    match tensor.dtype() {
        Dtype::F32 => {
            assert_eq!(bloq, 5, "output bloq must be 5 for F32 input");
            Ok(build_ray_f32(bytes, &shape, bloq))
        }
        Dtype::F16 | Dtype::BF16 => {
            // If output is @rs, convert f16/bf16 to f32
            assert_eq!(bloq, 5, "f16/bf16 -> other bloq not yet supported");
            let mut f32_bytes = Vec::with_capacity(bytes.len() * 2);
            for chunk in bytes.chunks_exact(2) {
                let raw = u16::from_le_bytes([chunk[0], chunk[1]]);
                let val: f32 = match tensor.dtype() {
                    Dtype::F16 => half::f16::from_bits(raw).to_f32(),
                    Dtype::BF16 => half::bf16::from_bits(raw).to_f32(),
                    _ => unreachable!(),
                };
                f32_bytes.extend_from_slice(&val.to_le_bytes());
            }
            Ok(build_ray_f32(&f32_bytes, &shape, bloq))
        }
        other => bail!("unsupported dtype: {:?}", other),
    }
}

/// Helper: build a linear-weights noun [w b] from two tensors.
/// `w` is wrapped as [%fp r=ray] or [%q8 r=ray scale=@rs].
fn linear(
    tensors: &SafeTensors,
    w_name: &str,
    b_name: &str,
    bloq: u8,
    quantize: bool,
) -> Result<Noun> {
    let w_tensor = tensors.tensor(w_name)?;
    let w_shape = w_tensor.shape().to_vec();
    let w_bytes = bytes_as_f32_vec(&w_tensor)?;

    let w_noun = if quantize {
        make_q8_weight(&w_bytes, &w_shape)
    } else {
        wrap_fp(build_ray_f32(&w_bytes, &w_shape, bloq))
    };

    // Bias: reshape to [1 d_out] so it nests with [1 d_out] rows in add-bias.
    let b_tensor = tensors.tensor(b_name)?;
    let mut b_shape = b_tensor.shape().to_vec();
    if b_shape.len() == 1 {
        b_shape.insert(0, 1);
    }
    let b_bytes = bytes_as_f32_vec(&b_tensor)?;
    let b_ray = build_ray_f32(&b_bytes, &b_shape, bloq);
    Ok(Noun::Cell(Cell::from([w_noun, b_ray])))
}

/// Read a tensor as f32 bytes regardless of input dtype.
fn bytes_as_f32_vec(tensor: &safetensors::tensor::TensorView) -> Result<Vec<u8>> {
    match tensor.dtype() {
        Dtype::F32 => Ok(tensor.data().to_vec()),
        Dtype::F16 => {
            let mut out = Vec::with_capacity(tensor.data().len() * 2);
            for chunk in tensor.data().chunks_exact(2) {
                let raw = u16::from_le_bytes([chunk[0], chunk[1]]);
                let v = half::f16::from_bits(raw).to_f32();
                out.extend_from_slice(&v.to_le_bytes());
            }
            Ok(out)
        }
        Dtype::BF16 => {
            let mut out = Vec::with_capacity(tensor.data().len() * 2);
            for chunk in tensor.data().chunks_exact(2) {
                let raw = u16::from_le_bytes([chunk[0], chunk[1]]);
                let v = half::bf16::from_bits(raw).to_f32();
                out.extend_from_slice(&v.to_le_bytes());
            }
            Ok(out)
        }
        other => bail!("unsupported dtype: {:?}", other),
    }
}

/// Slice a [d_model, 3*d_model] c_attn.weight into Q, K, V each [d_model, d_model].
/// Returns (wq_w, wk_w, wv_w) as byte slices ready to become rays.
fn split_qkv_weight<'a>(bytes: &'a [u8], rows: usize, cols: usize) -> [&'a [u8]; 3] {
    // weight is stored as [rows * cols] in row-major, each element 4 bytes (f32).
    // We want to split each row into 3 equal parts of cols/3 each.
    // This is NOT a simple byte slice — we'd need to rearrange.
    // But for GPT-2, c_attn stacks [Q_weights; K_weights; V_weights] along the column
    // axis, meaning for each input row, columns 0..d map to Q, d..2d to K, 2d..3d to V.
    //
    // We need to emit three [rows, cols/3] matrices.
    // For simplicity we'll do a proper reshape in the caller.
    let _ = (rows, cols);
    [bytes, bytes, bytes]  // placeholder
}

/// For GPT-2's packed c_attn, split into Q, K, V rays.
/// Input tensor shape: [d_model, 3*d_model] for weight, [3*d_model] for bias.
fn split_gpt2_qkv(
    tensors: &SafeTensors,
    prefix: &str,
    d_model: usize,
    bloq: u8,
    quantize: bool,
) -> Result<(Noun, Noun, Noun)> {
    let w_name = format!("{}.attn.c_attn.weight", prefix);
    let b_name = format!("{}.attn.c_attn.bias", prefix);

    let w = tensors.tensor(&w_name)?;
    let b = tensors.tensor(&b_name)?;

    let w_shape = w.shape();
    assert_eq!(w_shape.len(), 2);
    let rows = w_shape[0];
    let cols = w_shape[1];
    assert_eq!(cols, 3 * d_model, "c_attn weight expected [d_model, 3*d_model]");

    // Convert to f32 bytes regardless of input dtype
    let w_f32: Vec<u8> = match w.dtype() {
        Dtype::F32 => w.data().to_vec(),
        Dtype::F16 => {
            let mut out = Vec::with_capacity(w.data().len() * 2);
            for chunk in w.data().chunks_exact(2) {
                let raw = u16::from_le_bytes([chunk[0], chunk[1]]);
                let val = half::f16::from_bits(raw).to_f32();
                out.extend_from_slice(&val.to_le_bytes());
            }
            out
        }
        other => bail!("unsupported c_attn weight dtype: {:?}", other),
    };

    let b_f32: Vec<u8> = match b.dtype() {
        Dtype::F32 => b.data().to_vec(),
        _ => bail!("unexpected c_attn bias dtype"),
    };

    // Split weight: for each row, columns [0..d], [d..2d], [2d..3d]
    // These become the Q, K, V weights each of shape [rows, d_model]
    let row_stride = cols * 4; // bytes per row
    let d_bytes = d_model * 4;

    let mut wq_w = Vec::with_capacity(rows * d_bytes);
    let mut wk_w = Vec::with_capacity(rows * d_bytes);
    let mut wv_w = Vec::with_capacity(rows * d_bytes);
    for r in 0..rows {
        let row_start = r * row_stride;
        wq_w.extend_from_slice(&w_f32[row_start..row_start + d_bytes]);
        wk_w.extend_from_slice(&w_f32[row_start + d_bytes..row_start + 2 * d_bytes]);
        wv_w.extend_from_slice(&w_f32[row_start + 2 * d_bytes..row_start + 3 * d_bytes]);
    }

    // Split bias: [d_model] each
    let wq_b: Vec<u8> = b_f32[0..d_bytes].to_vec();
    let wk_b: Vec<u8> = b_f32[d_bytes..2 * d_bytes].to_vec();
    let wv_b: Vec<u8> = b_f32[2 * d_bytes..3 * d_bytes].to_vec();

    let shape_w = vec![rows, d_model];
    let shape_b = vec![1, d_model];

    let mk = |w: &[u8], b: &[u8]| -> Noun {
        let wn = if quantize {
            make_q8_weight(w, &shape_w)
        } else {
            wrap_fp(build_ray_f32(w, &shape_w, bloq))
        };
        let bn = build_ray_f32(b, &shape_b, bloq);
        Noun::Cell(Cell::from([wn, bn]))
    };
    Ok((mk(&wq_w, &wq_b), mk(&wk_w, &wk_b), mk(&wv_w, &wv_b)))
}

/// Build the model-weights noun for a GPT-2 model.
fn convert_gpt2(tensors: &SafeTensors, bloq: u8, quantize: bool) -> Result<Noun> {
    // Read shapes to infer config
    let tok = tensors.tensor("wte.weight").or_else(|_| tensors.tensor("transformer.wte.weight"))
        .context("missing token embedding")?;
    let d_model = tok.shape()[1];
    let vocab_size = tok.shape()[0];

    // Detect HF prefix style
    let has_prefix = tensors.tensor("transformer.wte.weight").is_ok();
    let p = |s: &str| if has_prefix { format!("transformer.{}", s) } else { s.to_string() };

    eprintln!("  d_model={} vocab={}", d_model, vocab_size);

    // Token embedding
    let tok_emb = tensor_to_ray(tensors, &p("wte.weight"), bloq)?;
    eprintln!("  tok_emb done");

    // Positional embedding
    let pos_emb = tensor_to_ray(tensors, &p("wpe.weight"), bloq)?;
    eprintln!("  pos_emb done");

    // Count layers
    let mut n_layers = 0usize;
    while tensors.tensor(&p(&format!("h.{}.ln_1.weight", n_layers))).is_ok() {
        n_layers += 1;
    }
    eprintln!("  n_layers={}", n_layers);

    // Build blocks list (from last to first so we can cons up a list)
    let mut blocks = Noun::Atom(Atom::from(0u8)); // null
    for i in (0..n_layers).rev() {
        let prefix = p(&format!("h.{}", i));
        let (wq, wk, wv) = split_gpt2_qkv(tensors, &prefix, d_model, bloq, quantize)?;
        let wo = linear(tensors,
            &format!("{}.attn.c_proj.weight", prefix),
            &format!("{}.attn.c_proj.bias", prefix), bloq, quantize)?;
        let ln1_g = tensor_to_ray(tensors, &format!("{}.ln_1.weight", prefix), bloq)?;
        let ln1_b = tensor_to_ray(tensors, &format!("{}.ln_1.bias", prefix), bloq)?;
        let ln2_g = tensor_to_ray(tensors, &format!("{}.ln_2.weight", prefix), bloq)?;
        let ln2_b = tensor_to_ray(tensors, &format!("{}.ln_2.bias", prefix), bloq)?;
        let ff1 = linear(tensors,
            &format!("{}.mlp.c_fc.weight", prefix),
            &format!("{}.mlp.c_fc.bias", prefix), bloq, quantize)?;
        let ff2 = linear(tensors,
            &format!("{}.mlp.c_proj.weight", prefix),
            &format!("{}.mlp.c_proj.bias", prefix), bloq, quantize)?;

        // block-weights = [wq wk wv wo ln1_g ln1_b ln2_g ln2_b ff1 ff2]
        let block = Noun::Cell(Cell::from([
            wq, wk, wv, wo, ln1_g, ln1_b, ln2_g, ln2_b, ff1, ff2,
        ]));
        blocks = Noun::Cell(Cell::from([block, blocks]));
        eprintln!("  block {} done", i);
    }

    // Final layer norm
    let ln_f_g = tensor_to_ray(tensors, &p("ln_f.weight"), bloq)?;
    let ln_f_b = tensor_to_ray(tensors, &p("ln_f.bias"), bloq)?;

    // Output projection: GPT-2 ties with token embeddings. We need [d_model, vocab_size].
    // The wte.weight is [vocab_size, d_model], so we need to transpose.
    // For now, just reuse wte.weight with a shape swap — this is INCORRECT because
    // GPT-2 ties output projection with token embeddings.
    // wte.weight is [vocab_size, d_model]; for output projection we need
    // [d_model, vocab_size] so we must transpose.
    let wte = tensors.tensor(&p("wte.weight"))?;
    let wte_shape = wte.shape();
    assert_eq!(wte_shape.len(), 2);
    let v = wte_shape[0];   // vocab_size
    let dm = wte_shape[1];  // d_model
    let wte_f32 = bytes_as_f32_vec(&wte)?;

    // transpose: out[d_m, v] = wte[v, d_m]
    let mut transposed = vec![0u8; wte_f32.len()];
    for i in 0..v {
        for j in 0..dm {
            let src = (i * dm + j) * 4;
            let dst = (j * v + i) * 4;
            transposed[dst..dst + 4].copy_from_slice(&wte_f32[src..src + 4]);
        }
    }
    let out_proj = build_ray_f32(&transposed, &[dm, v], bloq);
    eprintln!("  final layers done");

    // model-weights = [tok-emb pos-emb blocks ln-f-g ln-f-b out-proj]
    Ok(Noun::Cell(Cell::from([
        tok_emb, pos_emb, blocks, ln_f_g, ln_f_b, out_proj,
    ])))
}

// ===== Qwen3 / MLX 2-bit =====

/// Build a ray of packed uint32 data (bloq=5, kind=%uint).
/// Used for MLX-packed int2 weights: each uint32 holds 16 int2 values LSB-first.
fn build_ray_u32(bytes: &[u8], shape: &[usize]) -> Noun {
    assert_eq!(bytes.len() % 4, 0);
    let mut data_bytes = Vec::with_capacity(bytes.len() + 1);
    data_bytes.extend_from_slice(bytes);
    data_bytes.push(0x01); // MSB pin

    let data = Atom::from(data_bytes);

    let mut shape_noun = Noun::Atom(Atom::from(0u8));
    for &dim in shape.iter().rev() {
        shape_noun = Noun::Cell(Cell::from([
            Noun::Atom(Atom::from(dim as u64)),
            shape_noun,
        ]));
    }

    let meta = Noun::Cell(Cell::from([
        shape_noun,
        Noun::Atom(Atom::from(5u64)), // bloq=5 (32-bit words)
        Noun::Atom(Noun::from(kind_uint()).into_atom().unwrap()),
        Noun::Atom(Atom::from(0u8)),
    ]));

    Noun::Cell(Cell::from([meta, Noun::Atom(data)]))
}

/// Read an fp16/bf16/fp32 tensor, promote to fp32, and return as an fp32 ray.
fn f16_tensor_to_fp32_ray(tensors: &SafeTensors, name: &str) -> Result<Noun> {
    let t = tensors.tensor(name)
        .with_context(|| format!("missing tensor: {}", name))?;
    let f32_bytes = bytes_as_f32_vec(&t)?;
    Ok(build_ray_f32(&f32_bytes, t.shape(), 5))
}

/// Build an MLX-2bit weight noun: [%mlx2 w=ray scales=ray biases=ray group-size=@ud].
/// Reads `{prefix}.weight` (uint32 packed, shape [out, in/16]),
/// `{prefix}.scales` and `.biases` (f16, shape [out, in/group_size]).
/// Scales and biases are promoted to fp32.
fn mlx2_weight(tensors: &SafeTensors, prefix: &str, group_size: usize) -> Result<Noun> {
    let w_name = format!("{}.weight", prefix);
    let s_name = format!("{}.scales", prefix);
    let b_name = format!("{}.biases", prefix);

    let w = tensors.tensor(&w_name)
        .with_context(|| format!("missing tensor: {}", w_name))?;
    let s = tensors.tensor(&s_name)
        .with_context(|| format!("missing tensor: {}", s_name))?;
    let b = tensors.tensor(&b_name)
        .with_context(|| format!("missing tensor: {}", b_name))?;

    // safetensors exposes mlx-packed quantized weights as uint32.
    match w.dtype() {
        Dtype::U32 => {}
        other => bail!("mlx2 weight expected U32, got {:?} for {}", other, w_name),
    }

    let w_ray = build_ray_u32(w.data(), w.shape());

    let s_f32 = bytes_as_f32_vec(&s)?;
    let b_f32 = bytes_as_f32_vec(&b)?;
    let s_ray = build_ray_f32(&s_f32, s.shape(), 5);
    let b_ray = build_ray_f32(&b_f32, b.shape(), 5);

    Ok(Noun::Cell(Cell::from([
        Noun::Atom(tag_mlx2()),
        w_ray,
        s_ray,
        b_ray,
        Noun::Atom(Atom::from(group_size as u64)),
    ])))
}

/// Build the model-weights noun for a Qwen3 model quantized to MLX 2-bit.
///
/// Schema (matches planned `$model-weights-qwen3` in /lib/maroon.hoon):
///   [tok-emb blocks ln-f]
///
/// where:
///   tok-emb     = mlx2-weight       ::  [%mlx2 w scales biases group-size]
///   blocks      = (list block-weights-qwen3)
///   ln-f        = ray               ::  final RMSNorm gamma (fp32)
///
/// and each block is:
///   [q-proj k-proj v-proj o-proj
///    gate-proj up-proj down-proj
///    input-ln post-attn-ln
///    q-norm k-norm]
fn convert_qwen3(tensors: &SafeTensors, group_size: usize) -> Result<Noun> {
    let mut n_layers = 0usize;
    while tensors.tensor(&format!("model.layers.{}.input_layernorm.weight", n_layers)).is_ok() {
        n_layers += 1;
    }
    eprintln!("  n_layers={}", n_layers);

    // Token embedding (mlx2-quantized in this model).
    let tok_emb = mlx2_weight(tensors, "model.embed_tokens", group_size)?;
    eprintln!("  tok_emb done");

    // Build blocks list, consing from last to first.
    let mut blocks = Noun::Atom(Atom::from(0u8));
    for i in (0..n_layers).rev() {
        let p = format!("model.layers.{}", i);

        let q_proj = mlx2_weight(tensors, &format!("{}.self_attn.q_proj", p), group_size)?;
        let k_proj = mlx2_weight(tensors, &format!("{}.self_attn.k_proj", p), group_size)?;
        let v_proj = mlx2_weight(tensors, &format!("{}.self_attn.v_proj", p), group_size)?;
        let o_proj = mlx2_weight(tensors, &format!("{}.self_attn.o_proj", p), group_size)?;

        let gate_proj = mlx2_weight(tensors, &format!("{}.mlp.gate_proj", p), group_size)?;
        let up_proj   = mlx2_weight(tensors, &format!("{}.mlp.up_proj",   p), group_size)?;
        let down_proj = mlx2_weight(tensors, &format!("{}.mlp.down_proj", p), group_size)?;

        let input_ln     = f16_tensor_to_fp32_ray(tensors, &format!("{}.input_layernorm.weight", p))?;
        let post_attn_ln = f16_tensor_to_fp32_ray(tensors, &format!("{}.post_attention_layernorm.weight", p))?;
        let q_norm       = f16_tensor_to_fp32_ray(tensors, &format!("{}.self_attn.q_norm.weight", p))?;
        let k_norm       = f16_tensor_to_fp32_ray(tensors, &format!("{}.self_attn.k_norm.weight", p))?;

        // block = [q-proj k-proj v-proj o-proj gate-proj up-proj down-proj
        //          input-ln post-attn-ln q-norm k-norm]
        let block = Noun::Cell(Cell::from([
            q_proj, k_proj, v_proj, o_proj,
            gate_proj, up_proj, down_proj,
            input_ln, post_attn_ln,
            q_norm, k_norm,
        ]));
        blocks = Noun::Cell(Cell::from([block, blocks]));
        eprintln!("  block {} done", i);
    }

    let ln_f = f16_tensor_to_fp32_ray(tensors, "model.norm.weight")?;
    eprintln!("  final norm done");

    // model-weights-qwen3 = [tok-emb blocks ln-f]
    Ok(Noun::Cell(Cell::from([
        tok_emb, blocks, ln_f,
    ])))
}

// ===== BBPE tokenizer (gpt2 / qwen3 / any HF tokenizer.json) =====

/// Encode a Python-side `str_to_atom`: bytes interpreted as little-endian int.
fn str_to_atom(s: &str) -> Atom {
    if s.is_empty() {
        Atom::from(0u8)
    } else {
        Atom::from(s.as_bytes().to_vec())
    }
}

/// GPT-2's bytes_to_unicode mapping: each of 256 bytes -> a unique displayable
/// unicode codepoint. Identical for GPT-2 / Qwen / all BBPE tokenizers.
fn bytes_to_unicode() -> Vec<(u8, char)> {
    let mut bs: Vec<u32> = Vec::new();
    bs.extend((b'!' as u32)..=(b'~' as u32));
    bs.extend(0xA1u32..=0xACu32);
    bs.extend(0xAEu32..=0xFFu32);
    let mut cs: Vec<u32> = bs.clone();
    let mut n = 0u32;
    for b in 0u32..256u32 {
        if !bs.contains(&b) {
            bs.push(b);
            cs.push(256 + n);
            n += 1;
        }
    }
    bs.iter().zip(cs.iter())
        .map(|(b, c)| (*b as u8, char::from_u32(*c).unwrap()))
        .collect()
}

/// Build a balanced BST noun from key/value pairs in a fixed order.
/// Tree shape matches `+$tree` in /lib/*-tokenizer.hoon:
///   empty = 0
///   leaf  = [[k v] 0]
///   inner = [[k v] [left right]]
/// In-order traversal of the tree yields `items`.
fn build_tree(items: Vec<(Noun, Noun)>) -> Noun {
    let mut items: Vec<Option<(Noun, Noun)>> = items.into_iter().map(Some).collect();
    fn build(items: &mut [Option<(Noun, Noun)>], lo: usize, hi: usize) -> Noun {
        if lo >= hi {
            return Noun::Atom(Atom::from(0u8));
        }
        if lo + 1 == hi {
            let (k, v) = items[lo].take().unwrap();
            return Noun::Cell(Cell::from([
                Noun::Cell(Cell::from([k, v])),
                Noun::Atom(Atom::from(0u8)),
            ]));
        }
        let mid = (lo + hi) / 2;
        let (k, v) = items[mid].take().unwrap();
        let l = build(items, lo, mid);
        let r = build(items, mid + 1, hi);
        Noun::Cell(Cell::from([
            Noun::Cell(Cell::from([k, v])),
            Noun::Cell(Cell::from([l, r])),
        ]))
    }
    let len = items.len();
    build(&mut items, 0, len)
}

/// Convert a HuggingFace tokenizer.json (BBPE: gpt2, qwen, llama, etc.)
/// into a jam noun matching `$tokenizer` in our lib:
///   [vocab inverse-vocab merges byte-map inverse-byte-map]
fn convert_tokenizer(path: &std::path::Path) -> Result<Noun> {
    let bytes = std::fs::read(path)
        .with_context(|| format!("reading tokenizer.json: {:?}", path))?;
    let json: serde_json::Value = serde_json::from_slice(&bytes)
        .context("parsing tokenizer.json")?;
    let model = json.get("model").context("missing .model in tokenizer.json")?;

    // vocab: { string: id }
    let vocab_obj = model.get("vocab").and_then(|v| v.as_object())
        .context("missing .model.vocab")?;
    eprintln!("  vocab entries: {}", vocab_obj.len());

    let mut vocab_pairs: Vec<(String, u64)> = vocab_obj.iter()
        .map(|(k, v)| (k.clone(), v.as_u64().unwrap_or(0)))
        .collect();
    // Sort by key so the BST in-order traversal is deterministic.
    vocab_pairs.sort_by(|a, b| a.0.cmp(&b.0));

    // Collect special tokens from added_tokens (e.g. <|im_start|>, <|im_end|>).
    // These get two treatments:
    //   1. Added to vocab / inverse-vocab so decode can resolve them to text.
    //   2. Emitted separately in a `specials` field so the Hoon encoder can
    //      recognize them as verbatim strings and emit their single ID,
    //      rather than byte-level-encoding each character.
    let mut specials_pairs: Vec<(String, u64)> = Vec::new();
    if let Some(added) = json.get("added_tokens").and_then(|v| v.as_array()) {
        for tok in added {
            let id = tok.get("id").and_then(|v| v.as_u64()).unwrap_or(0);
            let content = tok.get("content").and_then(|v| v.as_str()).unwrap_or("");
            if content.is_empty() { continue; }
            specials_pairs.push((content.to_string(), id));
            if !vocab_obj.contains_key(content) {
                vocab_pairs.push((content.to_string(), id));
            }
        }
        if !specials_pairs.is_empty() {
            vocab_pairs.sort_by(|a, b| a.0.cmp(&b.0));
            eprintln!("  added special tokens: {}", specials_pairs.len());
        }
    }

    let vocab_items: Vec<(Noun, Noun)> = vocab_pairs.iter()
        .map(|(s, id)| (Noun::Atom(str_to_atom(s)), Noun::Atom(Atom::from(*id))))
        .collect();
    let vocab_noun = build_tree(vocab_items);

    // inverse-vocab: { id: string } — sort by id.
    let mut inv_pairs: Vec<(u64, String)> = vocab_pairs.iter()
        .map(|(s, id)| (*id, s.clone()))
        .collect();
    inv_pairs.sort_by_key(|x| x.0);
    let inv_items: Vec<(Noun, Noun)> = inv_pairs.iter()
        .map(|(id, s)| (Noun::Atom(Atom::from(*id)), Noun::Atom(str_to_atom(s))))
        .collect();
    let inverse_noun = build_tree(inv_items);

    // merges: array of "a b" strings (or [a, b] arrays in newer HF formats),
    // index in array = rank.
    let merges_arr = model.get("merges").and_then(|v| v.as_array())
        .context("missing .model.merges")?;
    eprintln!("  merges: {}", merges_arr.len());
    let mut merge_pairs: Vec<((String, String), u64)> = Vec::with_capacity(merges_arr.len());
    for (rank, m) in merges_arr.iter().enumerate() {
        let (a, b) = if let Some(s) = m.as_str() {
            // older format: "a b"
            let mut it = s.splitn(2, ' ');
            let a = it.next().unwrap_or("").to_string();
            let b = it.next().unwrap_or("").to_string();
            (a, b)
        } else if let Some(arr) = m.as_array() {
            // newer format: [a, b]
            let a = arr.get(0).and_then(|v| v.as_str()).unwrap_or("").to_string();
            let b = arr.get(1).and_then(|v| v.as_str()).unwrap_or("").to_string();
            (a, b)
        } else {
            bail!("unexpected merge entry shape: {:?}", m);
        };
        merge_pairs.push(((a, b), rank as u64));
    }
    // Sort by the [a b] cell key so the tree in-order traversal is sorted —
    // matches Hoon `map` ordering needs (vocab and inverse already sorted).
    merge_pairs.sort_by(|x, y| x.0.cmp(&y.0));
    let merge_items: Vec<(Noun, Noun)> = merge_pairs.iter()
        .map(|((a, b), rank)| {
            let key = Noun::Cell(Cell::from([
                Noun::Atom(str_to_atom(a)),
                Noun::Atom(str_to_atom(b)),
            ]));
            (key, Noun::Atom(Atom::from(*rank)))
        })
        .collect();
    let merges_noun = build_tree(merge_items);

    // byte-map and inverse-byte-map (universal BBPE byte mapping).
    let b2u = bytes_to_unicode();

    let mut bm_pairs: Vec<(u8, String)> = b2u.iter()
        .map(|(b, c)| (*b, c.to_string()))
        .collect();
    bm_pairs.sort_by_key(|x| x.0);
    let bm_items: Vec<(Noun, Noun)> = bm_pairs.iter()
        .map(|(b, s)| (Noun::Atom(Atom::from(*b as u64)), Noun::Atom(str_to_atom(s))))
        .collect();
    let byte_map_noun = build_tree(bm_items);

    let mut ibm_pairs: Vec<(String, u8)> = b2u.iter()
        .map(|(b, c)| (c.to_string(), *b))
        .collect();
    ibm_pairs.sort_by(|a, b| a.0.cmp(&b.0));
    let ibm_items: Vec<(Noun, Noun)> = ibm_pairs.iter()
        .map(|(s, b)| (Noun::Atom(str_to_atom(s)), Noun::Atom(Atom::from(*b as u64))))
        .collect();
    let inverse_byte_map_noun = build_tree(ibm_items);

    // specials: (tree [@t @ud]) — literal string → token ID.  Hoon side
    // scans the input for any of these verbatim and emits the ID directly
    // instead of byte-encoding each char through BPE.
    specials_pairs.sort_by(|a, b| a.0.cmp(&b.0));
    let specials_items: Vec<(Noun, Noun)> = specials_pairs.iter()
        .map(|(s, id)| (Noun::Atom(str_to_atom(s)), Noun::Atom(Atom::from(*id))))
        .collect();
    let specials_noun = build_tree(specials_items);

    // [vocab inverse merges byte-map inverse-byte-map specials]
    Ok(Noun::Cell(Cell::from([
        vocab_noun, inverse_noun, merges_noun, byte_map_noun, inverse_byte_map_noun, specials_noun,
    ])))
}

/// Jam a noun and write to a file.
fn jam_to_file(noun: Noun, path: &std::path::Path, label: &str) -> Result<()> {
    let jammed = noun.jam();
    let bytes = jammed.as_bytes();
    std::fs::write(path, bytes)
        .with_context(|| format!("writing {:?}", path))?;
    eprintln!("  {} wrote {} bytes to {:?}", label, bytes.len(), path);
    Ok(())
}

/// Emit one jam per layer into `out_dir/`.
///   tok-emb.jam   = tok-emb mlx2 weight
///   block-N.jam   = block N for N=0..n_layers-1
///   ln-f.jam      = final RMSNorm gamma
///   manifest.jam  = [n-layers group-size]
fn convert_qwen3_multifile(tensors: &SafeTensors, group_size: usize, out_dir: &std::path::Path) -> Result<()> {
    let mut n_layers = 0usize;
    while tensors.tensor(&format!("model.layers.{}.input_layernorm.weight", n_layers)).is_ok() {
        n_layers += 1;
    }
    eprintln!("  n_layers={}", n_layers);

    // Token embedding (one file, ~96 MB for Qwen3 1.7B at 2-bit).
    let tok_emb = mlx2_weight(tensors, "model.embed_tokens", group_size)?;
    jam_to_file(tok_emb, &out_dir.join("tok-emb.jam"), "tok-emb")?;

    // One file per block.
    for i in 0..n_layers {
        let p = format!("model.layers.{}", i);

        let q_proj = mlx2_weight(tensors, &format!("{}.self_attn.q_proj", p), group_size)?;
        let k_proj = mlx2_weight(tensors, &format!("{}.self_attn.k_proj", p), group_size)?;
        let v_proj = mlx2_weight(tensors, &format!("{}.self_attn.v_proj", p), group_size)?;
        let o_proj = mlx2_weight(tensors, &format!("{}.self_attn.o_proj", p), group_size)?;

        let gate_proj = mlx2_weight(tensors, &format!("{}.mlp.gate_proj", p), group_size)?;
        let up_proj   = mlx2_weight(tensors, &format!("{}.mlp.up_proj",   p), group_size)?;
        let down_proj = mlx2_weight(tensors, &format!("{}.mlp.down_proj", p), group_size)?;

        let input_ln     = f16_tensor_to_fp32_ray(tensors, &format!("{}.input_layernorm.weight", p))?;
        let post_attn_ln = f16_tensor_to_fp32_ray(tensors, &format!("{}.post_attention_layernorm.weight", p))?;
        let q_norm       = f16_tensor_to_fp32_ray(tensors, &format!("{}.self_attn.q_norm.weight", p))?;
        let k_norm       = f16_tensor_to_fp32_ray(tensors, &format!("{}.self_attn.k_norm.weight", p))?;

        let block = Noun::Cell(Cell::from([
            q_proj, k_proj, v_proj, o_proj,
            gate_proj, up_proj, down_proj,
            input_ln, post_attn_ln,
            q_norm, k_norm,
        ]));
        jam_to_file(block, &out_dir.join(format!("block-{}.jam", i)), &format!("block-{}", i))?;
    }

    // Final norm.
    let ln_f = f16_tensor_to_fp32_ray(tensors, "model.norm.weight")?;
    jam_to_file(ln_f, &out_dir.join("ln-f.jam"), "ln-f")?;

    // Manifest: tiny file with n-layers and group-size. Lets the loader enumerate.
    let manifest = Noun::Cell(Cell::from([
        Noun::Atom(Atom::from(n_layers as u64)),
        Noun::Atom(Atom::from(group_size as u64)),
    ]));
    jam_to_file(manifest, &out_dir.join("manifest.jam"), "manifest")?;

    Ok(())
}

fn main() -> Result<()> {
    let args = Args::parse();

    // Tokenizer mode reads JSON, not safetensors — short-circuit before mmap.
    if args.arch == "tokenizer" {
        eprintln!("converting tokenizer.json from {:?}", args.input);
        let tok = convert_tokenizer(&args.input)?;
        eprintln!("jamming...");
        let jammed = tok.jam();
        let bytes: &[u8] = jammed.as_bytes();
        eprintln!("writing {} bytes to {:?}...", bytes.len(), args.output);
        std::fs::write(&args.output, bytes)
            .with_context(|| format!("writing output: {:?}", args.output))?;
        return Ok(());
    }

    let file = File::open(&args.input)
        .with_context(|| format!("opening input: {:?}", args.input))?;
    let mmap = unsafe { Mmap::map(&file)? };
    let tensors = SafeTensors::deserialize(&mmap)?;

    eprintln!("loaded {} tensors from {:?}", tensors.names().len(), args.input);

    match args.arch.as_str() {
        "gpt2" => {
            let weights = convert_gpt2(&tensors, args.bloq, args.quantize)?;
            eprintln!("jamming...");
            let jammed = weights.jam();
            eprintln!("writing to {:?}...", args.output);
            let bytes: &[u8] = jammed.as_bytes();
            std::fs::write(&args.output, bytes)
                .with_context(|| format!("writing output: {:?}", args.output))?;
            eprintln!("wrote {} bytes", bytes.len());
        }
        "qwen3" => {
            if args.split {
                // Multi-file: emit one jam per layer into a directory.
                let out_dir = &args.output;
                std::fs::create_dir_all(out_dir)
                    .with_context(|| format!("creating output dir: {:?}", out_dir))?;
                convert_qwen3_multifile(&tensors, args.group_size, out_dir)?;
            } else {
                let weights = convert_qwen3(&tensors, args.group_size)?;
                eprintln!("jamming...");
                let jammed = weights.jam();
                let bytes: &[u8] = jammed.as_bytes();
                eprintln!("writing {} bytes to {:?}...", bytes.len(), args.output);
                std::fs::write(&args.output, bytes)
                    .with_context(|| format!("writing output: {:?}", args.output))?;
            }
        }
        other => bail!("unsupported architecture: {}", other),
    };

    Ok(())
}
