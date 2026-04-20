  ::
::::  Maroon: MAchine leaRning in hOON
::
::  Transformer inference library.
::  Forward pass only. Weights are nouns.
::  Built on Lagoon (arrays) and Saloon (activations).
::
/-  ls=lagoon
/+  *lagoon,
    math,
    saloon
::
~%  %maroon  ..part  ~
|%
::  Types
::
+$  tensor  ray:ls
::
::  Weight types for a transformer model.
::
::  A tensor as stored in model-weights:
::    [%fp r=ray]            — full precision (@rs/@rd etc)
::    [%q8 r=ray scale=@rs]  — int8 quantized (bloq=3 %uint), dequant = r*scale - offset
::  For simplicity we use symmetric quantization: no offset, zero stays zero.
::  Int8 values are stored as @uint bytes (0-255), interpreted as two's complement.
::
+$  weight-tensor
  $%  [%fp r=tensor]
      [%q8 r=tensor scale=@rs]
      ::  MLX 2-bit packed:
      ::    wq      = [out, in/16]    bloq=5 kind=%uint, 16 int2s per uint32
      ::    scales  = [out, in/gsz]   fp32 (per-group)
      ::    biases  = [out, in/gsz]   fp32 (per-group)
      ::  Dequant (fused with transpose) produces [in, out] fp32 — ready for
      ::  direct mmul as `x @ W` without a separate transpose step.
      [%mlx2 wq=tensor scales=tensor biases=tensor group-size=@ud]
  ==
::
::  A single linear projection: W=[d_in d_out], b=[1 d_out]
+$  linear-weights
  $:  w=weight-tensor  ::  weight matrix (optionally quantized)
      b=tensor         ::  bias vector (always fp)
  ==
::
::  A single transformer block (pre-norm architecture)
+$  block-weights
  $:  ::  attention projections
      wq=linear-weights
      wk=linear-weights
      wv=linear-weights
      wo=linear-weights
      ::  layer norms
      ln1-g=tensor  ln1-b=tensor  ::  pre-attention
      ln2-g=tensor  ln2-b=tensor  ::  pre-ffn
      ::  feed-forward
      ff1=linear-weights          ::  expand
      ff2=linear-weights          ::  contract
  ==
::
::  Full model weights (GPT-2 / vanilla pre-norm transformer)
+$  model-weights
  $:  tok-emb=tensor              ::  [vocab_size d_model]
      pos-emb=tensor              ::  [max_seq d_model]
      blocks=(list block-weights)
      ln-f-g=tensor               ::  final layer norm gamma
      ln-f-b=tensor               ::  final layer norm beta
      out-proj=tensor             ::  [d_model vocab_size]
  ==
::
::  Qwen3 block: GQA attention + per-head q/k RMSNorm + SwiGLU MLP + 2 RMSNorms.
::  No biases in linear projections.
+$  block-weights-qwen3
  $:  q-proj=weight-tensor         ::  [d_model d_model]
      k-proj=weight-tensor         ::  [d_model n_kv_heads*head_dim]
      v-proj=weight-tensor         ::  [d_model n_kv_heads*head_dim]
      o-proj=weight-tensor         ::  [d_model d_model]
      gate-proj=weight-tensor      ::  [d_model d_ff]
      up-proj=weight-tensor        ::  [d_model d_ff]
      down-proj=weight-tensor      ::  [d_ff d_model]
      input-ln=tensor              ::  [d_model] pre-attn RMSNorm gamma
      post-attn-ln=tensor          ::  [d_model] pre-MLP RMSNorm gamma
      q-norm=tensor                ::  [head_dim] per-head Q RMSNorm gamma
      k-norm=tensor                ::  [head_dim] per-head K RMSNorm gamma
  ==
::
::  Qwen3 model: tied embeddings (no separate out-proj, no pos-emb — uses RoPE)
+$  model-weights-qwen3
  $:  tok-emb=weight-tensor        ::  mlx2 [vocab d_model]
      blocks=(list block-weights-qwen3)
      ln-f=tensor                  ::  final RMSNorm gamma [d_model]
  ==
::
::  Qwen3 config
+$  model-config-qwen3
  $:  d-model=@ud
      n-heads=@ud
      n-kv-heads=@ud               ::  GQA: groups of n-heads/n-kv-heads
      n-layers=@ud
      d-ff=@ud
      vocab-size=@ud
      max-seq=@ud
      head-dim=@ud
      rms-eps=@rs                  ::  RMSNorm epsilon (typically 1e-6)
      rope-theta=@rs               ::  RoPE base frequency (typically 1e6 for Qwen3)
      yarn-factor=@rs              ::  YaRN interpolation factor
      yarn-orig-max-seq=@ud        ::  original training context length
      bloq=@ud
  ==
::
::  Model config
+$  model-config
  $:  d-model=@ud                 ::  embedding dimension
      n-heads=@ud                 ::  number of attention heads
      n-layers=@ud                ::  number of transformer blocks
      d-ff=@ud                    ::  feed-forward hidden dimension
      vocab-size=@ud              ::  vocabulary size
      max-seq=@ud                 ::  maximum sequence length
      =bloq                       ::  precision (5=@rs, 6=@rd)
  ==
::
::  +load-weights: cue a jammed atom into model-weights.
::
++  load-weights
  |=  jammed=@
  ^-  model-weights
  ;;(model-weights (cue jammed))
::
::  +dequant-q8-ray: int8 ray + scale -> fp32 ray.  JETTED as %dequant-q8.
::  Pure Hoon reference; the C jet does this much faster.
::
++  dequant-q8-ray
  ~/  %dequant-q8
  |=  [r=tensor scale=@rs]
  ^-  tensor
  =/  la  (lake %n)
  =/  rs-door  ~(. rs:math [%n .1e-5])
  =/  bytes  (ravel:la r)
  =/  shape-out  shape.meta.r
  =/  new-meta=meta:ls  [shape-out 5 %i754 ~]
  =/  f32-vals=(list @)
    %+  turn  bytes
    |=  b=@
    ?:  =(0 b)  .0
    ?:  (lth b 128)
      (mul:rs-door (sun:rs-door b) scale)
    =/  mag  (sun:rs-door (sub 256 b))
    (mul:rs-door (mul:rs-door mag .-1) scale)
  =/  data-out  (con data:(zeros:la new-meta) (rep 5 f32-vals))
  [new-meta data-out]
::
::  +dequant-mlx2-ray: MLX 2-bit packed weight -> fp32 ray.
::    w shape       = [out, in/16]    bloq=5 kind=%uint   (16 int2s per uint32)
::    scales shape  = [out, in/G]     bloq=5 kind=%i754   (one fp32 per group of G)
::    biases shape  = [out, in/G]     same as scales
::    out shape     = [out, in]       bloq=5 kind=%i754
::  Per-element: fp = scales[o, i/G] * (q & 0x3) + biases[o, i/G]
::                where q = (w[o, i/16] >> ((i % 16) * 2))
::  Pure Hoon reference; will need a jet for production use.
::
::  +dequant-mlx2-row: dequant a single row of an mlx2-packed weight to fp32.
::  Lets the embedding lookup avoid materializing the full [vocab, d_model]
::  fp32 table (1.2 GB on Qwen3 1.7B) just to read a handful of rows.
::    w shape:       [out, in/16] uint32
::    scales/biases: [out, in/G]  fp32
::    out:           [in]         fp32 (1-D)
::
++  dequant-mlx2-row
  ~/  %dequant-mlx2-row
  |=  [w=tensor scales=tensor biases=tensor group-size=@ud row=@ud]
  ^-  tensor
  =/  la  (lake %n)
  =/  rs-door  ~(. rs:math [%n .1e-5])
  =/  packed-cols     (snag 1 shape.meta.w)
  =/  in-features     (mul 16 packed-cols)
  =/  groups-per-row  (div in-features group-size)
  =/  w-data  data.w
  =/  s-data  data.scales
  =/  b-data  data.biases
  =/  out-vals=(list @)  ~
  =/  wc  0
  |-  ^-  tensor
  ?:  =(wc packed-cols)
    =/  out-meta=meta:ls  [~[in-features] 5 %i754 ~]
    =/  data-out  (con data:(zeros:la out-meta) (rep 5 (flop out-vals)))
    [out-meta data-out]
  =/  word    (cut 5 [(add (mul row packed-cols) wc) 1] w-data)
  =/  i-base  (mul wc 16)
  =/  out-vals-after-word
    =/  k  0
    =/  acc  out-vals
    |-  ^-  (list @)
    ?:  =(k 16)  acc
    =/  i      (add i-base k)
    =/  grp    (div i group-size)
    =/  scale  `@rs`(cut 5 [(add (mul row groups-per-row) grp) 1] s-data)
    =/  bias   `@rs`(cut 5 [(add (mul row groups-per-row) grp) 1] b-data)
    =/  q      (cut 0 [(mul k 2) 2] word)
    =/  qf     (sun:rs-door q)
    =/  val    `@`(add:rs-door (mul:rs-door scale qf) bias)
    $(k +(k), acc [val acc])
  $(wc +(wc), out-vals out-vals-after-word)
::
++  dequant-mlx2-ray
  ~/  %dequant-mlx2
  |=  [w=tensor scales=tensor biases=tensor group-size=@ud]
  ^-  tensor
  =/  la  (lake %n)
  =/  rs-door  ~(. rs:math [%n .1e-5])
  ?>  =(2 (lent shape.meta.w))
  =/  out-features    (snag 0 shape.meta.w)
  =/  packed-cols     (snag 1 shape.meta.w)
  =/  in-features     (mul 16 packed-cols)
  =/  groups-per-row  (div in-features group-size)
  =/  w-data  data.w
  =/  s-data  data.scales
  =/  b-data  data.biases
  ::  Output shape is [in_features, out_features] (fused with transpose).
  ::  Walk i outer, o inner — row-major emission matches the output layout.
  ::  For each (i, o) pair, look up: word = w[o, i/16], scale/bias[o, i/G].
  =/  out-meta=meta:ls  [~[in-features out-features] 5 %i754 ~]
  =/  out-vals=(list @)  ~
  =/  i  0
  |-  ^-  tensor
  ?:  =(i in-features)
    =/  data-out  (con data:(zeros:la out-meta) (rep 5 (flop out-vals)))
    [out-meta data-out]
  =/  word-col  (div i 16)
  =/  bit-off   (mul (mod i 16) 2)
  =/  grp       (div i group-size)
  =/  out-vals-after-i
    =/  o  0
    =/  acc  out-vals
    |-  ^-  (list @)
    ?:  =(o out-features)  acc
    =/  word   (cut 5 [(add (mul o packed-cols) word-col) 1] w-data)
    =/  q      (cut 0 [bit-off 2] word)
    =/  scale  `@rs`(cut 5 [(add (mul o groups-per-row) grp) 1] s-data)
    =/  bias   `@rs`(cut 5 [(add (mul o groups-per-row) grp) 1] b-data)
    =/  qf     (sun:rs-door q)
    =/  val    `@`(add:rs-door (mul:rs-door scale qf) bias)
    $(o +(o), acc [val acc])
  $(i +(i), out-vals out-vals-after-i)
::
::  +logits-tied-mlx2-fused: compute [1, vocab] = x @ dequant(wte).T where wte
::  is mlx2-packed. Jetted as %logits-tied-mlx2; Hoon fallback is too slow for
::  real vocabs (151k × per-row dequant in pure Hoon). At top level so the
::  C jet can dispatch against it without navigating through the `mr` core.
::
++  logits-tied-mlx2-fused
  ~/  %logits-tied-mlx2
  |=  [x=tensor w=tensor scales=tensor biases=tensor group-size=@ud]
  ^-  tensor
  =/  la  (lake %n)
  =/  rs-door  ~(. rs:math [%n .1e-5])
  ?>  =(2 (lent shape.meta.w))
  =/  vocab        (snag 0 shape.meta.w)
  =/  packed-cols  (snag 1 shape.meta.w)
  =/  d-model      (mul 16 packed-cols)
  =/  out-meta=meta:ls  [~[1 vocab] 5 %i754 ~]
  =/  vals=(list @)  ~
  =/  v  0
  |-  ^-  tensor
  ?:  =(v vocab)
    =/  data-out  (con data:(zeros:la out-meta) (rep 5 (flop vals)))
    [out-meta data-out]
  =/  wte-v  (dequant-mlx2-row w scales biases group-size v)
  =/  wte-2d  (reshape:la wte-v ~[d-model 1])
  =/  dot  (mmul:la x wte-2d)
  =/  val  `@`(get-item:la dot ~[0 0])
  $(v +(v), vals [val vals])
::
::  +mmul-mlx2: fused mlx2-dequant+matmul.  y = x @ dequant(w, scales, biases).
::    x:      [S, in_features] fp32
::    w:      [out_features, in_features/16] uint32 (2-bit packed)
::    scales: [out_features, in_features/group_size] fp32
::    biases: [out_features, in_features/group_size] fp32
::    output: [S, out_features] fp32
::
::  Jetted as %mmul-mlx2.  The C jet routes to backend_mmul_mlx2 which, when
::  CUDA is built in, runs a fused kernel against VRAM-cached weight buffers
::  (no per-call dequant materialization).  Pure-Hoon fallback composes the
::  two jetted primitives we already have — correct but slow.
::
++  mmul-mlx2
  ~/  %mmul-mlx2
  |=  [x=tensor w=tensor scales=tensor biases=tensor group-size=@ud]
  ^-  tensor
  =/  la  (lake %n)
  (mmul:la x (dequant-mlx2-ray w scales biases group-size))
::
::  +rms-norm-row: row-wise RMSNorm on an [S, D] fp32 tensor, applying
::  gamma[D].  Jetted as %rms-norm-row; GPU path keeps determinism via
::  sequential fmaf reduction per row.  Hoon fallback composes the
::  existing saloon primitives; correct but catastrophically slow at
::  transformer shapes because of per-element set-item on D-wide atoms.
::
++  rms-norm-row
  ~/  %rms-norm-row
  |=  [x=tensor gamma=tensor eps=@rs]
  ^-  tensor
  =/  la  (lake %n)
  =/  n-rows  (snag 0 shape.meta.x)
  =/  d       (snag 1 shape.meta.x)
  =/  i       0
  =/  out     x
  |-  ^-  tensor
  ?:  =(i n-rows)  out
  =/  row     (get-row:la out ~[i])
  =/  r1d     (reshape:la row ~[d])
  =/  normed  (rms-norm:sa:saloon r1d gamma eps)
  =/  n2d     (reshape:la normed ~[1 d])
  $(i +(i), out (set-row:la out ~[i] n2d))
::
++  mr
  =+  [rnd=*rounding-mode]
  |%
  ::
  ::  +linear: apply a linear projection
  ::
  ::  x=[S d_in] @ W=[d_in d_out] + b=[1 d_out] -> [S d_out]
  ::
  ++  linear
    |=  [x=tensor lw=linear-weights]
    ^-  tensor
    =,  (lake rnd)
    =/  w-fp  (dequantize w.lw)
    =/  out  (mmul x w-fp)
    ::  broadcast bias across rows: add b to each row of out
    (add-bias out b.lw)
  ::
  ::  +dequantize: return a tensor as fp32, dequantizing from int8 if needed.
  ::  Calls the top-level jetted +dequant-q8-ray.
  ::
  ++  dequantize
    |=  w=weight-tensor
    ^-  tensor
    ?-  -.w
      %fp    r.w
      %q8    (dequant-q8-ray r.w scale.w)
      %mlx2  (dequant-mlx2-ray wq.w scales.w biases.w group-size.w)
    ==
  ::
  ::  +add-bias: add a [1 D] or [D] bias to each row of [S D]
  ::
  ++  add-bias
    |=  [x=tensor b=tensor]
    ^-  tensor
    =,  (lake rnd)
    =/  n-rows  (snag 0 shape.meta.x)
    =/  i  0
    =/  out  x
    |-  ^-  tensor
    ?:  =(i n-rows)  out
    =/  row  (get-row out ~[i])
    =/  new-row  (add row b)
    $(i +(i), out (set-row out ~[i] new-row))
  ::
  ::  +attention: scaled dot-product attention
  ::
  ::  Q=[S d_k], K=[S d_k], V=[S d_v] -> [S d_v]
  ::  scores = softmax(Q @ K^T / sqrt(d_k))
  ::  output = scores @ V
  ::
  ::  +attention: scaled dot-product attention with causal mask
  ::
  ::  Q=[S d_k], K=[S d_k], V=[S d_v] -> [S d_v]
  ::  scores = softmax(Q @ K^T / sqrt(d_k) + mask)
  ::  output = scores @ V
  ::
  ++  attention
    |=  [q=tensor k=tensor v=tensor]
    ^-  tensor
    =,  (lake rnd)
    ::  d_k = last dimension of Q
    =/  d-k  (snag 1 shape.meta.q)
    =/  seq-len  (snag 0 shape.meta.q)
    ::  scores = (Q @ K^T) / sqrt(d_k)
    =/  kt  (transpose2d k)
    =/  scores  (mmul q kt)
    =/  dk-float  (fsun:sa:saloon bloq.meta.q kind.meta.q d-k)
    =/  sqrt-dk   (fsqrt bloq.meta.q kind.meta.q dk-float)
    =/  scores  (div-scalar scores sqrt-dk)
    ::  apply causal mask: set scores[i][j] to -inf where j > i
    =/  neg-inf  (fcon:sa:saloon bloq.meta.q kind.meta.q %neg-inf)
    =/  i  0
    =/  scores
      |-  ^-  tensor
      ?:  =(i seq-len)  scores
      =/  j  +(i)
      =.  scores
        |-  ^-  tensor
        ?:  =(j seq-len)  scores
        $(j +(j), scores (set-item scores ~[i j] neg-inf))
      $(i +(i), scores scores)
    ::  apply softmax row-by-row
    =/  i  0
    =/  attn  scores
    |-  ^-  tensor
    ?:  =(i seq-len)
      ::  attn @ V -> [S d_v]
      (mmul attn v)
    =/  row  (get-row scores ~[i])
    =/  sm-row  (softmax:sa:saloon row)
    $(i +(i), attn (set-row attn ~[i] sm-row))
  ::
  ::  +multi-head-attention: split into heads, attend, concat, project
  ::
  ::  x=[S d_model], weights -> [S d_model]
  ::
  ++  multi-head-attention
    |=  [x=tensor n-heads=@ud wq=linear-weights wk=linear-weights wv=linear-weights wo=linear-weights]
    ^-  tensor
    =,  (lake rnd)
    ::  project to Q, K, V  [S d_model]
    =/  q  (linear x wq)
    =/  k  (linear x wk)
    =/  v  (linear x wv)
    =/  d-model  (snag 1 shape.meta.q)
    =/  d-k  (^div d-model n-heads)
    ::  split into heads, attend, concat
    =/  h  0
    =/  results=(list tensor)  ~
    |-  ^-  tensor
    ?:  =(h n-heads)
      ::  concat all head results along columns
      =/  out
        ?~  results  !!
        =/  acc  i.results
        =/  rest  t.results
        |-  ^-  tensor
        ?~  rest  acc
        $(acc (hstack-2d acc i.rest), rest t.rest)
      ::  output projection
      (linear out wo)
    ::  extract columns [h*d_k, (h+1)*d_k - 1] for this head
    =/  col-start  (^mul h d-k)
    =/  col-end  (dec (^mul +(h) d-k))
    =/  q-h  (cols q col-start col-end)
    =/  k-h  (cols k col-start col-end)
    =/  v-h  (cols v col-start col-end)
    ::  attend this head
    =/  head-out  (attention q-h k-h v-h)
    $(h +(h), results (snoc results head-out))
  ::
  ::  +feed-forward: two-layer FFN with GELU
  ::
  ::  x=[S d_model] -> [S d_model]
  ::  hidden = gelu(x @ W1 + b1)
  ::  out = hidden @ W2 + b2
  ::
  ++  feed-forward
    |=  [x=tensor ff1=linear-weights ff2=linear-weights]
    ^-  tensor
    =/  hidden  (gelu:sa:saloon (linear x ff1))
    (linear hidden ff2)
  ::
  ::  +layer-norm-2d: apply layer-norm row-by-row on a [S D] tensor
  ::  gamma and beta are [D] (1D), x is [S D] (2D)
  ::
  ++  layer-norm-2d
    |=  [x=tensor gamma=tensor beta=tensor]
    ^-  tensor
    =/  la  (lake rnd)
    =/  n-rows  (snag 0 shape.meta.x)
    =/  i  0
    =/  out  x
    |-  ^-  tensor
    ?:  =(i n-rows)  out
    =/  row  (get-row:la x ~[i])
    ::  reshape row from [1 D] to [D] for layer-norm
    =/  d  (snag 1 shape.meta.x)
    =/  row-1d  (reshape:la row ~[d])
    =/  normed  (layer-norm:sa:saloon row-1d gamma beta)
    ::  reshape back to [1 D] for set-row
    =/  normed-2d  (reshape:la normed ~[1 d])
    $(i +(i), out (set-row:la out ~[i] normed-2d))
  ::
  ::  +transformer-block: one transformer block (pre-norm)
  ::
  ::  x=[S d_model] -> [S d_model]
  ::
  ++  transformer-block
    |=  [x=tensor bw=block-weights n-heads=@ud]
    ^-  tensor
    =,  (lake rnd)
    ::  pre-norm attention
    =/  normed  (layer-norm-2d x ln1-g.bw ln1-b.bw)
    =/  attn-out
      (multi-head-attention normed n-heads wq.bw wk.bw wv.bw wo.bw)
    ::  residual
    =/  x  (add x attn-out)
    ::  pre-norm feed-forward
    =/  normed  (layer-norm-2d x ln2-g.bw ln2-b.bw)
    =/  ff-out  (feed-forward normed ff1.bw ff2.bw)
    ::  residual
    (add x ff-out)
  ::
  ::  +forward: full model forward pass
  ::
  ::  tokens=[S] (list of token indices) -> logits=[vocab_size]
  ::  Returns logits for the LAST token position.
  ::
  ++  forward
    |=  [tokens=(list @ud) weights=model-weights config=model-config]
    ^-  tensor
    =,  (lake rnd)
    =/  seq-len  (lent tokens)
    ::  token embeddings: look up each token
    =/  x  (embed tokens tok-emb.weights bloq.config)
    =/  d-model  (snag 1 shape.meta.x)
    ::  add positional embeddings (first seq-len rows).
    ::  NOTE: don't use `submatrix` here — when seq-len=1 the slice becomes
    ::  `[0 0]` which lagoon reads as `[start=0 end=unset]` (whole dim).
    ::  Copy rows explicitly.
    =/  pos=tensor
      =/  init  (zeros [~[seq-len d-model] bloq.config %i754 ~])
      =/  i  0
      |-  ^-  tensor
      ?:  =(i seq-len)  init
      =/  row  (get-row pos-emb.weights ~[i])
      $(i +(i), init (set-row init ~[i] row))
    =/  x  (add x pos)
    =/  last-idx  (dec seq-len)
    =/  dbg-first5
      |=  t=tensor  ^-  (list @rs)
      :~  `@rs`(get-item t ~[last-idx 0])
          `@rs`(get-item t ~[last-idx 1])
          `@rs`(get-item t ~[last-idx 2])
          `@rs`(get-item t ~[last-idx 3])
          `@rs`(get-item t ~[last-idx 4])
      ==
    ~&  >  ['HN embed+pos ' (dbg-first5 x)]
    ::  Block 0 inlined with per-step debug, so we can pinpoint divergence.
    =/  blk0  (snag 0 blocks.weights)
    =/  ln1-out  (layer-norm-2d x ln1-g.blk0 ln1-b.blk0)
    ~&  >  ['HN blk0 ln1 ' (dbg-first5 ln1-out)]
    =/  q-full  (linear ln1-out wq.blk0)
    ~&  >  ['HN blk0 Q ' (dbg-first5 q-full)]
    =/  k-full  (linear ln1-out wk.blk0)
    ~&  >  ['HN blk0 K ' (dbg-first5 k-full)]
    =/  v-full  (linear ln1-out wv.blk0)
    ~&  >  ['HN blk0 V ' (dbg-first5 v-full)]
    ::  inline MHA head 0 (cols 0..d-head-1) for debug
    =/  d-head  (^div d-model n-heads.config)
    =/  q-h0  (cols q-full 0 (dec d-head))
    =/  k-h0  (cols k-full 0 (dec d-head))
    =/  v-h0  (cols v-full 0 (dec d-head))
    ~&  >  ['HN blk0 q-h0 ' (dbg-first5 q-h0)]
    ~&  >  ['HN blk0 k-h0 ' (dbg-first5 k-h0)]
    =/  kt-h0  (transpose2d k-h0)
    =/  scores0  (mmul q-h0 kt-h0)
    =/  dk-f  (fsun:sa:saloon bloq.meta.q-h0 kind.meta.q-h0 d-head)
    =/  sqrt-dk  (fsqrt bloq.meta.q-h0 kind.meta.q-h0 dk-f)
    =/  scores-scaled  (div-scalar scores0 sqrt-dk)
    ::  print last row of scaled scores (all 5 positions)
    =/  dbg-row5
      |=  t=tensor  ^-  (list @rs)
      :~  `@rs`(get-item t ~[last-idx 0])
          `@rs`(get-item t ~[last-idx 1])
          `@rs`(get-item t ~[last-idx 2])
          `@rs`(get-item t ~[last-idx 3])
          `@rs`(get-item t ~[last-idx 4])
      ==
    ~&  >  ['HN blk0 scaled-scores last-row ' (dbg-row5 scores-scaled)]
    ::  apply causal mask
    =/  neg-inf  (fcon:sa:saloon bloq.meta.q-h0 kind.meta.q-h0 %neg-inf)
    =/  sc-masked
      =/  ii  0
      =/  sc  scores-scaled
      |-  ^-  tensor
      ?:  =(ii seq-len)  sc
      =/  jj  +(ii)
      =.  sc
        |-  ^-  tensor
        ?:  =(jj seq-len)  sc
        $(jj +(jj), sc (set-item sc ~[ii jj] neg-inf))
      $(ii +(ii), sc sc)
    ~&  >  ['HN blk0 masked-scores last-row ' (dbg-row5 sc-masked)]
    ::  full softmax (row-by-row)
    =/  sm
      =/  ii  0
      =/  a  sc-masked
      |-  ^-  tensor
      ?:  =(ii seq-len)  a
      =/  row  (get-row sc-masked ~[ii])
      =/  smr  (softmax:sa:saloon row)
      $(ii +(ii), a (set-row a ~[ii] smr))
    ~&  >  ['HN blk0 softmax last-row ' (dbg-row5 sm)]
    =/  head0-out  (mmul sm v-h0)
    ~&  >  ['HN blk0 head0 ' (dbg-first5 head0-out)]
    =/  attn-out
      (multi-head-attention ln1-out n-heads.config wq.blk0 wk.blk0 wv.blk0 wo.blk0)
    ~&  >  ['HN blk0 attn ' (dbg-first5 attn-out)]
    =/  x  (add x attn-out)
    ~&  >  ['HN blk0 after-res1 ' (dbg-first5 x)]
    =/  ln2-out  (layer-norm-2d x ln2-g.blk0 ln2-b.blk0)
    ~&  >  ['HN blk0 ln2 ' (dbg-first5 ln2-out)]
    =/  ff-out  (feed-forward ln2-out ff1.blk0 ff2.blk0)
    ~&  >  ['HN blk0 ff ' (dbg-first5 ff-out)]
    =/  x  (add x ff-out)
    ~&  >  ['HN blk0 out ' (dbg-first5 x)]
    ::  run remaining blocks (1..N-1)
    =/  blks  (slag 1 blocks.weights)
    =/  blk-idx  1
    |-  ^-  tensor
    ?~  blks
      ::  final layer norm
      =/  x  (layer-norm-2d x ln-f-g.weights ln-f-b.weights)
      ~&  >  ['HN after final-LN ' (dbg-first5 x)]
      ::  project last position to vocab logits
      =/  last-row  (get-row x ~[last-idx])
      =/  logits
        (linear last-row [[%fp out-proj.weights] (zeros [~[1 vocab-size.config] bloq.config %i754 ~])])
      ~&  >  ['HN logit[262] ' `@rs`(get-item logits ~[0 262])]
      ~&  >  ['HN logit[257] ' `@rs`(get-item logits ~[0 257])]
      ~&  >  ['HN logit[49.994] ' `@rs`(get-item logits ~[0 49.994])]
      ~&  >  ['HN logit[0] ' `@rs`(get-item logits ~[0 0])]
      ~&  >  ['HN logit[1] ' `@rs`(get-item logits ~[0 1])]
      ~&  >  ['HN logit[last] ' `@rs`(get-item logits ~[0 (dec vocab-size.config)])]
      logits
    =/  x-out  (transformer-block x i.blks n-heads.config)
    ~&  >  ['HN blk' blk-idx (dbg-first5 x-out)]
    $(blks t.blks, x x-out, blk-idx +(blk-idx))
  ::
  ::  +embed: look up token embeddings
  ::
  ::  tokens=(list @ud) -> [S d_model]
  ::
  ++  embed
    |=  [tokens=(list @ud) emb-table=tensor =bloq]
    ^-  tensor
    =,  (lake rnd)
    =/  d-model  (snag 1 shape.meta.emb-table)
    =/  seq-len  (lent tokens)
    =/  out  (zeros [~[seq-len d-model] bloq %i754 ~])
    =/  i  0
    |-  ^-  tensor
    ?~  tokens  out
    =/  row  (get-row emb-table ~[i.tokens])
    $(tokens t.tokens, i +(i), out (set-row out ~[i] row))
  ::
  ::  =======================================================================
  ::  Qwen3-family primitives (RMSNorm, SwiGLU, RoPE+YaRN, GQA).
  ::  Generic transformer pieces used by any Llama/Qwen-style model.
  ::  =======================================================================
  ::
  ::  +rms-norm-2d: apply RMSNorm row-by-row on a [S D] tensor.
  ::  gamma is [D]; x is [S D]. No bias.
  ::
  ++  rms-norm-2d
    |=  [x=tensor gamma=tensor eps=@rs]
    ^-  tensor
    ::  Delegate to the top-level jetted arm.  Keeping this wrapper so
    ::  existing callers (forward, forward-qwen3) don't have to change.
    (rms-norm-row x gamma eps)
  ::
  ::  +silu-mul: element-wise silu(a) * b. Used in SwiGLU MLP.
  ::
  ++  silu-mul
    |=  [a=tensor b=tensor]
    ^-  tensor
    =,  (lake rnd)
    (mul (silu:sa:saloon a) b)
  ::
  ::  +swiglu-mlp: Qwen/Llama gated MLP. No biases.
  ::    hidden = silu(gate-proj(x)) * up-proj(x)
  ::    out    = down-proj(hidden)
  ::
  ++  swiglu-mlp
    |=  $:  x=tensor
            gate-w=weight-tensor
            up-w=weight-tensor
            down-w=weight-tensor
        ==
    ^-  tensor
    =,  (lake rnd)
    =/  gate   (linear-nobias x gate-w)
    =/  up     (linear-nobias x up-w)
    =/  hidden  (silu-mul gate up)
    (linear-nobias hidden down-w)
  ::
  ::  +linear-nobias: x @ W.T for a weight-tensor with no bias.
  ::  MLX (and HuggingFace) stores linear weights as [out, in]; our mmul
  ::  expects [in, out], so transpose after dequant. No-op in the equivalent
  ::  fp path if caller already stored in [in, out].
  ::
  ++  linear-nobias
    |=  [x=tensor w=weight-tensor]
    ^-  tensor
    =,  (lake rnd)
    ::  For %mlx2 weights, route to the fused +mmul-mlx2 — jetted to a single
    ::  dequant+matmul kernel that runs on GPU with VRAM-cached weights when
    ::  CUDA is built in.  %fp and %q8 dequant first, then plain mmul.
    ?-  -.w
      %fp    (mmul x r.w)
      %q8    (mmul x (dequant-q8-ray r.w scale.w))
      %mlx2  (mmul-mlx2 x wq.w scales.w biases.w group-size.w)
    ==
  ::
  ::  +fast-transpose: in-place shape-wise 2D transpose of a fp32 ray.
  ::  Both lagoon's jetted transpose (has _check oddities) and maroon's
  ::  transpose2d (pure-Hoon per-element set-item) OOM on 2k-dim weights.
  ::  This builds a fresh output atom via cut on input data + list/rep on
  ::  the way out, so one allocation at the end, no tree walks.
  ::
  ++  fast-transpose
    |=  a=tensor
    ^-  tensor
    =,  (lake rnd)
    ?>  =(2 (lent shape.meta.a))
    =/  rows  (snag 0 shape.meta.a)
    =/  cols  (snag 1 shape.meta.a)
    =/  a-data  data.a
    =/  out-meta=meta:ls  [~[cols rows] bloq.meta.a kind.meta.a ~]
    ::  Emit in column-major order so transposed element (i, j) comes from
    ::  a[j, i] = a-data at linear index j*cols + i. Traverse j outer, i inner
    ::  to build row-major output where row = old column.
    =/  vals=(list @)  ~
    =/  i  0
    |-  ^-  tensor
    ?:  =(i cols)
      =/  data-out  (con data:(zeros out-meta) (rep bloq.meta.a (flop vals)))
      [out-meta data-out]
    =/  vals-after-i
      =/  j  0
      =/  acc  vals
      |-  ^-  (list @)
      ?:  =(j rows)  acc
      =/  v  (cut bloq.meta.a [(^add (^mul j cols) i) 1] a-data)
      $(j +(j), acc [v acc])
    $(i +(i), vals vals-after-i)
  ::
  ::  +rope-inv-freq: compute [head_dim/2] inverse frequencies with YaRN scaling.
  ::    inv_freq[j] = 1 / base^(2j/head-dim)
  ::    YaRN piecewise modifies:
  ::      wavelen = 2pi / inv_freq
  ::      wavelen < high_freq_wavelen:  keep
  ::      wavelen > low_freq_wavelen:   divide by factor
  ::      else:                         smooth interpolate
  ::  low_freq_factor=1, high_freq_factor=32 are HF defaults for Qwen3.
  ::
  ++  rope-inv-freq
    |=  $:  head-dim=@ud
            base=@rs
            orig-max=@ud
            factor=@rs       ::  yarn factor (1 = plain RoPE)
        ==
    ^-  tensor
    =,  (lake rnd)
    =/  rs  ~(. rs:math [%n .1e-5])
    =/  half       (^div head-dim 2)
    =/  meta       `meta:ls`[~[half] 5 %i754 ~]
    =/  out        (zeros meta)
    =/  low-fac    .1
    =/  high-fac   .32
    =/  pi2        (mul:rs .2 .3.14159265)
    =/  low-wav    (div:rs (sun:rs orig-max) low-fac)   ::  8192
    =/  high-wav   (div:rs (sun:rs orig-max) high-fac)  ::  256
    =/  hd-rs      (sun:rs head-dim)
    =/  j          0
    |-  ^-  tensor
    ?:  =(j half)  out
    ::  exponent = 2j/head_dim
    =/  two-j      (sun:rs (^mul 2 j))
    =/  expnt      (div:rs two-j hd-rs)
    ::  base_pow = base ** expnt = exp(expnt * ln(base))
    =/  ln-base    (log:rs base)
    =/  base-pow   (exp:rs (mul:rs expnt ln-base))
    =/  inv        (div:rs .1 base-pow)
    =/  wavelen    (div:rs pi2 inv)
    ::  YaRN piecewise (factor=1 short-circuits to plain RoPE)
    =/  scaled
      ?:  =(factor .1)  inv
      ?:  (lth:rs wavelen high-wav)  inv
      ?:  (gth:rs wavelen low-wav)   (div:rs inv factor)
      ::  smooth ramp
      =/  s  (div:rs (sub:rs (div:rs (sun:rs orig-max) wavelen) low-fac) (sub:rs high-fac low-fac))
      =/  one-minus-s  (sub:rs .1 s)
      (add:rs (mul:rs one-minus-s (div:rs inv factor)) (mul:rs s inv))
    $(j +(j), out (set-item out ~[j] scaled))
  ::
  ::  +rope-attention-factor: YaRN attention scaling applied uniformly to cos/sin.
  ::    attention_factor = 0.1 * ln(factor) + 1.0 for factor > 1; else 1.0
  ::
  ++  rope-attention-factor
    |=  factor=@rs  ^-  @rs
    =/  rs  ~(. rs:math [%n .1e-5])
    ?:  =(factor .1)  .1
    (add:rs (mul:rs .0.1 (log:rs factor)) .1)
  ::
  ::  +rope-cos-sin: build [seq-len, head-dim] cos and sin tables.
  ::    For each position p, each dim j in [0, head_dim):
  ::      j < head_dim/2:  angle = inv_freq[j] * p
  ::      j >= head_dim/2: angle = inv_freq[j - head_dim/2] * p  (mirrored)
  ::    cos[p, j] = cos(angle) * attn_factor
  ::    sin[p, j] = sin(angle) * attn_factor
  ::
  ++  rope-cos-sin
    |=  [seq-len=@ud head-dim=@ud inv-freq=tensor attn-factor=@rs]
    ^-  [cos=tensor sin=tensor]
    =,  (lake rnd)
    =/  rs  ~(. rs:math [%n .1e-5])
    =/  meta=meta:ls  [~[seq-len head-dim] 5 %i754 ~]
    =/  half  (^div head-dim 2)
    =/  cos-out  (zeros meta)
    =/  sin-out  (zeros meta)
    =/  p  0
    |-  ^-  [tensor tensor]
    ?:  =(p seq-len)  [cos-out sin-out]
    ::  Inner j-loop returns [cos-out sin-out] as a pair so both mutations
    ::  propagate back out (a single `=.` would only yield one variable).
    =/  pq=[tensor tensor]
      =/  j  0
      |-  ^-  [tensor tensor]
      ?:  =(j head-dim)  [cos-out sin-out]
      =/  j-mod  ?:((^lth j half) j (^sub j half))
      =/  inv    `@rs`(get-item inv-freq ~[j-mod])
      =/  angle  (mul:rs (sun:rs p) inv)
      =/  c      (mul:rs (cos:rs angle) attn-factor)
      =/  s      (mul:rs (sin:rs angle) attn-factor)
      %=  $
        j        +(j)
        cos-out  (set-item cos-out ~[p j] c)
        sin-out  (set-item sin-out ~[p j] s)
      ==
    $(p +(p), cos-out -.pq, sin-out +.pq)
  ::
  ::  +rope-apply: rotate a [S, H, D_head] tensor in half-rotated convention.
  ::    cos, sin are [S, D_head] tables from +rope-cos-sin.
  ::    out[p, h, j]            = x[p, h, j] * cos[p, j] + rotate_half(x)[p, h, j] * sin[p, j]
  ::    where rotate_half(x)[p, h, j] =
  ::      j <  D_head/2:  -x[p, h, j + D_head/2]
  ::      j >= D_head/2:   x[p, h, j - D_head/2]
  ::
  ++  rope-apply
    |=  [x=tensor cos=tensor sin=tensor]
    ^-  tensor
    =,  (lake rnd)
    =/  rs  ~(. rs:math [%n .1e-5])
    =/  seq-len  (snag 0 shape.meta.x)
    =/  n-heads  (snag 1 shape.meta.x)
    =/  d-head   (snag 2 shape.meta.x)
    =/  half     (^div d-head 2)
    ::  Walk (p, h, j) in output row-major (j innermost), computing each value;
    ::  rep them into one atom at the end. Avoids per-element set-item on a
    ::  4M-word ray (catastrophic allocation).
    =/  out-meta=meta:ls  [~[seq-len n-heads d-head] 5 %i754 ~]
    =/  vals=(list @)  ~
    =/  p  0
    |-  ^-  tensor
    ?:  =(p seq-len)
      =/  data-out  (con data:(zeros out-meta) (rep 5 (flop vals)))
      [out-meta data-out]
    =/  vals-after-p
      =/  h  0
      =/  acc  vals
      |-  ^-  (list @)
      ?:  =(h n-heads)  acc
      =/  acc-after-h
        =/  j  0
        =/  acc2  acc
        |-  ^-  (list @)
        ?:  =(j d-head)  acc2
        =/  xj     `@rs`(get-item x ~[p h j])
        =/  cp     `@rs`(get-item cos ~[p j])
        =/  sp     `@rs`(get-item sin ~[p j])
        =/  rj-idx  ?:((^lth j half) (^add j half) (^sub j half))
        =/  xr     `@rs`(get-item x ~[p h rj-idx])
        =/  rot    ?:((^lth j half) (mul:rs xr .-1) xr)
        =/  o      `@`(add:rs (mul:rs xj cp) (mul:rs rot sp))
        $(j +(j), acc2 [o acc2])
      $(h +(h), acc acc-after-h)
    $(p +(p), vals vals-after-p)
  ::
  ::  +per-head-rms-norm: apply RMSNorm to each head's [D_head] vector.
  ::    x:     [S, H, D_head]
  ::    gamma: [D_head]
  ::    out:   same shape; each head's feature vec normalized independently.
  ::
  ++  per-head-rms-norm
    |=  [x=tensor gamma=tensor eps=@rs]
    ^-  tensor
    =,  (lake rnd)
    =/  rs  ~(. rs:math [%n .1e-5])
    =/  seq-len  (snag 0 shape.meta.x)
    =/  n-heads  (snag 1 shape.meta.x)
    =/  d-head   (snag 2 shape.meta.x)
    =/  x-data   data.x
    =/  g-data   data.gamma
    =/  d-rs     (sun:rs d-head)
    =/  out-meta=meta:ls  [~[seq-len n-heads d-head] 5 %i754 ~]
    ::  Walk (p, h) pairs. For each, compute rms scalar, then emit D_head vals.
    =/  vals=(list @)  ~
    =/  p  0
    |-  ^-  tensor
    ?:  =(p seq-len)
      =/  data-out  (con data:(zeros out-meta) (rep 5 (flop vals)))
      [out-meta data-out]
    =/  vals-after-p
      =/  h  0
      =/  acc  vals
      |-  ^-  (list @)
      ?:  =(h n-heads)  acc
      =/  base  (^mul (^add (^mul p n-heads) h) d-head)
      ::  Compute sum of squares for this head.
      =/  ss
        =/  j  0
        =/  s  `@rs`.0
        |-  ^-  @rs
        ?:  =(j d-head)  s
        =/  v  `@rs`(cut 5 [(^add base j) 1] x-data)
        $(j +(j), s (add:rs s (mul:rs v v)))
      =/  ms     (div:rs ss d-rs)
      =/  rms-v  (sqrt:rs (add:rs ms eps))
      ::  Emit D_head values: gamma[j] * x[p,h,j] / rms_v
      =/  acc-after-h
        =/  j  0
        =/  a  acc
        |-  ^-  (list @)
        ?:  =(j d-head)  a
        =/  xv   `@rs`(cut 5 [(^add base j) 1] x-data)
        =/  gv   `@rs`(cut 5 [j 1] g-data)
        =/  out  `@`(div:rs (mul:rs gv xv) rms-v)
        $(j +(j), a [out a])
      $(h +(h), acc acc-after-h)
    $(p +(p), vals vals-after-p)
  ::
  ::  +reshape-heads: [S, H*D_head] -> [S, H, D_head].
  ::  Same data layout in memory; just reinterpret the shape. `reshape` from
  ::  lagoon does this in-place with no copy.
  ::
  ++  reshape-heads
    |=  [x=tensor n-heads=@ud d-head=@ud]
    ^-  tensor
    =,  (lake rnd)
    =/  seq-len  (snag 0 shape.meta.x)
    (reshape x ~[seq-len n-heads d-head])
  ::
  ::  +flatten-heads: [S, H, D_head] -> [S, H*D_head]. Same as reshape-heads
  ::  in reverse — just reinterpret the 3-D shape as 2-D.
  ::
  ++  flatten-heads
    |=  x=tensor
    ^-  tensor
    =,  (lake rnd)
    =/  seq-len  (snag 0 shape.meta.x)
    =/  n-heads  (snag 1 shape.meta.x)
    =/  d-head   (snag 2 shape.meta.x)
    (reshape x ~[seq-len (^mul n-heads d-head)])
  ::
  ::  +gqa-attention: grouped-query attention over pre-RoPE'd q,k,v tensors.
  ::    q:    [S, n_heads, D_head]
  ::    k, v: [S, n_kv_heads, D_head]
  ::    out:  [S, n_heads*D_head] (flattened)
  ::  Each query head h uses kv head h / (n_heads/n_kv_heads).
  ::  Standard scaled dot-product attention with causal mask.
  ::
  ++  gqa-attention
    |=  [q=tensor k=tensor v=tensor]
    ^-  tensor
    =,  (lake rnd)
    =/  seq-len     (snag 0 shape.meta.q)
    =/  n-heads     (snag 1 shape.meta.q)
    =/  d-head      (snag 2 shape.meta.q)
    =/  n-kv-heads  (snag 1 shape.meta.k)
    =/  group       (^div n-heads n-kv-heads)
    =/  d-model     (^mul n-heads d-head)
    ::  Stream per-head attention outputs into a flat list in column order
    ::  out[p, c] where c = h*d_head + j. Linear index = p*d_model + h*d_head + j.
    ::  Build as a reversed list, then rep at end — avoids set-item on [S, d_model].
    ::
    ::  But the per-head output has layout [S, D_h] while the final layout
    ::  demands [S, d_model]. To write into out[p, :] row-major, we need all
    ::  n_heads head outputs for row p BEFORE moving to row p+1. Rather than
    ::  accumulating per-head tensors and then re-streaming, we simply compute
    ::  all head outputs as [S, D_h] tensors once, then stream row-by-row.
    =/  head-outs=(list tensor)
      =/  h  0
      =|  acc=(list tensor)
      |-
      ?:  =(h n-heads)  (flop acc)
      =/  kv-h  (^div h group)
      =/  q-h  (extract-head-2d q h)
      =/  k-h  (extract-head-2d k kv-h)
      =/  v-h  (extract-head-2d v kv-h)
      =/  head-out  (attention q-h k-h v-h)          ::  [S, D_h]
      $(h +(h), acc [head-out acc])
    ::  Now stream out row-by-row across all heads, packing into one atom.
    =/  out-meta=meta:ls  [~[seq-len d-model] 5 %i754 ~]
    =/  vals=(list @)  ~
    =/  p  0
    |-  ^-  tensor
    ?:  =(p seq-len)
      =/  data-out  (con data:(zeros out-meta) (rep 5 (flop vals)))
      [out-meta data-out]
    =/  vals-after-p
      =/  acc  vals
      =/  h  0
      =/  heads  head-outs
      |-  ^-  (list @)
      ?~  heads  acc
      =/  head-data  data.i.heads
      =/  base       (^mul p d-head)
      =/  acc-after-head
        =/  j  0
        =/  a  acc
        |-  ^-  (list @)
        ?:  =(j d-head)  a
        $(j +(j), a [`@`(cut 5 [(^add base j) 1] head-data) a])
      $(heads t.heads, h +(h), acc acc-after-head)
    $(p +(p), vals vals-after-p)
  ::
  ::  +extract-head-2d: slice head h out of a [S, H, D_head] tensor into a
  ::  fresh [S, D_head] tensor without per-element set-item allocations.
  ::
  ++  extract-head-2d
    |=  [x=tensor h=@ud]
    ^-  tensor
    =,  (lake rnd)
    =/  seq-len  (snag 0 shape.meta.x)
    =/  n-heads  (snag 1 shape.meta.x)
    =/  d-head   (snag 2 shape.meta.x)
    =/  x-data   data.x
    =/  out-meta=meta:ls  [~[seq-len d-head] 5 %i754 ~]
    =/  vals=(list @)  ~
    =/  p  0
    |-  ^-  tensor
    ?:  =(p seq-len)
      =/  data-out  (con data:(zeros out-meta) (rep 5 (flop vals)))
      [out-meta data-out]
    =/  base  (^mul (^add (^mul p n-heads) h) d-head)
    =/  vals-after-p
      =/  j  0
      =/  acc  vals
      |-  ^-  (list @)
      ?:  =(j d-head)  acc
      $(j +(j), acc [`@`(cut 5 [(^add base j) 1] x-data) acc])
    $(p +(p), vals vals-after-p)
  ::
  ::  +transformer-block-qwen3: one Qwen3 block.
  ::    1. x1 = rms-norm(x, input-ln)
  ::    2. q,k,v = proj(x1)  (no bias)
  ::    3. q,k = reshape to [S,H,D_h]; apply q-norm/k-norm per head
  ::    4. q,k = apply RoPE (cos,sin precomputed for whole forward)
  ::    5. attn_out = gqa_attention(q,k,v)
  ::    6. x = x + o-proj(attn_out)
  ::    7. x2 = rms-norm(x, post-attn-ln)
  ::    8. mlp_out = down(silu(gate(x2)) * up(x2))
  ::    9. x = x + mlp_out
  ::
  ++  transformer-block-qwen3
    |=  $:  x=tensor
            bw=block-weights-qwen3
            cfg=model-config-qwen3
            cos=tensor
            sin=tensor
        ==
    ^-  tensor
    =,  (lake rnd)
    =/  x1      (rms-norm-2d x input-ln.bw rms-eps.cfg)
    =/  q-flat  (linear-nobias x1 q-proj.bw)
    =/  k-flat  (linear-nobias x1 k-proj.bw)
    =/  v-flat  (linear-nobias x1 v-proj.bw)
    =/  q3      (reshape-heads q-flat n-heads.cfg head-dim.cfg)
    =/  k3      (reshape-heads k-flat n-kv-heads.cfg head-dim.cfg)
    =/  v3      (reshape-heads v-flat n-kv-heads.cfg head-dim.cfg)
    =/  q3      (per-head-rms-norm q3 q-norm.bw rms-eps.cfg)
    =/  k3      (per-head-rms-norm k3 k-norm.bw rms-eps.cfg)
    =/  q3      (rope-apply q3 cos sin)
    =/  k3      (rope-apply k3 cos sin)
    =/  attn    (gqa-attention q3 k3 v3)
    =/  o       (linear-nobias attn o-proj.bw)
    =/  x       (add x o)
    =/  x2      (rms-norm-2d x post-attn-ln.bw rms-eps.cfg)
    =/  mlp     (swiglu-mlp x2 gate-proj.bw up-proj.bw down-proj.bw)
    (add x mlp)
  ::
  ::  +forward-qwen3: full forward pass through a Qwen3 model.
  ::    tokens: (list @ud) prompt token IDs
  ::    returns [1, vocab_size] logits for the last token position
  ::
  ++  forward-qwen3
    |=  [tokens=(list @ud) weights=model-weights-qwen3 cfg=model-config-qwen3]
    ^-  tensor
    =,  (lake rnd)
    ::  Compose the streamed stages so callers with no HTTP tick budget
    ::  can still do a whole forward in one call.
    =/  [x=tensor cos=tensor sin=tensor]
      (forward-qwen3-embed tokens weights cfg)
    =/  blks  blocks.weights
    =/  x-all
      |-  ^-  tensor
      ?~  blks  x
      $(blks t.blks, x (transformer-block-qwen3 x i.blks cfg cos sin))
    (forward-qwen3-final x tokens weights cfg)
  ::
  ::  +forward-qwen3-embed: stage 1 of streamed Qwen3 forward.  Returns the
  ::  token embeddings and RoPE cos/sin tables that the per-block stage
  ::  needs.  Agents run this once per prompt, then drive the block loop
  ::  tick-by-tick with +transformer-block-qwen3.
  ::
  ++  forward-qwen3-embed
    |=  [tokens=(list @ud) weights=model-weights-qwen3 cfg=model-config-qwen3]
    ^-  [x=tensor cos=tensor sin=tensor]
    =/  seq-len  (lent tokens)
    =/  x  (embed-tied-mlx2 tokens tok-emb.weights cfg)
    =/  inv-freq  (rope-inv-freq head-dim.cfg rope-theta.cfg yarn-orig-max-seq.cfg yarn-factor.cfg)
    =/  attn-fac  (rope-attention-factor yarn-factor.cfg)
    =/  cs        (rope-cos-sin seq-len head-dim.cfg inv-freq attn-fac)
    [x cos.cs sin.cs]
  ::
  ::  +forward-qwen3-final: stage 3 of streamed Qwen3 forward.  Runs the
  ::  final RMSNorm and the tied output projection over x, producing the
  ::  [1, vocab_size] logits for the last token.
  ::
  ++  forward-qwen3-final
    |=  [x=tensor tokens=(list @ud) weights=model-weights-qwen3 cfg=model-config-qwen3]
    ^-  tensor
    =,  (lake rnd)
    =/  last-idx  (dec (lent tokens))
    =/  x-n       (rms-norm-2d x ln-f.weights rms-eps.cfg)
    =/  last-row  (get-row x-n ~[last-idx])
    (logits-tied-mlx2 last-row tok-emb.weights cfg)
  ::
  ::  +embed-tied-mlx2: token embedding lookup for an mlx2-packed wte.
  ::  Builds [seq_len, d_model] by dequanting only the needed rows.
  ::
  ++  embed-tied-mlx2
    |=  [tokens=(list @ud) wte=weight-tensor cfg=model-config-qwen3]
    ^-  tensor
    =,  (lake rnd)
    ?>  ?=(%mlx2 -.wte)
    =/  seq-len  (lent tokens)
    =/  d-model  d-model.cfg
    =/  out  (zeros [~[seq-len d-model] 5 %i754 ~])
    =/  i  0
    =/  toks  tokens
    |-  ^-  tensor
    ?~  toks  out
    =/  row-1d  (dequant-mlx2-row wq.wte scales.wte biases.wte group-size.wte i.toks)
    =/  row-2d  (reshape row-1d ~[1 d-model])
    $(toks t.toks, i +(i), out (set-row out ~[i] row-2d))
  ::
  ::  +logits-tied-mlx2: compute [1, vocab] logits = x @ wte.T where wte is
  ::  mlx2-packed [vocab, d_model]. Uses the jetted full-tensor dequant + an
  ::  ordinary transpose+mmul; materializes the full fp32 wte (~1.2 GB on
  ::  Bonsai-1.7B), so it needs --loom 33 or larger. Faster than per-row
  ::  (151k pure-Hoon dequants) by orders of magnitude even with a bigger peak.
  ::
  ++  logits-tied-mlx2
    |=  [last-row=tensor wte=weight-tensor cfg=model-config-qwen3]
    ^-  tensor
    =,  (lake rnd)
    ?>  ?=([%mlx2 *] wte)
    ::  Calls the top-level jetted +logits-tied-mlx2-fused.
    (logits-tied-mlx2-fused last-row wq.wte scales.wte biases.wte group-size.wte)
  ::
  ::  =======================================================================
  ::  End of Qwen3 primitives.  Sampling/argmax helpers below are shared.
  ::  =======================================================================
  ::
  ::  +argmax-token: get the token index with highest logit
  ::
  ++  argmax-token
    |=  logits=tensor
    ^-  @ud
    (argmax:la logits)
  ::
  ::  +sampling: flat struct of knobs.
  ::  Defaults mean "disabled":
  ::    temp=.1         — no temperature scaling
  ::    top-k=0         — no top-k filter
  ::    top-p=.1        — no top-p filter
  ::    rep-penalty=.1  — no repetition penalty
  ::  Fully-default sampling is equivalent to greedy (argmax).
  ::
  +$  sampling
    $:  temp=@rs
        top-k=@ud
        top-p=@rs
        rep-penalty=@rs
    ==
  ::
  ++  default-sampling
    ^-  sampling
    [.1 0 .1 .1]
  ::
  ::  +sample-token: apply sampling pipeline and draw a token.
  ::    rep-penalty (uses `context`) → temp → top-k → top-p → softmax → sample
  ::  If every knob is at its default, returns argmax (greedy).
  ::  `context` is the token history used for repetition penalty; pass ~ to skip.
  ::
  ++  sample-token
    |=  [logits=tensor strategy=sampling context=(list @ud) eny=@]
    ^-  @ud
    =/  la  (lake rnd)
    ::  greedy fast path: every knob at default
    ?:  ?&  =(.1 temp.strategy)
            =(0 top-k.strategy)
            =(.1 top-p.strategy)
            =(.1 rep-penalty.strategy)
        ==
      (argmax-token logits)
    ::  1. repetition penalty
    =/  logits
      ?:  =(.1 rep-penalty.strategy)  logits
      (apply-rep-penalty logits context rep-penalty.strategy)
    ::  2. temperature
    =/  logits
      ?:  =(.1 temp.strategy)  logits
      (div-scalar:la logits temp.strategy)
    ::  3. top-k mask
    =/  logits
      ?:  =(0 top-k.strategy)  logits
      (mask-top-k logits top-k.strategy)
    ::  4. top-p mask
    =/  logits
      ?:  =(.1 top-p.strategy)  logits
      (mask-top-p logits top-p.strategy)
    ::  5. softmax + sample
    =/  probs  (softmax:sa:saloon logits)
    (sample-from-dist probs eny)
  ::
  ::  +sample-from-dist: sample an index from a probability distribution.
  ::  Uses inverse CDF: generate r in [0,1), find first index where cumsum >= r.
  ::
  ++  sample-from-dist
    |=  [probs=tensor eny=@]
    ^-  @ud
    =/  la  (lake rnd)
    =/  els  (ravel:la probs)
    ::  r = (eny mod 1_000_000) / 1_000_000 as @rs in [0, 1)
    =/  r  (fnormalize bloq.meta.probs kind.meta.probs (mod eny 1.000.000))
    =/  i  0
    =/  cum  (fzero bloq.meta.probs kind.meta.probs)
    |-  ^-  @ud
    ?~  els  (dec i)    :: fallback: last index
    =.  cum  (fadd bloq.meta.probs kind.meta.probs cum i.els)
    ?:  (fgte bloq.meta.probs kind.meta.probs cum r)
      i
    $(i +(i), els t.els, cum cum)
  ::
  ::  +mask-top-k: set all but top-k logits to -inf (works on any shape).
  ::
  ++  mask-top-k
    |=  [logits=tensor k=@ud]
    ^-  tensor
    =/  la  (lake rnd)
    =/  els  (ravel:la logits)
    =/  n  (lent els)
    =/  sorted-els  (sort els (fgth-gate bloq.meta.logits kind.meta.logits))
    =/  threshold  ?:((lte k n) (snag (dec k) sorted-els) (snag (dec n) sorted-els))
    =/  neg-inf  (fcon:sa:saloon bloq.meta.logits kind.meta.logits %neg-inf)
    =/  new-els=(list @)
      %+  turn  els
      |=  x=@
      ?:((fgte bloq.meta.logits kind.meta.logits x threshold) x neg-inf)
    :-  meta.logits
    (con data:(zeros:la meta.logits) (rep bloq.meta.logits new-els))
  ::
  ::  +mask-top-p: nucleus sampling mask. Keep the smallest set of tokens
  ::  whose cumulative softmax probability >= p; mask the rest to -inf.
  ::
  ++  mask-top-p
    |=  [logits=tensor p=@rs]
    ^-  tensor
    =/  la  (lake rnd)
    =/  els  (ravel:la logits)
    =/  n  (lent els)
    ?:  =(0 n)  logits
    =/  bloq  bloq.meta.logits
    =/  kind  kind.meta.logits
    ::  sort logits descending; softmax in that order; find cum-prob cutoff
    =/  sorted-els  (sort els (fgth-gate bloq kind))
    =/  sorted-meta=meta:ls  [~[n] bloq kind ~]
    =/  sorted-ray=tensor
      :-  sorted-meta
      (con data:(zeros:la sorted-meta) (rep bloq sorted-els))
    =/  probs  (softmax:sa:saloon sorted-ray)
    =/  probs-list  (ravel:la probs)
    ::  walk until cumulative >= p, record the logit at that index as threshold
    =/  threshold
      =|  cum=@
      =.  cum  (fzero bloq kind)
      =/  i  0
      =/  sl  sorted-els
      =/  pl  probs-list
      |-  ^-  @
      ?~  pl  (snag (dec n) sorted-els)
      =.  cum  (fadd bloq kind cum i.pl)
      ?:  (fgte bloq kind cum p)
        ?~  sl  (snag (dec n) sorted-els)
        i.sl
      $(sl ?~(sl ~ t.sl), pl t.pl, i +(i))
    =/  neg-inf  (fcon:sa:saloon bloq kind %neg-inf)
    =/  new-els=(list @)
      %+  turn  els
      |=  x=@
      ?:((fgte bloq kind x threshold) x neg-inf)
    :-  meta.logits
    (con data:(zeros:la meta.logits) (rep bloq new-els))
  ::
  ::  +apply-rep-penalty: divide-or-multiply logits at indices of recent tokens.
  ::  Standard formulation: logit/penalty for positive logits, logit*penalty for
  ::  negative. `penalty > 1` discourages repetition.
  ::
  ++  apply-rep-penalty
    |=  [logits=tensor context=(list @ud) penalty=@rs]
    ^-  tensor
    =/  la  (lake rnd)
    ?:  =(~ context)  logits
    =/  bloq  bloq.meta.logits
    =/  kind  kind.meta.logits
    =/  zero  (fzero bloq kind)
    =/  seen=(set @ud)  (~(gas in *(set @ud)) context)
    =/  toks=(list @ud)  ~(tap in seen)
    =/  out  logits
    |-  ^-  tensor
    ?~  toks  out
    =/  idx=(list @ud)
      ?:  =(1 (lent shape.meta.logits))  ~[i.toks]
      ~[0 i.toks]
    =/  val  (get-item:la out idx)
    =/  new-val
      ?:  (fgth bloq kind val zero)
        (fdiv bloq kind val penalty)
      (fmul bloq kind val penalty)
    $(toks t.toks, out (set-item:la out idx new-val))
  ::
  ::  +generate: generate `n` tokens given a prompt.
  ::  Returns the full token sequence (prompt + generated).
  ::
  ++  generate
    |=  $:  prompt=(list @ud)
            n=@ud
            weights=model-weights
            config=model-config
            strategy=sampling
            eny=@
        ==
    ^-  (list @ud)
    =/  tokens  prompt
    =/  step  0
    |-  ^-  (list @ud)
    ?:  =(step n)  tokens
    =/  logits  (forward tokens weights config)
    =/  tok  (sample-token logits strategy tokens (mix eny step))
    $(step +(step), tokens (snoc tokens tok))
  ::
  ::  Scalar helpers
  ::
  ::  +hstack-2d: column-concatenate two 2D tensors. Workaround for a bug
  ::  in lagoon's `stack`/`hstack` that iterates over the wrong dimension.
  ::
  ++  hstack-2d
    |=  [a=tensor b=tensor]
    ^-  tensor
    =/  la  (lake rnd)
    ?>  =(2 (lent shape.meta.a))
    ?>  =(2 (lent shape.meta.b))
    =/  rows  (snag 0 shape.meta.a)
    ?>  =(rows (snag 0 shape.meta.b))
    =/  cols-a  (snag 1 shape.meta.a)
    =/  cols-b  (snag 1 shape.meta.b)
    =/  out-cols  (^add cols-a cols-b)
    =/  out  (zeros:la [~[rows out-cols] bloq.meta.a kind.meta.a ~])
    =/  i  0
    |-  ^-  tensor
    ?:  =(i rows)  out
    ::  copy a's row i, columns 0..cols-a-1
    =.  out
      =/  j  0
      |-  ^-  tensor
      ?:  =(j cols-a)  out
      $(j +(j), out (set-item:la out ~[i j] (get-item:la a ~[i j])))
    ::  copy b's row i, columns cols-a..out-cols-1
    =.  out
      =/  j  0
      |-  ^-  tensor
      ?:  =(j cols-b)  out
      $(j +(j), out (set-item:la out ~[i (^add cols-a j)] (get-item:la b ~[i j])))
    $(i +(i), out out)
  ::
  ::  +cols: extract columns [c-start..c-end] inclusive from a 2D tensor.
  ::  Lagoon's submatrix has a Hoon gotcha where [0 0] is misparsed as
  ::  [start=0 end=unset], so we do it directly.
  ::
  ++  cols
    |=  [a=tensor c-start=@ud c-end=@ud]
    ^-  tensor
    =/  la  (lake rnd)
    ?>  =(2 (lent shape.meta.a))
    =/  rows  (snag 0 shape.meta.a)
    =/  new-cols  +((sub c-end c-start))
    =/  out  (zeros:la [~[rows new-cols] bloq.meta.a kind.meta.a ~])
    =/  i  0
    |-  ^-  tensor
    ?:  =(i rows)  out
    =/  j  0
    =.  out
      |-  ^-  tensor
      ?:  =(j new-cols)  out
      =/  v  (get-item:la a ~[i (add c-start j)])
      $(j +(j), out (set-item:la out ~[i j] v))
    $(i +(i), out out)
  ::
  ::  +transpose2d: working 2D transpose (workaround for lagoon bug)
  ::
  ++  transpose2d
    |=  a=tensor
    ^-  tensor
    =/  la  (lake rnd)
    ?>  (check:la a)
    ?>  =(2 (lent shape.meta.a))
    =/  rows  (snag 0 shape.meta.a)
    =/  cols  (snag 1 shape.meta.a)
    =/  out-shape=(list @)  ~[cols rows]
    =/  out  (zeros:la [out-shape bloq.meta.a kind.meta.a ~])
    =/  i  0
    |-  ^-  tensor
    ?:  =(i rows)  out
    =/  j  0
    =.  out
      |-  ^-  tensor
      ?:  =(j cols)  out
      $(j +(j), out (set-item:la out ~[j i] (get-item:la a ~[i j])))
    $(i +(i), out out)
  ::
  ++  fsqrt
    |=  [=bloq =kind a=@]
    ^-  @
    ?>  =(%i754 kind)
    ?+  bloq  !!
      %7  (~(sqrt rq:math [rnd .~~~1e-10]) a)
      %6  (~(sqrt rd:math [rnd .~1e-10]) a)
      %5  (~(sqrt rs:math [rnd .1e-5]) a)
      %4  (~(sqrt rh:math [rnd .~~1e-2]) a)
    ==
  ++  fzero
    |=  [=bloq =kind]
    ^-  @
    ?>  =(%i754 kind)
    ?+(bloq !! %7 .~~~0, %6 .~0, %5 .0, %4 .~~0)
  ++  fadd
    |=  [=bloq =kind a=@ b=@]
    ^-  @
    ?>  =(%i754 kind)
    ?+  bloq  !!
      %7  (~(add rq:math [rnd .~~~0]) a b)
      %6  (~(add rd:math [rnd .~0]) a b)
      %5  (~(add rs:math [rnd .0]) a b)
      %4  (~(add rh:math [rnd .~~0]) a b)
    ==
  ++  fmul
    |=  [=bloq =kind a=@ b=@]
    ^-  @
    ?>  =(%i754 kind)
    ?+  bloq  !!
      %7  (~(mul rq:math [rnd .~~~0]) a b)
      %6  (~(mul rd:math [rnd .~0]) a b)
      %5  (~(mul rs:math [rnd .0]) a b)
      %4  (~(mul rh:math [rnd .~~0]) a b)
    ==
  ++  fdiv
    |=  [=bloq =kind a=@ b=@]
    ^-  @
    ?>  =(%i754 kind)
    ?+  bloq  !!
      %7  (~(div rq:math [rnd .~~~0]) a b)
      %6  (~(div rd:math [rnd .~0]) a b)
      %5  (~(div rs:math [rnd .0]) a b)
      %4  (~(div rh:math [rnd .~~0]) a b)
    ==
  ++  fgth
    |=  [=bloq =kind a=@ b=@]
    ^-  ?
    ?>  =(%i754 kind)
    ?+  bloq  !!
      %7  (~(gth rq:math [rnd .~~~0]) a b)
      %6  (~(gth rd:math [rnd .~0]) a b)
      %5  (~(gth rs:math [rnd .0]) a b)
      %4  (~(gth rh:math [rnd .~~0]) a b)
    ==
  ++  fgte
    |=  [=bloq =kind a=@ b=@]
    ^-  ?
    ?>  =(%i754 kind)
    ?+  bloq  !!
      %7  (~(gte rq:math [rnd .~~~0]) a b)
      %6  (~(gte rd:math [rnd .~0]) a b)
      %5  (~(gte rs:math [rnd .0]) a b)
      %4  (~(gte rh:math [rnd .~~0]) a b)
    ==
  ++  fgth-gate
    |=  [=bloq =kind]
    ^-  $-([@ @] ?)
    ?>  =(%i754 kind)
    ?+  bloq  !!
      %7  ~(gth rq:math [rnd .~~~0])
      %6  ~(gth rd:math [rnd .~0])
      %5  ~(gth rs:math [rnd .0])
      %4  ~(gth rh:math [rnd .~~0])
    ==
  ::  Normalize integer n in [0, 1_000_000) to float in [0, 1).
  ++  fnormalize
    |=  [=bloq =kind n=@ud]
    ^-  @
    ?>  =(%i754 kind)
    ?+  bloq  !!
      %7  (~(div rq:math [rnd .~~~1e-10]) (~(sun rq:math [rnd .~~~1e-10]) n) .~~~1e6)
      %6  (~(div rd:math [rnd .~1e-10]) (~(sun rd:math [rnd .~1e-10]) n) .~1e6)
      %5  (~(div rs:math [rnd .1e-5]) (~(sun rs:math [rnd .1e-5]) n) .1e6)
      %4  (~(div rh:math [rnd .~~0.01]) (~(sun rh:math [rnd .~~0.01]) n) .~~1e3)
    ==
  --
--
