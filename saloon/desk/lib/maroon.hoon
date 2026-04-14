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
::  Full model weights
+$  model-weights
  $:  tok-emb=tensor              ::  [vocab_size d_model]
      pos-emb=tensor              ::  [max_seq d_model]
      blocks=(list block-weights)
      ln-f-g=tensor               ::  final layer norm gamma
      ln-f-b=tensor               ::  final layer norm beta
      out-proj=tensor             ::  [d_model vocab_size]
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
::  +load-weights: cue a jammed atom into model-weights
::  The jammed atom should be produced by weights_to_noun.py
::
++  load-weights
  |=  jammed=@
  ^-  model-weights
  ;;(model-weights (cue jammed))
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
  ::  +dequantize: return the tensor as @rs-typed tensor,
  ::  dequantizing from int8 if needed.
  ::
  ++  dequantize
    |=  w=weight-tensor
    ^-  tensor
    ?-    -.w
        %fp  r.w
        %q8
      =/  la  (lake rnd)
      ::  r.w stores bytes in a @uint bloq=3 ray.
      ::  For each byte: if b < 128 signed = b, else signed = b - 256.
      ::  Float value = signed * scale.
      =/  rs-door  ~(. rs:math [rnd .1e-5])
      =/  bytes  (ravel:la r.w)
      =/  shape-out  shape.meta.r.w
      =/  new-meta=meta:ls  [shape-out 5 %i754 ~]
      =/  f32-vals=(list @)
        %+  turn  bytes
        |=  b=@
        ?:  =(0 b)  .0
        ?:  (lth b 128)
          (mul:rs-door (sun:rs-door b) scale.w)
        ::  b >= 128: signed = b - 256, float = -(256 - b) * scale
        =/  mag  (sun:rs-door (sub 256 b))
        (mul:rs-door (mul:rs-door mag .-1) scale.w)
      =/  data-out  (con data:(zeros:la new-meta) (rep 5 f32-vals))
      [new-meta data-out]
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
      =/  scores
        |-  ^-  tensor
        ?:  =(j seq-len)  scores
        $(j +(j), scores (set-item scores ~[i j] neg-inf))
      $(i +(i))
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
        $(acc (hstack acc i.rest), rest t.rest)
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
    ::  add positional embeddings (first seq-len rows)
    =/  pos  (submatrix ~[`[`0 `(dec seq-len)] ~] pos-emb.weights)
    =/  x  (add x pos)
    ::  run through transformer blocks
    =/  blks  blocks.weights
    |-  ^-  tensor
    ?~  blks
      ::  final layer norm
      =/  x  (layer-norm-2d x ln-f-g.weights ln-f-b.weights)
      ::  project last position to vocab logits
      =/  last-row  (get-row x ~[(dec seq-len)])
      (linear last-row [[%fp out-proj.weights] (zeros [~[1 vocab-size.config] bloq.config %i754 ~])])
    $(blks t.blks, x (transformer-block x i.blks n-heads.config))
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
  ::  +argmax-token: get the token index with highest logit
  ::
  ++  argmax-token
    |=  logits=tensor
    ^-  @ud
    (argmax:la logits)
  ::
  ::  +sample-token: sample from a logits distribution.
  ::  strategy:
  ::    [%greedy]          — argmax (deterministic)
  ::    [%temperature t]   — scale by 1/t, then sample from full softmax
  ::    [%top-k k t]       — keep top-k, apply temperature, sample
  ::  eny: entropy atom from the bowl (changes each request)
  ::
  +$  sampling
    $%  [%greedy ~]
        [%temperature t=@rs]
        [%top-k k=@ud t=@rs]
    ==
  ::
  ++  sample-token
    |=  [logits=tensor strategy=sampling eny=@]
    ^-  @ud
    =/  la  (lake rnd)
    ?-    -.strategy
        %greedy  (argmax-token logits)
        %temperature
      =/  scaled  (div-scalar:la logits t.strategy)
      =/  probs  (softmax:sa:saloon scaled)
      (sample-from-dist probs eny)
        %top-k
      =/  scaled  (div-scalar:la logits t.strategy)
      =/  masked  (mask-top-k scaled k.strategy)
      =/  probs  (softmax:sa:saloon masked)
      (sample-from-dist probs eny)
    ==
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
  ::  +mask-top-k: set all but top-k logits to -inf.
  ::
  ++  mask-top-k
    |=  [logits=tensor k=@ud]
    ^-  tensor
    =/  la  (lake rnd)
    ?>  =(1 (lent shape.meta.logits))
    =/  n  (snag 0 shape.meta.logits)
    =/  els  (ravel:la logits)
    ::  sort descending to find the k-th largest value as threshold
    =/  sorted-els  (sort els (fgth-gate bloq.meta.logits kind.meta.logits))
    =/  threshold  ?:((lte k n) (snag (dec k) sorted-els) (snag (dec n) sorted-els))
    =/  neg-inf  (fcon:sa:saloon bloq.meta.logits kind.meta.logits %neg-inf)
    ::  mask anything below threshold to -inf
    =/  new-els=(list @)
      %+  turn  els
      |=  x=@
      ?:((fgte bloq.meta.logits kind.meta.logits x threshold) x neg-inf)
    :-  meta.logits
    (con data:(zeros:la meta.logits) (rep bloq.meta.logits new-els))
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
    =/  tok  (sample-token logits strategy (mix eny step))
    $(step +(step), tokens (snoc tokens tok))
  ::
  ::  Scalar helpers
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
    =/  out
      |-  ^-  tensor
      ?:  =(j new-cols)  out
      =/  v  (get-item:la a ~[i (add c-start j)])
      $(j +(j), out (set-item:la out ~[i j] v))
    $(i +(i))
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
    =/  out
      |-  ^-  tensor
      ?:  =(j cols)  out
      $(j +(j), out (set-item:la out ~[j i] (get-item:la a ~[i j])))
    $(i +(i))
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
