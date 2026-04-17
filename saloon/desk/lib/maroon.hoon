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
      %fp  r.w
      %q8  (dequant-q8-ray r.w scale.w)
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
      (linear last-row [[%fp out-proj.weights] (zeros [~[1 vocab-size.config] bloq.config %i754 ~])])
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
