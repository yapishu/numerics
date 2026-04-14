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
::  A single linear projection: W=[d_in d_out], b=[1 d_out]
+$  linear-weights
  $:  w=tensor    ::  weight matrix
      b=tensor    ::  bias vector
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
    =/  out  (mmul x w.lw)
    ::  broadcast bias across rows: add b to each row of out
    (add-bias out b.lw)
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
      (linear last-row [out-proj.weights (zeros [~[1 vocab-size.config] bloq.config %i754 ~])])
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
  --
--
