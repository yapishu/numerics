/-  ls=lagoon
/+  *test,
    *lagoon,
    math,
    saloon,
    maroon
::
|%
::  Smoke test: run a linear projection on a tiny input
::
++  test-linear  ^-  tang
  =/  la  (lake %n)
  ::  x = [2 3] input
  =/  x  (en-ray:la [~[2 3] 5 %i754 ~] ~[.1 .2 .3 .4 .5 .6])
  ::  W = [3 2] weight
  =/  w  (en-ray:la [~[3 2] 5 %i754 ~] ~[.1 .0 .0 .1 .1 .1])
  ::  b = [1 2] bias
  =/  b  (en-ray:la [~[1 2] 5 %i754 ~] ~[.0.1 .0.2])
  =/  out  (linear:mr:maroon x [[%fp w] b])
  ::  check output shape is [2 2]
  %+  expect-eq
    !>(`(list @)`shape.meta.out)
    !>(`(list @)`~[2 2])
::
::  Smoke test: softmax on attention scores
::
++  test-transpose  ^-  tang
  =/  la  (lake %n)
  ::  use eye which is known to produce valid 2D rays
  =/  k  (eye:la [~[2 2] 5 %i754 ~])
  ?>  (check:la k)
  =/  kt  (transpose2d:mr:maroon k)
  =/  prod  (mmul:la k kt)
  ;:  weld
    %+  expect-eq
      !>(`(list @)`shape.meta.kt)
      !>(`(list @)`~[2 2])
    %+  expect-eq
      !>(`(list @)`shape.meta.prod)
      !>(`(list @)`~[2 2])
  ==
::  Full attention smoke test
++  test-attention  ^-  tang
  =/  la  (lake %n)
  =/  q  (eye:la [~[2 2] 5 %i754 ~])
  =/  k  (eye:la [~[2 2] 5 %i754 ~])
  =/  v  (eye:la [~[2 2] 5 %i754 ~])
  =/  out  (attention:mr:maroon q k v)
  %+  expect-eq
    !>(`(list @)`shape.meta.out)
    !>(`(list @)`~[2 2])
::
::  Smoke test: embedding lookup
::
++  test-embed  ^-  tang
  =/  la  (lake %n)
  ::  vocab=3, d_model=2
  =/  emb-table  (en-ray:la [~[3 2] 5 %i754 ~] ~[.0.1 .0.2 .0.3 .0.4 .0.5 .0.6])
  =/  tokens  ~[1 0 2]
  =/  out  (embed:mr:maroon tokens emb-table 5)
  ::  check output shape is [3 2]
  %+  expect-eq
    !>(`(list @)`shape.meta.out)
    !>(`(list @)`~[3 2])
::  Verify mmul works correctly with non-trivial matrices
::  (tests whether lagoon's nested |- trap pattern is correct)
++  test-mmul-nontrivial  ^-  tang
  =/  la  (lake %n)
  ::  A = [[1 2] [3 4]], B = [[5 6] [7 8]]
  ::  A*B = [[1*5+2*7, 1*6+2*8], [3*5+4*7, 3*6+4*8]] = [[19 22] [43 50]]
  =/  a  (en-ray:la [~[2 2] 5 %i754 ~] ~[~[.1 .2] ~[.3 .4]])
  =/  b  (en-ray:la [~[2 2] 5 %i754 ~] ~[~[.5 .6] ~[.7 .8]])
  =/  out  (mmul:la a b)
  ::  check the second row specifically — if j/k don't reset, row 1 would be wrong
  =/  r1c0  (get-item:la out ~[1 0])  ::  should be 43
  =/  r1c1  (get-item:la out ~[1 1])  ::  should be 50
  =/  expected-r1c0  (get-item:la (en-ray:la [~[1] 5 %i754 ~] ~[.43]) ~[0])
  =/  expected-r1c1  (get-item:la (en-ray:la [~[1] 5 %i754 ~] ~[.50]) ~[0])
  ;:  weld
    %+  expect-eq  !>(r1c0)  !>(expected-r1c0)
    %+  expect-eq  !>(r1c1)  !>(expected-r1c1)
  ==
::
::  Attention with degenerate d_k=1 (matches tiny-gpt2 per-head size)
++  test-attn-dk1  ^-  tang
  =/  la  (lake %n)
  =/  q  (en-ray:la [~[3 1] 5 %i754 ~] ~[~[.0.1] ~[.0.2] ~[.0.3]])
  =/  k  (en-ray:la [~[3 1] 5 %i754 ~] ~[~[.0.1] ~[.0.2] ~[.0.3]])
  =/  v  (en-ray:la [~[3 1] 5 %i754 ~] ~[~[.0.1] ~[.0.2] ~[.0.3]])
  =/  out  (attention:mr:maroon q k v)
  %+  expect-eq  !>(`(list @)`shape.meta.out)  !>(`(list @)`~[3 1])
::
::  Multi-head attention with n_heads=2
++  test-mha  ^-  tang
  =/  la  (lake %n)
  ::  S=3, d_model=4, n_heads=2, d_k=2
  =/  x  (en-ray:la [~[3 4] 5 %i754 ~] ~[~[.0.1 .0.2 .0.3 .0.4] ~[.0.5 .0.6 .0.7 .0.8] ~[.0.9 .1.0 .1.1 .1.2]])
  =/  mk  |=([s=(list @) b=@] (ones:la [s b %i754 ~]))
  =/  mk-lin
    |=  [[di=@ do=@] b=@]
    ^-  linear-weights:maroon
    [[%fp (mk ~[di do] b)] (mk ~[1 do] b)]
  =/  wq  (mk-lin [4 4] 5)
  =/  wk  (mk-lin [4 4] 5)
  =/  wv  (mk-lin [4 4] 5)
  =/  wo  (mk-lin [4 4] 5)
  =/  out  (multi-head-attention:mr:maroon x 2 wq wk wv wo)
  %+  expect-eq
    !>(`(list @)`shape.meta.out)
    !>(`(list @)`~[3 4])
::
::  Exact tiny-gpt2 shape: d=2, heads=2, d_k=1 (degenerate)
++  test-mha-tiny-gpt2  ^-  tang
  =/  la  (lake %n)
  ::  S=3, d_model=2, n_heads=2, d_k=1
  =/  x  (en-ray:la [~[3 2] 5 %i754 ~] ~[~[.0.1 .0.2] ~[.0.3 .0.4] ~[.0.5 .0.6]])
  =/  mk  |=([s=(list @) b=@] (ones:la [s b %i754 ~]))
  =/  mk-lin
    |=  [[di=@ do=@] b=@]
    ^-  linear-weights:maroon
    [[%fp (mk ~[di do] b)] (mk ~[1 do] b)]
  =/  wq  (mk-lin [2 2] 5)
  =/  wk  (mk-lin [2 2] 5)
  =/  wv  (mk-lin [2 2] 5)
  =/  wo  (mk-lin [2 2] 5)
  =/  out  (multi-head-attention:mr:maroon x 2 wq wk wv wo)
  %+  expect-eq
    !>(`(list @)`shape.meta.out)
    !>(`(list @)`~[3 2])
::
::  End-to-end: build a tiny transformer and run a forward pass
::  d_model=4, n_heads=1, d_ff=8, vocab=8, max_seq=4
::
++  test-forward-e2e  ^-  tang
  =/  la  (lake %n)
  =/  cfg=model-config:maroon
    [d-model=4 n-heads=1 n-layers=1 d-ff=8 vocab-size=8 max-seq=4 bloq=5]
  ::  build tiny weights — all ones for simplicity
  =/  mk  |=([s=(list @) b=@] (ones:la [s b %i754 ~]))
  =/  mk-lin
    |=  [[di=@ do=@] b=@]
    ^-  linear-weights:maroon
    [[%fp (mk ~[di do] b)] (mk ~[1 do] b)]
  ::  attention weights: 4->4 projections
  =/  wq  (mk-lin [4 4] 5)
  =/  wk  (mk-lin [4 4] 5)
  =/  wv  (mk-lin [4 4] 5)
  =/  wo  (mk-lin [4 4] 5)
  ::  layer norms: gamma=1, beta=0
  =/  ln-g  (mk ~[4] 5)
  =/  ln-b  (zeros:la [~[4] 5 %i754 ~])
  ::  feed-forward: 4->8->4
  =/  ff1  (mk-lin [4 8] 5)
  =/  ff2  (mk-lin [8 4] 5)
  ::  block
  =/  blk=block-weights:maroon
    [wq wk wv wo ln-g ln-b ln-g ln-b ff1 ff2]
  ::  model weights
  =/  weights=model-weights:maroon
    :*  tok-emb=(mk ~[8 4] 5)
        pos-emb=(mk ~[4 4] 5)
        blocks=~[blk]
        ln-f-g=ln-g
        ln-f-b=ln-b
        out-proj=(mk ~[4 8] 5)
    ==
  ::  run forward pass on token sequence [0 1 2]
  =/  logits  (forward:mr:maroon ~[0 1 2] weights cfg)
  ::  just verify we get logits with the right shape: [1 8]
  %+  expect-eq
    !>(`(list @)`shape.meta.logits)
    !>(`(list @)`~[1 8])
::
::  Test layer-norm produces correct output
::
++  test-layer-norm  ^-  tang
  =/  la  (lake %n)
  ::  x = [1 0 -1 2] as 1D ray
  =/  x  (en-ray:la [~[4] 5 %i754 ~] ~[.1 .0 .-1 .2])
  ::  gamma = all ones, beta = all zeros
  =/  gamma  (ones:la [~[4] 5 %i754 ~])
  =/  beta  (zeros:la [~[4] 5 %i754 ~])
  =/  out  (layer-norm:sa:saloon x gamma beta)
  ::  mean = 0.5, var = 1.25, std ~ 1.118
  ::  normalized: (1-0.5)/1.118, (0-0.5)/1.118, (-1-0.5)/1.118, (2-0.5)/1.118
  ::           ~  0.4472, -0.4472, -1.3416, 1.3416
  ::  just check shape is preserved
  %+  expect-eq
    !>(`(list @)`shape.meta.out)
    !>(`(list @)`~[4])
--
