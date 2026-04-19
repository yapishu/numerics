::  Khan thread: load qwen3-bonsai weights, run greedy forward on a hardcoded
::  prompt, return [argmax-token decoded-text].
::
::  Invoke via:
::    click -k -i /tmp/click-qwen3.hoon ~/.urbit/per
::  where /tmp/click-qwen3.hoon contains:
::    =/  m  (strand ,vase)
::    ;<  =bowl:spider  bind:m  get-bowl
::    ;<  v=vase  bind:m
::      (start-thread:strandio %fard our.bowl %saloon %qwen3-fwd-test %noun !>(~))
::    (pure:m v)
::
/-  ls=lagoon
/+  *strandio, *lagoon, math, saloon, maroon
::
=,  strand=strand:rand
::
^-  thread:rand
|=  arg=vase
=/  m  (strand ,vase)
^-  form:m
;<  jam=@  bind:m  (scry:strandio @ /cx/saloon/weights/qwen3-bonsai/jam)
::  Cue and cast to model-weights-qwen3.
=/  weights  ;;(model-weights-qwen3:maroon (cue jam))
::  Qwen3 1.7B Bonsai config (matches gguf2jam output for this model).
=/  cfg=model-config-qwen3:maroon
  :*  d-model=2.048
      n-heads=16
      n-kv-heads=8
      n-layers=28
      d-ff=6.144
      vocab-size=151.669
      max-seq=32.768
      head-dim=128
      rms-eps=.1e-6
      rope-theta=.1e6
      yarn-factor=.4
      yarn-orig-max-seq=8.192
      bloq=5
  ==
::  Hardcoded smoke-test prompt: 4 tokens (use real Qwen3 tokens later).
::  These IDs are placeholders; real run will pass via vase arg.
=/  tokens=(list @ud)  ~[1.124 1.234 1.345 1.456]
=/  logits  (forward-qwen3:mr:maroon tokens weights cfg)
=/  next-tok  (argmax-token:mr:maroon logits)
(pure:m !>(next-tok))
