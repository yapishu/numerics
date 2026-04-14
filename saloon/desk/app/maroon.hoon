  ::
::::  %maroon - on-ship transformer inference agent
::
::  Holds model weights in state. Accepts pokes for inference.
::  Weights are a noun. Architecture is Hoon. Forward pass is
::  Lagoon matmuls + Saloon activations.
::
/-  ls=lagoon
/+  default-agent,
    dbug,
    *lagoon,
    math,
    saloon,
    maroon
::
|%
+$  versioned-state
  $%  [%0 state-0]
  ==
+$  state-0
  $:  weights=(unit model-weights:maroon)
      config=(unit model-config:maroon)
  ==
+$  card  card:agent:gall
--
::
%-  agent:dbug
=|  state-0
=*  state  -
^-  agent:gall
|_  =bowl:gall
+*  this  .
    def   ~(. (default-agent this %|) bowl)
::
++  on-init
  ^-  (quip card _this)
  ~&  >  '%maroon initialized — no model loaded'
  `this
::
++  on-save   !>(state)
++  on-load
  |=  old-state=vase
  ^-  (quip card _this)
  =/  old  (mule |.(!<(versioned-state old-state)))
  ?:  ?=(%| -.old)
    ~&  >  '%maroon: resetting state on load'
    `this
  ?-  -.p.old
    %0  `this(state +.p.old)
  ==
::
++  on-poke
  |=  [=mark =vase]
  ^-  (quip card _this)
  ?+    mark  (on-poke:def mark vase)
    ::
    ::  Load model weights from a jammed atom
    ::  Poke with [%load-weights config=model-config:maroon jammed=@]
    ::
      %maroon-load
    =/  payload  !<([model-config:maroon @] vase)
    =/  cfg  -.payload
    =/  jammed  +.payload
    ~&  >  "loading model: d={<d-model.cfg>} heads={<n-heads.cfg>} layers={<n-layers.cfg>} vocab={<vocab-size.cfg>}"
    =/  w  (load-weights:maroon jammed)
    ~&  >  '%maroon model loaded successfully'
    `this(weights `w, config `cfg)
    ::
    ::  Run inference on token sequence
    ::  Poke with [%infer tokens=(list @ud)]
    ::
      %maroon-infer
    =/  tokens  !<((list @ud) vase)
    ?~  weights
      ~&  >>>  'no model loaded — poke %maroon-load first or run +saloon!maroon-load-gpt2'
      `this
    ?~  config
      ~&  >>>  'no model config'
      `this
    ~&  >  "running inference on {<(lent tokens)>} tokens..."
    =/  logits  (forward:mr:maroon tokens u.weights u.config)
    =/  next-token  (argmax-token:mr:maroon logits)
    ~&  >  "next token: {<next-token>}"
    `this
  ==
::
++  on-watch  on-watch:def
++  on-leave  on-leave:def
++  on-peek
  |=  =path
  ^-  (unit (unit cage))
  ?+    path  (on-peek:def path)
    ::
    ::  /x/status — is a model loaded?
    ::
      [%x %status ~]
    ``noun+!>(?~(weights %no-model %model-loaded))
    ::
    ::  /x/config — model config
    ::
      [%x %config ~]
    ?~  config  ~
    ``noun+!>(u.config)
  ==
::
++  on-agent  on-agent:def
++  on-arvo   on-arvo:def
++  on-fail   on-fail:def
--
