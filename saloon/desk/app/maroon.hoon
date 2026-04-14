  ::
::::  %maroon - on-ship transformer inference agent
::
::  Holds model weights in state. HTTP endpoint at /apps/maroon/chat.
::
/-  ls=lagoon
/+  default-agent,
    dbug,
    server,
    *lagoon,
    math,
    saloon,
    maroon,
    tokenizer
::
|%
+$  versioned-state
  $%  [%0 state-0]
  ==
+$  state-0
  $:  weights=(unit model-weights:maroon)
      config=(unit model-config:maroon)
      last-output=(list @ud)
      tok=(unit tokenizer-maps:tokenizer)
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
  ~&  >  '%maroon initialized — bound at /apps/maroon/chat'
  :_  this
  :~  [%pass /eyre/connect %arvo %e %connect [~ /apps/maroon/chat] dap.bowl]
  ==
::
++  on-save   !>(state)
++  on-load
  |=  old-state=vase
  ^-  (quip card _this)
  =/  old  (mule |.(!<(versioned-state old-state)))
  ::  re-bind on every reload to survive agent revives
  =/  rebind-cards=(list card)
    :~  [%pass /eyre/connect %arvo %e %connect [~ /apps/maroon/chat] dap.bowl]
    ==
  ?:  ?=(%| -.old)
    ~&  >  '%maroon: resetting state on load'
    [rebind-cards this]
  ?-  -.p.old
    %0  [rebind-cards this(state +.p.old)]
  ==
::
++  on-poke
  |=  [=mark =vase]
  ^-  (quip card _this)
  ?+    mark  (on-poke:def mark vase)
    ::
    ::  Handle HTTP request on /apps/maroon/chat
    ::
      %handle-http-request
    |^
    =+  !<([eyre-id=@ta req=inbound-request:eyre] vase)
    =/  rl=request-line:server  (parse-request-line:server url.request.req)
    =/  site=(list @t)  site.rl
    ?.  ?=([%apps %maroon %chat *] site)
      :_  this
      (not-found eyre-id)
    ?:  =(%'GET' method.request.req)
      :_  this
      (give-help eyre-id)
    ?.  =(%'POST' method.request.req)
      :_  this
      (give-http eyre-id 405 ~[['content-type' 'text/plain']] (some (as-octs:mimes:html 'method not allowed')))
    =/  body
      ?~  body.request.req  ''
      q.u.body.request.req
    =/  parsed  (de:json:html body)
    ?~  parsed
      :_  this
      (give-http eyre-id 400 ~ (some (as-octs:mimes:html '{"error":"invalid JSON"}')))
    =/  tokens-opt=(unit (list @ud))
      %.  u.parsed
      %-  ot:dejs-soft:format
      :~  [%tokens (ar:dejs-soft:format ni:dejs-soft:format)]
      ==
    =/  prompt-opt=(unit @t)
      %.  u.parsed
      %-  ot:dejs-soft:format
      :~  [%prompt so:dejs-soft:format]
      ==
    =/  n=(unit @ud)
      %.  u.parsed
      %-  ot:dejs-soft:format
      :~  [%n ni:dejs-soft:format]
      ==
    =/  temperature=(unit @ta)
      %.  u.parsed
      %-  ot:dejs-soft:format
      :~  [%temperature no:dejs-soft:format]
      ==
    ?~  weights
      :_  this
      (give-http eyre-id 503 ~ (some (as-octs:mimes:html '{"error":"no model loaded"}')))
    ?~  config
      :_  this
      (give-http eyre-id 503 ~ (some (as-octs:mimes:html '{"error":"no model config"}')))
    ::  Determine input tokens: from "tokens" array, or encode "prompt" via tokenizer
    =/  tokens=(list @ud)
      ?^  tokens-opt  u.tokens-opt
      ?~  prompt-opt  ~
      ?~  tok  ~
      (encode:tokenizer u.tok u.prompt-opt)
    ?:  =(~ tokens)
      :_  this
      (give-http eyre-id 400 ~ (some (as-octs:mimes:html '{"error":"provide tokens or prompt (and load tokenizer)"}')))
    =/  n-tokens  (fall n 10)
    =/  strategy  ?~(temperature [%greedy ~] [%temperature (slav %rs u.temperature)])
    ~&  >  "HTTP /apps/maroon/chat: generating {<n-tokens>} tokens..."
    =/  out
      %:  generate:mr:maroon
        tokens  n-tokens  u.weights  u.config
        strategy
        eny.bowl
      ==
    ::  Build response: include both tokens and text if tokenizer loaded
    =/  body-json=json
      ?~  tok
        [%o (~(gas by *(map @t json)) ~[['output' [%a (turn out numb:enjs:format)]]])]
      =/  text  (decode:tokenizer u.tok out)
      %-  pairs:enjs:format
      :~  ['output' [%a (turn out numb:enjs:format)]]
          ['text' s+text]
      ==
    :-  (give-json eyre-id body-json)
    this(last-output out)
    ::
    ++  give-http
      |=  [eyre-id=@ta status=@ud headers=(list [@t @t]) body=(unit octs)]
      ^-  (list card)
      %+  give-simple-payload:app:server  eyre-id
      [[status headers] body]
    ::
    ++  give-json
      |=  [eyre-id=@ta jon=json]
      ^-  (list card)
      %+  give-simple-payload:app:server  eyre-id
      (json-response:gen:server jon)
    ::
    ++  not-found
      |=  eyre-id=@ta
      ^-  (list card)
      (give-http eyre-id 404 ~[['content-type' 'text/plain']] (some (as-octs:mimes:html 'not found')))
    ::
    ++  give-help
      |=  eyre-id=@ta
      ^-  (list card)
      =/  msg  'POST JSON {"tokens":[...], "n":N, "temperature":T?} to this endpoint'
      (give-http eyre-id 200 ~[['content-type' 'text/plain']] (some (as-octs:mimes:html msg)))
    --
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
    ::  Load tokenizer from jammed atom (vocab + merges + byte maps)
    ::
      %maroon-load-tokenizer
    =/  jammed  !<(@ vase)
    ~&  >  "loading tokenizer..."
    =/  t  (cue-tokenizer:tokenizer jammed)
    ~&  >  '%maroon tokenizer loaded successfully'
    `this(tok `t)
    ::
      %maroon-infer
    =/  tokens  !<((list @ud) vase)
    ?~  weights
      ~&  >>>  'no model loaded'
      `this
    ?~  config
      `this
    ~&  >  "running inference on {<(lent tokens)>} tokens..."
    =/  logits  (forward:mr:maroon tokens u.weights u.config)
    =/  next-token  (argmax-token:mr:maroon logits)
    ~&  >  "next token: {<next-token>}"
    `this(last-output ~[next-token])
    ::
      %maroon-generate
    =/  req  !<([prompt=(list @ud) n=@ud strategy=sampling:mr:maroon] vase)
    ?~  weights
      ~&  >>>  'no model loaded'
      `this
    ?~  config
      `this
    ~&  >  "generating {<n.req>} tokens from prompt of {<(lent prompt.req)>}..."
    =/  out
      %:  generate:mr:maroon
        prompt.req  n.req
        u.weights  u.config  strategy.req
        eny.bowl
      ==
    ~&  >  "generated: {<out>}"
    `this(last-output out)
  ==
::
++  on-watch
  |=  =path
  ^-  (quip card _this)
  ?+    path  (on-watch:def path)
    [%http-response *]  `this
  ==
++  on-leave  on-leave:def
++  on-peek
  |=  =path
  ^-  (unit (unit cage))
  ?+    path  (on-peek:def path)
    [%x %status ~]       ``noun+!>(?~(weights %no-model %model-loaded))
    [%x %config ~]       ?~(config ~ ``noun+!>(u.config))
    [%x %last-output ~]  ``noun+!>(last-output)
    ::  /x/tok-stats — sizes of tokenizer maps (debug)
      [%x %tok-stats ~]
    ?~  tok  ~
    =/  stats=[vocab=@ud inv=@ud merges=@ud bytemap=@ud ibm=@ud]
      :*  ~(wyt by vocab.u.tok)
          ~(wyt by inverse-vocab.u.tok)
          ~(wyt by merges.u.tok)
          ~(wyt by byte-map.u.tok)
          ~(wyt by inverse-byte-map.u.tok)
      ==
    ``noun+!>(stats)
    ::  /x/tok-get-vocab/KEY  — debug: lookup in vocab (given as @ud atom)
      [%x %tok-get-vocab @ ~]
    ?~  tok  ~
    ``noun+!>((~(get by vocab.u.tok) (@t (slav %ud i.t.t.path))))
    ::  /x/tok-get-inv/ID  — debug: look up ID in inverse-vocab
      [%x %tok-get-inv @ ~]
    ?~  tok  ~
    ``noun+!>((~(get by inverse-vocab.u.tok) (slav %ud i.t.t.path)))
    ::  /x/decode/~[id1 id2 ...] — decode token IDs to text (for debugging)
      [%x %decode *]
    ?~  tok  ~
    =/  ids=(list @ud)
      %+  turn  t.t.path
      |=  n=@ta
      (slav %ud n)
    ``noun+!>((decode:tokenizer u.tok ids))
    ::  /x/encode/~['text'] — encode text to token IDs (for debugging)
      [%x %encode @ ~]
    ?~  tok  ~
    ``noun+!>((encode:tokenizer u.tok (@t i.t.t.path)))
  ==
::
++  on-agent  on-agent:def
++  on-arvo
  |=  [=wire =sign-arvo]
  ^-  (quip card _this)
  ?+  wire  (on-arvo:def wire sign-arvo)
    [%eyre %connect ~]  `this
  ==
++  on-fail   on-fail:def
--
