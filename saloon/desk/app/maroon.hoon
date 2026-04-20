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
    tokenizer=gpt2-tokenizer
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
      gen=(unit gen-state)
      weights-qwen3=(unit model-weights-qwen3:maroon)
      config-qwen3=(unit model-config-qwen3:maroon)
  ==
::  Generation is tick-based so long CPU forwards don't exceed Vere's ~45s
::  HTTP chunk idle timeout.  Two tick modes:
::    %single       - whole forward per tick.  Used when the forward is
::                    fast enough (GPU-ready models, or any model small
::                    enough that one token fits under the timeout).
::    %block-stream - GPT-2 legacy: embed+pos on the first tick, one
::                    transformer block per subsequent tick, emit
::                    keepalive pings between ticks.  Needed when the
::                    CPU forward would exceed the timeout.
::
::  Which mode a given load uses lives in +default-tick-mode.  Anything
::  added to +forward-loaded with no stream impl defaults to %single.
+$  tick-mode  ?(%single %block-stream)
+$  gen-state
  $:  eyre-id=@ta
      tokens=(list @ud)        ::  prompt + tokens generated so far
      n-remaining=@ud
      strategy=sampling:mr:maroon
      step=@ud                  ::  for entropy mixing
      start=@da                 ::  when generation began
      last-tick=@da             ::  when the last tick event fired
      n-prompt=@ud              ::  length of the original prompt
      mode=tick-mode
      phase=?(%new %block %final)  ::  only used when mode = %block-stream
      block-idx=@ud             ::  next block to compute (mode = %block-stream)
      x=(unit tensor:maroon)    ::  intermediate activations across events
      cos=(unit tensor:maroon)  ::  RoPE cos table (qwen3 block-stream only)
      sin=(unit tensor:maroon)  ::  RoPE sin table (qwen3 block-stream only)
  ==
+$  card  card:agent:gall
::
::  Build an SSE data event body from a JSON payload.
::  Format: "data: <payload>\n\n"
::
++  sse-event-data
  |=  payload=@t
  ^-  octs
  =/  body  (rap 3 ~['data: ' payload (rap 3 ~[10 10])])
  [(met 3 body) body]
::
::  SSE comment (keepalive). Clients ignore lines starting with ':'.
::  Sending one resets Vere's HTTP chunk idle timeout.
::
++  ping-verbs
  ^-  (list @t)
  :~  'Accomplishing'  'Actioning'  'Actualizing'  'Architecting'
      'Baking'  'Beaming'  'Beboppin\''  'Befuddling'  'Billowing'
      'Blanching'  'Bloviating'  'Boogieing'  'Boondoggling'  'Booping'
      'Bootstrapping'  'Brewing'  'Bunning'  'Burrowing'  'Calculating'
      'Canoodling'  'Caramelizing'  'Cascading'  'Catapulting'
      'Cerebrating'  'Channeling'  'Channelling'  'Choreographing'
      'Churning'  'Clauding'  'Coalescing'  'Cogitating'  'Combobulating'
      'Composing'  'Computing'  'Concocting'  'Considering'
      'Contemplating'  'Cooking'  'Crafting'  'Creating'  'Crunching'
      'Crystallizing'  'Cultivating'  'Deciphering'  'Deliberating'
      'Determining'  'Dilly-dallying'  'Discombobulating'  'Doing'
      'Doodling'  'Drizzling'  'Ebbing'  'Effecting'  'Elucidating'
      'Embellishing'  'Enchanting'  'Envisioning'  'Evaporating'
      'Fermenting'  'Fiddle-faddling'  'Finagling'  'Flambéing'
      'Flibbertigibbeting'  'Flowing'  'Flummoxing'  'Fluttering'
      'Forging'  'Forming'  'Frolicking'  'Frosting'  'Gallivanting'
      'Galloping'  'Garnishing'  'Generating'  'Gesticulating'
      'Germinating'  'Gitifying'  'Grooving'  'Gusting'  'Harmonizing'
      'Hashing'  'Hatching'  'Herding'  'Honking'  'Hullaballooing'
      'Hyperspacing'  'Ideating'  'Imagining'  'Improvising'
      'Incubating'  'Inferring'  'Infusing'  'Ionizing'  'Jitterbugging'
      'Julienning'  'Kneading'  'Leavening'  'Levitating'  'Lollygagging'
      'Manifesting'  'Marinating'  'Meandering'  'Metamorphosing'
      'Misting'  'Moonwalking'  'Moseying'  'Mulling'  'Mustering'
      'Musing'  'Nebulizing'  'Nesting'  'Newspapering'  'Noodling'
      'Nucleating'  'Orbiting'  'Orchestrating'  'Osmosing'
      'Perambulating'  'Percolating'  'Perusing'  'Philosophising'
      'Photosynthesizing'  'Pollinating'  'Pondering'  'Pontificating'
      'Pouncing'  'Precipitating'  'Prestidigitating'  'Processing'
      'Proofing'  'Propagating'  'Puttering'  'Puzzling'  'Quantumizing'
      'Razzle-dazzling'  'Razzmatazzing'  'Recombobulating'
      'Reticulating'  'Roosting'  'Ruminating'  'Sautéing'  'Scampering'
      'Schlepping'  'Scurrying'  'Seasoning'  'Shenaniganing'
      'Shimmying'  'Simmering'  'Skedaddling'  'Sketching'  'Slithering'
      'Smooshing'  'Sock-hopping'  'Spelunking'  'Spinning'  'Sprouting'
      'Stewing'  'Sublimating'  'Swirling'  'Swooping'  'Symbioting'
      'Synthesizing'  'Tempering'  'Thinking'  'Thundering'  'Tinkering'
      'Tomfoolering'  'Topsy-turvying'  'Transfiguring'  'Transmuting'
      'Twisting'  'Undulating'  'Unfurling'  'Unravelling'  'Vibing'
      'Waddling'  'Wandering'  'Warping'  'Whatchamacalliting'
      'Whirlpooling'  'Whirring'  'Whisking'  'Wibbling'  'Working'
      'Wrangling'  'Zesting'  'Zigzagging'
  ==
::
++  sse-event-ping
  |=  eny=@
  ^-  octs
  =/  vs  ping-verbs
  =/  verb  (snag (mod eny (lent vs)) vs)
  =/  body  (rap 3 ~[': ' verb '...' (rap 3 ~[10 10])])
  [(met 3 body) body]
::
::  Model dispatch.  The orchestration code (HTTP handler, gen-tick)
::  never references a specific model by name — it goes through these
::  helpers.  Adding a new model type:
::    1. add (unit ...) state fields for its weights + config
::    2. add a clause to +forward-loaded
::    3. add a clause to +default-tick-mode iff the model needs
::       block-stream (otherwise it gets %single, which works for any
::       model whose forward fits under Vere's HTTP chunk timeout).
::
::  +model-loaded: does any model have both weights and config?
::
++  model-loaded
  |=  s=state-0
  ^-  ?
  ?|  ?&(?=(^ weights-qwen3.s) ?=(^ config-qwen3.s))
      ?&(?=(^ weights.s) ?=(^ config.s))
  ==
::
::  +forward-loaded: run a whole-forward for the loaded model, returning
::  logits, or ~ if nothing is loaded.  Called by scries and by the
::  %single-mode tick.
::
++  forward-loaded
  |=  [s=state-0 tokens=(list @ud)]
  ^-  (unit tensor:maroon)
  ?:  ?&(?=(^ weights-qwen3.s) ?=(^ config-qwen3.s))
    `(forward-qwen3:mr:maroon tokens u.weights-qwen3.s u.config-qwen3.s)
  ?:  ?&(?=(^ weights.s) ?=(^ config.s))
    `(forward:mr:maroon tokens u.weights.s u.config.s)
  ~
::
::  +default-tick-mode: per-loaded-model tick strategy.  GPT-2 is the
::  only backend with a %block-stream implementation today; everything
::  else defaults to %single.
::
++  default-tick-mode
  |=  s=state-0
  ^-  tick-mode
  ::  All current models use the streamed block-by-block path so CPU
  ::  forwards don't blow the HTTP chunk timeout.  New models with a
  ::  fast enough single-shot forward can pick %single here.
  %block-stream
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
    ::  accept numbers OR strings for float params — JSON numbers lose
    ::  precision for fp32 literals, so clients often send strings.
    =/  num-or-str=$-(json (unit @ta))
      |=  j=json  ^-  (unit @ta)
      ?+  j  ~
        [%n *]  `p.j
        [%s *]  `p.j
      ==
    =/  temperature=(unit @ta)
      %.  u.parsed
      %-  ot:dejs-soft:format
      :~  [%temperature num-or-str]
      ==
    =/  top-k-opt=(unit @ud)
      %.  u.parsed
      %-  ot:dejs-soft:format
      :~  ['top_k' ni:dejs-soft:format]
      ==
    =/  top-p-opt=(unit @ta)
      %.  u.parsed
      %-  ot:dejs-soft:format
      :~  ['top_p' num-or-str]
      ==
    =/  rep-pen-opt=(unit @ta)
      %.  u.parsed
      %-  ot:dejs-soft:format
      :~  ['repetition_penalty' num-or-str]
      ==
    ?.  (model-loaded state)
      :_  this
      (give-http eyre-id 503 ~ (some (as-octs:mimes:html '{"error":"no model loaded"}')))
    =/  tokens=(list @ud)
      ?^  tokens-opt  u.tokens-opt
      ?~  prompt-opt  ~
      ?~  tok  ~
      (encode:tokenizer u.tok u.prompt-opt)
    ?:  =(~ tokens)
      :_  this
      (give-http eyre-id 400 ~ (some (as-octs:mimes:html '{"error":"provide tokens or prompt (and load tokenizer)"}')))
    =/  n-tokens  (fall n 10)
    ::  Defaults: temp=0.7, top-p=0.9, rep-penalty=1.2, top-k disabled.
    ::  Clients can override any of them; setting top_p=1.0 disables it, etc.
    ::  slav %rs needs the `.` prefix (so '.5' parses as 5.0 but '5' bails);
    ::  normalize so JSON `5.0` and `"5.0"` both work.
    =/  to-rs
      |=  c=@ta  ^-  @rs
      =/  s=@t  ?:(=('.' (end 3 c)) c (rap 3 ~['.' c]))
      (slav %rs s)
    =/  strategy=sampling:mr:maroon
      :*  temp=?~(temperature .0.7 (to-rs u.temperature))
          top-k=(fall top-k-opt 0)
          top-p=?~(top-p-opt .0.9 (to-rs u.top-p-opt))
          rep-penalty=?~(rep-pen-opt .1.2 (to-rs u.rep-pen-opt))
      ==
    ~&  >  "SSE /apps/maroon/chat: streaming {<n-tokens>} tokens..."
    ::  Open SSE stream: send headers and the prompt as the first event
    =/  sse-headers
      ^-  (list [@t @t])
      :~  ['content-type' 'text/event-stream']
          ['cache-control' 'no-cache']
          ['x-accel-buffering' 'no']
      ==
    =/  sampling-json=json
      :-  %o
      %-  ~(gas by *(map @t json))
      :~  ['temperature' s+(scot %rs temp.strategy)]
          ['top_k' (numb:enjs:format top-k.strategy)]
          ['top_p' s+(scot %rs top-p.strategy)]
          ['repetition_penalty' s+(scot %rs rep-penalty.strategy)]
      ==
    =/  prompt-event
      =/  prompt-json=json
        :-  %o
        %-  ~(gas by *(map @t json))
        :~  ['type' s+'prompt']
            ['tokens' [%a (turn tokens numb:enjs:format)]]
            ['sampling' sampling-json]
        ==
      (sse-event-data (en:json:html prompt-json))
    =/  resp-header=response-header:http  [200 sse-headers]
    =/  cards=(list card)
      :~  [%give %fact ~[/http-response/[eyre-id]] %http-response-header !>(resp-header)]
          [%give %fact ~[/http-response/[eyre-id]] %http-response-data !>(`prompt-event)]
          [%pass /gen-tick %arvo %b %wait now.bowl]
      ==
    ::  Stash the in-progress generation, picking tick mode per model.
    =/  new-gen=gen-state
      :*  eyre-id  tokens  n-tokens  strategy  0
          now.bowl  now.bowl  (lent tokens)
          (default-tick-mode state)
          %new  0  ~  ~  ~
      ==
    [cards this(gen `new-gen)]
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
    ::  Load Qwen3 weights (mlx2-quantized). Payload is [qwen3-cfg jam-atom].
    ::  Weights are cued off the jam atom and stored in state.
    ::
      %maroon-load-qwen3
    =/  payload  !<([model-config-qwen3:maroon @] vase)
    =/  cfg  -.payload
    =/  jammed  +.payload
    ~&  >  "loading qwen3: d={<d-model.cfg>} heads={<n-heads.cfg>} kv-heads={<n-kv-heads.cfg>} layers={<n-layers.cfg>} vocab={<vocab-size.cfg>}"
    =/  w  ;;(model-weights-qwen3:maroon (cue jammed))
    ~&  >  '%maroon qwen3 model loaded successfully'
    `this(weights-qwen3 `w, config-qwen3 `cfg)
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
    ::  /x/wo-bias-shape — inspect first block's wo.b shape (debug)
      [%x %wo-bias-shape ~]
    ?~  weights  ~
    =/  blks  blocks.u.weights
    ?~  blks  ~
    ``noun+!>(`(list @)`shape.meta.b.wo.i.blks)
    ::  /x/wo-w-tag — inspect first block's wo.w tag (%fp or %q8)
      [%x %wo-w-tag ~]
    ?~  weights  ~
    =/  blks  blocks.u.weights
    ?~  blks  ~
    ``noun+!>(-.w.wo.i.blks)
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
    ::  /x/test-transpose-mlx2/noun
    ::    Dequant block-0 q-proj, then transpose, return first 8 fp32 vals
    ::    of row 0 of the TRANSPOSED matrix. Isolates whether transpose works
    ::    on a dequanted tensor.
      [%x %test-transpose-mlx2 ~]
    =/  jres  (mule |.(.^(@ %cx /(scot %p our.bowl)/saloon/(scot %da now.bowl)/weights/qwen3-bonsai/jam)))
    ?:  ?=(%| -.jres)  ~
    =/  ws  ;;(model-weights-qwen3:maroon (cue p.jres))
    =/  blk0   (snag 0 blocks.ws)
    =/  qproj  q-proj.blk0
    ?>  ?=([%mlx2 *] qproj)
    =/  fp  (dequant-mlx2-ray:maroon wq.qproj scales.qproj biases.qproj group-size.qproj)
    =/  fp-t  fp   :: dequant now produces [in, out] = transposed by construction
    =/  vals=(list @rs)
      :~  `@rs`(get-item:la fp-t ~[0 0])
          `@rs`(get-item:la fp-t ~[0 1])
          `@rs`(get-item:la fp-t ~[0 2])
          `@rs`(get-item:la fp-t ~[0 3])
          `@rs`(get-item:la fp-t ~[0 4])
          `@rs`(get-item:la fp-t ~[0 5])
          `@rs`(get-item:la fp-t ~[0 6])
          `@rs`(get-item:la fp-t ~[0 7])
      ==
    ``noun+!>(vals)
    ::  /x/test-dequant-mlx2/noun
    ::    Reads /weights/qwen3-bonsai/jam, cues, pulls block 0 q-proj (which
    ::    is an mlx2-packed weight), dequants it, returns first 8 fp32 values
    ::    of row 0 as a (list @rs).
    ::    Compare against numpy reference (saloon/tools/dequant_q_proj.py):
    ::      [-0.02282715 -0.02282715 -0.02282715 0 0.02282715 -0.02282715 0 0]
    ::    Exists here so click threads can validate dequant correctness on
    ::    real weights without /+ importing maroon. Remove when validated.
    ::  /x/test-dequant-raw/N/noun — dequant q-proj, return all of:
    ::    - shape of output meta
    ::    - cut 5 [N 1] data
    ::    - get-item [0, N]
    ::    - data's met 3 (byte count)
    ::  to cross-check get-item vs direct cut.
      [%x %test-dequant-raw @ ~]
    =/  jres  (mule |.(.^(@ %cx /(scot %p our.bowl)/saloon/(scot %da now.bowl)/weights/qwen3-bonsai/jam)))
    ?:  ?=(%| -.jres)  ~
    =/  ws  ;;(model-weights-qwen3:maroon (cue p.jres))
    =/  blk0   (snag 0 blocks.ws)
    =/  qproj  q-proj.blk0
    ?.  ?=(%mlx2 -.qproj)  ~
    =/  deq  (dequant-mlx2-ray:maroon wq.qproj scales.qproj biases.qproj group-size.qproj)
    =/  n  (slav %ud i.t.t.path)
    ``noun+!>(`@ux`(get-item:la deq ~[0 n]))
    ::
    ::  /x/test-raw-word/N/noun — return w-data word at linear offset N
    ::  (diagnostic for jet vs hoon layout verification)
      [%x %test-raw-word @ ~]
    =/  jres  (mule |.(.^(@ %cx /(scot %p our.bowl)/saloon/(scot %da now.bowl)/weights/qwen3-bonsai/jam)))
    ?:  ?=(%| -.jres)  ~
    =/  ws  ;;(model-weights-qwen3:maroon (cue p.jres))
    =/  blk0   (snag 0 blocks.ws)
    =/  qproj  q-proj.blk0
    ?>  ?=([%mlx2 *] qproj)
    =/  n  (slav %ud i.t.t.path)
    ``noun+!>(`@ux`(cut 5 [n 1] data.wq.qproj))
    ::
      [%x %test-dequant-mlx2 ~]
    =/  jres  (mule |.(.^(@ %cx /(scot %p our.bowl)/saloon/(scot %da now.bowl)/weights/qwen3-bonsai/jam)))
    ?:  ?=(%| -.jres)  ~
    =/  ws  ;;(model-weights-qwen3:maroon (cue p.jres))
    =/  blk0   (snag 0 blocks.ws)
    =/  qproj  q-proj.blk0
    ?>  ?=([%mlx2 *] qproj)
    =/  fp  (dequant-mlx2-ray:maroon wq.qproj scales.qproj biases.qproj group-size.qproj)
    =/  vals=(list @rs)
      :~  `@rs`(get-item:la fp ~[0 0])
          `@rs`(get-item:la fp ~[0 1])
          `@rs`(get-item:la fp ~[0 2])
          `@rs`(get-item:la fp ~[0 3])
          `@rs`(get-item:la fp ~[0 4])
          `@rs`(get-item:la fp ~[0 5])
          `@rs`(get-item:la fp ~[0 6])
          `@rs`(get-item:la fp ~[0 7])
      ==
    ``noun+!>(vals)
    ::  /x/qwen3-say/<prompt>/noun — end-to-end inference with decoded output.
    ::    Encodes <prompt> via the loaded tokenizer, runs forward-qwen3 on the
    ::    loaded qwen3 weights, returns [next-id=@ud next-text=@t]. Requires
    ::    prior pokes: %maroon-load-qwen3 and %maroon-load-tokenizer.
      [%x %qwen3-say @ ~]
    ?~  weights-qwen3
      ~&  >>>  'no qwen3 weights loaded — poke %maroon-load-qwen3 first'
      ~
    ?~  config-qwen3  ~
    ?~  tok
      ~&  >>>  'no tokenizer loaded — poke %maroon-load-tokenizer first'
      ~
    =/  prompt=@t  (@t i.t.t.path)
    =/  ids=(list @ud)  (encode:tokenizer u.tok prompt)
    =/  logits  (forward-qwen3:mr:maroon ids u.weights-qwen3 u.config-qwen3)
    =/  next-id  (argmax-token:mr:maroon logits)
    =/  next-text=@t  (decode:tokenizer u.tok ~[next-id])
    ``noun+!>([next-id next-text])
    ::
    ::  /x/forward-qwen3/<id1>/<id2>/.../noun
    ::    Reads /weights/qwen3-bonsai/jam, cues to model-weights-qwen3,
    ::    runs forward on the supplied token IDs, returns next argmax token.
    ::    Bonsai-1.7B config is hardcoded; this is for testing, not production.
    ::    Exists here so click threads (which can't /+ import maroon) can drive
    ::    the forward-qwen3 path; remove when an inference scry surface lands.
      [%x %forward-qwen3 *]
    =/  ids=(list @ud)
      %+  turn  t.t.path
      |=  n=@ta
      (slav %ud n)
    =/  jres  (mule |.(.^(@ %cx /(scot %p our.bowl)/saloon/(scot %da now.bowl)/weights/qwen3-bonsai/jam)))
    ?:  ?=(%| -.jres)  ~
    =/  ws  ;;(model-weights-qwen3:maroon (cue p.jres))
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
    =/  logits  (forward-qwen3:mr:maroon ids ws cfg)
    ``noun+!>((argmax-token:mr:maroon logits))
  ==
::
++  on-agent  on-agent:def
++  on-arvo
  |=  [=wire =sign-arvo]
  ^-  (quip card _this)
  ?+    wire  (on-arvo:def wire sign-arvo)
      [%eyre %connect ~]  `this
      [%gen-tick ~]
    ?~  gen  `this
    =/  g  u.gen
    =/  la  (lake %n)
    =/  seq-len  (lent tokens.g)
    =/  ping-card=card
      :*  %give  %fact  ~[/http-response/[eyre-id.g]]
          %http-response-data  !>(`(sse-event-ping (mix eny.bowl step.g)))
      ==
    =/  tick-card=card  [%pass /gen-tick %arvo %b %wait now.bowl]
    ?-    mode.g
        ::  +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        ::  Single-shot: one tick = one token.  Model-agnostic.
        ::  +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        %single
      =/  logits-u  (forward-loaded state tokens.g)
      ?~  logits-u  `this(gen ~)
      =/  next-tok
        (sample-token:mr:maroon u.logits-u strategy.g tokens.g (mix eny.bowl step.g))
      =/  text-chunk=@t  ?~(tok '' (decode:tokenizer u.tok ~[next-tok]))
      =/  total=@dr  (sub now.bowl start.g)
      ~&  >  ['step' +(step.g) 'tok' next-tok 'text' text-chunk 'total' total]
      =/  token-card=card
        =/  chunk-json=json
          :-  %o
          %-  ~(gas by *(map @t json))
          :~  ['type' s+'token']  ['id' (numb:enjs:format next-tok)]
              ['text' s+text-chunk]
          ==
        :*  %give  %fact  ~[/http-response/[eyre-id.g]]
            %http-response-data
            !>(`(sse-event-data (en:json:html chunk-json)))
        ==
      =/  new-tokens  (snoc tokens.g next-tok)
      =/  remaining   (dec n-remaining.g)
      ?:  =(0 remaining)
        =/  gen-toks  (slag n-prompt.g new-tokens)
        =/  full-text=@t  ?~(tok '' (decode:tokenizer u.tok gen-toks))
        ~&  >  ['DONE tokens' (lent gen-toks) 'total' total 'text' full-text]
        =/  done-card=card
          =/  done-json=json  [%o (~(gas by *(map @t json)) ~[['type' s+'done']])]
          :*  %give  %fact  ~[/http-response/[eyre-id.g]]
              %http-response-data
              !>(`(sse-event-data (en:json:html done-json)))
          ==
        =/  kick-card=card  [%give %kick ~[/http-response/[eyre-id.g]] ~]
        :_  this(gen ~, last-output new-tokens)
        ~[token-card done-card kick-card]
      :_  this(gen `g(tokens new-tokens, n-remaining remaining, step +(step.g), last-tick now.bowl))
      ~[token-card tick-card]
    ::
        ::  +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        ::  Block-stream: embed on the first tick, one transformer block
        ::  per subsequent tick, final+sample on the last.  Keepalive
        ::  pings fire between every tick so the HTTP chunk stream stays
        ::  alive even under slow CPU forwards.  Each phase dispatches
        ::  per-model by inspecting which weights are loaded.
        ::  +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        %block-stream
      ?-    phase.g
          %new
        ::  Qwen3: embed + precompute RoPE cos/sin.
        ?:  ?&(?=(^ weights-qwen3) ?=(^ config-qwen3))
          =/  ws   u.weights-qwen3
          =/  cfg  u.config-qwen3
          =/  emb  (forward-qwen3-embed:mr:maroon tokens.g ws cfg)
          :_  this(gen `g(phase %block, block-idx 0, x `x.emb, cos `cos.emb, sin `sin.emb, last-tick now.bowl))
          ~[ping-card tick-card]
        ::  GPT-2: embed + positional embeddings.
        ?~  weights  `this
        ?~  config   `this
        =/  w  u.weights
        =/  c  u.config
        =/  x0  (embed:mr:maroon tokens.g tok-emb.w bloq.c)
        =/  d-model  (snag 1 shape.meta.x0)
        =/  pos=tensor:maroon
          =/  init  (zeros:la [~[seq-len d-model] bloq.c %i754 ~])
          =/  i  0
          |-  ^-  tensor:maroon
          ?:  =(i seq-len)  init
          =/  row  (get-row:la pos-emb.w ~[i])
          $(i +(i), init (set-row:la init ~[i] row))
        =/  x1  (add:la x0 pos)
        :_  this(gen `g(phase %block, block-idx 0, x `x1, last-tick now.bowl))
        ~[ping-card tick-card]
      ::
          %block
        ?~  x.g  `this
        ::  Qwen3
        ?:  ?&(?=(^ weights-qwen3) ?=(^ config-qwen3))
          ?~  cos.g  `this
          ?~  sin.g  `this
          =/  ws   u.weights-qwen3
          =/  cfg  u.config-qwen3
          =/  blk  (snag block-idx.g blocks.ws)
          =/  x-next
            (transformer-block-qwen3:mr:maroon u.x.g blk cfg u.cos.g u.sin.g)
          =/  next-idx  +(block-idx.g)
          =/  n-layers  (lent blocks.ws)
          =/  next-phase=?(%new %block %final)
            ?:  =(next-idx n-layers)  %final
            %block
          :_  this(gen `g(phase next-phase, block-idx next-idx, x `x-next, last-tick now.bowl))
          ~[ping-card tick-card]
        ::  GPT-2
        ?~  weights  `this
        ?~  config   `this
        =/  w  u.weights
        =/  c  u.config
        =/  blk  (snag block-idx.g blocks.w)
        =/  x-next  (transformer-block:mr:maroon u.x.g blk n-heads.c)
        =/  next-idx  +(block-idx.g)
        =/  n-layers  (lent blocks.w)
        =/  next-phase=?(%new %block %final)
          ?:  =(next-idx n-layers)  %final
          %block
        :_  this(gen `g(phase next-phase, block-idx next-idx, x `x-next, last-tick now.bowl))
        ~[ping-card tick-card]
      ::
          %final
        ?~  x.g  `this
        ::  Compute logits per model.
        =/  logits=tensor:maroon
          ?:  ?&(?=(^ weights-qwen3) ?=(^ config-qwen3))
            (forward-qwen3-final:mr:maroon u.x.g tokens.g u.weights-qwen3 u.config-qwen3)
          ?~  weights  !!
          ?~  config   !!
          =/  w  u.weights
          =/  c  u.config
          =/  x-norm  (layer-norm-2d:mr:maroon u.x.g ln-f-g.w ln-f-b.w)
          =/  last-row  (get-row:la x-norm ~[(dec seq-len)])
          =/  bias-zeros  (zeros:la [~[1 vocab-size.c] bloq.c %i754 ~])
          (linear:mr:maroon last-row [[%fp out-proj.w] bias-zeros])
        =/  next-tok
          (sample-token:mr:maroon logits strategy.g tokens.g (mix eny.bowl step.g))
        =/  text-chunk=@t  ?~(tok '' (decode:tokenizer u.tok ~[next-tok]))
        =/  total=@dr  (sub now.bowl start.g)
        ~&  >  ['step' +(step.g) 'tok' next-tok 'text' text-chunk 'total' total]
        =/  token-card=card
          =/  chunk-json=json
            :-  %o
            %-  ~(gas by *(map @t json))
            :~  ['type' s+'token']  ['id' (numb:enjs:format next-tok)]
                ['text' s+text-chunk]
            ==
          :*  %give  %fact  ~[/http-response/[eyre-id.g]]
              %http-response-data
              !>(`(sse-event-data (en:json:html chunk-json)))
          ==
        =/  new-tokens  (snoc tokens.g next-tok)
        =/  remaining   (dec n-remaining.g)
        ?:  =(0 remaining)
          =/  gen-toks  (slag n-prompt.g new-tokens)
          =/  full-text=@t  ?~(tok '' (decode:tokenizer u.tok gen-toks))
          ~&  >  ['DONE tokens' (lent gen-toks) 'total' total 'text' full-text]
          =/  done-card=card
            =/  done-json=json  [%o (~(gas by *(map @t json)) ~[['type' s+'done']])]
            :*  %give  %fact  ~[/http-response/[eyre-id.g]]
                %http-response-data
                !>(`(sse-event-data (en:json:html done-json)))
            ==
          =/  kick-card=card  [%give %kick ~[/http-response/[eyre-id.g]] ~]
          :_  this(gen ~, last-output new-tokens)
          ~[token-card done-card kick-card]
        :_  this(gen `g(tokens new-tokens, n-remaining remaining, step +(step.g), last-tick now.bowl, phase %new, block-idx 0, x ~, cos ~, sin ~))
        ~[token-card tick-card]
      ==
    ==
  ==
++  on-fail   on-fail:def
--
