::  Smoke test: cue the qwen3-bonsai single-file jam.
::
::  Reads /weights/qwen3-bonsai/jam from Clay (~538 MB), confirms cue succeeds,
::  shape-probes the top-level. No schema validation yet.
::
::  Requires 64-bit Vere (32-bit's u3r_met caps single atoms at ~512 MB).
::
::  Usage: +saloon!maroon-load-qwen3
::
:-  %say
|=  [[now=@da eny=@uv bec=beak] ~ ~]
:-  %noun
=/  path  /(scot %p p.bec)/(scot %tas q.bec)/(scot %da now)/weights/qwen3-bonsai/jam
=/  jam-res  (mule |.(.^(@ %cx path)))
?:  ?=(%| -.jam-res)
  ~|  %need-qwen3-bonsai-jam
  !!
~&  >  "loaded qwen3-bonsai.jam ({<(met 3 p.jam-res)>} bytes) — cueing..."
=/  cue-res  (mule |.((cue p.jam-res)))
?:  ?=(%| -.cue-res)
  ~&  >>>  'cue failed (loom?)'
  ~|  %cue-failed
  !!
=/  top  p.cue-res
~&  >  'cue succeeded'
~&  >  ?@(top 'WARN: top is atom, expected cell' 'top is cell as expected')
::  top should be [tok-emb blocks ln-f]
?@  top  !!
=/  tok-emb  -.top
~&  >  ?@(tok-emb 'WARN: tok-emb is atom' [%tok-emb-tag head=-.tok-emb])
'qwen3-bonsai-smoke-ok'
