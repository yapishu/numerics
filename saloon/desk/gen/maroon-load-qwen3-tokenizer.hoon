::  Load Qwen3 BBPE tokenizer from /weights/qwen3-tokenizer.jam.
::
::  Generate with:
::    gguf2jam --arch tokenizer <hf-tokenizer.json> -o qwen3-tokenizer.jam
::  Put into your pier at saloon/weights/qwen3-tokenizer.jam then |commit.
::
::  Pokes %maroon with %maroon-load-tokenizer (same mark as gpt2; the on-poke
::  handler just calls cue-tokenizer:tokenizer which is BBPE-generic). Encode
::  will only produce GPT-2-correct splits until we add a qwen3-tokenizer.hoon
::  with Qwen3's pre-tokenize regex; decode-via-inverse-vocab works today.
::
:-  %say
|=  [[now=@da eny=@uv bec=beak] ~ ~]
:-  %maroon-load-tokenizer
::
=/  path  /(scot %p p.bec)/(scot %tas q.bec)/(scot %da now)/weights/qwen3-tokenizer/jam
=/  jam-res  (mule |.(.^(@ %cx path)))
?:  ?=(%| -.jam-res)
  ~&  >>>  'tokenizer not found at /weights/qwen3-tokenizer.jam'
  ~&  >>>  'generate with: gguf2jam --arch tokenizer <hf-tokenizer.json> -o qwen3-tokenizer.jam'
  ~|  %no-tokenizer-file
  !!
~&  >  "loaded qwen3-tokenizer.jam ({<(met 3 p.jam-res)>} bytes)"
p.jam-res
