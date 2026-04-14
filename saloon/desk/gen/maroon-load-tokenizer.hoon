::  Load GPT-2 BPE tokenizer from /weights/gpt2-tokenizer.jam
::
::  Generate with:
::    python3 tools/tokenizer_to_noun.py --output gpt2-tokenizer.jam
::  Put into your pier at saloon/weights/gpt2-tokenizer.jam then |commit.
::
/-  ls=lagoon
/+  *lagoon, math, saloon, maroon
::
:-  %say
|=  [[now=@da eny=@uv bec=beak] ~ ~]
:-  %maroon-load-tokenizer
::
=/  path  /(scot %p p.bec)/(scot %tas q.bec)/(scot %da now)/weights/gpt2-tokenizer/jam
=/  jam-res  (mule |.(.^(@ %cx path)))
?:  ?=(%| -.jam-res)
  ~&  >>>  'tokenizer not found at /weights/gpt2-tokenizer.jam'
  ~&  >>>  'generate with: python3 tools/tokenizer_to_noun.py --output gpt2-tokenizer.jam'
  ~|  %no-tokenizer-file
  !!
~&  >  "loaded gpt2-tokenizer.jam ({<(met 3 p.jam-res)>} bytes)"
p.jam-res
