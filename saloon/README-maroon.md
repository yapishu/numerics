# Maroon — On-Ship Transformer Inference

Maroon is a Hoon library + Gall agent for running small transformer models
on your Urbit ship. The model's weights live as a noun in your ship's state.
The forward pass is pure Hoon, built on Lagoon (arrays) and Saloon (activations).

## What's here

- `desk/lib/maroon.hoon` — inference library (linear, attention, feed-forward,
  transformer blocks, full forward pass)
- `desk/app/maroon.hoon` — Gall agent that holds model weights in state and
  runs inference on demand
- `desk/mar/maroon-load.hoon`, `desk/mar/maroon-infer.hoon` — poke marks
- `desk/gen/maroon-test.hoon` — generator that builds tiny test weights inline
  and runs a forward pass (no external weights needed)
- `desk/gen/maroon-load-gpt2.hoon` — generator that loads GPT-2 weights from
  `/weights/gpt2.jam` in your pier
- `tools/weights_to_noun.py` — converts HuggingFace model weights to a jammed
  noun matching the `$model-weights` type

## Quick test — no model needed

```
+saloon!maroon-test
```

Builds tiny dummy weights, runs a complete transformer forward pass, prints
the predicted next token.

## Loading a real model

The GPT-2 jam file is ~650MB and is NOT distributed with the desk. You need
to generate it yourself and drop it into your pier.

1. Install the Python deps:
   ```
   pip install torch transformers numpy
   ```

2. Convert GPT-2 small (~10 minutes):
   ```
   cd saloon/tools
   python3 weights_to_noun.py --model gpt2 --output gpt2.jam
   ```

3. Copy the jam file into your pier's saloon desk at `/weights/gpt2.jam`.
   (Mount the desk with `|mount %saloon` if it's not already mounted.)

4. Commit the desk from the dojo:
   ```
   |commit %saloon
   ```

5. Load the weights into the running agent:
   ```
   +saloon!maroon-load-gpt2
   ```
   This runs the generator which cues the jam file and pokes the agent.
   Loading takes a few moments (cueing + type-checking 650MB of nouns).

6. Run inference on a token sequence:
   ```
   :maroon &maroon-infer ~[0 1 2]
   ```

If no weights are loaded, the agent prints a warning instead of crashing.

## Architecture

The agent holds `weights=(unit model-weights)` and `config=(unit model-config)`
in its state. Pokes:
- `%maroon-load` with `[model-config jammed-weights=@]` — loads a model
- `%maroon-infer` with `(list @ud)` — runs inference on a token sequence

The full forward pass is:
```
tokens -> embed -> add pos-emb -> [transformer blocks] -> final layer-norm -> project -> logits
```

Each transformer block (pre-norm):
```
x -> layer-norm -> multi-head attention -> add residual
  -> layer-norm -> feed-forward (GELU) -> add residual
```

Multi-head attention splits Q/K/V into `n_heads` heads of dimension
`d_model / n_heads`, runs scaled dot-product attention on each
(with causal mask), concatenates, and projects.

## Performance

The current implementation is pure Hoon. Every matmul element is computed
via repeated `fun-scalar` calls. For real use, you need jetted Lagoon
operations. Jets exist for lagoon's add, sub, mul, div, mmul, etc.

The eml primitive (`exp(x) - ln(y)`) has a jet hint and a C implementation
in `../vere/pkg/noun/jets/e/math_rs.c` — rebuild vere to enable it.

## Chat workflow (tokenize → generate → detokenize)

Chat requires the GPT-2 BPE tokenizer. Since BPE isn't implemented in Hoon
yet, we use `chat.py` as a Python sidecar that wraps the ship interaction.

1. Install HuggingFace transformers:
   ```
   pip install transformers
   ```

2. Encode a prompt:
   ```
   python3 tools/chat.py --prompt "Once upon a time" --n 10
   ```
   This prints a dojo command like:
   ```
   :maroon &maroon-generate [~[7454 2402 257 640] 10 [%greedy ~]]
   ```

3. Paste that into your ship's dojo. The agent will generate N tokens
   (slow without jets — minutes per token on real GPT-2).

4. Retrieve the output token IDs with a scry:
   ```
   .^((list @ud) %gx /=maroon=/last-output/noun)
   ```

5. Decode back to text:
   ```
   python3 tools/chat.py --decode '7454 2402 257 640 198 198 198 198'
   ```

### Sampling strategies

Replace `[%greedy ~]` with one of:
- `[%greedy ~]` — always pick the highest-logit token (deterministic)
- `[%temperature .0.8]` — scale logits by 1/t then sample (lower = sharper)
- `[%top-k 40 .0.9]` — keep top-k, temperature-scale, sample

## Credits

Based on the EML architecture: Odrzywołek (2026), "All elementary functions
from a single operator" (arXiv:2603.21852). Built on top of Saloon/Lagoon
by @sigilante and the Urbit numerics team.
