## Expected wiring

The node accepts and returns **`CONDITIONING` only**. It does not connect to the model.

Use it after text encoding and before any reference-conditioning node:

```text
CLIP Text Encode (positive)
        ↓
Nunchaku FLUX.2 Klein Text Enhancer
        ↓
Multi Reference Latent          [only for image editing]
        ↓
KSampler / CFG Guider positive
```

For ordinary text-to-image:

```text
CLIP Loader
    ↓
CLIP Text Encode
    ↓
Nunchaku FLUX.2 Klein Text Enhancer
    ↓
KSampler positive
```

For reference editing:

```text
CLIP Text Encode
    ↓
Text Enhancer
    ↓
Multi Reference Latent
    ↓
Sampler positive
```

Putting the enhancer **before** `Multi Reference Latent` is conceptually cleaner: it modifies only the text-conditioning tensor, while the later node attaches reference latents as metadata. The enhancer itself preserves conditioning metadata and changes only the active token slice. 

## Negative conditioning

For the first test, use your normal negative path unchanged:

```text
positive text → Text Enhancer → positive
positive text → Conditioning Zero Out → negative
```

At CFG `1.0`, the negative branch generally does not provide a useful comparison. For a basic correctness test, keep CFG, seed, sampler, resolution, prompt, LoRAs, and references fixed.

You may later test the enhancer on both branches, but that is a separate semantic experiment. Do not initially apply it to the zeroed-negative conditioning.

---

## Controls

### `magnitude`

Multiplies active text embeddings.

* `1.0`: neutral
* below `1.0`: lower embedding magnitude
* above `1.0`: higher embedding magnitude
* `0.0`: active text slice becomes zero after the other transforms

This is not equivalent to CFG. It scales the Qwen conditioning before the model processes it.

### `contrast`

Changes differences between active tokens relative to their sequence mean.

* `0.0`: neutral
* positive: tokens become more differentiated
* negative: tokens become more similar

The implementation uses:

```text
positive contrast: scale = 1 + contrast
negative contrast: scale = exp(contrast)
```

So `contrast=-1.0` does not invert tokens; it compresses deviations to about `0.368` of their original size.

### `normalize_strength`

Equalizes the norms—roughly the magnitudes—of active token embeddings.

* `0.0`: neutral
* `1.0`: each active token is rescaled toward the active region’s mean token norm

### `skip_bos`

When enabled, token `0` is excluded. The original node treats this as the BOS token, whose norm may be unusually large.

Keep it enabled for normal usage.

### `debug`

Logs:

* active token range;
* initial mean norm;
* final mean norm.

This is useful for proving that the node executed, although the output image may not change dramatically at moderate settings.

---

# Recommended first test

Use text-to-image without references or LoRAs. This isolates the node.

Prompt:

```text
A close-up portrait of a woman wearing an ornate red jacket,
dramatic side lighting, detailed skin, cinematic photography.
```

Keep:

```text
seed:       0
steps:      8
CFG:        1.0
sampler:    Euler
scheduler:  your normal Klein scheduler
resolution: 1024 × 1024 or your established test resolution
```

Generate these four outputs.

### Test 1 — bypass

Connect `CLIP Text Encode` directly to the sampler.

### Test 2 — neutral node

```text
magnitude:          1.0
contrast:           0.0
normalize_strength: 0.0
skip_bos:           true
debug:              true
```

Expected:

* The node returns the exact original conditioning object through its neutral fast path.
* The result should follow the same ordinary path as bypassing the node. 
* Because Nunchaku has known fused-kernel nondeterminism, separately queued generations might not be pixel-identical, so routing and tensor-level tests are stronger evidence than decoded-image identity.

### Test 3 — obvious magnitude change

```text
magnitude:          2.0
contrast:           0.0
normalize_strength: 0.0
skip_bos:           true
debug:              true
```

Expected:

* Console final mean norm should be approximately twice the initial mean norm.
* Image behavior may show stronger prompt adherence, but this is not guaranteed to be visually linear.

### Test 4 — obvious combined transformation

```text
magnitude:          1.5
contrast:           1.0
normalize_strength: 0.75
skip_bos:           true
debug:              true
```

Expected:

* The node logs a changed final mean norm.
* The output should differ materially from neutral, although exactly how it differs is model-dependent.

---

## Useful A/B workflow

Use a conditioning switch or queue the settings manually:

```text
A: bypass
B: neutral
C: magnitude 2.0
D: magnitude 1.5, contrast 1.0, normalize 0.75
```

Everything downstream must stay identical.

Do not compare outputs generated with:

* different prompts;
* different seeds;
* changed reference order;
* different LoRA state;
* different resolution;
* different sampler schedules.

Those variables can overwhelm the conditioning change.

---

## Test each control independently

After the initial check, isolate each parameter:

| Test               | Magnitude | Contrast | Normalize |
| ------------------ | --------: | -------: | --------: |
| Neutral            |       1.0 |      0.0 |       0.0 |
| Weak magnitude     |       0.5 |      0.0 |       0.0 |
| Strong magnitude   |       2.0 |      0.0 |       0.0 |
| Positive contrast  |       1.0 |      1.0 |       0.0 |
| Negative contrast  |       1.0 |     -1.0 |       0.0 |
| Full normalization |       1.0 |      0.0 |       1.0 |

This makes it possible to diagnose a single control rather than interpreting three transformations at once.

## What to look for in the console

With `debug=true`, expect messages resembling:

```text
Text Enhancer item 0: active tokens [1:N], initial mean norm ...
Text Enhancer item 0: final mean norm ...
```

With typical masked Qwen conditioning, `N` is derived from the last nonzero position in `attention_mask`. Without a valid two-dimensional mask—or with an all-zero mask—the compatibility node preserves the original node’s legacy fallback of up to 77 tokens. 

The first practical validation should therefore be:

1. Confirm the node appears under
   `Nunchaku/FLUX.2 Klein/Enhancer/Text`.
2. Run neutral with debug enabled.
3. Run `magnitude=2.0`.
4. Verify the final logged mean norm changes substantially.
5. Compare decoded outputs only after confirming the tensor transform executed.
