---
title: "Why my college Counter-Strike lagged, and how that led me to understand GPUs"
author: "Suresh Rajendiran"
pubDatetime: 2026-10-17T09:00:00+05:30
tags:
  - gpu
  - ai
  - hardware
description: "It started with a laggy game on a government laptop. It ended with me finally understanding why GPUs — not CPUs — run modern AI. Matrix math, a real benchmark, and the hardware behind it."
---

Back in college, a few of us would install Counter-Strike on the government-issued laptops we got and play during the free hours. The laptops were fine for notes and labs. For CS, they were miserable. The game ran, but it *crawled* — jerky movement, frames dropping every time something exploded on screen.

For years I just filed it under "cheap laptop." It was only much later, when I started poking at how AI actually runs, that I understood what was really missing: those laptops had no real GPU. And the exact thing that made CS lag is the exact thing that makes modern AI need GPUs.

So this post is me connecting those dots. If you've ever wondered why AI runs on GPUs and not CPUs — and what's actually inside a GPU — this is the explanation I wish someone had given me.

## Table of contents

## It all comes down to one operation: matrix multiplication

Here's the thing that surprised me most: whether it's rendering a game or running ChatGPT, the computer is mostly doing the *same kind of math* underneath — **matrix multiplication**.

A matrix is just a grid of numbers. Nothing scary. And matrix multiplication is just a way to apply the same transformation to a whole batch of data at once.

Let me show it with something concrete instead of abstract math.

### Example: a photo filter

Every pixel on your screen is three numbers — Red, Green, Blue, each from 0 to 255. A "sepia" filter (that warm, old-photo brown) is literally a small recipe for mixing those three numbers into new ones:

```
New Red   = 0.4 × old R  +  0.8 × old G  +  0.2 × old B
New Green = 0.3 × old R  +  0.7 × old G  +  0.2 × old B
New Blue  = 0.3 × old R  +  0.5 × old G  +  0.1 × old B
```

Those nine numbers are the "sepia matrix." Take a pixel — say R:100, G:150, B:200 (a soft blue) — run the recipe, and you get R:200, G:175, B:125 (a warm tan). That's the filter, applied to one pixel.

A photo has *millions* of pixels. Same recipe, applied to every single one. That's a matrix multiplication.

### Where AI comes in

Now the part that made it click for me. A neural network layer does the **exact same thing**:

```
output = weights × input
```

The only difference from the sepia filter is that nobody writes the "recipe" by hand — the network *learns* those numbers from data. They're called **weights**.

A tiny example — predicting a house price from size, bedrooms, and age:

```
price = 150 × size  +  30 × bedrooms  +  (−20) × age
```

The network figured out `150`, `30`, `−20` by looking at thousands of past sales. Feed in a house, get a prediction. Same shape as the filter.

And a language model like ChatGPT? Same idea again. It turns words into numbers, multiplies them by huge learned weight matrices, and the result is scores for what word comes next:

```
"my"  →  [numbers]  →  × weights  →  scores for every possible next word  →  "love"
```

It does this one word at a time — which is why you see ChatGPT type out its answer word by word. Each word is another matrix multiplication.

> The whole insight: **graphics, house prices, and ChatGPT are all the same operation — `output = weights × input` — just at wildly different scales.** A real language model does *billions* of these per response.

### Why a matrix, if it's the same number of operations?

This one bugged me for a while. Writing it as a matrix doesn't reduce the math — it's the same multiplications and additions either way. So why bother?

Because the matrix *tells the hardware something*: "here is a big batch of operations, and none of them depend on each other." Every output number is computed independently. That independence is the whole game — it's what lets you do them all at once instead of one after another.

Which is exactly where the GPU comes in.

## Why a GPU beats a CPU at this

A CPU and a GPU are built on opposite philosophies.

A **CPU** is a handful of very smart, very fast cores — maybe 8 to 64 of them. Think of a few brilliant professors. They're great at complicated, step-by-step problems where each step depends on the last.

A **GPU** is thousands of simpler cores. Think of a stadium full of students. Each one is slower than a professor, but there are *thousands* of them, and they can all do the same simple thing at the same time.

Now look back at the photo filter. Two million pixels, same recipe, every pixel independent. A CPU walks through them with its few cores. A GPU hands each pixel to its own core and does huge batches at once.

> This is the answer to my Counter-Strike question. Rendering a game means recomputing millions of pixels 60+ times a second. My government laptop was doing that on a weak CPU with no real GPU — so it crawled. A gaming machine does it on thousands of GPU cores in parallel, and it's smooth.

The same trick that renders games fast is the trick that runs AI fast. Neural networks are just enormous stacks of these independent matrix operations, which is a GPU's dream workload.

There's a catch worth knowing: the GPU only wins when the problem is actually parallel. Hand it a small or sequential, decision-heavy job and the CPU wins — because sending work to the GPU has a setup cost. GPUs win on *big batches*, not small errands.

## Let's actually measure it

Enough theory. I ran a real benchmark: multiply two 8000×8000 matrices (that's about 128 billion multiply-add operations) on a CPU versus a GPU.

I used PyTorch, because the CPU and GPU versions are almost identical — the *only* thing that changes is the device name. That by itself makes the point: same math, same one-line call, different hardware.

```python
import time, torch

N = 8000  # two 8000 x 8000 matrices

def benchmark(device):
    a = torch.rand(N, N, device=device)
    b = torch.rand(N, N, device=device)

    # Warm-up: the first call has one-time setup, so don't time it.
    _ = a @ b
    if device == "cuda":
        torch.cuda.synchronize()

    start = time.perf_counter()
    _ = a @ b                       # the matrix multiply
    if device == "cuda":
        torch.cuda.synchronize()    # wait for the GPU to actually finish
    return time.perf_counter() - start

cpu = benchmark("cpu")
print(f"CPU:        {cpu:.3f} s")

if torch.cuda.is_available():
    print("GPU name:", torch.cuda.get_device_name(0))
    gpu = benchmark("cuda")
    print(f"GPU (CUDA): {gpu:.3f} s")
    print(f"Speedup:    {cpu/gpu:.1f}x")
```

Two small things in that code are easy to miss but important:

- **The warm-up call.** The first GPU operation pays a one-time setup cost. Run one throwaway multiply before timing, or your numbers are garbage.
- **`torch.cuda.synchronize()`.** The GPU runs *asynchronously* — Python fires off the work and moves on. Without synchronizing, you'd be timing how long it takes to *launch* the work, not *finish* it, and you'd get a fake, absurdly fast number.

I ran this on a free NVIDIA Tesla T4 (Google Colab — Runtime → Change runtime type → GPU, no credit card needed). The result:

```
CPU:        7.531 s
GPU name:   Tesla T4
GPU (CUDA): 0.223 s
Speedup:    33.8x
```

**33.8× faster.** Same math. One line of code different.

One honest caveat so you don't over-read that number: part of that 33.8× is a *weak* Colab CPU. On my own Mac (M4 Pro), the CPU did the same job in about 0.37s — roughly 20× faster than Colab's CPU. The point isn't the exact multiplier; it's that for big, parallel matrix math, the gap between CPU and GPU is enormous. Always compare a speedup to the baseline it was measured against.

## What's actually inside a GPU: CUDA cores vs Tensor cores

When I started reading GPU spec sheets, two terms kept showing up: **CUDA cores** and **Tensor cores**. They're not the same thing, and the difference matters a lot for AI.

**CUDA cores** are the general-purpose workers — the "thousands of students." They do ordinary math, one operation at a time: `a × b`. This is the big number in the marketing ("16,384 CUDA cores").

**Tensor cores** are specialists. Instead of one multiply at a time, a Tensor core does a *whole small matrix multiply-and-add in one shot*. Since that's literally the operation neural networks are made of, Tensor cores give huge speedups for AI.

> This is the single biggest reason NVIDIA dominates AI: they put dedicated matrix-multiply hardware on the chip.

A couple of things I got wrong at first, so you don't have to:

- **Not every GPU has Tensor cores.** They arrived in 2017. Older cards (like the GTX 1080) have thousands of CUDA cores but *zero* Tensor cores — they can still run AI, just slower.
- **"CUDA cores" and "Tensor cores" are NVIDIA-only names.** AMD calls them Stream Processors and Matrix Cores. Apple Silicon (my Mac) has neither — its AI acceleration lives in a separate Neural Engine.

And raw core count doesn't decide "speed" on its own. It's core count *and* clock speed *and* memory bandwidth *and* whether your workload actually fits the hardware. More on that bandwidth point below.

### VRAM: the other half of the story

Cores are useless if you can't feed them data fast enough. That's what **VRAM** (the GPU's own dedicated memory) is for — huge bandwidth to keep thousands of cores busy. It's also why, for AI, VRAM *size* is often the make-or-break spec: your model has to fit in it. A 24 GB card can run models an 8 GB card simply can't, no matter how fast its cores are.

(My Mac is the odd one out here — Apple Silicon uses *unified memory* shared between CPU and GPU, so there's no separate VRAM number at all.)

## The current NVIDIA lineup, roughly

To put the T4 I used in context, here's the rough landscape (approximate figures — exact specs vary by model and go stale fast):

| GPU | Type | VRAM | Tensor cores | Notes |
| --- | --- | --- | --- | --- |
| T4 | Data-center (older) | 16 GB | Yes | What I benchmarked on; everywhere in the cloud |
| RTX 4090 | Consumer flagship | 24 GB | Yes | Great value for home ML |
| A100 | Data-center AI | 40 / 80 GB | Yes | The AI workhorse |
| H100 | Data-center AI | 80 GB | Yes | Trained most modern LLMs |
| B200 | Data-center (newest) | ~192 GB | Yes | Built for giant models |

The detail I find most telling: the **A100 has fewer CUDA cores than an RTX 4090** (~6,912 vs ~16,384), yet it's a far better AI chip — because of its much larger VRAM, higher bandwidth, and AI-tuned Tensor cores. That's the whole "core count isn't everything" lesson in one row.

## So, back to the laggy laptop

My college laptops lagged on Counter-Strike for the same reason a CPU is slow at AI: they were trying to do a massively parallel job (millions of pixels, over and over) on hardware built for a few sequential tasks at a time.

A GPU flips that — thousands of simple cores doing the same operation on different data, all at once. Games figured this out decades ago. AI inherited the exact same hardware, because a neural network is, underneath it all, just a mountain of matrix multiplications.

Funny how a laggy game in a college lab turned out to be the same story as the AI boom. I just didn't have the words for it back then.
