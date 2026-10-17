---
title: "The need for GPUs: from laggy games to running AI"
author: "Suresh Rajendiran"
pubDatetime: 2026-10-17T09:00:00+05:30
tags:
  - gpu
  - ai
  - hardware
description: "Why does AI run on GPUs and not CPUs? It comes down to one operation — matrix multiplication — and the kind of hardware that's good at it. Matrix math, a real benchmark, and what's actually inside a GPU."
---

Back in college, a few of us would install Counter-Strike on the government-issued laptops we got and play during the free hours. The laptops were fine for notes and labs. For CS, they were miserable — jerky movement, frames dropping every time something exploded on screen. For years I just filed it under "cheap laptop."

It was only much later that I understood what those laptops were really missing: a real GPU. And the more I dug into *why* games need one, the more I realized the same idea quietly runs underneath a lot of computing today. This post is the explanation I wish someone had given me — what a GPU actually does, why it beats a CPU at certain jobs, and what's inside one.

## Table of contents

## It all comes down to one operation: matrix multiplication

Here's the thing that surprised me most: whether it's rendering a game or running ChatGPT, the computer is mostly doing the *same kind of math* underneath — **matrix multiplication**.

A matrix is just a grid of numbers. Nothing scary. And matrix multiplication is just a way to apply the same transformation to a whole batch of data at once.

Let me show it with something concrete instead of abstract math.

### Example: the "warm filter" on a photo

You know those warm, old-photo filters on Instagram — the ones that make everything look a bit golden and vintage? That's "sepia," and it's a great way to see what matrix multiplication actually is.

Start with how a computer sees color. Every pixel on your screen is just three numbers: how much **Red**, **Green**, and **Blue** it has, each from 0 (none) to 255 (full). Mix those three and you get any color. A pixel of `R:100, G:150, B:200`, for instance, is a soft blue.

A sepia filter is nothing more than a fixed **recipe** for remixing those three numbers into three new ones:

```
New Red   = 0.4 × old R  +  0.8 × old G  +  0.2 × old B
New Green = 0.3 × old R  +  0.7 × old G  +  0.2 × old B
New Blue  = 0.3 × old R  +  0.5 × old G  +  0.1 × old B
```

Read one line at a time. The new red is built by taking a little of the old red, a lot of the old green, a bit of the old blue, and adding them up. Same idea for green and blue, with different amounts. Those nine numbers *are* the filter — they're the "sepia matrix."

Let's run our soft-blue pixel through it:

```
New Red   = 0.4 × 100 + 0.8 × 150 + 0.2 × 200 = 200
New Green = 0.3 × 100 + 0.7 × 150 + 0.2 × 200 = 175
New Blue  = 0.3 × 100 + 0.5 × 150 + 0.1 × 200 = 125
```

So `R:100, G:150, B:200` (soft blue) becomes `R:200, G:175, B:125` (a warm tan). The filter nudged a cool color into a warm one — exactly what sepia does.

That's the whole operation for *one* pixel. Now here's the catch: a single photo has **millions** of pixels, and every one gets the same nine-number recipe. Applying one fixed recipe to a huge batch of inputs, all at once — that is matrix multiplication.

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

A quick honesty note: a real LLM is far more involved than `output = weights × input`. There's attention, many stacked layers, normalization, non-linear steps in between, and a lot more. But strip all that away and the heavy lifting — the part that eats the compute — is still matrix multiplication, over and over. That's the piece I'm zooming in on here; the rest is a story for another post.

### Why a matrix, if it's the same number of operations?

This one bugged me for a while. Writing it as a matrix doesn't reduce the math — it's the same multiplications and additions either way. So why bother?

Because the matrix *tells the hardware something*: "here is a big batch of operations, and none of them depend on each other." Every output number is computed independently. That independence is the whole game — it's what lets you do them all at once instead of one after another.

Which is exactly where the GPU comes in.

## Why a GPU beats a CPU at this

A CPU and a GPU are built on opposite philosophies, and the easiest way to feel the difference is to *watch* it. Picture our photo as a grid of pixels, each one waiting for the sepia recipe. The **CPU has a single worker (core)** that ticks through the pixels one at a time; the **GPU has many simple workers** that each grab a pixel and color them all at once:

<figure class="chip-demo">
  <div class="chip-demo__panels">
    <div class="chip-demo__panel">
      <div class="chip-demo__head">
        <h4>CPU — 1 powerful core</h4>
        <span class="chip-demo__badge chip-demo__badge--cpu">working…</span>
      </div>
      <div class="chip-demo__cores" aria-hidden="true">
        <span class="chip-demo__core chip-demo__core--cpu"></span>
      </div>
      <p class="chip-demo__sub">One worker, one pixel at a time.</p>
      <div class="chip-demo__grid chip-demo__grid--cpu" aria-hidden="true">
        <span></span><span></span><span></span><span></span><span></span><span></span>
        <span></span><span></span><span></span><span></span><span></span><span></span>
        <span></span><span></span><span></span><span></span><span></span><span></span>
        <span></span><span></span><span></span><span></span><span></span><span></span>
        <span></span><span></span><span></span><span></span><span></span><span></span>
        <span></span><span></span><span></span><span></span><span></span><span></span>
      </div>
    </div>
    <div class="chip-demo__panel">
      <div class="chip-demo__head">
        <h4>GPU — many simple cores</h4>
        <span class="chip-demo__badge chip-demo__badge--gpu">done ✓</span>
      </div>
      <div class="chip-demo__cores" aria-hidden="true">
        <span class="chip-demo__core chip-demo__core--gpu"></span>
        <span class="chip-demo__core chip-demo__core--gpu"></span>
        <span class="chip-demo__core chip-demo__core--gpu"></span>
        <span class="chip-demo__core chip-demo__core--gpu"></span>
        <span class="chip-demo__core chip-demo__core--gpu"></span>
        <span class="chip-demo__core chip-demo__core--gpu"></span>
      </div>
      <p class="chip-demo__sub">Many workers, all pixels at once.</p>
      <div class="chip-demo__grid chip-demo__grid--gpu" aria-hidden="true">
        <span></span><span></span><span></span><span></span><span></span><span></span>
        <span></span><span></span><span></span><span></span><span></span><span></span>
        <span></span><span></span><span></span><span></span><span></span><span></span>
        <span></span><span></span><span></span><span></span><span></span><span></span>
        <span></span><span></span><span></span><span></span><span></span><span></span>
        <span></span><span></span><span></span><span></span><span></span><span></span>
      </div>
    </div>
  </div>
  <figcaption>Each square is one pixel getting the sepia recipe. Same photo, same total work — the GPU's many cores finish it all together while the CPU's single core is still ticking through one pixel at a time.</figcaption>
</figure>

<style>
  .chip-demo { margin: 2rem 0; }
  .chip-demo__panels {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 1.5rem;
  }
  @media (max-width: 640px) {
    .chip-demo__panels { grid-template-columns: 1fr; }
  }
  .chip-demo__panel {
    border: 1px solid var(--border);
    border-radius: 0.75rem;
    padding: 1rem;
    background: color-mix(in srgb, var(--background) 92%, var(--accent));
  }
  .chip-demo__head {
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 0.5rem;
  }
  .chip-demo__head h4 { margin: 0; font-size: 1rem; }
  .chip-demo__badge {
    font-size: 0.7rem;
    font-weight: 600;
    padding: 0.1rem 0.5rem;
    border-radius: 999px;
    letter-spacing: 0.02em;
    white-space: nowrap;
  }
  .chip-demo__badge--cpu {
    color: var(--muted-foreground);
    background: color-mix(in srgb, var(--foreground) 10%, transparent);
    animation: chip-cpu-badge 6s infinite steps(1, end);
  }
  .chip-demo__badge--gpu {
    color: var(--accent-foreground);
    background: var(--accent);
    opacity: 0;
    animation: chip-gpu-badge 6s infinite steps(1, end);
  }

  /* the worker cores */
  .chip-demo__cores {
    display: flex;
    gap: 6px;
    margin-top: 0.75rem;
    min-height: 26px;
    align-items: center;
  }
  .chip-demo__core {
    border-radius: 5px;
    background: color-mix(in srgb, var(--foreground) 20%, transparent);
  }
  .chip-demo__core--cpu {
    width: 46px;
    height: 24px;
    animation: chip-core-pulse 0.5s infinite alternate ease-in-out;
  }
  .chip-demo__core--gpu {
    width: 22px;
    height: 24px;
    animation: chip-gpu-core 6s infinite steps(1, end);
  }

  .chip-demo__sub { margin: 0.6rem 0 0.9rem; font-size: 0.8rem; opacity: 0.75; }
  .chip-demo__grid {
    display: grid;
    grid-template-columns: repeat(6, 1fr);
    gap: 5px;
  }
  .chip-demo__grid span {
    --cell: color-mix(in srgb, var(--foreground) 12%, transparent);
    aspect-ratio: 1 / 1;
    border-radius: 4px;
    background: var(--cell);
  }

  /*
    One full loop is 6s.
    - GPU: all cells fill together near the start, hold, then clear at 100%.
    - CPU: cells fill one-by-one across the loop, every cell HOLDS its color,
      and they ALL clear together at the very end (100%) so the next loop
      starts from a completely blank grid.
  */

  .chip-demo__grid--gpu span { animation: chip-gpu-fill 6s infinite steps(1, end); }

  .chip-demo__grid--cpu span { animation-duration: 6s; animation-iteration-count: infinite; animation-timing-function: steps(1, end); }
  /* Each cell has its OWN keyframes: fill at its moment, hold, clear at 100%. */
  .chip-demo__grid--cpu span:nth-child(1)  { animation-name: cpu-c1; }
  .chip-demo__grid--cpu span:nth-child(2)  { animation-name: cpu-c2; }
  .chip-demo__grid--cpu span:nth-child(3)  { animation-name: cpu-c3; }
  .chip-demo__grid--cpu span:nth-child(4)  { animation-name: cpu-c4; }
  .chip-demo__grid--cpu span:nth-child(5)  { animation-name: cpu-c5; }
  .chip-demo__grid--cpu span:nth-child(6)  { animation-name: cpu-c6; }
  .chip-demo__grid--cpu span:nth-child(7)  { animation-name: cpu-c7; }
  .chip-demo__grid--cpu span:nth-child(8)  { animation-name: cpu-c8; }
  .chip-demo__grid--cpu span:nth-child(9)  { animation-name: cpu-c9; }
  .chip-demo__grid--cpu span:nth-child(10) { animation-name: cpu-c10; }
  .chip-demo__grid--cpu span:nth-child(11) { animation-name: cpu-c11; }
  .chip-demo__grid--cpu span:nth-child(12) { animation-name: cpu-c12; }
  .chip-demo__grid--cpu span:nth-child(13) { animation-name: cpu-c13; }
  .chip-demo__grid--cpu span:nth-child(14) { animation-name: cpu-c14; }
  .chip-demo__grid--cpu span:nth-child(15) { animation-name: cpu-c15; }
  .chip-demo__grid--cpu span:nth-child(16) { animation-name: cpu-c16; }
  .chip-demo__grid--cpu span:nth-child(17) { animation-name: cpu-c17; }
  .chip-demo__grid--cpu span:nth-child(18) { animation-name: cpu-c18; }
  .chip-demo__grid--cpu span:nth-child(19) { animation-name: cpu-c19; }
  .chip-demo__grid--cpu span:nth-child(20) { animation-name: cpu-c20; }
  .chip-demo__grid--cpu span:nth-child(21) { animation-name: cpu-c21; }
  .chip-demo__grid--cpu span:nth-child(22) { animation-name: cpu-c22; }
  .chip-demo__grid--cpu span:nth-child(23) { animation-name: cpu-c23; }
  .chip-demo__grid--cpu span:nth-child(24) { animation-name: cpu-c24; }
  .chip-demo__grid--cpu span:nth-child(25) { animation-name: cpu-c25; }
  .chip-demo__grid--cpu span:nth-child(26) { animation-name: cpu-c26; }
  .chip-demo__grid--cpu span:nth-child(27) { animation-name: cpu-c27; }
  .chip-demo__grid--cpu span:nth-child(28) { animation-name: cpu-c28; }
  .chip-demo__grid--cpu span:nth-child(29) { animation-name: cpu-c29; }
  .chip-demo__grid--cpu span:nth-child(30) { animation-name: cpu-c30; }
  .chip-demo__grid--cpu span:nth-child(31) { animation-name: cpu-c31; }
  .chip-demo__grid--cpu span:nth-child(32) { animation-name: cpu-c32; }
  .chip-demo__grid--cpu span:nth-child(33) { animation-name: cpu-c33; }
  .chip-demo__grid--cpu span:nth-child(34) { animation-name: cpu-c34; }
  .chip-demo__grid--cpu span:nth-child(35) { animation-name: cpu-c35; }
  .chip-demo__grid--cpu span:nth-child(36) { animation-name: cpu-c36; }

  /* fills over the first ~90% of the loop; everything clears at 98% so the
     grid is blank at 100% / 0% of the next loop. */
  @keyframes cpu-c1  { 0%,2.3%{background:var(--cell)} 2.5%,97%{background:var(--accent)} 98%,100%{background:var(--cell)} }
  @keyframes cpu-c2  { 0%,4.8%{background:var(--cell)} 5%,97%{background:var(--accent)} 98%,100%{background:var(--cell)} }
  @keyframes cpu-c3  { 0%,7.3%{background:var(--cell)} 7.5%,97%{background:var(--accent)} 98%,100%{background:var(--cell)} }
  @keyframes cpu-c4  { 0%,9.8%{background:var(--cell)} 10%,97%{background:var(--accent)} 98%,100%{background:var(--cell)} }
  @keyframes cpu-c5  { 0%,12.3%{background:var(--cell)} 12.5%,97%{background:var(--accent)} 98%,100%{background:var(--cell)} }
  @keyframes cpu-c6  { 0%,14.8%{background:var(--cell)} 15%,97%{background:var(--accent)} 98%,100%{background:var(--cell)} }
  @keyframes cpu-c7  { 0%,17.3%{background:var(--cell)} 17.5%,97%{background:var(--accent)} 98%,100%{background:var(--cell)} }
  @keyframes cpu-c8  { 0%,19.8%{background:var(--cell)} 20%,97%{background:var(--accent)} 98%,100%{background:var(--cell)} }
  @keyframes cpu-c9  { 0%,22.3%{background:var(--cell)} 22.5%,97%{background:var(--accent)} 98%,100%{background:var(--cell)} }
  @keyframes cpu-c10 { 0%,24.8%{background:var(--cell)} 25%,97%{background:var(--accent)} 98%,100%{background:var(--cell)} }
  @keyframes cpu-c11 { 0%,27.3%{background:var(--cell)} 27.5%,97%{background:var(--accent)} 98%,100%{background:var(--cell)} }
  @keyframes cpu-c12 { 0%,29.8%{background:var(--cell)} 30%,97%{background:var(--accent)} 98%,100%{background:var(--cell)} }
  @keyframes cpu-c13 { 0%,32.3%{background:var(--cell)} 32.5%,97%{background:var(--accent)} 98%,100%{background:var(--cell)} }
  @keyframes cpu-c14 { 0%,34.8%{background:var(--cell)} 35%,97%{background:var(--accent)} 98%,100%{background:var(--cell)} }
  @keyframes cpu-c15 { 0%,37.3%{background:var(--cell)} 37.5%,97%{background:var(--accent)} 98%,100%{background:var(--cell)} }
  @keyframes cpu-c16 { 0%,39.8%{background:var(--cell)} 40%,97%{background:var(--accent)} 98%,100%{background:var(--cell)} }
  @keyframes cpu-c17 { 0%,42.3%{background:var(--cell)} 42.5%,97%{background:var(--accent)} 98%,100%{background:var(--cell)} }
  @keyframes cpu-c18 { 0%,44.8%{background:var(--cell)} 45%,97%{background:var(--accent)} 98%,100%{background:var(--cell)} }
  @keyframes cpu-c19 { 0%,47.3%{background:var(--cell)} 47.5%,97%{background:var(--accent)} 98%,100%{background:var(--cell)} }
  @keyframes cpu-c20 { 0%,49.8%{background:var(--cell)} 50%,97%{background:var(--accent)} 98%,100%{background:var(--cell)} }
  @keyframes cpu-c21 { 0%,52.3%{background:var(--cell)} 52.5%,97%{background:var(--accent)} 98%,100%{background:var(--cell)} }
  @keyframes cpu-c22 { 0%,54.8%{background:var(--cell)} 55%,97%{background:var(--accent)} 98%,100%{background:var(--cell)} }
  @keyframes cpu-c23 { 0%,57.3%{background:var(--cell)} 57.5%,97%{background:var(--accent)} 98%,100%{background:var(--cell)} }
  @keyframes cpu-c24 { 0%,59.8%{background:var(--cell)} 60%,97%{background:var(--accent)} 98%,100%{background:var(--cell)} }
  @keyframes cpu-c25 { 0%,62.3%{background:var(--cell)} 62.5%,97%{background:var(--accent)} 98%,100%{background:var(--cell)} }
  @keyframes cpu-c26 { 0%,64.8%{background:var(--cell)} 65%,97%{background:var(--accent)} 98%,100%{background:var(--cell)} }
  @keyframes cpu-c27 { 0%,67.3%{background:var(--cell)} 67.5%,97%{background:var(--accent)} 98%,100%{background:var(--cell)} }
  @keyframes cpu-c28 { 0%,69.8%{background:var(--cell)} 70%,97%{background:var(--accent)} 98%,100%{background:var(--cell)} }
  @keyframes cpu-c29 { 0%,72.3%{background:var(--cell)} 72.5%,97%{background:var(--accent)} 98%,100%{background:var(--cell)} }
  @keyframes cpu-c30 { 0%,74.8%{background:var(--cell)} 75%,97%{background:var(--accent)} 98%,100%{background:var(--cell)} }
  @keyframes cpu-c31 { 0%,77.3%{background:var(--cell)} 77.5%,97%{background:var(--accent)} 98%,100%{background:var(--cell)} }
  @keyframes cpu-c32 { 0%,79.8%{background:var(--cell)} 80%,97%{background:var(--accent)} 98%,100%{background:var(--cell)} }
  @keyframes cpu-c33 { 0%,82.3%{background:var(--cell)} 82.5%,97%{background:var(--accent)} 98%,100%{background:var(--cell)} }
  @keyframes cpu-c34 { 0%,84.8%{background:var(--cell)} 85%,97%{background:var(--accent)} 98%,100%{background:var(--cell)} }
  @keyframes cpu-c35 { 0%,87.3%{background:var(--cell)} 87.5%,97%{background:var(--accent)} 98%,100%{background:var(--cell)} }
  @keyframes cpu-c36 { 0%,89.8%{background:var(--cell)} 90%,97%{background:var(--accent)} 98%,100%{background:var(--cell)} }

  @keyframes chip-gpu-fill {
    0%, 5%    { background: var(--cell); }
    8%, 97%   { background: var(--accent); }
    98%, 100% { background: var(--cell); }
  }
  @keyframes chip-gpu-core {
    0%, 5%    { background: color-mix(in srgb, var(--foreground) 20%, transparent); }
    8%, 97%   { background: var(--accent); }
    98%, 100% { background: color-mix(in srgb, var(--foreground) 20%, transparent); }
  }
  @keyframes chip-core-pulse {
    from { background: color-mix(in srgb, var(--foreground) 20%, transparent); }
    to   { background: var(--accent); }
  }
  /* GPU "done" shows up early; CPU badge stays "working" until near the end. */
  @keyframes chip-gpu-badge {
    0%, 7%    { opacity: 0; }
    8%, 97%   { opacity: 1; }
    98%, 100% { opacity: 0; }
  }
  @keyframes chip-cpu-badge {
    0%, 90%   { opacity: 1; }
    92%, 100% { opacity: 0.25; }
  }

  @media (prefers-reduced-motion: reduce) {
    .chip-demo__grid span,
    .chip-demo__core,
    .chip-demo__badge { animation: none; }
    .chip-demo__grid span { background: color-mix(in srgb, var(--accent) 45%, transparent); }
    .chip-demo__core { background: var(--accent); }
    .chip-demo__badge--gpu { opacity: 1; }
  }
</style>

That's the core idea: a **CPU** has a few very fast cores, so it ticks through the pixels a small group at a time. A **GPU** has thousands of simpler cores, so it colors the whole batch at once and is done while the CPU is still going. For work this repetitive and independent, "many simple hands all at once" wins easily.

> Rendering a game is the same shape of problem: recomputing millions of pixels 60+ times a second. Do that on a weak CPU with no real GPU and it crawls (hello, college laptop). Do it on thousands of GPU cores in parallel and it's smooth.

### But one isn't "better" than the other

It's tempting to walk away thinking the GPU is just the faster chip. It isn't — they're built for different jobs.

A GPU's cores are **simple**. They shine when you have the *same* operation to run over a huge pile of data (every pixel, every number in a matrix).

A CPU's few cores are **complex and clever**. They're built exactly for that branchy, sequential work: running your operating system, your browser's logic, a database query, the parts of a program that have to happen in order.

So the honest picture:

- **CPU** → great at *complex, sequential, decision-heavy* tasks.
- **GPU** → great at *simple, repetitive* tasks done across tons of data at once.

Neither wins outright. That's why every computer has both and hands each the kind of work it's good at. AI leans on the GPU because a neural network is exactly the GPU's happy case: the same matrix math, repeated across mountains of numbers.

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

(The full script is [here as a GitHub gist](https://gist.github.com/sureshr-dev/a24d45cdf82d8b6da33fef1dd3da8f16) — paste it straight into a Colab cell.)

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

One caveat: this exact number will differ depending on the CPU and GPU you run it on. The point isn't the precise multiplier — it's that for big, parallel matrix math, the gap between CPU and GPU is enormous.

## What's actually inside a GPU: CUDA cores vs Tensor cores

When I started reading GPU spec sheets, two terms kept showing up: **CUDA cores** and **Tensor cores**. They're not the same thing, and the difference matters a lot for AI.

**CUDA cores** are the general-purpose workers. They do ordinary math **one operation at a time**: `a × b`, then the next, then the next. This is the big number in the marketing ("16,384 CUDA cores").

**Tensor cores** are specialists. Instead of one multiply at a time, a Tensor core chews through a **whole 4×4 matrix multiply-and-add in a single step** — dozens of multiplies and adds at once, as one operation.

Why does that matter so much for AI? Remember that a neural network is, underneath it all, just stacks of matrix multiplications. A CUDA core would grind through such a matrix one `a × b` at a time; a Tensor core swallows a 4×4 block in one go. Since AI is *nothing but* these matrix blocks repeated billions of times, hardware that does a whole block per step is enormously faster — that's exactly what a Tensor core is built for.

> This is the single biggest reason NVIDIA dominates AI: they put dedicated matrix-multiply hardware on the chip.

A couple of things I got wrong at first, so you don't have to:

- **Not every GPU has Tensor cores.** They arrived in 2017. Older cards (like the GTX 1080) have thousands of CUDA cores but *zero* Tensor cores — they can still run AI, just slower.
- **"CUDA cores" and "Tensor cores" are NVIDIA-only names.** AMD calls them Stream Processors and Matrix Cores. Apple Silicon (my Mac) has neither — its AI acceleration lives in a separate Neural Engine.

And raw core count doesn't decide "speed" on its own. It's core count *and* clock speed *and* memory bandwidth *and* whether your workload actually fits the hardware. More on that bandwidth point below.

### VRAM: the other half of the story

Cores are useless if you can't feed them data fast enough. That's what **VRAM** (the GPU's own dedicated memory) is for — huge bandwidth to keep thousands of cores busy. It's also why, for AI, VRAM *size* is often the make-or-break spec: your model has to fit in it. A 24 GB card can run models an 8 GB card simply can't, no matter how fast its cores are.

Here's the rule of thumb that makes it concrete. A model's "size" is its number of **parameters** (those learned weights from earlier). Each parameter is a number that takes up memory — commonly 2 bytes in the half-precision formats used for AI. So roughly:

> **VRAM needed ≈ parameters × 2 bytes** (just to *load* the model — running it needs a bit more).

| Model size | Parameters | VRAM just to load (fp16) |
| --- | --- | --- |
| Small | 1 billion | ~2 GB |
| Mid | 7 billion | ~14 GB |
| Large | 70 billion | ~140 GB |
| Frontier | 400+ billion | ~800 GB+ |

Look at that 70-billion-parameter row: ~140 GB won't fit on *any* single GPU (an H100 has 80 GB). That's why big models must be **split across multiple GPUs** — and why frontier models, with hundreds of billions of parameters, need whole clusters just to hold the weights, let alone train them.

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

And to put the *scale* in perspective: a single H100 is a monster, but training a frontier model like Claude Opus or GPT-4 doesn't use one — it uses **tens of thousands of them clustered together**. Rough public estimates put these training runs around **~16,000–25,000 H100-class GPUs** working in parallel for weeks or months (the exact numbers aren't officially published, so treat these as ballpark). The hardware that renders a game and the hardware that trains an AI are the same idea — just scaled up by a factor that's hard to picture.

## Wrapping up

The need for a GPU comes down to one thing: some problems are massively parallel — millions of independent operations of the same shape — and a CPU's few powerful cores just aren't built for that. A GPU flips the design: thousands of simple cores doing the same operation on different data, all at once.

Games figured this out decades ago. AI inherited the exact same hardware, because a neural network is, underneath it all, just a mountain of matrix multiplications.

And that laggy Counter-Strike in a college lab? Same story as the AI boom, really. I just didn't have the words for it back then.
