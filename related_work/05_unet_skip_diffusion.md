# Related Work: U-Net Skip Connections in Diffusion Models

## Summary

The U-Net used by virtually all pixel- and latent-space diffusion models is not a
symmetric stack: it has *skip connections* that route encoder activations
directly into the decoder at every resolution, bypassing the bottleneck `h`.
Editing-time interventions on diffusion models (Asyrp, BoundaryDiffusion, h-space
directions, etc.) traditionally treat `h` as the "semantic latent" and modify it
in isolation. A growing body of follow-up work shows that this picture is
incomplete: the skip path carries most of the per-pixel signal during a single
denoising step, so a perturbation `Δh` applied only at the bottleneck is
*partially erased* by the unmodified encoder features that the decoder
concatenates back in. Our finding — that supervised `Δh_smile` is a near-perfect
linear classifier in `h`-space (d' = 4.23) but moves the decoded image by only
1.25 mean-abs pixel at `t = 500` against an 11.10 noise floor — is a direct,
quantitative measurement of this asymmetry. Multi-step Asyrp-style injection
recovers semantic control precisely because each denoised `x_{t-1}` re-enters
the encoder, so the modification is re-encoded into the skip features at the
next step.

## Key papers

1. **Kwon, Jeong, Uh — "Diffusion Models already have a Semantic Latent Space"
   (Asyrp, ICLR 2023, arXiv:2210.10960).** Defines `h`-space as the deepest
   U-Net bottleneck and claims it is the natural semantic latent of frozen
   DPMs (homogeneity, linearity, robustness, consistency). Critically, the
   *method* applies `Δh_t` at **every** reverse-process step via the
   asymmetric reverse process — the paper does not isolate the single-step
   contribution of `Δh`, which is the regime our experiment probes.

2. **Si, Huang, Jiang, Liu — "FreeU: Free Lunch in Diffusion U-Net" (CVPR
   2024, arXiv:2309.11497).** Fourier-domain analysis of U-Net feature maps
   shows the backbone path carries low-frequency / semantic content while the
   *skip path injects most of the high-frequency content* into the decoder.
   The authors argue the skip path is so dominant that it "causes the network
   to overlook the backbone semantics," and propose two scalar rescalings
   `(b, s)` that boost the backbone and (in the Fourier sense) attenuate
   the skip — improving sample quality without retraining. This is the
   closest existing paper to our finding: it qualitatively reports the same
   skip/backbone imbalance we measure quantitatively.

3. **Jiang et al. — "SCEdit: Efficient and Controllable Image Diffusion
   Generation via Skip Connection Editing" (CVPR 2024 Highlight,
   arXiv:2312.11392).** Inserts trainable SC-Tuner modules *on the skip
   connections* rather than the backbone, and shows that editing the skips
   alone is sufficient for high-quality conditional generation (edge / depth /
   pose / segmentation control), at 7.9% of ControlNet's parameter count.
   Concrete evidence that the skip path is a load-bearing control surface.

4. **Jeong, Kwon, Uh — "Training-free Content Injection using h-space in
   Diffusion Models" (InjectFusion, WACV 2024, arXiv:2303.15403).** The same
   group as Asyrp explicitly acknowledges the bottleneck-skip coupling:
   "if one directly changes the bottleneck only, it distorts the relation
   between the skip connection and the bottleneck." They introduce a
   **skip-calibration** step that re-normalizes the bottleneck so its
   statistics stay consistent with the un-edited skips. This is an implicit
   admission that single-shot `Δh` is insufficient.

5. **Cipriano et al. — "Training-Free Style and Content Transfer by
   Leveraging U-Net Skip Connections in Stable Diffusion" (SkipInject,
   arXiv:2501.14524).** Probes each skip group individually and finds that
   `h`-space has "an almost imperceptible effect on the final image,"
   that the innermost skips (closest to `h`) have limited effect, and that
   *mid-resolution skips (l = 4, 5) carry the content signal*. Mirrors our
   single-step `Δh` ineffectiveness result and identifies *which* skip
   groups matter.

6. **Park, Kwon, Choi, Jo, Uh — "Understanding the Latent Space of Diffusion
   Models through the Lens of Riemannian Geometry" (NeurIPS 2023,
   arXiv:2307.12868).** Computes the pullback metric of the U-Net encoder
   at `h` and finds the *local* latent basis. Provides a geometric
   explanation of why `h`-space is "info-rich" (the encoder Jacobian has a
   well-defined top-singular subspace) without claiming it controls the
   final image — consistent with our d' = 4.23 vs. SNR = 0.11 split.

7. **Zhu, Wu, Mihalcea, Pan — "Boundary Guided Learning-Free Semantic
   Control with Diffusion Models" (BoundaryDiffusion, NeurIPS 2023,
   arXiv:2302.08357).** Fits SVMs in `h`-space and forces the denoising
   trajectory to cross the SVM hyperplane at a critical mixing step. Like
   Asyrp, it works by applying the shift across *multiple* steps, again
   sidestepping single-step skip dominance.

8. **Lu et al. — "Hierarchical Diffusion Autoencoders and Disentangled
   Image Manipulation" (HDAE, WACV 2024, arXiv:2304.11829).** Extends the
   diff-AE bottleneck `z_sem` (Preechakul et al., CVPR 2022,
   arXiv:2111.15640) into a *hierarchical* latent — one code per resolution
   — explicitly because a single bottleneck code "fails to reflect the rich
   information of details." Architectural analogue of FreeU's frequency
   argument.

9. **Choi et al. — "Perception Prioritized Training of Diffusion Models"
   (P2-weighting, CVPR 2022, arXiv:2204.00227).** Identifies that
   mid-noise timesteps (the "content stage") are where rich visual
   concepts are learned. Relevant because our `t = 500` injection point
   sits in this stage, where skips already encode coarse structure that
   the decoder will mostly preserve regardless of `Δh`.

## How our work fits in

**Connection to FreeU (arXiv:2309.11497).** FreeU *qualitatively* observed
the same skip/backbone imbalance we now measure directly: skips dominate the
decoder. FreeU's response was a generation-quality patch — rescale the two
paths. Our contribution is the editing-side analogue: we show the imbalance
is so severe that a class-separating `Δh` (d' = 4.23, perfect linear
classifier in `h`-space) produces a sub-noise-floor pixel change (1.25 vs
11.10) when injected for a single step. FreeU's "skips overpower main path"
narrative and our "single-step `Δh` is locked out by skips" measurement are
two views of the same architectural fact.

**Connection to Asyrp (arXiv:2210.10960).** Asyrp claims `h`-space *is* the
semantic latent space. Our Phase 10 result is compatible with the *first
half* of that claim — `h`-space is genuinely information-rich (d' = 4.23,
near-perfect linear smile classifier) — while pinning down *why* Asyrp must
inject `Δh_t` at every step rather than once: each denoised `x_{t-1}` is
re-encoded, so skips at the next step finally "see" the modification.
Single-step h-only injection does not survive the skip lock; Asyrp's
multi-step protocol is therefore not optional, it is the mechanism by which
`h`-edits become visible. InjectFusion (arXiv:2303.15403) is the closest
prior acknowledgement of this, but framed it as a "calibration" issue
rather than a structural property of single-step editing.

## Open questions raised by our finding

- **Hybrid skip-aware injection.** Would `Δskip(t) + Δh(t)` — derived
  jointly from supervised class differences at every U-Net resolution —
  produce stronger single-step edits than `Δh` alone? SCEdit
  (arXiv:2312.11392) and SkipInject (arXiv:2501.14524) suggest yes, but
  neither uses a *supervised concept direction* in the skip space.
- **Per-resolution skip importance.** SkipInject already isolates the
  l = 4, 5 skips for content; our framework could measure d' per skip
  resolution and rank which skips are *necessary* for a given concept
  (e.g. smile may live in a different resolution than gender or age).
- **Frequency-resolved editing.** Combining FreeU's Fourier decomposition
  of the skip path with concept-direction supervision could give a
  frequency-band-targeted edit, possibly avoiding the global
  over-smoothing FreeU itself flags.
- **Where, in time, does the skip lock break?** Our 1.25-pixel ceiling at
  `t = 500` suggests the lock is strongest in the content stage of
  P2-weighting (arXiv:2204.00227). Sweeping `t` would reveal whether
  late-step `Δh` injection — when skips carry only fine detail — survives
  better.
