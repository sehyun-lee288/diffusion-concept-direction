# Related Work: Diffusion h-space / Bottleneck Concept Direction

## Summary

A line of work starting with Kwon et al.'s "Asyrp" (ICLR 2023) established the U-Net mid-block bottleneck of a frozen DDPM — the **h-space** — as a semantic latent space in which attribute directions can be discovered, added, and decoded into edited images. Subsequent papers extended Asyrp with unsupervised direction discovery (PCA, Riemannian / Jacobian spectral analysis, shift modules), with disentanglement via linear projection / orthogonal complement, and with multi-attribute / regional / training-free variants. Most h-space methods crucially apply the Δh **at every reverse step** (or to both the predicted noise and the predicted x0 term) rather than at a single timestep, an observation that is consistent with — and explains — our finding that single-step injection at t=500 is overwhelmed by encoder skip features.

## Key papers

### 1 — Diffusion Models Already Have a Semantic Latent Space (Kwon, Jeong, Uh; ICLR 2023; arXiv:2210.10960)
- **Idea**: A frozen pretrained DDPM/DDIM already exposes a semantic latent space — the bottleneck activation of the U-Net, called *h-space* — with homogeneity, linearity, robustness, and cross-timestep consistency.
- **Method**: An **asymmetric reverse process** ("Asyrp") learns a small network Δh_t that perturbs the predicted-x0 branch only, applied at every reverse step in a chosen editing interval; the noise branch is left untouched. Evaluated on CelebA-HQ, AFHQ-dog, LSUN-{church,bedroom}, METFACES across DDPM++, iDDPM, and ADM backbones.
- **Relation to us**: Direct foundation. Matches our finding (2) that **multi-step injection works** and is what they actually use; their asymmetric trick (inject only in the x0 branch) is one mechanism by which their edits survive the skip pathway. They never measure single-step failure as we do in finding (1).
- **What we could borrow**: The asymmetric (P-only) injection as an alternative to symmetric Δh that may further reduce skip dominance; their editing-interval / quality-deficiency formulation to choose timesteps.

### 2 — Discovering Interpretable Directions in the Semantic Latent Space of Diffusion Models (Haas, Huberman-Spiegelglas, Mulayoff, Graßhof, Brandt, Michaeli; FG 2024; arXiv:2303.11073)
- **Idea**: Treat h-space as h_{T:1}, the *concatenation of bottleneck activations across all timesteps*, and discover global semantic directions in it via PCA, supervised classifiers, and per-image Jacobian spectral analysis.
- **Method**: Incremental PCA over many sampled trajectories; supervised SVM directions from classifier-labeled samples; image-specific directions from the right singular vectors of the denoiser Jacobian. Disentanglement via the **orthogonal-complement projection** v = [I − V(V^T V)^−1 V^T] v_0. Δh is injected into both the P (predicted x0) and D (direction toward x_t) terms in **one forward pass per step**. Backbone: DDPM trained on CelebA-HQ (HuggingFace), exactly our setup.
- **Relation to us**: Closest sibling paper. Matches findings (3, 4): mean-shift / supervised directions on CelebA-HQ DDPM exhibit attribute entanglement and are cleaned by linear projection. Extends our finding (4) — instead of per-timestep Gram–Schmidt they use a single global projector across timesteps.
- **What we could borrow**: Their PCA-on-h baseline; the symmetric-injection (P+D) variant; their orthogonal-complement formula as the reference disentanglement protocol against which we should benchmark our per-timestep Gram–Schmidt and our channel-loading method.

### 3 — Understanding the Latent Space of Diffusion Models through the Lens of Riemannian Geometry (Park, Kwon, Choi, Jo, Uh; NeurIPS 2023; arXiv:2307.12868)
- **Idea**: Use the **pullback metric** of the encoder feature map to define a Riemannian geometry on x_t-space and derive a local latent basis whose principal vectors are interpretable editing directions.
- **Method**: At each x_t, compute the Jacobian of the encoder feature toward x_t; its right-singular vectors form a local basis. They edit by traversing along these vectors. They study how the geometry evolves with t and with conditioning.
- **Relation to us**: Complements our channel-loading view (finding 5). Where we classify *channels* of h as smile-pure / gender-pure, they classify *directions* of x_t by curvature/singular-value structure. Both are linear-algebraic decompositions of the same denoiser, viewed at different layers.
- **What we could borrow**: Pullback-metric SVD on the mid_block input as a principled alternative to mean-shift Δh; their finding that earlier timesteps edit coarse attributes and later timesteps edit details supports treating our channel partition as t-dependent.

### 4 — Unsupervised Discovery of Semantic Latent Directions in Diffusion Models (Park, Kwon, Jo, Uh; arXiv:2302.12469, 2023)
- **Idea**: Direct precursor to the Riemannian paper above — discover disentangled, globally-consistent editing directions in a frozen diffusion model without labels.
- **Method**: Use Riemannian-geometry analysis of the latent–feature map to extract directions; finds early-t directions are coarse and late-t are fine-grained.
- **Relation to us**: Provides the unsupervised baseline our supervised mean-shift Δh should be compared against; the timestep-coarseness finding is consistent with our choice of t=500 for global attribute (smile/gender) edits.
- **What we could borrow**: Their argument that disentanglement comes "for free" from the Riemannian basis — useful as a control to test whether our channel-loading partition really *adds* disentanglement on top of geometry-based directions.

### 5 — Unsupervised Discovery of Interpretable Directions in h-space of Pre-trained Diffusion Models (Zhang, Liu, Lin, Zhu, Zhao; arXiv:2310.09912, 2023)
- **Idea**: Learn a shift-control module that perturbs h-space in disentangled directions, supervised only by a reconstructor that must recover the shift type and magnitude (a diffusion analogue of GANSpace/Voynov-Babenko).
- **Method**: Shift module + reconstructor + discriminator, trained end-to-end through the full reverse process with gradient checkpointing for VRAM.
- **Relation to us**: An alternative to our supervised N=20+20 mean-shift; sidesteps the need for attribute labels but at the cost of training a module.
- **What we could borrow**: Could serve as an unsupervised competitor on the same CelebA-HQ DDPM; useful sanity check for whether mean-shift Δh from 20+20 already captures what unsupervised methods find.

### 6 — Boundary Guided Learning-Free Semantic Control with Diffusion Models (Zhu, Wu, Deng, Russakovsky, Yan; NeurIPS 2023; arXiv:2302.08357)
- **Idea**: Even in *unconditional* DDPMs there exist semantic subspace boundaries in the intermediate latent trajectory; you can guide denoising across these boundaries without any extra training.
- **Method**: Identify boundaries in intermediate latent spaces (not specifically h-space) and steer the trajectory across them. Evaluated on CelebA, CelebA-HQ, LSUN-church, LSUN-bedroom, AFHQ-dog.
- **Relation to us**: Operates at a different layer than h-space but shares the "frozen DDPM, no training" philosophy. Provides a competing inductive bias: linear half-space boundaries rather than additive Δh.
- **What we could borrow**: SVM/boundary-fit Δh as an alternative to mean-shift — this exactly parallels InterFaceGAN-style hyperplane attribute directions and would give us a per-channel hyperplane that connects naturally to our channel-loading classifier.

### 7 — Training-free Content Injection using h-space in Diffusion Models (Jeong, Kwon, Uh; arXiv:2303.15403, 2023)
- **Idea**: Use h-space for *content* injection rather than attribute editing — blend bottleneck features from one generation into another and **calibrate the skip connections** so the injected content survives.
- **Method**: Gradually blend bottleneck features of the two trajectories with normalization, then explicitly modify skip connections to keep the injected semantics coherent in the decoder output.
- **Relation to us**: Highly relevant to our finding (1). They essentially *acknowledge* that bottleneck-only manipulation is partially overridden by skips, and their fix is to calibrate skip connections explicitly. This is independent evidence for our **skip-dominance** observation.
- **What we could borrow**: Their skip-calibration trick as a way to make single-step Δh injection actually decode — directly addressing the failure mode in finding (1).

### 8 — Navigating h-Space for Multi-Attribute Editing in Diffusion Models (ICASSP 2024 / IEEE 10920714)
- **Idea**: Supervised multi-attribute editing in h-space (aging, gender, eyeglasses) with simultaneous control of several attributes.
- **Method**: Learns interpretable per-attribute directions in h-space and combines them under a multi-attribute objective so each edit minimally affects others.
- **Relation to us**: Directly targets attribute entanglement (our findings 3–4) but in the multi-attribute regime. Their approach is supervised and learns directions; ours uses simple mean-shift with post-hoc orthogonalization. (Note: full text was not accessible, so the methodological match is based on the abstract only.)
- **What we could borrow**: Their multi-attribute joint-edit benchmark protocol — a natural extension of our smile/gender 2-attribute setup to gender × smile × age × eyeglasses.

### 9 — Concept Sliders: LoRA Adaptors for Precise Control in Diffusion Models (Gandikota, Materzynska, Zhou, Torralba, Bau; arXiv:2311.12092)
- **Idea**: Continuous attribute control by learning a small LoRA adapter per concept that minimizes interference with other attributes.
- **Method**: Operates in **weight space** (not h-space) via LoRA on Stable-Diffusion-XL; supervised by either prompt pairs or image pairs. Disentanglement comes from an interference-minimizing loss.
- **Relation to us**: Same goal (controllable, disentangled attribute editing) at the opposite end of the design space — model weights rather than activation bottleneck. Useful as a strong baseline for "fidelity vs disentanglement" tradeoff.
- **What we could borrow**: Their interference loss formulation; their evaluation protocol (target-attribute Δ vs unintended-attribute Δ).

### 10 — Post-Hoc Concept Disentanglement: From Correlated to Isolated Concept Representations (Erogullari, Lapuschkin, Samek, Pahde; arXiv:2503.05522)
- **Idea**: General framework for disentangling correlated concept directions post-hoc via a non-orthogonality loss.
- **Method**: Optimize directions under a soft orthogonality penalty; evaluate on CelebA (beard ↔ necktie) and FunnyBirds. Applied to activation steering, including in generative models.
- **Relation to us**: Provides a principled alternative to our hard Gram–Schmidt (finding 4) — a *soft* orthogonality penalty may preserve more attribute-relevant variance than projecting it out entirely.
- **What we could borrow**: The soft non-orthogonality loss as an ablation against our hard per-timestep Gram–Schmidt.

## How our work fits in

**Established and replicated.** The existence of an editable bottleneck h-space in frozen CelebA-HQ DDPMs is firmly established (Asyrp, Haas et al.). Linear directions discovered by supervised mean-shift or PCA produce visually plausible attribute edits **when applied across the full reverse trajectory**. Linear-projection / orthogonal-complement disentanglement is the standard fix for attribute entanglement (Haas et al., InterFaceGAN heritage, Post-Hoc Concept Disentanglement). Our findings (2) and (3)–(4) replicate this prior knowledge on the same `google/ddpm-celebahq-256` backbone Haas et al. used.

**What is underexplored.** Three gaps are visible in the literature:
(i) **Skip dominance is mentioned but rarely measured.** Jeong et al. (2023) implicitly recognize it by adding skip calibration to their content-injection method, and Asyrp's asymmetric P-only injection is partly a workaround, but no paper we found explicitly quantifies "Δh shifts h-space classifier by d'=4.23 yet leaves the decoded image unchanged at one step" the way our finding (1) does.
(ii) **Channel-level structure of h.** Existing work treats h as a single vector and applies global linear operations (PCA, SVM, projection). The *per-channel attribute loading* perspective in our finding (5) — partitioning the 512 mid_block channels into smile-pure / gender-pure / joint / weak buckets — does not appear in the surveyed papers. The closest is Park et al.'s Riemannian basis, but that operates in x_t-space and gives per-image directions rather than a global channel taxonomy.
(iii) **Selective per-channel injection** as a disentanglement mechanism (our finding 6) is, to our knowledge, novel; prior disentanglement relies on global linear projection in the full 512-dim space.

**Our specific contribution.** We add an empirical falsification of naive single-step h-injection (the skip-dominance measurement, finding 1), a *channel-axis* disentanglement view (findings 5, 6) that is complementary to the direction-axis view of Haas et al. and Park et al., and a side-by-side comparison of per-timestep Gram–Schmidt (finding 4) against the standard global orthogonal-complement projection used in the literature.

## Open questions inherited

- Why does h-space classification (d'=4.23) decouple so strongly from h-space *decodability* at a single timestep? No prior paper measures this gap directly.
- Is the mid_block channel basis special, or do the same attribute partitions exist in up/down-block bottlenecks? Haas et al. concatenate across timesteps but not across blocks.
- How does the skip-calibration trick of Jeong et al. (arXiv:2303.15403) interact with channel-selective injection — can we get single-step editing to work by combining the two?
- The Riemannian basis (Park et al., 2302.12469 / 2307.12868) is per-image; is it consistent with our *global*, attribute-labeled channel loading, or do they disagree about which directions matter?
- Do soft orthogonality (Erogullari et al.) and hard per-timestep Gram–Schmidt give different identity-preservation / attribute-fidelity tradeoffs on the same CelebA-HQ DDPM?
- All h-space papers we found benchmark *image quality* and *attribute success*; almost none benchmark **classifier-space d' inside h itself**, which our pipeline produces — is this a missing standard metric for the field?
