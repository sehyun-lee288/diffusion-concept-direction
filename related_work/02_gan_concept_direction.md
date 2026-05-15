# Related Work 02 — GAN Concept / Latent Direction Discovery

## Summary

Our diffusion-side methods (mean-shift Δh, per-timestep Gram–Schmidt of one
attribute against another, attribute-paired 2D planes, per-channel loading
scatters (A_c, B_c), and selective single-channel injection) are direct
methodological descendants of a five-year literature on **finding linear
concept directions inside pretrained GANs**. That lineage splits into three
families that map almost one-to-one onto our pipeline:

1. **Supervised linear-direction discovery** — InterFaceGAN (mean-shift /
   SVM normal), which we adapt to h-space at each timestep.
2. **Unsupervised global decomposition** — GANSpace (PCA in W), SeFa
   (eigendecomposition of the first projection matrix), Voynov & Babenko
   (reconstructor-trained directions), Hessian Penalty (axis-aligned
   regularization), LatentCLR (contrastive). Our orthogonalization step
   and 2D plane analysis sit between this family and the supervised one.
3. **Channel-level analysis** — StyleSpace (Wu et al.) and StyleCLIP global
   directions, which are the closest GAN analogues of our Phase 15–16
   per-channel scatters and single-channel injection in the DDPM
   bottleneck.

The high-level transfer from GAN to diffusion is mostly clean (linear
direction in a feature space, applied additively), but **two things change
structurally**: (i) GAN edits are single-pass while ours run over T denoising
steps, so a "direction" is really a *trajectory* of vectors; (ii) GANs have a
dedicated, low-dimensional, axis-aligned style space S, whereas the DDPM
bottleneck h has no AdaIN-style channel modulation, which makes the
single-channel result in our Phase 15 a non-trivial empirical claim rather
than a built-in property.

## Key papers

### 1. Shen et al., InterFaceGAN (arXiv:1907.10786, CVPR 2020)
- **Idea.** For any binary face attribute, a single hyperplane in the GAN
  latent space cleanly separates positive/negative samples.
- **Method.** Generate ~500K samples, score with a pretrained attribute
  classifier, fit a linear SVM per attribute. The hyperplane normal n is
  the edit direction. For entangled attributes, project one direction
  against another: **n₁ − (n₁ᵀ n₂) n₂**. They prefer W space over Z.
- **Relation to us.** Our mean-shift Δh = mean(h | attr=1) − mean(h | attr=0)
  is the closed-form "center-of-mass" cousin of the SVM normal (under
  Gaussian-class assumptions they coincide up to a covariance whitening
  factor). Our per-timestep Gram–Schmidt step is exactly InterFaceGAN's
  conditional manipulation formula, applied independently at each t.
- **What we could borrow.** Replace mean-shift by an SVM normal on
  Δh-projected features as a robustness check; also adopt their >95%
  classifier-validation accuracy protocol as a sanity gate before injection.

### 2. Härkönen et al., GANSpace (arXiv:2004.02546, NeurIPS 2020)
- **Idea.** Important directions in W (StyleGAN) or early-layer features
  (BigGAN) can be recovered unsupervised by PCA.
- **Method.** Sample N latents, compute PCA of their W codes; the top
  components yield interpretable axes (viewpoint, lighting, age). Apply
  edits only at chosen layer ranges for spatial localization.
- **Relation to us.** Our attribute-paired 2D plane (two Δh axes spanning
  a 2-D subspace) is a *supervised* analogue of GANSpace's top-2 PCA
  basis. Their layer-wise application is the GAN analogue of our
  per-timestep restriction.
- **What we could borrow.** Run PCA on the cross-timestep covariance of
  h to obtain an unsupervised baseline; compare its top components to
  our supervised Δh directions to test whether attribute axes are also
  the principal axes of natural variation in h-space.

### 3. Wu, Lischinski, Shechtman, StyleSpace Analysis (arXiv:2011.12799, CVPR 2021)
- **Idea.** StyleGAN2's per-channel style space S (9,088 channels) is far
  more disentangled than W or W+, and many attributes are controlled by
  *one* channel.
- **Method.** Compute attribute-conditioned channel statistics; rank
  channels by a normalized difference score; verify with an Attribute
  Dependency metric. Specific findings: gender = layer 9 / ch 6, smile =
  layer 6 / ch 501, lipstick = layer 15 / ch 45, gray hair = layer 11 /
  ch 286. Broader attributes (glasses, hairstyle) need a small set.
- **Relation to us.** This is the most direct ancestor of our Phase 15–16.
  Our (A_c, B_c) per-channel loading scatter and the discovery of channels
  110 (smile-pure) and 173 (gender-pure) are a diffusion-bottleneck
  reenactment of StyleSpace.
- **What we could borrow.** Their Attribute Dependency metric — measuring
  how much *other* classifier scores move when you inject one channel —
  would be a clean, quantitative replacement for our current visual
  inspection of single-channel injections.

### 4. Shen & Zhou, SeFa (arXiv:2007.06600, CVPR 2021)
- **Idea.** Semantic directions are baked into generator *weights*, not
  data — extract them in closed form.
- **Method.** Eigendecompose AᵀA, where A is the first projection matrix
  in the generator (y = Ax + b). The top eigenvectors of A are the
  semantic directions; no images or labels needed.
- **Relation to us.** No direct analogue in our current pipeline: the
  DDPM bottleneck is not produced by a single projection from a code,
  so SeFa-style weight factorization does not transfer literally.
- **What we could borrow.** Apply SeFa to the convolution kernels that
  produce h at each timestep — eigenvectors of those weight matrices
  could give an *unsupervised, label-free* candidate set of h-space
  directions to compare against Δh.

### 5. Voynov & Babenko, Unsupervised Discovery of Interpretable Directions (arXiv:2002.03754, ICML 2020)
- **Idea.** Learn K direction vectors jointly with a reconstructor that,
  given before/after images, predicts which direction was used and by
  how much.
- **Method.** Directions are parameters of a linear map; the reconstructor
  is a small CNN; train with cross-entropy on direction index plus
  regression on shift magnitude.
- **Relation to us.** Provides a label-free way to discover directions
  if we ever want to scale beyond CelebA's ~40 labeled attributes.
- **What we could borrow.** Plug the same reconstructor objective into
  h-space at a fixed t to mine *new* concept axes (beyond classifier
  labels) and then test whether they too concentrate on a few channels.

### 6. Peebles et al., The Hessian Penalty (arXiv:2008.10599, ECCV 2020)
- **Idea.** Penalize off-diagonal entries of the generator's input
  Hessian to push generators toward axis-aligned disentanglement.
- **Method.** A Hutchinson stochastic estimator of off-diagonal Hessian
  energy added to the training loss.
- **Relation to us.** Predicts that a *trained-for-disentanglement* model
  should have approximately single-channel attribute codes. Our DDPM was
  not trained with this prior, so the fact that we still find pure
  single-channel attributes (110, 173) is the empirically interesting
  observation.

### 7. Abdal et al., StyleFlow (arXiv:2008.02401, TOG 2021)
- **Idea.** Use conditional continuous normalizing flows in W+ to do
  attribute-conditioned sampling and editing.
- **Method.** Train a CNF p(W+ | attributes); invert the flow to edit.
- **Relation to us.** Suggests directions are nonlinear in W+; for h-space
  this means our linear Δh may be a first-order approximation that could
  be refined per-timestep by a small flow conditioned on (t, attribute).

### 8. Patashnik et al., StyleCLIP (arXiv:2103.17249, ICCV 2021)
- **Idea.** Use CLIP as the supervision signal to find StyleSpace
  directions for arbitrary text prompts.
- **Method.** Three variants — latent optimization, latent mapper, and a
  global StyleSpace direction obtained from prompt-difference + channel
  ranking.
- **Relation to us.** The "global direction" variant is essentially Δh
  with CLIP-derived pseudo-labels. Drop-in candidate for extending our
  pipeline to text-defined attributes without retraining a classifier.

## How our work fits in — GAN → diffusion transfer

**What translated cleanly.**
- *Mean-shift / SVM hyperplane (InterFaceGAN).* Linear separability of
  positive vs negative attribute samples holds in our h-space at every
  timestep we inspected, validating the most basic assumption of the
  GAN literature.
- *Conditional / orthogonal manipulation.* The Gram–Schmidt projection
  n₁ − (n₁ᵀ n₂) n₂ continues to work, but **only when applied
  per-timestep**: a single global orthogonalization (using Δh averaged
  over t) leaks the conditioning attribute back in at the timesteps where
  the two directions are most correlated. This is a real diffusion-specific
  finding — the angle between Δh_smile(t) and Δh_gender(t) is not constant
  in t.
- *Single-channel manipulation (StyleSpace).* Our Phase 15 result that
  channels 110 and 173 carry smile-pure and gender-pure signal respectively
  is a strong analogue of Wu et al.'s "one channel per attribute" claim.
  Channel 110 / 173 in the DDPM mid-block plays the same role as their
  layer-6 / ch 501 (smile) and layer-9 / ch 6 (gender) in StyleGAN2.

**What did not translate cleanly.**
- *Single-pass vs T-step injection.* A GAN edit is one matrix-vector add;
  ours is T such adds with the U-Net re-entangling channels at every
  layer between bottleneck applications. A direction that is "pure" at
  the bottleneck can be partially undone by later layers, so per-channel
  purity at h does not automatically imply per-channel purity in the
  output. Our (A_c, B_c) scatter is therefore the *necessary* condition,
  and StyleSpace-style attribute-dependency metrics on the final image
  are the still-needed *sufficient* one.
- *Closed-form factorization (SeFa).* The DDPM bottleneck is not a
  single projection from a code, so SeFa-style weight decomposition
  does not give a direct decomposition of h. PCA on collected h vectors
  is closer in spirit, but unsupervised in a different sense.
- *Style-space inductive bias.* GANs come with an AdaIN-modulated S
  space that is *built* to be channel-axis-aligned. The DDPM has no
  such inductive bias, so the channel-level purity we observe is a
  property of the *learned* feature, not the architecture.

**Specific answers to the project questions.**
- *Does StyleSpace's "single channels carry attribute meaning" hold for
  us?* Yes, in the same qualitative sense — Phase 15 isolated channel 110
  as smile-pure and channel 173 as gender-pure with the (A_c, B_c)
  scatter cleanly off-axis on one coordinate. The scale is consistent
  with Wu et al.: roughly one dominant channel for clean attributes,
  a small set for broader ones.
- *Does InterFaceGAN orthogonalization work in our per-timestep setup?*
  Partially. The formula is correct, but the orthogonalization must be
  applied per t (or equivalently per timestep slice of the Δh
  trajectory). A single global orthogonalization leaves residual
  conditioning leakage at the timesteps where the two raw directions
  are most aligned.

## Open questions
1. Is the (channel 110, channel 173) discovery seed-stable across DDPM
   retrainings, or is it a property of one trained checkpoint?
2. Does SeFa-on-conv-weights or GANSpace-on-h give us unsupervised
   directions that line up with the classifier-derived Δh?
3. Can a CLIP-loss objective (StyleCLIP global) bypass the CelebA
   classifier and reveal channels for attributes not in CelebA?
4. How does the per-timestep angle ∠(Δh_a(t), Δh_b(t)) behave — does it
   monotonically decrease as t → 0, and can a learned per-t
   reweighting beat global orthogonalization?
5. Hessian-Penalty-style training of a small diffusion model: does
   adding axis-alignment regularization sharpen single-channel purity
   further, or does it hurt sample quality?
