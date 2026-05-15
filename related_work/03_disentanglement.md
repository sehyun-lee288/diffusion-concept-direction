# Related Work: Disentanglement and Orthogonal Concept Directions in Generative Models

## Summary

Disentanglement in image generative models has been pursued along two complementary
axes: (i) *training-time* objectives that bake an independence prior into the
latent code (β-VAE, β-TCVAE, FactorVAE), and (ii) *post-hoc* analyses of an
already-trained generator's internal space (InterFaceGAN, GANSpace, StyleSpace,
Asyrp, Concept Algebra, Concept Sliders, Post-Hoc CAV disentanglement). Our work
falls firmly in the second camp: we take a frozen diffusion U-Net, identify a
direction `Δh_smile` in the bottleneck `h`-space by labelled mean-shift, and then
*decorrelate* it from `Δh_gender` per-timestep with Gram–Schmidt. The empirical
phenomenon we observe — `corr(Δh_smile, Δh_gender) ≈ +0.56` before
orthogonalization, `+0.13` after, but recoupling at large edit strength
`|s|=6` — is consistent with what this literature predicts: linear
disentanglement is real but local, and downstream non-linearities re-mix any
basis you try to impose from outside the training loop.

## Key Papers

- **β-VAE (Higgins et al., ICLR 2017)** and **β-TCVAE (Chen et al.,
  arXiv:1802.04942)**. Train-time disentanglement by reweighting the KL /
  total-correlation term. Produces axis-aligned independent latents *by
  construction*, but at a steep reconstruction cost and only on simple
  datasets. β-TCVAE introduces the Mutual Information Gap (MIG) metric.
- **Locatello et al., "Challenging Common Assumptions in the Unsupervised
  Learning of Disentangled Representations" (arXiv:1811.12359, ICML 2019).**
  The impossibility result: without inductive bias on either the model or
  the data, disentangled representations are not identifiable. Practical
  implication for us — any post-hoc orthogonalization is necessarily picking
  *one* basis out of an equivalence class, and its quality is bounded by the
  generator's own structure.
- **InterFaceGAN (Shen et al., arXiv:2005.09635 / CVPR 2020).** Finds an SVM
  hyperplane normal `n_a` for each attribute in StyleGAN's W-space, and —
  crucially for us — explicitly proposes *conditional manipulation*:
  `n_a − (n_a · n_b) n_b` to remove the gender component from a smile
  direction. This is the exact one-pair Gram–Schmidt step we use, just in
  W-space rather than h-space.
- **GANSpace (Härkönen et al., arXiv:2004.02546, NeurIPS 2020).**
  Unsupervised PCA of W (or early-layer features) yields ~100 interpretable
  directions. Establishes that the leading principal components of the
  generator's latent distribution are already approximately independent.
- **StyleFlow (Abdal et al., arXiv:2008.02401, ACM TOG 2021).** Conditional
  continuous normalizing flow over W+, conditioned on the attribute vector,
  so that editing one attribute samples from `p(W | a')` while holding the
  others fixed. A *non-linear* alternative to Gram–Schmidt that explicitly
  models attribute correlations rather than projecting them out.
- **StyleSpace (Wu et al., arXiv:2011.12799, CVPR 2021).** Most directly
  relevant to our "purity > 0.85 channel" approach. They show that the
  channel-wise style-modulation space `S` of StyleGAN2 is *significantly more
  disentangled* than W or W+, and propose ranking channels by an
  **Attribute Dependency (AD)** metric — essentially how much a target
  attribute changes versus how much *other* attributes co-change. Top-ranked
  channels per attribute act as local, single-channel sliders.
- **StylEx (Lang et al., arXiv:2104.13369, ICCV 2021).** Trains a
  classifier-aware StyleGAN so that StyleSpace coordinates are forced to
  align with classifier-relevant attributes; counterfactual editing is then
  per-coordinate.
- **Asyrp (Kwon et al., arXiv:2210.10960, ICLR 2023).** Introduces
  `h`-space (the U-Net bottleneck) and empirically demonstrates its
  homogeneity, linearity, robustness, and cross-timestep consistency —
  the very assumptions our mean-shift direction relies on.
- **Concept Algebra for Score-Based Text-Controlled Models (Wang et al.,
  arXiv:2302.03693, NeurIPS 2023).** Formalizes "concepts as subspaces" of
  the score representation and shows that text-conditioned diffusion models
  admit an algebraic decomposition where one concept's subspace can be
  projected out of another. This is the theoretical scaffolding for what we
  do empirically in h-space.
- **Haas et al., "Discovering Interpretable Directions in the Semantic
  Latent Space of Diffusion Models" (arXiv:2303.11073).** Directly relevant:
  finds h-space directions both unsupervised (PCA / Jacobian spectrum) and
  supervised (classifier on generated samples), and **explicitly proposes
  linear projection to disentangle correlated directions** — the same
  operation we apply, scaled to multi-step diffusion.
- **Concept Sliders (Gandikota et al., arXiv:2311.12092, ECCV 2024).**
  LoRA adaptors trained with a *preservation* objective that suppresses
  drift on listed attributes (e.g. preserve race/gender while editing age).
  A learnable counterpart to our hard Gram–Schmidt subtraction.
- **Post-Hoc Concept Disentanglement (arXiv:2503.05522, 2025).** Adds a
  non-orthogonality penalty to CAV training so concept directions come out
  orthonormal. Closest existing analogue to a *soft* version of our
  multi-attribute joint orthogonalization.

## How Our Work Fits In

**Post-hoc Gram–Schmidt versus training-time independence priors.**
β-VAE-style training-time methods *cannot* be applied to a frozen Stable
Diffusion / DDPM — by the time we get the model, the smile/gender entanglement
is already baked into the U-Net weights. Our line of work is in the
InterFaceGAN / GANSpace / Concept-Algebra / Haas-et-al. family: take the
model as given, find directions, then *clean them up* by linear projection.
The novel piece we add is doing it **per-timestep** in h-space — InterFaceGAN
projects once in W-space, Haas et al. project a single global h-direction,
whereas the bottleneck attribute structure in a diffusion U-Net actually
shifts across `t`, so the gender-component of `Δh_smile(t)` is itself
t-dependent. Our drop from `+0.559` to `+0.134` confirms this is worth doing,
and the residual `0.134` plus the recoupling at `|s|=6` confirms Locatello's
warning — a frozen network has a *finite* linear-disentanglement budget.

**Is "purity > 0.85 channel selection" a known idea?** Yes, but only in
StyleGAN, not in diffusion h-space. StyleSpace (Wu et al. 2020) and StylEx
(Lang et al. 2021) both rank channels of `S` by an attribute-specific metric
(AD or classifier influence) and edit only the top channels. Our 110
"smile-pure" channels with purity `(Δsmile − Δgender) / (Δsmile + Δgender) >
0.85` is the diffusion h-space analogue. The two design choices that appear
new are: (a) doing this in the diffusion U-Net bottleneck rather than a GAN
modulation space, where the channels are not nominally an "isolated style
basis" but turn out to behave that way on a sizable subset; (b) using a
*hard* purity cut rather than a continuous AD-style ranking, which gives us
the clean `|s| ≤ 4` editing window.

**Is the clean `|s| ≤ 4` range a new observation?** Partly. Asyrp and
DiffStyle both report that linearity in h-space is robust *for moderate edit
strength* and degrades for large shifts, and StyleFlow's whole motivation is
that linear edits become entangled for large step sizes. So the *existence*
of a clean regime is expected. What is — to our knowledge — under-reported
is the *quantitative break point* on diffusion bottleneck channels under a
purity-selected basis, and the asymmetry that selective injection on
pure channels stays clean further than full-vector Gram–Schmidt does.

## Open Questions

- **When does orthogonalization fail?** Locatello's impossibility plus our
  recoupling at `|s|=6` suggest there is a non-linear neighbourhood of the
  identity inside which Gram–Schmidt is a good local model and outside which
  the U-Net's non-linearities re-mix attributes. Characterising this
  neighbourhood as a function of `t`, channel index, and image content is
  open. The Jacobian-spectrum approach of Haas et al. 2024 is the obvious
  tool.
- **Soft purity weighting.** Replacing the hard `purity > 0.85` cut with a
  continuous weight `w_c ∈ [0, 1]` per channel — closer to StyleSpace's AD
  metric and to the soft non-orthogonality penalty of arXiv:2503.05522 —
  could trade a small purity loss for a larger usable edit range.
- **Multi-attribute joint orthogonalization.** Pairwise Gram–Schmidt
  (smile ⟂ gender) does not guarantee `smile ⟂ {gender, age, eyeglasses, …}`
  jointly. Concept Algebra's subspace projection and Concept Sliders'
  multi-preservation loss both attack this, and a per-timestep multi-attribute
  QR / oblique projection in h-space is a natural next step.
- **Does the rank-1-ish channel split predict editability?** If channels
  cluster into roughly attribute-pure groups, an SVD of the
  `(attributes × channels)` loading matrix should expose this structure —
  and the singular spectrum's decay rate would quantify how close the
  bottleneck is to truly rank-1.
