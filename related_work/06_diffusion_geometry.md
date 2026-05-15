# Related Work: The Geometry of Diffusion Models

## Summary

"Diffusion geometry" is a loose cluster of work that treats the diffusion
generative process not as a black-box sampler but as a *geometric object* whose
structure can be measured, differentiated, and traversed. Four threads recur:
(i) **Riemannian geometry of latent / h-space** — assigning a metric (usually a
pullback) to the noisy state space and computing local bases, geodesics, and
curvature; (ii) **geometry of the PF-ODE / reverse-SDE trajectory itself** —
asking whether trajectories are straight, where they curve, and where they
branch; (iii) **score-field geometry** — critical points, Hessian/Jacobian
eigenstructure, basins of attraction, and the normal bundle of the data
manifold; and (iv) **mode / memorization structure** — when and how a
trajectory commits to a data mode, framed as a symmetry-breaking phase
transition. The unifying message for our project is that the *informative
geometric state of a diffusion model is the trajectory across noise scales, not
a single timestep's activations* — which is exactly why our single-timestep
activation-polytope transfer failed.

## Key papers

**1. Understanding the Latent Space of Diffusion Models through the Lens of
Riemannian Geometry** (Park, Kwon, Choi, Jo, Uh; NeurIPS 2023; arXiv:2307.12868).
*Idea:* the h-space (U-Net bottleneck) has no intrinsic metric, so import one by
pullback from x-space. *Method:* use the Jacobian `J_x = ∇_x h` to define a
pullback metric; its top singular vectors give a *local* latent basis, used for
training-free editing by a single-timestep traversal. They show the geometry is
time-varying and coarse-to-fine. *Relation to our work:* this is the closest
existing "geometry of h-space" and it explicitly works at a single timestep —
but note it *edits once* and reports success on Stable Diffusion x-space, which
is mid-trajectory. It does not give a trajectory-level boundary; its basis is a
tangent-space object, not a partition of state space. It partly explains our
polytope failure: the relevant structure is a smooth metric that *changes every
step*, not a fixed sign-pattern cell.

**2. Unsupervised Discovery of Semantic Latent Directions in Diffusion Models**
(Park, Kwon, Jo, Uh; 2023; arXiv:2302.12469). *Idea:* find semantic directions
without labels. *Method:* power-iterate the Jacobian of the denoiser to extract
the subspace that most influences a region of interest; directions are applied
across timesteps. *Relation to our work:* the Asyrp-style multi-step application
matches our finding (3) that the edit must accumulate over the trajectory.
Confirms a single Jacobian slice is a local linearization, not a controller.

**3. The Spacetime of Diffusion Models: An Information Geometry Perspective**
(Karczewski, Heinonen, Pouplin, Hauberg, Garg; ICLR 2026 Oral; arXiv:2505.17517).
*Idea:* the standard deterministic-PF-ODE pullback is provably degenerate — it
forces geodesics to decode as straight lines in pixel space, ignoring data
geometry. *Method:* use the stochastic reverse-SDE decoder and an
information-geometric Fisher-Rao metric on the *latent spacetime* `z = (x_t, t)`
indexing both state and noise scale; denoising distributions form an exponential
family with tractable geodesics and a principled edit distance. *Relation to our
work:* **this is the strongest candidate framework for a trajectory-aware
boundary (T6).** It makes "the trajectory" — `(x_t, t)` across all `t` — the
geometric object, and gives a metric on it. A decision boundary could be defined
as a Fisher-Rao separatrix in this spacetime. It also directly explains why a
single-timestep construct (polytope, pullback) has "no controlling power": the
deterministic single-`t` view collapses the metric.

**4. Dynamical regimes of diffusion models** (Biroli, Bonnaire, de Bortoli,
Mézard; Nature Communications 2024). *Idea:* the reverse process has distinct
dynamical regimes separated by sharp transition times. *Method:* statistical-
physics analysis of reverse dynamics under near-exact scores; identifies a
**speciation time** (trajectory commits to a class/mode) and a later
**collapse/condensation time** (commits to a specific training example).
*Relation to our work:* this is essentially the diffusion analogue of a decision
boundary — the speciation time is *when the trajectory commits to an attribute*.
A "trajectory-aware boundary" for an attribute could be operationalized as the
speciation manifold at that attribute's speciation time.

**5. Measuring Semantic Information Production in Generative Diffusion Models**
(Handke, Koulischer, Raya, Ambrogioni; ICLR DeLTa 2025; arXiv:2506.10433).
*Idea:* measure *when* class-semantic decisions happen. *Method:* an online
optimal-Bayes classifier estimates conditional entropy of the class given `x_t`;
its time-derivative locates peak information transfer. Finds semantic decisions
peak at intermediate `t` and, crucially, **different classes/features commit at
different times**. *Relation to our work:* directly supports a trajectory-aware
notion — the "boundary" for each attribute lives at its own characteristic `t`.
For our CelebA-HQ attributes, this predicts attribute-specific commit windows,
not a universal `t=500`.

**6. Symmetry-breaking / phase-transition view of generation** (multiple:
spontaneous symmetry breaking in diffusion, Raya & Ambrogioni NeurIPS 2023; and
follow-ups on class speciation). *Idea:* generation is a sequence of noise-driven
symmetry-breaking transitions; trajectory branching happens when multiple
datapoints remain compatible with `x_t` and the model is forced to pick.
*Relation to our work:* gives the mechanism of "commitment": a bifurcation in
the score field. The boundary is the unstable set separating basins *before*
symmetry breaking — a genuinely trajectory-level structure.

**7. Your diffusion model secretly knows the dimension of the data manifold**
(Stanczuk, Batzolis, Deveney, Schönlieb; ICML 2024; arXiv:2212.12611). *Idea:*
at low noise the score points toward the data manifold and spans its normal
bundle; counting score directions estimates intrinsic dimension. *Relation to
our work:* explains the encoder-skip dominance we saw (1): near the end of the
trajectory the score is dominated by manifold-normal correction, so a bottleneck
injection is overwritten by skip-carried manifold geometry.

**8. Losing dimensions: Geometric memorization in generative diffusion**
(Achilli, Ventura, Silvestri, Pham, Raya, Krotov, Lucibello, Ambrogioni; 2024;
arXiv:2410.08727). *Idea:* memorization is a progressive collapse of degrees of
freedom in the score field — salient directions freeze first, then fine detail.
*Relation to our work:* frames mode structure as eigenvalue collapse of the
score Jacobian along the trajectory; suggests the score-Jacobian spectrum is a
trajectory-level diagnostic of where structure (and attributes) lock in.

**9. Score-field critical-point / geometric-asymptotics analyses** (e.g.
geometric asymptotics of score mixing, 2025/2026). *Idea:* for empirical
(Dirac-mixture) data the limiting potential is piecewise-quadratic with
Voronoi-type structure; trajectories converge to critical points. *Relation to
our work:* the *Voronoi* picture is the honest analogue of "linear regions" for
diffusion — but it lives in the score potential over the whole trajectory, not
in a single layer's activation signs.

## How our work connects

**Best framework for trajectory-aware boundary (T6).** The Spacetime /
information-geometry view (paper 3) is the most promising scaffold: it makes the
*whole trajectory* `(x_t,t)` the geometric carrier and equips it with a
well-posed Fisher-Rao metric. Our "trajectory-aware boundary" can be posed as a
codimension-1 separatrix in this spacetime — the locus where the Fisher-Rao
geodesic distance to two attribute-conditioned endpoint distributions is equal.
This is well-defined precisely where a single-timestep polytope is not, because
it integrates over noise scales rather than slicing one. The
speciation/symmetry-breaking line (papers 4, 6) supplies the dynamical content:
the boundary is the *unstable manifold* of the score field at the attribute's
speciation time.

**Has anyone studied where the trajectory commits to an attribute?** Yes, but
incompletely. Speciation time (paper 4) and semantic-information peaks (paper 5)
both localize *when* class commitment happens, and both find it is
attribute-specific and at intermediate `t`. However, this work studies *class*
commitment (which mode) on simple datasets (GMMs, CIFAR-10), not *attribute*
commitment (smiling vs. not) on a fixed identity in CelebA-HQ, and it does not
construct an explicit boundary surface — it reports a scalar transition time.
Constructing the actual separating manifold, and at the attribute granularity,
is the open gap our project can fill.

**What's underexplored.** (a) No one has built an explicit *decision-boundary
surface* in diffusion state space — only scalar transition times or local
tangent bases. (b) The h-space geometry work (papers 1, 2) and the
trajectory/speciation work (papers 3-6) are disjoint literatures; nobody has
asked whether h-space directions *are* the symmetry-breaking modes. (c)
Attribute-level (not class-level) speciation on a held-fixed identity is
untouched.

## Open questions / concrete leads for the next phase

- **Measure attribute speciation time.** Run our `d'` classifier on `x_t` (not
  just h) across all 50 DDIM steps; the `t` where `d'` rises sharply is the
  CelebA-HQ analogue of speciation. Compare against the constant `t=500` we
  currently assume — paper 5 predicts it is attribute-dependent.
- **Score-Jacobian spectrum as a trajectory diagnostic.** Following papers 7-8,
  track eigenvalues of `∇_{x_t} ε_θ` along the trajectory; the step where an
  attribute-aligned eigendirection becomes dominant is a candidate "commit
  point" and a trajectory-level boundary signal.
- **Fisher-Rao separatrix.** Adopt the spacetime metric of paper 3 and define
  the attribute boundary as the equidistant locus between two attribute-
  conditioned denoising distributions — a principled replacement for the failed
  sign-pattern polytope.
- **Test the bifurcation hypothesis.** At the measured speciation `t`, perturb
  `x_t` along a candidate concept direction and check for basin-flip (paper 6);
  a sharp, history-independent flip would confirm a genuine boundary, whereas a
  smooth drift would not.
- **Open caveat.** Papers 4-6 are validated mostly on low-dimensional or
  CIFAR-scale data; whether sharp speciation survives at CelebA-HQ 256 and at
  attribute (not class) granularity is unverified and worth a dedicated check.
