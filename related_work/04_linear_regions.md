# Related Work: Decision Boundary Visualization and Linear-Region Theory

## Summary

The "linear-region paradigm" treats a deep network with piecewise-linear (PWL)
activations — most commonly ReLU — as a continuous piecewise-affine map. Each
neuron defines a hyperplane in input space; the joint pattern of activation
signs across all neurons partitions the input into convex polytopes
("linear regions"), on each of which the network reduces to a single affine
function. Two threads use this geometry: (i) **expressivity theory**, which
counts or bounds the number of such regions to argue that depth multiplies
capacity, and (ii) **boundary visualization**, which slices the input space by
a 2-D plane and renders either the polytope structure or the level set
`f(x) = 0` directly. The paradigm makes two implicit claims that matter for
our diffusion project: (a) the activation pattern is the right "state" of the
network at a point, and (b) the decision boundary on a 2-D slice reveals
semantically meaningful structure because it tracks where the affine piece
changes.

## Key papers

**Montúfar, Pascanu, Cho & Bengio (2014), arXiv:1402.1869** — the original
linear-region count for deep ReLU/maxout nets. They prove that a depth-`L`
network with `n` units per layer can attain on the order of
`(n/n_0)^((L−1)·n_0) · n^{n_0}` regions, giving the headline "exponential in
depth" result that motivates the whole paradigm.

**Serra, Tjandraatmadja & Ramalingam (2018), arXiv:1711.02114** — tighter
bounds and the first exact enumeration via mixed-integer linear programming.
Their key qualitative finding: deep nets only beat shallow ones in region
count once total neuron count exceeds input dimension, suggesting that
"region count = expressivity" is more nuanced than the 2014 bound implies.

**Hanin & Rolnick (2019a), arXiv:1901.09021** ("Complexity of Linear Regions
in Deep Networks") and **(2019b), arXiv:1906.00904** ("Deep ReLU Networks
Have Surprisingly Few Activation Patterns") — show that the *average* number
of regions along any 1-D slice grows only **linearly** in total neurons,
independent of depth, both at initialization and after training. The
worst-case exponential bound is essentially never realized in practice. This
is the central caveat: real networks live far below their theoretical
combinatorial capacity.

**Jeon, Lee, Kim & Moon (2020), arXiv:1912.05827** ("E-GBAS", AAAI 2020) —
applies the activation-pattern view to *generative* networks. They define a
"generative boundary" as the hyperplane on which an internal node's
pre-activation changes sign, then sample within polytopes bounded by the
most informative such hyperplanes to obtain semantically coherent variations
of a query image. This is the canonical 2-D-slice visualization for a
generator.

**Humayun, Balestriero, Balakrishnan & Baraniuk (2023), arXiv:2302.12828**
("SplineCam", CVPR 2023 Highlight) — the first *exact* (sampling-free)
algorithm that computes the full polytope partition and the decision
boundary on a bounded 2-D region of input space for any CPWL network. It
builds a planar graph by intersecting back-projected hyperplanes with the
slice boundary and enumerates faces via cycle search. Importantly, it
remains exact only for CPWL activations (ReLU, leaky-ReLU, abs, maxout).

**Park, Lee, Park & Yoon (2023), arXiv:2312.17285** ("Relaxed Decision
Region", RDR) — selects a sparse set of "principal" neurons whose joint
sign pattern carves out a region grouping instances with a coherent
concept. RDR essentially uses Top-K-balanced sign selection to define an
attribute-meaningful region, anticipating the same selection issue we hit.

**Black et al. (2022), arXiv:2211.12312** ("Interpreting Neural Networks
through the Polytope Lens", Anthropic) — argues that polytopes, not
directions or single neurons, are the right unit of mechanistic
interpretability, and shows empirically that *boundary density* in
activation space correlates with semantic transitions.

**Alfarra et al. (2020), arXiv:2002.08838** ("On the Decision Boundaries of
Deep Neural Networks") — characterizes the decision boundary as a tropical
hypersurface dual to the convex hull of two zonotopes; used for pruning,
lottery tickets, and adversarial generation, but inherits the ReLU /
piecewise-linear scaffolding.

## How our work fits in

Our pilot replicated the 2-D-slice visualization paradigm (E-GBAS /
SplineCam / RDR) inside a diffusion U-Net. Three observations matter:

1. **SiLU is not the obstruction.** SiLU is monotone in sign with its
   argument: `sign(SiLU(x)) = sign(x)`, so the sign-pattern induced
   partition is identical to what a ReLU net with the same pre-activations
   would yield. Our hyperplane boundaries `sign(spatial_mean(h_c)) = 0` and
   per-pixel `sign(h_c[i,j]) = 0` form the same arrangement of lines in
   `(α, β)` as in SplineCam. The polytope structure is real; the paradigm
   transfers at the level of geometry.

2. **What breaks is the *meaning* of a polytope edge.** In classification
   the boundary `f(x) = 0` is the object of interest because crossing it
   flips the model's output. In a diffusion U-Net at a single timestep, the
   relevant output is not a per-step neuron sign but the *integrated effect
   of the denoising trajectory*. A polytope edge marks where one channel's
   spatial-mean sign flips, but the generated image at the end of sampling
   depends on hundreds of such per-step affine pieces composed along the
   trajectory, plus noise. A static sign-pattern map captures one timestep
   of the dynamical system and loses the rest.

3. **Phase 14 makes this concrete.** Even on a 2-D plane that is *attribute-
   meaningful* by construction (axes aligned with attribute-conditional
   directions), the Top-K-balanced channel selection produced **radial
   boundary clustering at the in-window center** — every boundary line was
   forced through the origin because that is where the selection criterion
   (balanced flips inside the window) is maximized. The boundaries that
   appear are an artifact of the selector, not of attribute regions. The
   per-pixel sign at a fixed `(i, j)` gives a uniform-direction arrangement
   (no radial artifact), confirming the model itself does not produce that
   pattern. RDR (arXiv:2312.17285) faces the same hazard: its "principal
   neurons" are selected for region coherence, so the resulting region
   shape inherits the selection bias. SplineCam (arXiv:2302.12828) sidesteps
   this only because it enumerates *all* hyperplanes exhaustively — it has
   no selection step to bias.

In short: the polytope view (Black et al., arXiv:2211.12312) — boundary
density = semantic density — assumes the network's affine partition *is* the
semantic partition. For classifiers that is approximately true; for a
diffusion U-Net it holds only along a trajectory, not at one timestep.

## Open questions

- **Trajectory-aware boundary analysis.** Is there a well-defined
  "decision boundary" of a diffusion model — e.g. the set of starting
  noises that, under a fixed schedule, denoise to images on opposite sides
  of an attribute? This is a boundary of a composed map, not of a single
  affine piece.
- **Multi-step linear-region counting.** Composing `T` U-Net steps with
  ReLU/SiLU produces a piecewise-affine map of depth `~T · L`. The
  Hanin–Rolnick (arXiv:1901.09021, arXiv:1906.00904) "linear in neurons"
  result suggests the effective region count along a 1-D noise slice
  stays modest even at large `T`; whether boundary *positions* concentrate
  near semantic transitions in this composed map is open.
- **Selection-free attribute boundaries.** Can we adapt SplineCam-style
  exhaustive enumeration to score polytope edges by an attribute classifier
  rather than by network sign, avoiding the Top-K artifact entirely?
- **What replaces "the polytope" for diffusion?** Candidates: probability-
  flow ODE basins of attraction, score-field critical sets, or flow-map
  Jacobian rank changes. Each is a trajectory-level analogue of the
  single-step linear region.
