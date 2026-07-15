# Instructions: rewrite of \subsubsection{Partition-of-unity head for sequential updates}

## Goal and audience

Rewrite the subsubsection `subsubsec:pou-head` in `journal_harry_hardcurl.tex`
so that a reader who has NEVER seen a partition of unity can follow it. The two primary
readers are the author and his supervisor — expert in FEM and porous-media flow, but
unfamiliar with: "partition of unity", "patch"/"window", "overlapping Cartesian grid",
"tensor product" (in this context), "one-dimensional bump", "normalized cosine window",
"linear probe". Every one of these must either be explained in plain words at first use
or removed.

Hard requirements:
- Smooth prose with flowing transitions between paragraphs — NO bullet lists, no
  enumerated design choices.
- Every concept in words BEFORE its symbol. A formula may only contain objects the
  reader has already been told about in plain language.
- Keep ALL existing equations, labels (`eq:pou_head`, `eq:pou_update`), citations, and
  the final cross-reference to the numerical studies. The mathematics does not change —
  only the exposition around it.
- Length may grow ~30–50% relative to the current version; clarity wins over brevity.
- Add the two TikZ figures specified below.

## Paragraph-by-paragraph structure

**Paragraph 1 — motivation (keep, lightly edited).** The current opening paragraph is
good: coupled problem, conductivity evolves with saturation, flux needed at every step,
retraining prohibitive, but the change between consecutive steps is small and localized
at the saturation front. End it one sentence earlier than now — do NOT yet name the
"partition-of-unity linear head"; end on the idea: "we freeze the trained network and
replace only its output layer by one whose coefficients can be updated in closed
form." This closing sentence is also where the term "head" gets its one-time
definition (see Terminology rules): "...its output layer (its `head', in
machine-learning terminology)...". The subsubsection title may keep "head" since the
definition arrives in the first paragraph.

**Paragraph 2 — why a single global output layer cannot do the job (NEW; this is the
bridge the current text is missing).** Explain: the trained network combines its W
hidden-layer functions with a single set of W output weights, valid over the whole
domain. Updating those weights moves the reconstructed flux EVERYWHERE at once, while
the change we need to track is confined to the neighborhood of the moving front; a
global correction must therefore compromise between regions that need change and
regions that must stay put. (One sentence may note this failure is observed in the
numerical studies.) The remedy: give different parts of the domain their OWN copies of
the output coefficients, so the update can act where the drift is and leave the rest
untouched. This paragraph sets up everything; write it carefully and plainly.

**Paragraph 3 — the covering by overlapping regions, and what "partition of unity"
means (NEW; all in words, no display math).** Describe the construction concretely:
cover the domain by N_w rectangular regions arranged on a regular grid, each somewhat
larger than the grid spacing so that neighboring regions overlap; a typical point lies
inside two to four regions. To each region k attach a weight function chi_k that equals
one near the center of its region and decreases smoothly to zero at its edge — a smooth
hill (for the profile, say in words: built from a squared cosine in each coordinate
direction, i.e., a smooth profile along x multiplied by the same profile along y; do
NOT use the phrase "tensor product" without this gloss). Because regions overlap, the
raw weights of the regions containing a point add up to more than one; dividing each
weight by their pointwise sum fixes this, so that at EVERY point the weights add up to
exactly one. A family of weights with this property is called a partition of unity —
literally, the number one is divided among the regions. Cite Melenk & Babuška
\citep{MelenkBabuska1996} HERE, at the moment the concept is introduced, with a
half-sentence gloss: the idea of blending local approximations through such weights
goes back to the partition-of-unity finite element method, where it is used to glue
local polynomial spaces into a globally conforming approximation. State the two consequences the
reader needs: (i) if all regions were given identical coefficients, the weighted blend
would return exactly the unpartitioned function — the partition by itself distorts
nothing; (ii) the normalized weights remain continuously differentiable, which matters
because the flux is a derivative of psi. Reference Figure A here.

**Paragraph 4 — the blended stream function (formula `eq:pou_head`, now earned).**
State the equation, then immediately read it aloud in one sentence: "at each point, the
stream function is the weighted average of the predictions of the N_w regional heads,
the weights being the window functions of the regions containing that point." Keep the
window-construction details that are already in the current draft, but compress them —
the prose burden was carried by Paragraph 3.

**Paragraph 5 — the shared features and the warm start.** First say what h(x) is in
words: the last hidden layer of the trained network evaluated at x — W scalar functions
that the training has shaped into a problem-adapted basis; the trained network's output
is a single fixed combination of them, psi_trained = h(x)^T w. Then the reduced basis:
rather than giving each region all W functions, give it r much smaller cleverly chosen
combinations — the FIRST is the trained combination itself (so "the trained solution"
is available to every region as a single basis function), and the remaining r−1 are the
directions in which the hidden features vary most over the domain (computed once by a
singular value decomposition of the feature values at the collocation points,
orthogonalized against w). Keep the display math for P. Then the warm-start identity in
plain words BEFORE the formal statement: if every region selects only the first basis
function and nothing else (theta_k = e_1), then, because the weights sum to one at
every point, the blend reproduces the trained network exactly — sequential updating
therefore starts from the trained solution, not from scratch.

**Paragraph 6 — the per-step update (`eq:pou_update`).** Keep the current content but
gloss each ingredient at first mention: Phi ("one row per dual face, expressing that
face's flux as a linear function of all regional coefficients"), the ridge anchor
("where a step's data say little about a region — for instance far from the front —
the penalty keeps that region's coefficients at their previous values instead of
letting them drift"), and the factorize-once economy ("the geometry, windows, and
features never change, so the matrix is assembled and factorized a single time; each
refresh afterwards costs only a pair of triangular solves"). Keep the conservation
sentence and the Proposition cross-reference exactly as now.

**Paragraph 7 — offline/online summary and lineage (rewrite of the current last
paragraph).** Replace "linear-probe argument" with a plain formulation: during
training, the deliberately narrow output layer — one weight per hidden unit — forces
the hidden layers to learn features through which a small linear combination can
already represent the flux well; that is what makes the frozen features reusable. Then:
online, adaptation happens only in the small regional coefficient spaces. Then the
lineage, one or two sentences, with each citation attached to a plain half-sentence
description of what that work does — never a bare bracket dump:
- \citep{ChenRFM2022} (random feature method): solves PDEs with FIXED randomly
  generated features on overlapping regions, determining the coefficients by linear
  least squares — the same feature-plus-linear-solve split used here, with learned
  rather than random features;
- \citep{LeeTrask2021} (partition-of-unity networks): combines learned partition
  functions with local polynomial coefficients obtained in closed form;
- \citep{Moseley2023} (finite-basis PINNs): trains a separate small network per
  overlapping region and blends them with a partition of unity, aimed at scaling
  physics-informed training to large domains;
- \citep{MelenkBabuska1996}: already cited in Paragraph 3; may be re-cited here in the
  "traces back to" clause without a new gloss.
The sentence should make clear in passing what THIS work adds relative to that line:
the blended field is a stream function, so every update inherits exact local
conservation, and the coefficients are refreshed sequentially inside a time-stepping
loop. Keep the equal-DOF locality-vs-richness remark but soften to match
the ablation ("at equal numbers of coefficients, adding regions is at least as
effective as enriching each region"), and keep the forward reference to the numerical
studies for the chosen grid, rank, and ridge. Reference Figure B in this paragraph or
in Paragraph 5.

## Figure A (TikZ, essential) — windows and the partition of unity in 1D

Two vertically stacked panels sharing an x-axis over [0,1]; place after Paragraph 3.
- Top panel: four raw cosine-squared bumps on an overlapping grid (centers at 0, 1/3,
  2/3, 1; support radius 1/3 so neighbors overlap), each drawn in a different line
  style/color; shade one overlap zone lightly and label it "overlap: point belongs to
  two regions". Mark one region's center c_k and its support width on the axis.
- Bottom panel: the same four bumps after normalization (each divided by the pointwise
  sum), plus a dashed horizontal line at height 1 labeled "sum of weights = 1
  everywhere". The visual message: individual weights deform, but their sum is the
  flat line.
- Caption (2–3 sentences, plain): raw smooth bumps on an overlapping grid; after
  dividing by their sum, the weights form a partition of unity: at every point the
  weights of the covering regions add to one, so blending regional predictions with
  these weights introduces no distortion of its own.
- TikZ hint: pgfplots with domain sampling of cos^2(pi*t/2) clipped to |t|<=1 is
  sufficient; normalization curve can be plotted from the same expressions. Keep it
  self-contained (no external data files).

## Figure B (TikZ, strongly recommended) — offline/online pipeline

A block diagram, placed near Paragraph 5 or 7. Left group in a rounded box labeled
"offline — trained once (~15 min)": input x → [hidden layers h(x), W units] →
[global output weights w] → psi_trained; below the trunk, an annotation "hidden
features frozen after training". Right group in a rounded box labeled "online — every
flux refresh (milliseconds)": h(x) → [projection P: r features, first = trained
combination] → N_w parallel small boxes theta_1 ... theta_Nw (regional heads) →
[blend with windows chi_k, sum = 1] → psi. An arrow from the offline box's h(x) into
the online box marked "reused, not retrained"; an arrow looping back into the theta
boxes labeled "closed-form update, eq. (pou_update)". Caption: one sentence per stage.
Keep the diagram monochrome-friendly (line styles, not color-dependent).

## Terminology rules

- Allowed after introduction: window, region, partition of unity, warm start.
- Must be introduced before use, exactly once, in plain words: partition of unity
  (Paragraph 3), reduced basis (Paragraph 5), ridge/anchor (Paragraph 6).
- "Head" (as in "PoU head", "global head", "regional heads") is machine-learning
  jargon and must be DEFINED at first use, in Paragraph 1, since the subsubsection
  title itself uses it. Required pattern: when the output layer is first mentioned,
  add the apposition — "the network's output layer (its `head', in machine-learning
  terminology)" — after which "head" may be used freely. NEVER write "global head";
  say "the single output layer shared by the whole domain" (Paragraph 2) or, after
  the definition, "the global output layer". "Regional heads" is allowed only after
  the Paragraph-1 definition AND Paragraph 3's introduction of regions.
- Banned outright: "tensor product" (say: a profile along x multiplied by the same
  profile along y), "bump" as an unexplained noun (say: smooth hill-shaped profile,
  then may use "bump"), "linear probe" (use the plain formulation in Paragraph 7),
  "Cartesian grid of patches" (say: regular grid of overlapping rectangular regions),
  "global head" (see above), "hard-curl" (banned everywhere per the main spec).
- The word "patch" may be replaced by "region" throughout for this audience; pick one
  and be consistent.

## Citations — summary

Four references, each cited once at the point its concept enters, each with a plain
half-sentence gloss (no bare citation lists): MelenkBabuska1996 in Paragraph 3 (origin
of the partition-of-unity idea, PUFEM); ChenRFM2022, LeeTrask2021, Moseley2023 in the
Paragraph-7 lineage with the glosses specified above. Verify the BibTeX keys match the
entries added per `journal_revision_spec.md` (do not invent new keys; if the .bib uses
different keys for these works, use the existing ones).

## What not to change

Equations `eq:pou_head` and `eq:pou_update` and the P display; all \label/\ref;
citations and their placement in the lineage sentence; the conservation statement and
its reference to the Proposition; the forward reference to the numerical studies.
Do not add new claims or numbers.
