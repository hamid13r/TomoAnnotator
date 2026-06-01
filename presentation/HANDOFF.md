# TomoAnnotator — Presentation Handoff for Design

**Purpose:** everything a designer needs to build the slide deck. Content is final;
visual treatment is yours. Where a slide needs an image, the asset (or a description
of what to make) is noted.

---

## At a glance

- **Project:** TomoAnnotator — teach a model to spot organelles in cryo-ET tomograms from a few brush strokes.
- **Team:** Grotjahn Lab, Scripps Research. Lead: Michaela Medina. Contributor: Hamidreza Rahmani.
- **Event:** Scripps Research Hackathon 2026 (May 29 – June 1).
- **Talk length:** ~5 minutes.
- **Audience / judges:** mixed — part technical, part scientific/domain.
- **Core narrative angle:** *idea → real, working tool, built with Claude.*
- **One-line pitch:** "Paint a little, let the model learn, detect everywhere."

---

## Key messages (the things the audience must walk away with)

1. Finding organelles in tomograms is slow, manual, expert-only work that doesn't scale.
2. TomoAnnotator turns it into: paint a few examples → train in minutes → auto-detect across new data.
3. It already runs end-to-end on real Grotjahn Lab tomograms — not a mockup.
4. It was taken from idea to a working 9-script pipeline during the hackathon, with Claude.
5. It's reusable: any cryo-ET lab can point it at their data and their own features.

---

## Visual identity / design direction

- **Mood:** dark, scientific, "instrument readout" feel. Deep navy/near-black background.
- **Suggested palette (from the working deck):**
  - Background ink: `#101420`
  - Panel/card: `#161B2E`
  - Accent blue (Claude/tech): `#4D9DFF`
  - Accent orange (organelle/biology): `#FF8C00`
  - Teal highlight: `#2DD4BF`
  - Text white: `#F5F7FB`; muted gray: `#9AA4BF`
- **Organelle class colors** (use consistently anywhere features are listed/overlaid):
  mitochondria = orange `#FF8C00`, ER = cyan `#00BFFF`, microtubules = green `#32CD32`,
  vesicles = violet `#EE82EE`, ribosomes = gold `#FFD700`, nuclear envelope = tomato `#FF6347`,
  filaments = magenta `#FF00FF`.
- **Type:** clean sans-serif (Arial/Inter/Helvetica). Big bold headlines, generous whitespace.
- **Tone:** confident and concrete; let the real tomogram imagery carry the "wow."

---

## Assets provided

- `presentation/assets/tomo_raw.png` — a real raw tomogram slice (grayscale). Use as the "before."
- `presentation/assets/tomo_overlay.png` — same slice with painted organelle classes overlaid in class colors. Use as the "after / what we paint."
- Source data shown: tomogram `MIM019_2_lam11_ts_003`, volume 250 × 720 × 512 voxels.
- A reference build of the deck already exists (`TomoAnnotator_Hackathon.pptx`) if you want a starting layout.

**Still to capture (nice-to-have, optional):** a screen-grab of the matplotlib painting
viewer in action, and a sample `results/report.md` or a few rows of `predictions.csv`
for the "output" moment.

---

## Slide-by-slide content

> 8 slides for a ~5-minute talk. Each block = on-slide text + a visual note + what the
> speaker says. Keep on-slide text terse; speaker notes carry the detail.

### Slide 1 — Title
- **On slide:** "TomoAnnotator" / sub: "Teach a model to spot organelles in cryo-ET tomograms — from a few brush strokes." / "Idea → working tool, built with Claude" / "Grotjahn Lab · Scripps Research Hackathon 2026 · Michaela Medina, Hamidreza Rahmani"
- **Visual:** full-bleed faint tomogram texture behind the title, or a clean dark cover with the accent rule.
- **Say:** "We built a tool that finds organelles in cryo-ET data automatically — and we got it working during the hackathon."

### Slide 2 — The problem
- **Headline:** "Finding organelles in tomograms is slow, manual, expert work."
- **Three cards:**
  - *Cryo-ET = 3D cell maps* — Each tomogram is a noisy 3D volume (250×720×512 voxels) of a frozen cell.
  - *Done by hand today* — A biologist scrolls slice-by-slice asking: is there mitochondria here? ER? microtubules?
  - *Doesn't scale* — Dozens of tomograms per session. Manual screening is the bottleneck before any analysis.
- **Footer line:** "The ask: a tool that tells us which organelles are present in each tomogram — automatically."
- **Visual:** the three cards; optionally the raw tomogram slice as a faint side image.
- **Say:** "Today this is eyeballed slice by slice. It's slow, it needs an expert, and it's the bottleneck."

### Slide 3 — The idea
- **Headline:** "Paint a little. Let the model learn. Detect everywhere."
- **Four steps (left→right, arrows between):**
  1. **Paint** — biologist paints a few examples of each feature in 1–2 tomograms.
  2. **Learn** — extract patches, train a small CNN (2D / 2.5D / 3D) in minutes.
  3. **Detect** — slide the model over new tomograms → presence/absence report.
  4. **Report** — Claude summarizes which tomograms are most interesting.
- **Footer line:** "No full segmentation needed — a few dozen brush strokes per class is enough."
- **Visual:** 4-step horizontal flow, color-code each step.
- **Say:** "The trick is weak supervision: you don't segment everything, you just paint a few examples."

### Slide 4 — It works on real data  *(the money slide)*
- **Headline:** "Real Grotjahn Lab tomograms, painted and detected."
- **Visual:** side-by-side — `tomo_raw.png` (label: "Raw tomogram slice") and `tomo_overlay.png` (label: "Painted organelle classes").
- **Caption:** "MIM019_2_lam11_ts_003 · 7 feature classes: mitochondria, ER, microtubules, vesicles, ribosomes, nuclear envelope, filaments."
- **Say:** "This is our actual data. Left is raw; right is what we paint. From a handful of strokes like these, the model learns to find these features everywhere."

### Slide 5 — Built with Claude
- **Headline:** "From a one-line idea to a 9-script pipeline."
- **Four points:**
  - *Scaffolded the whole pipeline* — preprocess → paint → extract → train → detect → report, plus config-driven feature classes.
  - *Wrote the hard parts* — a matplotlib painting viewer, balanced patch sampling, and 2D/2.5D/3D CNNs that swap with one config line.
  - *Handled the infra* — conda envs (GPU + CPU), SLURM jobs for Garibaldi, S3 sync, EC2 GPU launch recipe.
  - *Wired up the AI report* — Claude on Bedrock turns the predictions CSV into a plain-English summary.
- **Visual:** four stacked rows with accent bars; optional small Claude/AI motif.
- **Say:** "This is the hackathon story: Claude let us go from idea to a real, infrastructure-complete tool in days."

### Slide 6 — Under the hood
- **Headline:** "One config, three model modes, GPU-optional."
- **Visual:** 7-box horizontal pipeline: Raw .mrc › preprocess › paint › extract patches › train CNN › detect › Claude report.
- **Three detail cards:**
  - *Model modes* — 2D · 2.5D (default) · 3D, switch with one line in config.yaml. 2.5D sidesteps the missing-wedge along Z.
  - *Config-driven* — feature classes, patch size, thresholds all in config.yaml. Add a class, no code changes.
  - *Runs anywhere* — CPU laptop for everything but training; GPU on Garibaldi (SLURM) or EC2 g4dn.
- **Say:** "It's flexible — three model types from one config line — and it runs whether or not you have a GPU."

### Slide 7 — Why it matters
- **Headline:** "Turns hours of manual screening into minutes."
- **Three stat cards:**
  - **minutes** — to train, not days
  - **7** — organelle classes, extensible
  - **1–2** — tomograms to paint, then auto-detect the rest
- **Footer line:** "Reusable beyond this lab: any cryo-ET group can point it at their data, paint their own features, and screen tomograms at scale."
- **Say:** "The payoff: minutes instead of hours, and it generalizes to any lab's data."

### Slide 8 — Closing
- **Headline:** "Painted once. Detects everywhere."
- **Sub:** "A real cryo-ET tool, idea-to-working in a hackathon — with Claude."
- **Small print:** "Next: more painted data, validation against expert labels, napari integration. · Grotjahn Lab · Scripps Research · 2026"
- **Say:** "Painted once, detects everywhere. Thank you."

---

## Notes for the designer

- Slide 4 is the emotional peak — give the two images maximum size and let them breathe.
- Keep body text to the phrases above; resist adding paragraphs. The speaker fills in.
- Maintain the organelle color legend consistently (slides 4, 6, 7 reference the classes).
- If you add an animation, the slide-3 four-step flow and the slide-6 pipeline are the natural candidates (reveal left to right).
- Logos to source if desired: Scripps Research, Grotjahn Lab. Claude/Anthropic mark optional on slide 5.

---

*Reference for credibility line if needed: Medina, Rahmani et al., "Surface Morphometrics
reveals local membrane thickness variation in organellar subcompartments," J Cell Biol
2025, PMID: 41474626.*
