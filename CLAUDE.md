# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Orientation: this is `gh-pages`, not the model code

Despite the path being `neurips2026_sub/VolFill/`, the checked-out branch is **`gh-pages`** — a static project website, not the training/inference codebase. There is no PyTorch, no `volfill` package, no training entry points here. The model code lives on the **`main`** branch of the same repository (under its own `volfill` conda env, per the parent `../CLAUDE.md`).

Do not look here for: model architecture, training scripts, dataset loaders, evaluation entry points. Switch to `main` first (`git checkout main`) — but be aware that this working tree is committed to `gh-pages`, so changes will collide.

The parent `../CLAUDE.md` describes the multi-project workspace; this file covers only the website.

## What's on this branch

A static GitHub-Pages site for the VolFill paper. No build step, no bundler, no package manager — `index.html` is served as-is, with two vanilla-JS modules and a CSS file. GitHub Pages auto-deploys whatever lands on `gh-pages` (`.nojekyll` disables Jekyll processing). Serve locally with `python -m http.server 8000`.

Page structure (single-file):
- `index.html` — hero / abstract / method / qualitative viewer / quantitative tables / ablations / limitations / bibtex / references. Tables, method text, and the references list are inline HTML; the two viewers are JS-driven. The references `<ol class="references">` (numbered `[n]` via a CSS counter, baselines + eval datasets) lives inside the `#bibtex` section after the `<pre>`.
- `css/style.css` — all page styles, including the `viewer-*` classes that the JS depends on for thumb wiring.
- `js/teaser-viewer.js` — drives the single `<model-viewer id="teaser-viewer">` at the top.
- `js/viewer-swap.js` — drives the side-by-side baseline-vs-ours comparison in the Qualitative Results section, with bidirectional camera sync.
- `js/gallery.js` — orphan video-gallery script not wired into the current `index.html` (kept for potential future use).

Both JS files are wrapped in IIFEs so their top-level `SCENES` / `MODELS_ROOT` consts don't collide.

3D rendering uses Google's `<model-viewer>` web component loaded from CDN (`ajax.googleapis.com/.../model-viewer@4.0.0`). It expects GLB files (binary glTF) at the URL given by the `src` attribute.

## Asset layout — the contract that drives the viewers

Each scene is a folder under `models/<sample_id>/`. Required files:

```
models/<sample_id>/
  input.jpg          # input RGB shown beside the viewers
  ours.glb           # VolFill prediction (point cloud as GLB)
  moge2.glb          # one .glb per baseline listed in BASELINES (see js/viewer-swap.js)
  da3.glb
  nova3r.glb
  lari.glb
```

The `<method>.glb` stem MUST match the `variant` field in the `BASELINES` array in `js/viewer-swap.js` — that's how the viewer maps a clicked thumb to a file path.

Two manifests govern which scenes the page shows:

- `models/valid_samples_teaser.txt` — **hand-curated** ordered list for the top teaser viewer. Edit directly.
- `models/valid_samples.txt` — **auto-generated** by `scripts/validate_samples.py`. Contains only samples that have every required `.glb` plus `input.jpg`. Do not hand-edit; regenerate.

`models/valid_samples_all.txt` is a snapshot of all candidates (informational; not consumed by JS).

## Workflow: adding new qualitative scenes

This is the most common task. The pipeline goes inference-outputs → GLB → validation:

1. **Convert raw inference outputs to web GLBs** (run from inside the `volfill` conda env on `main`, since it uses `open3d` and `trimesh`):
   ```bash
   python scripts/build_quali_glbs.py \
     --quali_root ../VolFill/results/quali_results \
     --out_dir models/
   ```
   This reads from `VolFill/results/quali_results/<method_dir>/<dataset>/<scene>/<frame>/` (one subdir per method — see the `METHOD_DIRS` mapping at the top of the script) and writes `models/<dataset>_<scene>_<frame>/<method>.glb` + `input.jpg`.

2. **Refresh the validated list:**
   ```bash
   python scripts/validate_samples.py --methods ours moge2 da3 nova3r lari
   ```
   This regenerates `models/valid_samples.txt`. The `--methods` list must match (or be a subset of) the `BASELINES` `variant` values in `js/viewer-swap.js`, otherwise the JS will request a `.glb` that doesn't exist and the viewer will silently 404.

3. **(Optional) Update the teaser list** in `models/valid_samples_teaser.txt` to highlight the best new scenes at the top of the page.

`scripts/build_glb_assets.py` is an older/simpler converter that takes a JSON manifest of source paths instead of scanning a results tree. `build_quali_glbs.py` is the one actively used; only fall back to `build_glb_assets.py` for one-off scenes outside the standard results layout.

`scripts/render_orbits.py` renders 360° orbit MP4s into `videos/orbit_<id>.mp4`. The current `index.html` doesn't embed orbit videos (the supplementary video section is commented out), so this script is dormant.

## Non-obvious things to know

- **Coordinate flip in `build_quali_glbs.py`.** `_normalize_and_orient()` (lines ~146–159) negates both Y *and* Z to convert from OpenCV (Y-down, Z-into-scene) to GLTF (Y-up, Z-toward-viewer). Flipping only one axis produces a left-handed reflection and the viewer renders a mirrored scene. If a new sample appears flipped or upside-down, this is where to look.
- **Baseline rename = three places.** Adding/removing/renaming a baseline requires editing all three: the `BASELINES` array in `js/viewer-swap.js`, the `--methods` arg passed to `validate_samples.py`, and the `METHOD_DIRS` mapping in `build_quali_glbs.py`. The page will silently break if any pair drifts apart. Each `BASELINES` entry also has a `group` field (currently `Pixel-aligned` for `moge2`/`da3`, `Amodal` for `nova3r`/`lari`) that only controls how the method thumbs are stacked/labelled in the second thumb row — it doesn't affect file paths.
- **Default sample IDs in `index.html` are hardcoded** in two places — the `src` of `<model-viewer id="teaser-viewer">` (currently `models/scrream_scene10_full_00_001005/ours.glb`) and the `<model-viewer id="viewer-right">` / `<model-viewer id="viewer-left">` pair (currently `models/scrream_scene01_full_00_000055/{nova3r,ours}.glb`) — as the initial scene shown before the JS swaps in the first thumb. If you delete those specific sample dirs, the page will 404 on first paint (the JS recovers when the user clicks a thumb, but it looks broken on load). Either keep the referenced dirs or update both `src` attributes. Note the side-by-side default `scrream_scene01_full_00_000055` exists on disk but is **not** in `valid_samples.txt` (that list has `..._000285`), so it is shown only as the first-paint default and is never reachable via a thumb — the thumbs come exclusively from `valid_samples.txt`.
- **`.glb` files are committed to the repo.** `models/*/*.glb` is tracked, and `models/` is the bulk of the branch's size. `.gitignore` has commented-out entries for hosting them externally — uncomment only if you switch to LFS or external hosting.
- **Pre-release links/IDs in `index.html` are not final.** The arXiv button now points to a concrete ID (`href="https://arxiv.org/abs/2605.31466"`) but it is a placeholder — the BibTeX block still carries a `TODO: replace once arXiv ID is assigned` and an unverified `@article{ngo2026volfill, ...}` year/journal. The Code button still points to `https://github.com/ngoductuanlhp/VolFill`. Treat arXiv ID, BibTeX, and Code URL as all needing a final pass before public release; don't assume the live arXiv link is real just because it isn't `#`. (The author/affiliation block is now finalized — real authors, with institution logos from `figures/icons/` — so the old author-list `TODO` has been removed.)

## Common commands

```bash
# Serve the site locally
python -m http.server 8000   # then open http://localhost:8000

# Regenerate valid_samples.txt after adding new sample dirs or editing BASELINES
python scripts/validate_samples.py --methods ours moge2 da3 nova3r lari

# Convert a fresh batch of inference outputs into web GLBs
python scripts/build_quali_glbs.py \
  --quali_root ../VolFill/results/quali_results \
  --out_dir models/

# Dry-run discovery (lists samples that would be converted)
python scripts/build_quali_glbs.py --dry_run

# Deploy: commit and push to gh-pages
git add models/ index.html  # or whatever changed
git commit -m "..."
git push origin gh-pages    # GitHub Pages picks up automatically
```

There is no linter, no test suite, no CI on this branch.
