# CLAUDE.md

Guidance for Claude Code when working on this branch.

## Orientation: this is `gh-pages`, not the model code

This branch is the **static project website** for VolFill (NeurIPS 2026). There is no PyTorch, no `volfill` package and no training code here. The model code is on the `main` branch of the same repo (and the full dev code in the private `VolFill_private` repo). GitHub Pages serves this branch as-is (`.nojekyll` disables Jekyll).

No build step, bundler or package manager: plain HTML, CSS (`@layer`), and vanilla ES modules. Serve locally with `python scripts/serve.py` (port 8000). Plain `python -m http.server` ignores HTTP Range requests, so the animation videos can't seek there; GitHub Pages handles ranges.

## File map

```
index.html                  semantic sections, no inline styles; KaTeX (CDN + SRI) renders $...$ math
css/tokens.css              palette (dark default, [data-theme=light]), type scale, spacing, @font-face
css/base.css                reset + base + utilities (wrap-read 720px, wrap-wide 1200px, .reveal)
css/components.css          above-the-fold components (buttons, topbar, hero, stage/viewers, stats); render-blocking
css/sections.css            everything below the fold (chapters, animation figures, widgets, compare, tables, chart,
                            lightbox); loaded async via rel=preload so first paint isn't blocked
js/main.js                  boot: theme, KaTeX, scroll-spy + progress, reveals, tabs, lightbox, BibTeX copy,
                            and lazy imports (IntersectionObserver) of the modules below
js/theme.js                 data-theme on <html>, persisted in localStorage('volfill-theme'); fires 'themechange'
                            (an inline <head> script applies the stored theme before first paint)
js/viewers/model-viewer.js  loads <model-viewer> 4.0.0 from CDN with SRI; carousel, loading bar, fullscreen helpers
js/viewers/teaser.js        hero viewer: ours.glb for models/valid_samples_teaser.txt. Shows media/teaser-poster.webp
                            first; loads <model-viewer> on idle (desktop) or on tap of "Explore in 3D" (<=720px)
js/viewers/compare.js       baseline-vs-ours viewer with camera sync; holds the BASELINES array
js/chart.js                 SVG bar chart (occluded APD / complete FS); numbers inline, mirror Tables 1-2
js/anim-player.js           inline Manim videos: loop while >=50% visible, pause off-screen; beat chips under
                            each clip highlight the current beat and seek to it (marks from media/anim/manifest.json)
js/widgets/tudf-slider.js   W1: tau/eps slider over media/widgets/tudf-slice.{png,json}
js/widgets/flow-scrubber.js W2: t-slider over media/widgets/flow/<scene>/step_XX.webp (real Euler steps)
fonts/                      self-hosted WOFF2: Inter, Newsreader (+italic), JetBrains Mono
figures/opt/                WebP figures at 800/1600/2400 (scripts/optimize_images.py); originals stay in figures/
figures/logo/               Blender/Cycles logo + favicons (scripts/logo/)
media/anim/                 s1..s3 x {dark,light} x {webm,mp4,jpg poster} + manifest.json (Problem, TUDF, VAE)
media/widgets/              widget data (scripts/export_widget_data.py)
animations/                 Manim sources (not loaded by the page): palette.py, common.py, scenes/*.py, render.sh
scripts/screenshot.py       Playwright verification harness (screenshots + smoke checks)
scripts/serve.py            local preview server with HTTP Range support (also used by the harness)
```

## Asset contract (drives both viewers)

```
models/<sample_id>/input.jpg  ours.glb  moge2.glb  da3.glb  nova3r.glb  lari.glb
```

- `<method>.glb` stems MUST match the `variant` fields of `BASELINES` in `js/viewers/compare.js`.
- `models/valid_samples_teaser.txt` is **hand-curated** (teaser order). `models/valid_samples.txt` is **auto-generated** by `scripts/validate_samples.py`; never hand-edit it. The comparison viewer's first scene is the first line of `valid_samples.txt`.
- **Baseline rename = three places:** `BASELINES` in `js/viewers/compare.js`, the `--methods` passed to `validate_samples.py`, and `METHOD_DIRS` in `build_quali_glbs.py`. `group` only affects UI grouping (Pixel-aligned / Amodal).
- `.glb` files are committed (`models/` is ~740 MB). `<model-viewer>` uses `loading="lazy"` and the viewers initialise only near the viewport.

## Workflows

**Add qualitative scenes** (run in the `volfill` env, which has open3d/trimesh):
```bash
python scripts/build_quali_glbs.py --quali_root ../VolFill/results/quali_results --out_dir models/
python scripts/validate_samples.py --methods ours moge2 da3 nova3r lari
# optionally edit models/valid_samples_teaser.txt
```
`build_quali_glbs.py` negates Y *and* Z (OpenCV -> glTF); flipping only one mirrors the scene.

**Re-render the animations** (needs manim, ffmpeg with libvpx-vp9/libx264, and LaTeX with dvisvgm; TinyTeX works):
```bash
bash animations/render.sh                 # all scenes, both themes, 1080p30, encode + manifest
bash animations/render.sh S2TUDF          # one scene
QUALITY=-ql bash animations/render.sh     # quick preview
```
Each scene calls `self.mark()` at the start of every beat. The number of marks must equal the number of `.beats button` chips in the matching `[data-anim-fig="sN"]` figure of `index.html` (chip i seeks to mark i). Colours come from `animations/palette.py`, which mirrors `css/tokens.css`; keep them in sync. Clips are encoded with a 15-frame GOP so seeking stays responsive. Keep each clip ≤ ~3 MB.

**Widget data:**
```bash
python scripts/export_widget_data.py tudf --glb models/scrream_scene09_full_00_000275/ours.glb      # W1 (also feeds S1-S3)
python scripts/export_widget_data.py flow --traj_dir <run>/out/<scene>/<frame> --name scene09        # W2, per scene
```
W2 needs a denoising trajectory from `VolFill_private`: `python -m volfill.evaluation.eval_latent_flow_visible --mode inference_all_steps ...` (see `scripts/infer/infer_traj_webpage.sbatch` there). It writes `step_XXX/pred_tudf_256.npz` per Euler step. List the exported scenes in `media/widgets/flow/index.json`. The code integrates t from 1 → 0; the page shows the paper's convention (0 = noise, 1 = data).

**Figures:** `python scripts/optimize_images.py` regenerates `figures/opt/*.webp`. **Logo:** `scripts/logo/render_logo.py` (Blender) + `scripts/logo/make_assets.sh`.

**Verify before pushing:**
```bash
python scripts/screenshot.py --out /tmp/volfill_shots/check --sections        # desktop/tablet/mobile x dark/light
python scripts/screenshot.py --out /tmp/shots --gl swiftshader                # headless Linux (AWS) without a GPU
```
It checks for console errors, asset 404s, viewer loads and swaps, camera sync, theme persistence, inline animation playback and beat-chip seeking, and widget response, and exits non-zero on failure.

**Deploy:** commit to `gh-pages` and `git push origin gh-pages`; GitHub Pages redeploys automatically.

## Non-obvious things

- Pre-release metadata: BibTeX is an `@inproceedings` NeurIPS 2026 entry that keeps arXiv 2605.31466; update it when proceedings metadata exists.
- `media/og-image.jpg` (1200×630) is the social preview referenced by the OG/Twitter tags.
- Under `prefers-reduced-motion` the clips show native controls and nothing autoplays.
- `media/teaser-poster.webp` is rendered from the first teaser GLB: `python scripts/export_widget_data.py poster --glb models/$(head -1 models/valid_samples_teaser.txt)/ours.glb`. Regenerate it if you reorder the teaser list.
- Lighthouse (local, `scripts/serve.py`): mobile 90 / 100 / 100 / 100, desktop 100 across the board. The 3D-viewer script is ~1 MB, so keep it lazy.
- `scripts/build_glb_assets.py` and `scripts/render_orbits.py` are older and dormant. The supplementary-video section is not on the page (`videos/video_supp.mp4` is still committed).
