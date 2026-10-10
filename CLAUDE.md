# CLAUDE.md

Guidance for Claude Code when working on this branch.

## Orientation: this is `gh-pages`, not the model code

This branch is the **static project website** for VolFill (NeurIPS 2026). There is no PyTorch, no `volfill` package and no training code here. The model code is on the `main` branch of the same repo (and the full dev code in the private `VolFill_private` repo). GitHub Pages serves this branch as-is (`.nojekyll` disables Jekyll).

No build step, bundler or package manager: plain HTML, CSS (`@layer`), and vanilla ES modules. Serve locally with `python scripts/serve.py` (port 8000). Plain `python -m http.server` ignores HTTP Range requests, so the animation videos can't seek there; GitHub Pages handles ranges.

## File map

```
index.html                  semantic sections, no inline styles; KaTeX (CDN + SRI) renders $...$ math
css/tokens.css              palette ([data-theme=dark|light]; the page defaults to light), type scale, spacing, @font-face
css/base.css                reset + base + utilities (wrap-read 720px, wrap-wide 1200px, .reveal)
css/components.css          above-the-fold components (buttons, topbar, hero, stage/viewers, stats); render-blocking
css/sections.css            everything below the fold (chapters, animation figures, widgets, compare, tables, chart,
                            lightbox); loaded async via rel=preload so first paint isn't blocked
js/main.js                  boot: theme, KaTeX, scroll-spy + progress, reveals, tabs, lightbox, BibTeX copy,
                            and lazy imports (IntersectionObserver) of the modules below
js/theme.js                 data-theme on <html>: light unless the visitor toggled dark (only that choice is stored, in
                            localStorage('volfill-theme-choice')); updates meta theme-color; fires 'themechange'
                            (an inline <head> script applies the stored theme before first paint)
js/viewers/model-viewer.js  loads <model-viewer> 4.0.0 from CDN with SRI; carousel, loading bar, fullscreen helpers
js/viewers/teaser.js        hero viewer: ours.glb for models/valid_samples_teaser.txt. Shows media/teaser-poster.webp
                            first; loads <model-viewer> on idle (desktop) or on tap of "Explore in 3D" (<=720px)
js/viewers/compare.js       baseline-vs-ours viewer with camera sync; holds the BASELINES array
js/chart.js                 SVG bar chart (occluded APD / complete FS); numbers inline, mirror Tables 1-2
js/anim-player.js           inline Manim videos: loop while >=50% visible, pause off-screen; beat chips under
                            each clip highlight the current beat and seek to it (marks from media/anim/manifest.json)
js/hero-anim.js             hero logo loop: picks media/logo-anim.mov (WebKit: HEVC+alpha) or .webm (VP9+alpha) by
                            engine, attaches it on idle while on screen; reduced motion keeps the poster
js/widgets/tudf-slider.js   W1: tau/eps slider over media/widgets/tudf-slice.{png,json}
js/widgets/flow-scrubber.js W2: t-slider over media/widgets/flow/<scene>/step_XX.webp (real Euler steps)
fonts/                      self-hosted WOFF2: Inter, Newsreader (+italic), JetBrains Mono
figures/opt/                WebP figures at 800/1600/2400 (scripts/optimize_images.py); originals stay in figures/
figures/logo/               Cycles logo master (= frame 0 of the hero loop) + favicons (scripts/logo/make_assets.sh)
media/logo-anim.*           hero logo loop, 840 px, transparent: .mov (HEVC+alpha), .webm (VP9+alpha), -poster.webp
media/anim/                 s1..s3 x {dark,light} x {webm,mp4,jpg poster} + manifest.json (Problem, TUDF, VAE)
media/widgets/              widget data (scripts/export_widget_data.py)
media/{dit,vae}-*.webp      insets of the generation / VAE SVG diagrams (index.html), which mirror Figs. 3 / 2 of
                            the paper: same NRGB-D kitchen/001435 renders, visible = photo colour, hidden = --occluded.
                            Exported from the paper workspace (paper_figs/arch/render_assets.py); keep them in sync
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

**Add qualitative scenes** (needs numpy/scipy/pillow/trimesh; open3d optional, the `volfill` env on AWS has no open3d):
```bash
python scripts/build_quali_glbs.py --quali_root ../VolFill/results/quali_results --out_dir models/ \
    --intrinsics_root <moge_root>          # <moge_root>/<dataset>/<scene>/<frame>/moge.npz
python scripts/validate_samples.py --methods ours moge2 da3 nova3r lari
# optionally edit models/valid_samples_teaser.txt
```
Points are coloured as in the paper (`--color visibility`, default): first hit along each input-camera ray keeps the photo colour, everything else is `#6E8FC4` (hidden), judged on each method's own geometry. It needs the input camera: normalised MoGe intrinsics (the paper workspace caches them in `paper_figs/teaser/_cache/data/moge/ours/`) or `intrinsics` in the sample's metadata.json. Regenerate `media/teaser-poster.webp` afterwards (below).
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
python scripts/export_widget_data.py flow --traj_dir <run>/out/<scene>/<frame> --name scene09 \
    --intrinsics <moge_root>/scrream/scene09_full_00/000275/moge.npz                                 # W2, per scene
python scripts/export_widget_data.py vae --gt gt_tudf_256.npz --recon recon_tudf_256.npz \
    --occ occ_64.npy --meta metadata.json --height 0.35                                             # S3 data
```
VolFill grids are indexed `(z, y, x)` while `bbox_min/max` are `(x, y, z)`; `tudf_to_points` handles this (before Oct 2026 it swapped x and z, so the W2 frames showed transposed scenes). W2 frames use the same paper colouring as the GLBs. S3 reads `media/widgets/vae-slice-{gt,recon,occ}.png` + `vae-slice.json`: one gravity-aligned slice through the real VAE stages of NRGB-D kitchen/001435 (GT TUDF, decoder 64³ occupancy, reconstruction), produced with the paper VAE in the paper workspace (`paper_figs/arch/_cache/`).
W2 needs a denoising trajectory from `VolFill_private`: `python -m volfill.evaluation.eval_latent_flow_visible --mode inference_all_steps ...` (see `scripts/infer/infer_traj_webpage.sbatch` there). It writes `step_XXX/pred_tudf_256.npz` per Euler step. List the exported scenes in `media/widgets/flow/index.json`. The code integrates t from 1 → 0; the page shows the paper's convention (0 = noise, 1 = data).

**Figures:** `python scripts/optimize_images.py` regenerates `figures/opt/*.webp`. `figures/architecture_{vae,dit}.png` are 2400-px renders of the paper's Fig. 2 / Fig. 3 PDFs (opened by the "Paper figure" links). **Logo + hero loop:** rendered with Blender/Cycles in the paper workspace (`paper_figs/logo_anim/`: `prep_scene.py` turns the NRGB-D kitchen/001435 GT TUDF into the scene mesh + voxels, `animate_logo.py` renders the loop, `encode.sh web` writes `media/logo-anim.*`). The static master `figures/logo/volfill_logo.png` is frame 0 at 2048 px (`--res 2048 --samples 512 --motion-blur 0 --times 0`); derive the rest with `scripts/logo/make_assets.sh` and `optimize_images.py`'s `build_logo()`. `scripts/logo/render_logo.py` renders the earlier designs (2³ slot, offset block; `figures/logo/alt/`).

**Verify before pushing:**
```bash
python scripts/screenshot.py --out /tmp/volfill_shots/check --sections        # desktop/tablet/mobile x dark/light
python scripts/screenshot.py --out /tmp/shots --gl swiftshader                # headless Linux (AWS) without a GPU
```
It checks for console errors, asset 404s, viewer loads and swaps, camera sync, theme persistence, inline animation playback and beat-chip seeking, and widget response, and exits non-zero on failure.

**Deploy:** commit to `gh-pages` and `git push origin gh-pages`; GitHub Pages redeploys automatically.

## Non-obvious things

- BibTeX is the NeurIPS 2026 `@inproceedings` entry (title/authors as in the camera-ready). Add `volume`/`pages`/`publisher` once the official proceedings entry is online; arXiv 2605.31466 stays linked from the hero buttons.
- `media/og-image.jpg` (1200×630) is the social preview referenced by the OG/Twitter tags.
- Under `prefers-reduced-motion` the clips show native controls and nothing autoplays.
- `media/teaser-poster.webp` is rendered from the first teaser GLB: `python scripts/export_widget_data.py poster --glb models/$(head -1 models/valid_samples_teaser.txt)/ours.glb`. Regenerate it if you reorder the teaser list.
- Lighthouse (local, `scripts/serve.py`): mobile 90 / 100 / 100 / 100, desktop 100 across the board. The 3D-viewer script is ~1 MB, so keep it lazy.
- `scripts/build_glb_assets.py` and `scripts/render_orbits.py` are older and dormant. The supplementary-video section is not on the page (`videos/video_supp.mp4` is still committed).
