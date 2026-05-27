# VolFill — Project Webpage

Static GitHub-Pages-style site for the VolFill paper.

## Local preview

```bash
cd VolFill_ghpage
python3 -m http.server 8000
# open http://localhost:8000
```

That's the full toolchain — there is no build step.

## What's in this repo

```
index.html              # the page
css/style.css           # all styling (vanilla CSS, no framework)
js/gallery.js           # click-to-play video gallery
js/viewer-swap.js       # click-to-swap GLB in <model-viewer>
paper/volfill.pdf       # the paper
figures/                # rasterized figures
videos/                 # orbit videos + supplementary video
posters/                # first-frame JPGs for the gallery
models/<scene>/         # per-scene 3D assets (input.jpg + *.glb + thumb.png)
scripts/                # asset-pipeline scripts (run offline)
```

## Asset pipeline

The page ships with placeholder paths for the interactive viewer and the
orbit-video gallery. To populate them with real VolFill outputs:

1. Run VolFill inference on the scenes you want to feature (`.ply` point
   clouds and/or `.obj` meshes will be written to your output directory).
2. Write a manifest JSON describing each scene:

    ```json
    [
      {
        "id": "scene01",
        "input":   "/path/to/scene_01/image.jpg",
        "ours":    "/path/to/scene_01/volfill.ply",
        "gt":      "/path/to/scene_01/gt.ply",
        "nova3r":  "/path/to/scene_01/nova3r.ply",
        "trellis": "/path/to/scene_01/trellis.ply"
      }
    ]
    ```

3. Convert into web-ready `.glb` files:

    ```bash
    pip install trimesh pillow
    python scripts/build_glb_assets.py --manifest scenes.json --out models/
    ```

4. Render 360° orbit videos:

    ```bash
    pip install open3d 'imageio[ffmpeg]'
    python scripts/render_orbits.py --manifest scenes.json
    ```

Update `index.html` so the scene-thumbnail `data-stem` values and the
gallery `data-src` paths match the scene IDs you generated.

## Manual content edits (TODO)

`index.html` contains the following placeholders to fill in before
publication:

- Hero: author names, affiliations, profile URLs.
- Hero: arXiv link in the second action button.
- Quantitative tables: every `TBD` cell — copy the final numbers from
  the paper PDF.
- BibTeX block: replace the placeholder once the arXiv ID is assigned.

## Deployment

The site is fully static. Two common options:

- Push this directory to its own GitHub repo and enable Pages on `main`.
- Push as the `gh-pages` branch of an existing repo and enable Pages on
  that branch.

The `<model-viewer>` web component is loaded from a CDN, so no build /
bundle step is needed.

## Credits

Page template adapted from
[TrackCraft3R](https://cvlab-kaist.github.io/TrackCraft3R/) /
[St4RTrack](https://st4rtrack.github.io/). Interactive 3D viewer pattern
adapted from [DAGE](https://dage-3d.github.io/).
