#!/bin/bash
# Render the page animations (Manim) in both themes and encode them for the web.
#   bash animations/render.sh            # all scenes, 1080p30
#   bash animations/render.sh S2TUDF     # one scene
#   QUALITY=-ql bash animations/render.sh   # quick preview
# Needs: manim (pip), ffmpeg with libvpx-vp9 + libx264, a LaTeX with dvisvgm (TinyTeX works).
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(dirname "$HERE")"
OUT="$ROOT/media/anim"
WORK="${WORK:-/tmp/volfill_manim}"
MANIM="${MANIM:-manim}"
QUALITY="${QUALITY:--r 1920,1080 --fps 30}"
[ -d "$HOME/Library/TinyTeX/bin/universal-darwin" ] && export PATH="$HOME/Library/TinyTeX/bin/universal-darwin:$PATH"
[ -d "$HOME/tools/.TinyTeX/bin/x86_64-linux" ] && export PATH="$HOME/tools/.TinyTeX/bin/x86_64-linux:$PATH"

SCENES=(
  "s1 s1_problem S1Problem"
  "s2 s2_tudf S2TUDF"
  "s3 s3_vae S3VAE"
  "s4 s4_flow S4Flow"
  "s5 s5_conditioning S5Conditioning"
)
mkdir -p "$OUT" "$WORK"

for spec in "${SCENES[@]}"; do
  read -r short file cls <<<"$spec"
  if [ $# -gt 0 ] && [[ " $* " != *" $cls "* ]]; then continue; fi
  for theme in dark light; do
    echo "== $cls ($theme)"
    (cd "$HERE" && VOLFILL_THEME=$theme $MANIM $QUALITY --disable_caching --media_dir "$WORK/$theme" \
        -o "$short.mp4" "scenes/$file.py" "$cls" 2>&1 | grep -E "Rendered|rror" || true)
    src=$(find "$WORK/$theme/videos/$file" -name "$short.mp4" | head -1)
    base="$OUT/$short-$theme"
    ffmpeg -v error -y -i "$src" -an -c:v libvpx-vp9 -b:v 0 -crf 38 -row-mt 1 -deadline good -cpu-used 2 \
      -g 15 -keyint_min 15 -pix_fmt yuv420p "$base.webm"
    ffmpeg -v error -y -i "$src" -an -c:v libx264 -preset slow -crf 26 -g 15 -keyint_min 15 -sc_threshold 0 \
      -pix_fmt yuv420p -movflags +faststart "$base.mp4"
    poster_t=$(python3 -c "import json;m=json.load(open('$HERE/out/$cls.json'));print(max(0,(m['marks'][1] if len(m['marks'])>1 else m['duration'])-0.1))")
    ffmpeg -v error -y -ss "$poster_t" -i "$src" -frames:v 1 -vf scale=1280:-2 -q:v 4 "$base.jpg"
    printf '   %s  webm %s KB  mp4 %s KB\n' "$short-$theme" "$(( $(stat -f%z "$base.webm" 2>/dev/null || stat -c%s "$base.webm") / 1024 ))" \
      "$(( $(stat -f%z "$base.mp4" 2>/dev/null || stat -c%s "$base.mp4") / 1024 ))"
  done
done

python3 - "$HERE/out" "$OUT/manifest.json" <<'EOF'
import json, sys
from pathlib import Path
names = {"S1Problem": "s1", "S2TUDF": "s2", "S3VAE": "s3", "S4Flow": "s4", "S5Conditioning": "s5"}
out = {}
for cls, short in names.items():
    f = Path(sys.argv[1]) / f"{cls}.json"
    if f.exists():
        out[short] = json.loads(f.read_text())
Path(sys.argv[2]).write_text(json.dumps(out, indent=1))
print("manifest:", ", ".join(f"{k}={len(v['marks'])} steps/{v['duration']}s" for k, v in out.items()))
EOF
