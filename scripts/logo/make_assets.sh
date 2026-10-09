#!/bin/bash
# Derive sized logos, favicons and dark/light previews from the 2048px master render.
#   bash scripts/logo/make_assets.sh [figures/logo]
set -euo pipefail
D=${1:-figures/logo}
FF=${FFMPEG:-/opt/homebrew/bin/ffmpeg}
M=$D/volfill_logo.png

# Resize in premultiplied space so transparent (black) pixels don't fringe the edges.
resize() { # src size dst [extra filters]
  "$FF" -loglevel error -y -i "$1" -vf "format=rgba,premultiply=inplace=1,scale=$2:$2:flags=lanczos${4:-},unpremultiply=inplace=1,format=rgba" -frames:v 1 "$3"
}
on_bg() { # src colour canvas logo_size dst
  "$FF" -loglevel error -y -f lavfi -i "color=c=$2:s=$3x$3" -i "$1" -filter_complex \
    "[1:v]format=rgba,premultiply=inplace=1,scale=$4:$4:flags=lanczos,unpremultiply=inplace=1[l];[0:v][l]overlay=(W-w)/2:(H-h)/2:format=auto,format=rgb24" \
    -frames:v 1 "$5"
}

resize "$M" 512 "$D/volfill_logo_512.png"
resize "$M" 128 "$D/volfill_logo_128.png"
resize "$M" 512 "$D/favicon-512.png"
resize "$M" 192 "$D/favicon-192.png"
resize "$M" 32 "$D/favicon-32.png" ",unsharp=3:3:0.6"
on_bg "$M" 0x0E1015 180 148 "$D/favicon-180.png"
on_bg "$M" 0x0E1015 1024 880 "$D/preview_dark.png"
on_bg "$M" 0xFAF9F6 1024 880 "$D/preview_light.png"
ls -la "$D"
