#!/usr/bin/env bash
# Convert a recording from the game (press V) into blog-ready formats.
#
#   tools/clip.sh <input.webm> [outdir]
#
# Produces, next to each other:
#   <name>.mp4       H.264 + AAC, faststart — plays inline everywhere
#   <name>.gif       palette-optimised, 12 fps, 720px wide
#   <name>-poster.jpg  a frame for the <video poster> attribute
#
# Trim before converting if you want a shorter clip:
#   ffmpeg -i in.webm -ss 3 -t 8 -c copy trimmed.webm
set -euo pipefail

IN="${1:?usage: tools/clip.sh <input.webm> [outdir]}"
OUTDIR="${2:-$(dirname "$IN")}"
BASE="$(basename "${IN%.*}")"
mkdir -p "$OUTDIR"

echo "→ $BASE.mp4"
# yuv420p and the even-dimension scale are what make it play on Safari and iOS;
# without them you get a file that works everywhere except the phones most
# people will read the post on.
ffmpeg -hide_banner -loglevel error -y -i "$IN" \
  -c:v libx264 -preset slow -crf 19 -pix_fmt yuv420p \
  -vf "scale=trunc(iw/2)*2:trunc(ih/2)*2" \
  -c:a aac -b:a 160k -movflags +faststart \
  "$OUTDIR/$BASE.mp4"

echo "→ $BASE.gif"
# Two passes: build a palette from the whole clip, then map to it. A single
# pass quantises per-frame and the dithering in this game's output turns into
# crawling noise.
PAL="$(mktemp -t palette).png"
ffmpeg -hide_banner -loglevel error -y -i "$IN" \
  -vf "fps=12,scale=720:-1:flags=lanczos,palettegen=stats_mode=diff" "$PAL"
ffmpeg -hide_banner -loglevel error -y -i "$IN" -i "$PAL" \
  -lavfi "fps=12,scale=720:-1:flags=lanczos[x];[x][1:v]paletteuse=dither=bayer:bayer_scale=3" \
  "$OUTDIR/$BASE.gif"
rm -f "$PAL"

echo "→ $BASE-poster.jpg"
ffmpeg -hide_banner -loglevel error -y -ss 1 -i "$IN" -frames:v 1 -q:v 3 \
  "$OUTDIR/$BASE-poster.jpg"

echo
ls -lh "$OUTDIR/$BASE".{mp4,gif} "$OUTDIR/$BASE-poster.jpg" 2>/dev/null | awk '{print $5, $9}'
echo
echo "Blog embed:"
echo "  <video src=\"$BASE.mp4\" poster=\"$BASE-poster.jpg\" autoplay loop muted playsinline></video>"
