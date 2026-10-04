#!/bin/sh
# Regenerate every ojas logo raster from docs/brand/ojas-logo-source.png.
# Run from the repository root: sh docs/brand/render.sh
# Needs Google Chrome (rasterizes docs/brand/render.html) and ImageMagick
# (crop, alpha, and downsampling only). The social card loads its fonts from
# Google Fonts, so run it online.
set -eu

src=docs/brand/ojas-logo-source.png
out=site/assets
chrome="/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"
page="file://$(pwd)/docs/brand/render.html"

# 1. Transparent cut-out. The render is an object on pure black; alpha is the
#    brightest channel lifted x2.2 above a 1.5% noise floor, and colour is
#    un-premultiplied so the reds stay true over any dark background.
magick "$src" -alpha off -crop 704x704+160+146 +repage "$out/.crop.png"
magick "$out/.crop.png" -fx 'mx=max(u.r,max(u.g,u.b)); al=min(1,max(0,(mx-0.015)*2.2)); al>0.004 ? min(1,u/al) : 0' "$out/.rgb.png"
magick "$out/.crop.png" -fx 'min(1,max(0,(max(u.r,max(u.g,u.b))-0.015)*2.2))' -colorspace gray "$out/.alpha.png"
magick "$out/.rgb.png" "$out/.alpha.png" -alpha off -compose CopyOpacity -composite "$out/ojas-logo.png"
rm "$out/.crop.png" "$out/.rgb.png" "$out/.alpha.png"
magick "$out/ojas-logo.png" -quality 88 -define webp:alpha-quality=95 "$out/ojas-logo.webp"

# 2. Tile, plate and social card, rasterized by Chrome.
shot() { "$chrome" --headless --hide-scrollbars --allow-file-access-from-files "$@" 2>&1 | grep 'written' ; }
shot --default-background-color=00000000 --window-size=512,512 --screenshot="$out/ojas-tile-512.png" "$page?kind=tile"
shot --window-size=512,512 --screenshot="$out/icon-512.png" "$page?kind=plate"
shot --virtual-time-budget=5000 --window-size=1200,630 --screenshot="$out/ojas-og.png" "$page?kind=og"

# 3. Downsampled sizes.
magick "$out/ojas-tile-512.png" -filter Lanczos -resize 32x32 "$out/favicon-32.png"
magick "$out/icon-512.png" -filter Lanczos -resize 192x192 "$out/icon-192.png"
magick "$out/icon-512.png" -filter Lanczos -resize 180x180 -alpha off "$out/apple-touch-icon.png"

# 4. docs/ mirrors the one-page site for local previewing.
cp "$out"/* docs/assets/
rm -f docs/why.html docs/benchmarks.html docs/roadmap.html docs/guide/*.html
if [ -d site/assets/plots ]; then
  mkdir -p docs/assets/plots
  cp site/assets/plots/* docs/assets/plots/
fi
# The page, scripts, styles, reference pages, sitemap and search index are
# mirrored (and verified) by the generator.
go run scripts/sitegen/main.go
