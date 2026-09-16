#!/usr/bin/env bash
# Render every PlantUML source in docs/architecture/ to SVG.
#
# This script is the canonical way to regenerate the architecture
# diagrams from their .puml sources. Diagrams are generated from
# version-controlled source rather than hand-edited image files.
# Whenever a .puml source changes, run this script and commit the
# regenerated SVG alongside.
#
# Requirements:
#   * Java runtime (`java -version` works).
#   * PlantUML jar at $PLANTUML_JAR (defaults to
#     ~/.local/share/plantuml/plantuml.jar). The jar is a single-file
#     download from https://plantuml.com/download — kept under the
#     user's local share, not committed to this repo.
#
# Usage:
#   scripts/render_diagrams.sh           # render all .puml -> .svg
#   PLANTUML_JAR=/custom/path/plantuml.jar scripts/render_diagrams.sh
#
# Exit 0 on success. Non-zero when the jar is missing or any render
# fails so this script is safe to wire into CI later.
set -euo pipefail

PLANTUML_JAR="${PLANTUML_JAR:-$HOME/.local/share/plantuml/plantuml.jar}"
SOURCE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/docs/architecture"

if [ ! -f "$PLANTUML_JAR" ]; then
    echo "render_diagrams: PlantUML jar not found at $PLANTUML_JAR" >&2
    echo "Download from https://plantuml.com/download (single-file jar)" >&2
    echo "or set PLANTUML_JAR to a custom location." >&2
    exit 2
fi

if ! command -v java >/dev/null 2>&1; then
    echo "render_diagrams: java not on PATH — install JRE first." >&2
    exit 2
fi

# -tsvg     : emit SVG (preferred over raster)
# -nbthread : single-thread; the diagrams here are tiny and parallelism
#             adds noise to error reporting without helping latency.
# Pass each .puml explicitly so a missing source dir fails clearly.
shopt -s nullglob
sources=("$SOURCE_DIR"/*.puml)
if [ ${#sources[@]} -eq 0 ]; then
    echo "render_diagrams: no .puml sources in $SOURCE_DIR" >&2
    exit 2
fi

echo "render_diagrams: rendering ${#sources[@]} source(s) -> SVG"
java -Djava.awt.headless=true -jar "$PLANTUML_JAR" -tsvg -nbthread 1 "${sources[@]}"
echo "render_diagrams: done. Outputs in $SOURCE_DIR/*.svg"
