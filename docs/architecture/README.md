# Architecture diagrams

Version-controlled sources for callosum's architecture diagrams.

Per the active control plane, diagrams used in architecture docs must be
generated from version-controlled source, not maintained primarily as
hand-edited image files. Each `.puml` file in this directory is a
PlantUML source; the matching `.svg` next to it is the rendered output.
Regenerate when the source changes.

## Sources

- `request_lifecycle.puml` — sequence diagram of one request from
  inbound POST through routing decision, backend dispatch, upstream
  streaming, usage-log write, peer-quality capture, and the offline
  shadow label pass that can turn captured peer opinions into
  `quality_score` training candidates. Renders without external
  dependencies (just PlantUML + Java).
- `runtime_topology.puml` — component diagram of the loopback proxy,
  backend lanes, persistent state, capability harness, and scheduled
  auto-dev path. This is the quickest visual for understanding where
  model-research inputs feed the dev loop.

## Rendering

```
scripts/render_diagrams.sh
```

Output: `docs/architecture/*.svg` next to each `.puml` source. PNG
copies can be rendered on demand with the same PlantUML jar using
`-tpng`.

The script expects PlantUML's single-file jar at
`~/.local/share/plantuml/plantuml.jar`. If you haven't installed it
yet, the script prints a clear message and exits non-zero. The jar is
a single-file download from `https://plantuml.com/download`; it is
kept under the user's local share rather than committed to this repo.
Java is the only other requirement and is already standard on this
workstation.

## Why PlantUML and not Structurizr / Mermaid / diagrams-as-code

- **PlantUML + Java** runs with what is already on the workstation;
  no language runtime install, no `npm`/Chromium dependency.
- Pure-text source is diff-friendly. Source review catches drift
  between intent and what gets rendered.
- SVG output is the control plane's preferred public format for
  architecture artifacts.
- Sequence diagrams in PlantUML do not need Graphviz, which is the
  most common friction with `dot`-backed component diagrams.

Switching to Structurizr DSL later is reasonable if the diagram
inventory grows past a handful of files and the C4-model formalism
starts paying off; for the current scope (one sequence diagram) the
overhead would exceed the benefit.
