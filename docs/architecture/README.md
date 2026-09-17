# Architecture artifacts

Version-controlled architecture artifacts for Callosum.

Static call and control-flow graphs are generated from version-controlled
Python source. Regenerate them whenever the routing or dispatch architecture
changes.

## Generated artifacts

- `generated/call_graph_focus.dot`
- `generated/call_graph_focus.json`
- `generated/call_graph_focus.svg`
- `generated/control_flow_focus.dot`
- `generated/control_flow_focus.json`
- `generated/control_flow_focus.svg`
- `generated/README.md`

Prose architecture is documented in `ARCHITECTURE.md`.

## Regeneration

```bash
uv run python scripts/generate_static_graphs.py --out-dir docs/architecture/generated
```

The script uses Python's standard library and Graphviz `dot`.

## Why generated static graphs

- Generated graphs stay synchronized with source code.
- DOT, JSON, and SVG outputs are inspectable and reviewable.
- SVG output works in browsers and Markdown viewers.
- The generator uses no third-party Python packages.
