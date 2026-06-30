#!/usr/bin/env python3
"""Generate static call/control-flow reference graphs for architecture review.

This is intentionally conservative: it uses only Python's stdlib AST and emits
review aids, not a sound whole-program proof. The outputs are generated from
source and are meant to point follow-on review threads at concrete code nodes.
"""

from __future__ import annotations

import ast
import json
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
PACKAGE_ROOT = SRC_ROOT / "callosum"
OUT_DIR = REPO_ROOT / "docs" / "architecture" / "generated"

FOCUS_ROOTS = (
    "callosum.__main__:_run_server",
    "callosum.app:create_app",
    "callosum.app:_dispatch_route",
    "callosum.app:_dispatch_internal",
    "callosum.app:_dispatch_nonstream_with_cell_retry",
    "callosum.app:_dispatch_stream_with_cell_retry",
    "callosum.app:_dispatch_nonstream",
    "callosum.app:_dispatch_stream",
    "callosum.app:_log_attempt",
    "callosum.routing.factory:build_router",
    "callosum.routing.router:Router.route",
)

CFG_ROOTS = (
    "callosum.app:_dispatch_internal",
    "callosum.app:_dispatch_nonstream",
    "callosum.app:_dispatch_stream",
    "callosum.routing.router:Router.route",
)


@dataclass(frozen=True)
class FunctionDefn:
    qname: str
    module: str
    name: str
    path: Path
    lineno: int
    end_lineno: int
    node: ast.AST = field(compare=False, hash=False, repr=False)


class ModuleCollector(ast.NodeVisitor):
    def __init__(self, module: str, path: Path) -> None:
        self.module = module
        self.path = path
        self.imports: dict[str, str] = {}
        self.functions: dict[str, FunctionDefn] = {}
        self._class_stack: list[str] = []
        self._function_depth = 0

    def visit_Import(self, node: ast.Import) -> None:
        for alias in node.names:
            bound = alias.asname or alias.name.split(".", 1)[0]
            self.imports[bound] = alias.name

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        if node.module is None:
            return
        mod = "." * node.level + node.module if node.level else node.module
        for alias in node.names:
            if alias.name == "*":
                continue
            bound = alias.asname or alias.name
            self.imports[bound] = f"{mod}.{alias.name}"

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        self._class_stack.append(node.name)
        for stmt in node.body:
            self.visit(stmt)
        self._class_stack.pop()

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        self._record_function(node)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        self._record_function(node)

    def _record_function(self, node: ast.FunctionDef | ast.AsyncFunctionDef) -> None:
        if self._function_depth:
            return
        self._function_depth += 1
        parts = [*self._class_stack, node.name]
        name = ".".join(parts)
        qname = f"{self.module}:{name}"
        self.functions[qname] = FunctionDefn(
            qname=qname,
            module=self.module,
            name=name,
            path=self.path,
            lineno=node.lineno,
            end_lineno=getattr(node, "end_lineno", node.lineno),
            node=node,
        )
        # Keep imports discovered inside top-level functions out of alias
        # resolution. Those are often lazy imports and should appear as call
        # targets only when invoked explicitly.
        self._function_depth -= 1


class CallCollector(ast.NodeVisitor):
    def __init__(self, module: str, imports: dict[str, str]) -> None:
        self.module = module
        self.imports = imports
        self.calls: set[str] = set()

    def visit_Call(self, node: ast.Call) -> None:
        target = self._expr_name(node.func)
        if target:
            self.calls.add(target)
        self.generic_visit(node)

    def _expr_name(self, node: ast.AST) -> str | None:
        if isinstance(node, ast.Name):
            name = node.id
            imported = self.imports.get(name)
            if imported:
                return imported
            return f"{self.module}:{name}"
        if isinstance(node, ast.Attribute):
            if isinstance(node.value, ast.Name):
                base_name = node.value.id
                imported = self.imports.get(base_name)
                if imported:
                    return f"{imported}.{node.attr}"
                # A local object's method call (`ctx.set(...)`,
                # `backend.health(...)`, `fallback.record_attempt(...)`) cannot
                # be resolved soundly from AST alone. Returning None avoids the
                # misleading suffix match where every `.set()` call became
                # `PinState.set`.
                return None
            base = self._expr_name(node.value)
            if base and not base.startswith(f"{self.module}:"):
                return f"{base}.{node.attr}"
            return node.attr
        return None


def module_name(path: Path) -> str:
    rel = path.relative_to(SRC_ROOT).with_suffix("")
    return ".".join(rel.parts)


def collect() -> tuple[dict[str, FunctionDefn], dict[str, dict[str, str]]]:
    functions: dict[str, FunctionDefn] = {}
    imports_by_module: dict[str, dict[str, str]] = {}
    for path in sorted(PACKAGE_ROOT.rglob("*.py")):
        module = module_name(path)
        tree = ast.parse(path.read_text(), filename=str(path))
        collector = ModuleCollector(module, path)
        collector.visit(tree)
        functions.update(collector.functions)
        imports_by_module[module] = collector.imports
    return functions, imports_by_module


def resolve_call(raw: str, functions: dict[str, FunctionDefn]) -> str | None:
    if raw in functions:
        return raw
    if ":" in raw:
        module, name = raw.split(":", 1)
        direct = f"{module}:{name}"
        if direct in functions:
            return direct
        if "." in name:
            matches = [q for q in functions if q.startswith(f"{module}:") and q.endswith(f".{name}")]
            if len(matches) == 1:
                return matches[0]
        return None
    # import attribute shape, e.g. callosum.routing.factory.build_router
    for q in functions:
        module, name = q.split(":", 1)
        dotted = f"{module}.{name}"
        if dotted == raw:
            return q
    return None


def build_edges(
    functions: dict[str, FunctionDefn],
    imports_by_module: dict[str, dict[str, str]],
) -> dict[str, set[str]]:
    edges: dict[str, set[str]] = {q: set() for q in functions}
    for qname, fn in functions.items():
        collector = CallCollector(fn.module, imports_by_module.get(fn.module, {}))
        collector.visit(fn.node)
        for raw in collector.calls:
            resolved = resolve_call(raw, functions)
            if resolved and resolved != qname:
                edges[qname].add(resolved)
    return edges


def reachable(edges: dict[str, set[str]], roots: tuple[str, ...], depth: int = 4) -> set[str]:
    seen: set[str] = set()
    frontier = [(r, 0) for r in roots if r in edges]
    while frontier:
        node, dist = frontier.pop(0)
        if node in seen:
            continue
        seen.add(node)
        if dist >= depth:
            continue
        for nxt in sorted(edges.get(node, ())):
            frontier.append((nxt, dist + 1))
    return seen


def dot_quote(value: str) -> str:
    return json.dumps(value)


def rel(path: Path) -> str:
    return str(path.relative_to(REPO_ROOT))


def write_call_graph(functions: dict[str, FunctionDefn], edges: dict[str, set[str]]) -> None:
    focus = reachable(edges, FOCUS_ROOTS, depth=5)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    dot_path = OUT_DIR / "call_graph_focus.dot"
    json_path = OUT_DIR / "call_graph_focus.json"
    svg_path = OUT_DIR / "call_graph_focus.svg"

    lines = [
        "digraph call_graph_focus {",
        '  graph [rankdir=LR, bgcolor="white", label="callosum static call graph focus", labelloc=t];',
        '  node [shape=box, style="rounded,filled", fillcolor="#fff8dc", color="#555555", fontname="Helvetica"];',
        '  edge [color="#555555"];',
    ]
    for q in sorted(focus):
        fn = functions[q]
        color = "#f6d365" if q in FOCUS_ROOTS else "#fff8dc"
        label = f"{q}\\n{rel(fn.path)}:{fn.lineno}"
        lines.append(f"  {dot_quote(q)} [label={dot_quote(label)}, fillcolor={dot_quote(color)}];")
    for src in sorted(focus):
        for dst in sorted(edges.get(src, ())):
            if dst in focus:
                lines.append(f"  {dot_quote(src)} -> {dot_quote(dst)};")
    lines.append("}")
    dot_path.write_text("\n".join(lines) + "\n")

    payload = {
        "generated_from": "scripts/generate_static_graphs.py",
        "scope": "runtime entrypoints plus five static call levels",
        "roots": list(FOCUS_ROOTS),
        "nodes": [
            {
                "id": q,
                "path": rel(functions[q].path),
                "line": functions[q].lineno,
                "end_line": functions[q].end_lineno,
            }
            for q in sorted(focus)
        ],
        "edges": [{"source": s, "target": d} for s in sorted(focus) for d in sorted(edges.get(s, ())) if d in focus],
    }
    json_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    render_svg(dot_path, svg_path)


def cfg_label(node: ast.AST) -> str:
    if isinstance(node, ast.If):
        return f"if {ast.unparse(node.test)[:80]}"
    if isinstance(node, ast.For | ast.AsyncFor):
        return f"for {ast.unparse(node.target)[:40]} in {ast.unparse(node.iter)[:80]}"
    if isinstance(node, ast.While):
        return f"while {ast.unparse(node.test)[:80]}"
    if isinstance(node, ast.Try):
        return "try"
    if isinstance(node, ast.Return):
        return "return"
    if isinstance(node, ast.Raise):
        return "raise"
    if isinstance(node, ast.Expr) and isinstance(node.value, ast.Call):
        return f"call {ast.unparse(node.value.func)[:80]}"
    if isinstance(node, ast.Assign):
        return f"assign {ast.unparse(node.targets[0])[:80]}"
    return type(node).__name__


def cfg_edges_for(fn: FunctionDefn) -> tuple[list[tuple[str, str]], dict[str, str]]:
    edges: list[tuple[str, str]] = []
    labels: dict[str, str] = {}
    counter = 0

    def new_node(node: ast.AST, prefix: str = "n") -> str:
        nonlocal counter
        counter += 1
        node_id = f"{fn.qname}:{prefix}{counter}"
        labels[node_id] = f"{getattr(node, 'lineno', fn.lineno)}: {cfg_label(node)}"
        return node_id

    def walk_block(stmts: list[ast.stmt], incoming: list[str]) -> list[str]:
        exits = incoming
        for stmt in stmts:
            current = new_node(stmt)
            for prev in exits:
                edges.append((prev, current))
            if isinstance(stmt, ast.If):
                body_exits = walk_block(stmt.body, [current])
                else_exits = walk_block(stmt.orelse, [current]) if stmt.orelse else [current]
                exits = body_exits + else_exits
            elif isinstance(stmt, ast.For | ast.AsyncFor | ast.While):
                body_exits = walk_block(stmt.body, [current])
                for tail in body_exits:
                    edges.append((tail, current))
                exits = [current]
            elif isinstance(stmt, ast.Try):
                body_exits = walk_block(stmt.body, [current])
                handler_exits: list[str] = []
                for handler in stmt.handlers:
                    handler_exits.extend(walk_block(handler.body, [current]))
                final_exits = walk_block(stmt.finalbody, body_exits + handler_exits) if stmt.finalbody else []
                exits = final_exits or body_exits + handler_exits or [current]
            elif isinstance(stmt, ast.Return | ast.Raise):
                exits = []
            else:
                exits = [current]
        return exits

    start = f"{fn.qname}:start"
    labels[start] = f"{fn.qname}\\n{rel(fn.path)}:{fn.lineno}"
    body = getattr(fn.node, "body", [])
    walk_block(body, [start])
    return edges, labels


def write_cfgs(functions: dict[str, FunctionDefn]) -> None:
    dot_path = OUT_DIR / "control_flow_focus.dot"
    svg_path = OUT_DIR / "control_flow_focus.svg"
    json_path = OUT_DIR / "control_flow_focus.json"
    all_edges: list[dict[str, str]] = []
    all_nodes: list[dict[str, str]] = []
    lines = [
        "digraph control_flow_focus {",
        '  graph [rankdir=TB, bgcolor="white", label="callosum static control-flow focus", labelloc=t];',
        '  node [shape=box, style="rounded,filled", fillcolor="#e6f3ff", color="#555555", fontname="Helvetica"];',
        '  edge [color="#555555"];',
    ]
    for root in CFG_ROOTS:
        fn = functions.get(root)
        if fn is None:
            continue
        edges, labels = cfg_edges_for(fn)
        cluster_name = root.replace(":", "_").replace(".", "_")
        lines.append(f"  subgraph cluster_{cluster_name} {{")
        lines.append(f"    label={dot_quote(root)};")
        for node_id, label in labels.items():
            lines.append(f"    {dot_quote(node_id)} [label={dot_quote(label)}];")
            all_nodes.append({"id": node_id, "label": label, "function": root})
        for src, dst in edges:
            lines.append(f"    {dot_quote(src)} -> {dot_quote(dst)};")
            all_edges.append({"source": src, "target": dst, "function": root})
        lines.append("  }")
    lines.append("}")
    dot_path.write_text("\n".join(lines) + "\n")
    json_path.write_text(
        json.dumps(
            {
                "generated_from": "scripts/generate_static_graphs.py",
                "scope": "statement-level AST control-flow sketch for dispatch/router roots",
                "roots": list(CFG_ROOTS),
                "nodes": all_nodes,
                "edges": all_edges,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    render_svg(dot_path, svg_path)


def render_svg(dot_path: Path, svg_path: Path) -> None:
    try:
        subprocess.run(["dot", "-Tsvg", str(dot_path), "-o", str(svg_path)], check=True)
    except (FileNotFoundError, subprocess.CalledProcessError):
        # DOT/JSON remain the canonical generated references when graphviz is
        # unavailable; the caller can still inspect or render elsewhere.
        if svg_path.exists():
            svg_path.unlink()


def write_index(functions: dict[str, FunctionDefn], edges: dict[str, set[str]]) -> None:
    focus = reachable(edges, FOCUS_ROOTS, depth=5)
    inbound: dict[str, int] = {q: 0 for q in focus}
    outbound: dict[str, int] = {q: len(edges.get(q, ()) & focus) for q in focus}
    for src in focus:
        for dst in edges.get(src, ()):
            if dst in inbound:
                inbound[dst] += 1
    ranked = sorted(focus, key=lambda q: (inbound[q] + outbound[q], outbound[q], q), reverse=True)
    md = [
        "# Static Architecture Graph Reference",
        "",
        "Generated by `scripts/generate_static_graphs.py` from the Python AST.",
        "These graphs are review aids: dynamic dispatch, closures, FastAPI decorators, and protocol calls are necessarily approximate.",
        "",
        "## Artifacts",
        "",
        "- `call_graph_focus.dot` / `call_graph_focus.svg` / `call_graph_focus.json` - runtime-root call graph, five static call levels deep.",
        "- `control_flow_focus.dot` / `control_flow_focus.svg` / `control_flow_focus.json` - statement-level control-flow sketches for dispatch/router roots.",
        "",
        "## Focus Roots",
        "",
    ]
    md.extend(f"- `{root}`" for root in FOCUS_ROOTS)
    md.extend(["", "## High-Fan-In/Fan-Out Review Targets", ""])
    for q in ranked[:20]:
        fn = functions[q]
        md.append(f"- `{q}` - {rel(fn.path)}:{fn.lineno} (in={inbound[q]}, out={outbound[q]})")
    md.extend(
        [
            "",
            "## Suggested Drill-Down Threads",
            "",
            "- `DRILL-001` Request dispatch and retry: start at `callosum.app:_dispatch_internal`, then `_dispatch_*_with_cell_retry`, `_dispatch_nonstream`, and `_dispatch_stream`.",
            "- `DRILL-002` Router decision correctness: start at `callosum.routing.router:Router.route`, then `extract_features`, `CapabilityFilter.filter`, predictor, and selector nodes.",
            "- `DRILL-003` Observability and state persistence: start at `callosum.app:_log_attempt` and follow usage-log, estimator-finalize, peer-quality, and failure-registry calls.",
            "- `DRILL-004` Startup/background behavior: start at `callosum.__main__:_run_server`, `callosum.app:create_app`, `lifespan`, smoke tester, cooldown prober, catalog refresh, and capability harness scheduling.",
            "",
        ]
    )
    (OUT_DIR / "README.md").write_text("\n".join(md))


def main() -> None:
    functions, imports_by_module = collect()
    edges = build_edges(functions, imports_by_module)
    write_call_graph(functions, edges)
    write_cfgs(functions)
    write_index(functions, edges)


if __name__ == "__main__":
    main()
