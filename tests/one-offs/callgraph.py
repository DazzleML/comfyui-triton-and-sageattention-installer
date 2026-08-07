"""Static call-graph analyzer for the monolithic installer.

Answers the questions that matter before refactoring a ~4900-line single-file
program: who calls what, what is unreachable, and what is the blast radius of
changing a given function.

Resolves intra-class calls (`self.foo()`), qualified calls (`Cls.foo()`), and
module-level function calls. Purely static (ast) -- no imports, no execution.

Usage:
  python tests/one-offs/callgraph.py --dead
      List functions/methods with no callers (dead-code candidates).

  python tests/one-offs/callgraph.py --impact _find_matching_wheel
      Blast radius: everything that transitively reaches this function,
      plus which known entry points are affected.

  python tests/one-offs/callgraph.py --callers _get_wheel_configs
  python tests/one-offs/callgraph.py --callees upgrade_pip_setuptools

  python tests/one-offs/callgraph.py --entry-reach
      For each known entry point, how many functions it can reach.
"""
import argparse
import ast
import sys
from collections import defaultdict
from pathlib import Path

DEFAULT_TARGET = Path(__file__).parent.parent.parent / "comfyui_triton_sageattention.py"

# Entry points that matter for "does my change affect X?" questions.
ENTRY_POINTS = {
    "main": "CLI entry",
    "install": "--install / --upgrade (execution)",
    "preview_changes": "--dryrun (plan display)",
    "plan_installation": "plan construction (shared by dryrun + install)",
    "show_installed": "--show-installed",
    "cleanup_previous_installation": "--cleanup",
}


def build_graph(path: Path):
    """Return (defs, edges) where edges[caller] = set(callee names)."""
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))

    defs = {}      # name -> "Class.name" or "name"
    edges = defaultdict(set)

    class Collector(ast.NodeVisitor):
        def __init__(self):
            self.cls = None
            self.func = None

        def visit_ClassDef(self, node):
            prev, self.cls = self.cls, node.name
            self.generic_visit(node)
            self.cls = prev

        def _visit_func(self, node):
            qual = f"{self.cls}.{node.name}" if self.cls else node.name
            defs[node.name] = qual
            prev, self.func = self.func, node.name
            self.generic_visit(node)
            self.func = prev

        visit_FunctionDef = _visit_func
        visit_AsyncFunctionDef = _visit_func

        def visit_Call(self, node):
            if self.func is not None:
                callee = None
                f = node.func
                # self.foo(...) / obj.foo(...) / Cls.foo(...)
                if isinstance(f, ast.Attribute):
                    callee = f.attr
                # foo(...)
                elif isinstance(f, ast.Name):
                    callee = f.id
                if callee:
                    edges[self.func].add(callee)
            self.generic_visit(node)

    Collector().visit(tree)
    # Keep only edges pointing at functions defined in this file
    known = set(defs)
    pruned = {k: {c for c in v if c in known and c != k} for k, v in edges.items()}
    return defs, pruned


def reverse(edges):
    rev = defaultdict(set)
    for caller, callees in edges.items():
        for c in callees:
            rev[c].add(caller)
    return rev


def transitive(graph, start):
    """All nodes reachable from start (exclusive of start)."""
    seen, stack = set(), [start]
    while stack:
        cur = stack.pop()
        for nxt in graph.get(cur, ()):
            if nxt not in seen:
                seen.add(nxt)
                stack.append(nxt)
    seen.discard(start)
    return seen


def main():
    ap = argparse.ArgumentParser(description="static call-graph analyzer")
    ap.add_argument("--file", type=Path, default=DEFAULT_TARGET)
    ap.add_argument("--dead", action="store_true", help="list uncalled functions")
    ap.add_argument("--impact", metavar="FUNC", help="blast radius of changing FUNC")
    ap.add_argument("--callers", metavar="FUNC", help="direct callers of FUNC")
    ap.add_argument("--callees", metavar="FUNC", help="direct callees of FUNC")
    ap.add_argument("--entry-reach", action="store_true", help="reach of each entry point")
    args = ap.parse_args()

    defs, edges = build_graph(args.file)
    rev = reverse(edges)
    print(f"Parsed {args.file.name}: {len(defs)} functions/methods\n")

    if args.dead:
        # Dunder / entry points are called externally, not from this file.
        dead = sorted(
            n for n in defs
            if not rev.get(n)
            and not n.startswith("__")
            and n not in ENTRY_POINTS
        )
        print(f"UNCALLED within this file ({len(dead)}):")
        for n in dead:
            print(f"  {defs[n]}")
        print("\n(note: abstract methods, CLI entry points and externally-called")
        print(" helpers legitimately appear here -- verify before deleting)")

    if args.callers:
        f = args.callers
        direct = sorted(rev.get(f, ()))
        all_up = sorted(transitive(rev, f))
        print(f"DIRECT callers of {f} ({len(direct)}):")
        for c in direct:
            print(f"  {defs.get(c, c)}")
        print(f"\nTRANSITIVE callers ({len(all_up)}):")
        for c in all_up:
            print(f"  {defs.get(c, c)}")

    if args.callees:
        f = args.callees
        print(f"DIRECT callees of {f}:")
        for c in sorted(edges.get(f, ())):
            print(f"  {defs.get(c, c)}")

    if args.impact:
        f = args.impact
        if f not in defs:
            print(f"ERROR: {f} not found")
            return 1
        upstream = transitive(rev, f)
        direct = sorted(rev.get(f, ()))
        print(f"BLAST RADIUS of {defs[f]}")
        print(f"  direct callers      : {len(direct)}")
        for c in direct:
            print(f"      {defs.get(c, c)}")
        print(f"  transitive callers  : {len(upstream)}")
        affected = [(e, d) for e, d in ENTRY_POINTS.items()
                    if e in upstream or e == f]
        print(f"  ENTRY POINTS AFFECTED ({len(affected)}):")
        for e, d in affected:
            print(f"      {e:32} {d}")
        if not affected:
            print("      (none -- unreachable from known entry points)")

    if args.entry_reach:
        print("ENTRY POINT REACH:")
        for e, d in ENTRY_POINTS.items():
            if e in defs:
                print(f"  {e:32} reaches {len(transitive(edges, e)):3} functions   ({d})")

    return 0


if __name__ == "__main__":
    sys.exit(main())
