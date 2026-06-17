"""Detect CUDA-minor gaps between PyTorch and SageAttention wheel availability.

A "gap" is a CUDA minor that PyTorch publishes a Windows wheel for, but
SageAttention does NOT build a wheel for. For each gap, the only SAFE alias
target is the nearest LOWER CUDA minor in the same major that SA does build:
CUDA minor-version compatibility is forward (a newer runtime runs code built
for an older minor), so aliasing DOWN is safe; aliasing UP is not.

This is a read-only reporting tool -- it makes no changes. Re-run it whenever
PyTorch or SageAttention ship new releases to see if a new fillable gap appeared.

Usage:
  python tests/one-offs/detect_cuda_wheel_gaps.py
  python tests/one-offs/detect_cuda_wheel_gaps.py --json

Candidate CUDA minors to probe are listed in PYTORCH_CANDIDATES; extend it as
new CUDA toolkits appear.
"""
import argparse
import json
import re
import sys
import urllib.error
import urllib.request

# CUDA minors to probe on the PyTorch index. Extend as new toolkits ship.
PYTORCH_CANDIDATES = ["118", "121", "124", "126", "127", "128", "129", "130", "131", "132"]

PYTORCH_INDEX = "https://download.pytorch.org/whl/cu{code}/torch/"
SA_RELEASES_API = "https://api.github.com/repos/woct0rdho/SageAttention/releases"

# Aliases already wired into the installer (keep in sync with CUDA_WHEEL_ALIASES).
KNOWN_ALIASES = {"129": "128", "132": "130"}


def http_ok(url: str) -> bool:
    req = urllib.request.Request(url, method="HEAD")
    try:
        with urllib.request.urlopen(req, timeout=15) as r:
            return r.status == 200
    except urllib.error.HTTPError as e:
        return e.code == 200
    except Exception:
        return False


def pytorch_has_windows_wheels(code: str) -> bool:
    """True if PyTorch's cuXXX index has at least one win_amd64 torch wheel."""
    url = PYTORCH_INDEX.format(code=code)
    try:
        with urllib.request.urlopen(url, timeout=20) as r:
            if r.status != 200:
                return False
            body = r.read().decode("utf-8", "replace")
    except Exception:
        return False
    return "win_amd64" in body


def sa_cuda_codes() -> set:
    """CUDA codes SageAttention publishes wheels for (any release)."""
    req = urllib.request.Request(SA_RELEASES_API, headers={"Accept": "application/vnd.github+json"})
    try:
        with urllib.request.urlopen(req, timeout=20) as r:
            data = json.loads(r.read().decode("utf-8", "replace"))
    except Exception as e:
        print(f"ERROR: could not query SageAttention releases: {e}", file=sys.stderr)
        return set()
    codes = set()
    for rel in data:
        for asset in rel.get("assets", []):
            for m in re.findall(r"cu(\d+)", asset.get("name", "")):
                codes.add(m)
    return codes


def nearest_lower_same_major(gap: str, sa_codes: set):
    """Nearest LOWER CUDA minor in the same major that SA builds (safe target)."""
    gap_major, gap_minor = gap[:2], int(gap)
    candidates = [c for c in sa_codes if c[:2] == gap_major and int(c) < gap_minor]
    return max(candidates, key=int) if candidates else None


def main():
    ap = argparse.ArgumentParser(description="Detect CUDA-minor wheel gaps")
    ap.add_argument("--json", action="store_true", help="emit JSON")
    args = ap.parse_args()

    sa_codes = sa_cuda_codes()
    pt_codes = [c for c in PYTORCH_CANDIDATES if pytorch_has_windows_wheels(c)]

    rows = []
    for code in pt_codes:
        if code in sa_codes:
            rows.append({"cuda": code, "sa_wheel": True, "gap": False})
            continue
        target = nearest_lower_same_major(code, sa_codes)
        rows.append({
            "cuda": code,
            "sa_wheel": False,
            "gap": True,
            "safe_alias_target": target,
            "already_aliased": KNOWN_ALIASES.get(code),
        })

    if args.json:
        print(json.dumps({"pytorch": pt_codes, "sageattention": sorted(sa_codes), "rows": rows}, indent=2))
        return 0

    print(f"PyTorch Windows CUDA minors : {', '.join(pt_codes)}")
    print(f"SageAttention CUDA minors   : {', '.join(sorted(sa_codes))}")
    print()
    print(f"{'CUDA':<6} {'SA wheel':<9} {'gap':<5} {'safe target':<12} {'aliased?'}")
    print("-" * 50)
    actionable = []
    for r in rows:
        if not r["gap"]:
            print(f"{r['cuda']:<6} {'yes':<9} {'-':<5} {'-':<12} -")
        else:
            tgt = r["safe_alias_target"] or "(none - up-alias unsafe)"
            al = r["already_aliased"] or "NO"
            print(f"{r['cuda']:<6} {'no':<9} {'YES':<5} {tgt:<12} {al}")
            if r["safe_alias_target"] and not r["already_aliased"]:
                actionable.append(r)

    print()
    if actionable:
        print("FILLABLE GAPS NOT YET ALIASED (candidates for testing):")
        for r in actionable:
            print(f"  cu{r['cuda']} -> cu{r['safe_alias_target']}  (verify end-to-end, then add to CUDA_WHEEL_ALIASES)")
    else:
        print("No new fillable gaps. All safe-to-alias CUDA gaps are already covered.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
