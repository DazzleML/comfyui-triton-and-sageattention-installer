"""
One-time backfill of organicUniqueClones and totalOrganicUniqueClones into gist state.json.

Computes organic unique clones for each daily entry using the MIN formula:
  ciUniqueClones = MIN(round(uniqueClones * ciRate), ciRuns)
  organicUniqueClones = uniqueClones - ciUniqueClones

Also seeds totalOrganicUniqueClones and totalCiUniqueClones cumulative fields.

Idempotent: safe to run multiple times (recomputes from current data).

Usage:
    python tests/one-offs/backfill_organic_unique.py                    # dry-run triton (default)
    python tests/one-offs/backfill_organic_unique.py --write             # apply to triton
    python tests/one-offs/backfill_organic_unique.py --repo ncsi         # dry-run NCSI
    python tests/one-offs/backfill_organic_unique.py --repo ncsi --write # apply to NCSI
"""

import json
import math
import subprocess
import sys

CONFIGS = {
    "triton": {
        "gist_id": "77f23ace7465637447db0a6c79cf46ba",
        "label": "ComfyUI Triton & SageAttention",
    },
    "ncsi": {
        "gist_id": "1362078955559665832b72835b309e98",
        "label": "NCSI Resolver",
    },
}


def run_gh(args):
    result = subprocess.run(["gh"] + args, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"ERROR: gh {' '.join(args)}\n{result.stderr}")
        sys.exit(1)
    return result.stdout.strip()


def compute_organic_unique(entry):
    """Compute organic unique clones using MIN(percentage, ciRuns) formula."""
    raw_unique = entry.get("uniqueClones", 0)
    clones = entry.get("clones", 0)
    ci_checkouts = entry.get("ciCheckouts", 0)
    ci_runs = entry.get("ciRuns", 0)

    ci_rate = ci_checkouts / clones if clones > 0 else 0
    ci_unique_by_pct = round(raw_unique * ci_rate)
    ci_unique_ceiling = ci_runs
    ci_unique_clones = min(ci_unique_by_pct, ci_unique_ceiling)
    return max(0, raw_unique - ci_unique_clones), ci_unique_clones


def main():
    write_mode = "--write" in sys.argv

    repo_key = "triton"
    if "--repo" in sys.argv:
        idx = sys.argv.index("--repo")
        if idx + 1 < len(sys.argv):
            repo_key = sys.argv[idx + 1].lower()
    if repo_key not in CONFIGS:
        print(f"Unknown repo: {repo_key}. Options: {', '.join(CONFIGS.keys())}")
        sys.exit(1)

    config = CONFIGS[repo_key]
    gist_id = config["gist_id"]
    label = config["label"]

    print(f"{'=' * 60}")
    print(f"Backfill organicUniqueClones for {label}")
    print(f"Gist: {gist_id}")
    print(f"Mode: {'WRITE' if write_mode else 'DRY RUN'}")
    print(f"{'=' * 60}\n")

    # 1. Fetch current state
    print("Fetching gist state.json...")
    state_raw = run_gh(["api", f"gists/{gist_id}", "--jq", '.files["state.json"].content'])
    state = json.loads(state_raw)

    history = state.get("dailyHistory", [])
    print(f"  dailyHistory: {len(history)} entries")
    print(f"  totalUniqueClones: {state.get('totalUniqueClones', 'MISSING')}")
    print(f"  totalOrganicUniqueClones: {state.get('totalOrganicUniqueClones', 'MISSING')}")

    # 2. Compute organicUniqueClones for each daily entry
    print("\nComputing organicUniqueClones per entry...")
    total_ci_unique = 0
    patched = 0
    for entry in history:
        date_key = entry.get("date", "")[:10]
        organic_unique, ci_unique = compute_organic_unique(entry)
        total_ci_unique += ci_unique

        old_val = entry.get("organicUniqueClones", "MISSING")
        entry["organicUniqueClones"] = organic_unique

        raw_u = entry.get("uniqueClones", 0)
        ci_ck = entry.get("ciCheckouts", 0)
        ci_r = entry.get("ciRuns", 0)

        changed = old_val != organic_unique
        if changed:
            patched += 1

        marker = " *" if changed else ""
        detail = ""
        if ci_unique > 0:
            detail = f" (ciUnique={ci_unique}: pct={round(raw_u * (ci_ck/entry.get('clones',1)) if entry.get('clones',0) > 0 else 0)}, ceil={ci_r})"
        print(f"  {date_key}: unique={raw_u} -> organic={organic_unique}{detail}{marker}")

    # 3. Compute cumulative totals
    # SAFETY: Only set cumulative totals if they don't already exist.
    # Re-running after entries age out of the 31-day window would produce
    # incorrect (too low) totalCiUniqueClones since we'd sum fewer entries.
    # The workflow uses delta accumulation to avoid this — these seeds are
    # only needed for initial setup before the first workflow run.
    total_unique = state.get("totalUniqueClones", 0)
    total_organic_unique = max(0, total_unique - total_ci_unique)

    old_total = state.get("totalOrganicUniqueClones", "MISSING")
    old_ci_total = state.get("totalCiUniqueClones", "MISSING")

    if old_ci_total != "MISSING":
        print(f"\n  WARNING: totalCiUniqueClones already exists ({old_ci_total}).")
        print(f"  Summing from daily window gives {total_ci_unique}.")
        if total_ci_unique < old_ci_total:
            print(f"  Keeping existing value (higher = entries may have aged out).")
            total_ci_unique = old_ci_total
            total_organic_unique = max(0, total_unique - total_ci_unique)
        else:
            print(f"  Using new value (>= existing).")

    state["totalCiUniqueClones"] = total_ci_unique
    state["totalOrganicUniqueClones"] = total_organic_unique

    # 4. Summary
    print(f"\n{'=' * 60}")
    print(f"Summary:")
    print(f"  Daily entries patched: {patched}/{len(history)}")
    print(f"  totalUniqueClones (raw):     {total_unique}")
    print(f"  totalCiUniqueClones:         {old_ci_total} -> {total_ci_unique}")
    print(f"  totalOrganicUniqueClones:    {old_total} -> {total_organic_unique}")

    if not write_mode:
        print(f"\n[DRY RUN] No changes written to gist.")
        print(f"  Run with --write to apply changes.")
        return

    # 5. Write back to gist
    print("\nUpdating gist...")
    payload = {"files": {"state.json": {"content": json.dumps(state, indent=2)}}}
    payload_path = "tests/one-offs/_organic_unique_payload.json"
    with open(payload_path, "w") as f:
        json.dump(payload, f)

    updated_at = run_gh(["api", "--method", "PATCH", f"gists/{gist_id}",
                         "--input", payload_path, "--jq", ".updated_at"])
    print(f"Gist updated at {updated_at}")

    import os
    os.remove(payload_path)
    print("Done.")


if __name__ == "__main__":
    main()
