"""
One-time backfill of missing stats fields into triton gist state.json.

Adds fields introduced in NCSI v0.7.10-v0.7.11 that the older triton
workflow didn't collect: uniqueClones, uniqueViews, capturedAt per day,
and top-level totalUniqueClones, totalUniqueViews, popularPaths.

Usage:
    python tests/one-offs/backfill_stats_fields.py          # dry-run (default)
    python tests/one-offs/backfill_stats_fields.py --write   # apply changes
"""

import json
import subprocess
import sys
from datetime import datetime, timezone

GIST_ID = "77f23ace7465637447db0a6c79cf46ba"
OWNER = "DazzleML"
REPO = "comfyui-triton-and-sageattention-installer"


def run_gh(args):
    result = subprocess.run(["gh"] + args, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"ERROR: gh {' '.join(args)}\n{result.stderr}")
        sys.exit(1)
    return result.stdout.strip()


def main():
    write_mode = "--write" in sys.argv

    # 1. Fetch current state from gist
    print("Fetching gist state.json...")
    state_raw = run_gh(["api", f"gists/{GIST_ID}", "--jq", '.files["state.json"].content'])
    state = json.loads(state_raw)

    # Save pre-backfill state locally for safety
    with open("tests/one-offs/_pre_backfill_state.json", "w") as f:
        json.dump(state, f, indent=2)
    print("  Saved pre-backfill state to tests/one-offs/_pre_backfill_state.json")

    history = state.get("dailyHistory", [])
    print(f"  dailyHistory: {len(history)} entries")
    print(f"  Existing top-level fields: {sorted(state.keys())}")

    # 2. Fetch 14-day clone data with uniques
    print("\nFetching clone traffic (14-day window)...")
    clones_raw = run_gh(["api", f"repos/{OWNER}/{REPO}/traffic/clones?per=day"])
    clones_data = json.loads(clones_raw)

    # 3. Fetch 14-day view data with uniques
    print("Fetching view traffic (14-day window)...")
    views_raw = run_gh(["api", f"repos/{OWNER}/{REPO}/traffic/views?per=day"])
    views_data = json.loads(views_raw)

    # 4. Fetch popular paths
    print("Fetching popular paths...")
    paths_raw = run_gh(["api", f"repos/{OWNER}/{REPO}/traffic/popular/paths"])
    paths_data = json.loads(paths_raw)

    # 5. Build lookup maps: date -> uniques
    clone_uniques = {}
    for day in clones_data.get("clones", []):
        date_key = day["timestamp"][:10]
        clone_uniques[date_key] = day["uniques"]

    view_uniques = {}
    for day in views_data.get("views", []):
        date_key = day["timestamp"][:10]
        view_uniques[date_key] = day["uniques"]

    print(f"  Clone uniques available: {len(clone_uniques)} days, total {sum(clone_uniques.values())}")
    print(f"  View uniques available:  {len(view_uniques)} days, total {sum(view_uniques.values())}")
    print(f"  Popular paths: {len(paths_data)} entries")

    # 6. Patch dailyHistory entries
    print("\nPatching dailyHistory entries...")
    patched = 0
    for entry in history:
        date_key = entry.get("date", "")[:10]
        changed = False

        # Add uniqueClones (from API if available, else 0)
        if "uniqueClones" not in entry:
            entry["uniqueClones"] = clone_uniques.get(date_key, 0)
            changed = True

        # Add uniqueViews (from API if available, else 0)
        if "uniqueViews" not in entry:
            entry["uniqueViews"] = view_uniques.get(date_key, 0)
            changed = True

        # Add capturedAt (approximate from date + workflow schedule time)
        if "capturedAt" not in entry:
            entry["capturedAt"] = date_key + "T03:00:00Z"
            changed = True

        # Ensure ciCheckouts and organicClones exist (should already be there)
        if "ciCheckouts" not in entry:
            entry["ciCheckouts"] = 0
            changed = True
        if "organicClones" not in entry:
            entry["organicClones"] = entry.get("clones", 0)
            changed = True

        if changed:
            patched += 1
            print(f"  Patched {date_key}: uniqueClones={entry['uniqueClones']}, "
                  f"uniqueViews={entry['uniqueViews']}")

    # 7. Set top-level cumulative unique counts
    # Use API 14-day totals as starting point (best we can do)
    old_uc = state.get("totalUniqueClones", "N/A")
    old_uv = state.get("totalUniqueViews", "N/A")
    total_unique_clones = sum(clone_uniques.values())
    total_unique_views = sum(view_uniques.values())
    state["totalUniqueClones"] = total_unique_clones
    state["totalUniqueViews"] = total_unique_views

    # 8. Add popularPaths (current snapshot)
    state["popularPaths"] = [
        {"path": p["path"], "title": p["title"], "count": p["count"], "uniques": p["uniques"]}
        for p in paths_data
    ]

    # 9. Ensure totalCiCheckouts exists (should already)
    if "totalCiCheckouts" not in state:
        state["totalCiCheckouts"] = 0

    # 10. Ensure ciCheckouts map exists (should already)
    if "ciCheckouts" not in state:
        state["ciCheckouts"] = {}

    # Summary
    print(f"\n{'='*50}")
    print(f"Summary:")
    print(f"  dailyHistory entries patched: {patched}/{len(history)}")
    print(f"  totalUniqueClones: {old_uc} -> {total_unique_clones}")
    print(f"  totalUniqueViews:  {old_uv} -> {total_unique_views}")
    print(f"  popularPaths: {len(state['popularPaths'])} entries")
    print(f"  totalCiCheckouts: {state.get('totalCiCheckouts', 0)}")

    if not write_mode:
        print(f"\n[DRY RUN] No changes written to gist.")
        print(f"  Run with --write to apply changes.")

        # Show what the state would look like
        with open("tests/one-offs/_backfill_preview.json", "w") as f:
            json.dump(state, f, indent=2)
        print(f"  Preview saved to tests/one-offs/_backfill_preview.json")
        return

    # 11. Write back to gist
    print("\nUpdating gist...")
    payload = {"files": {"state.json": {"content": json.dumps(state, indent=2)}}}
    payload_path = "tests/one-offs/_backfill_payload.json"
    with open(payload_path, "w") as f:
        json.dump(payload, f)

    updated_at = run_gh(["api", "--method", "PATCH", f"gists/{GIST_ID}",
                         "--input", payload_path, "--jq", ".updated_at"])
    print(f"Gist updated at {updated_at}")

    # Cleanup payload (keep pre-backfill and preview for reference)
    import os
    os.remove(payload_path)
    print("Done.")


if __name__ == "__main__":
    main()
