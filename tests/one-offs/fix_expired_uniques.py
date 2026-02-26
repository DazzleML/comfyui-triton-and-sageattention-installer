"""
Remove misleading uniqueClones/uniqueViews=0 from daily entries where
the data was outside the 14-day API window at backfill time.

Entries with clones > 0 but uniqueClones == 0 are suspect — they mean
"we don't know" not "zero unique cloners". Removing these fields lets
the dashboard display a gap rather than a false zero.

Usage:
    python tests/one-offs/fix_expired_uniques.py                    # dry-run triton
    python tests/one-offs/fix_expired_uniques.py --write             # apply to triton
    python tests/one-offs/fix_expired_uniques.py --repo ncsi         # dry-run NCSI
    python tests/one-offs/fix_expired_uniques.py --repo ncsi --write # apply to NCSI
"""

import json
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
    print(f"Fix expired unique data for {label}")
    print(f"Gist: {gist_id}")
    print(f"Mode: {'WRITE' if write_mode else 'DRY RUN'}")
    print(f"{'=' * 60}\n")

    print("Fetching gist state.json...")
    state_raw = run_gh(["api", f"gists/{gist_id}", "--jq", '.files["state.json"].content'])
    state = json.loads(state_raw)

    history = state.get("dailyHistory", [])
    fields_to_check = ["uniqueClones", "uniqueViews", "organicUniqueClones"]
    fixed = 0

    print(f"Checking {len(history)} daily entries...\n")
    for entry in history:
        date_key = entry.get("date", "")[:10]
        clones = entry.get("clones", 0)
        views = entry.get("views", 0)

        # Suspect: has traffic but unique counts are 0
        # This means the data was outside the API window during backfill
        suspect_clones = clones > 0 and entry.get("uniqueClones") == 0
        suspect_views = views > 0 and entry.get("uniqueViews") == 0

        if suspect_clones or suspect_views:
            removed = []
            for field in fields_to_check:
                if field in entry:
                    del entry[field]
                    removed.append(field)
            if removed:
                fixed += 1
                print(f"  {date_key}: clones={clones}, views={views} -> removed {', '.join(removed)}")
        else:
            uq = entry.get("uniqueClones", "absent")
            uv = entry.get("uniqueViews", "absent")
            print(f"  {date_key}: clones={clones}, unique={uq}, views={views}, uniqueViews={uv} (ok)")

    print(f"\nEntries fixed: {fixed}/{len(history)}")

    if fixed == 0:
        print("No changes needed.")
        return

    if not write_mode:
        print(f"\n[DRY RUN] No changes written to gist.")
        print(f"  Run with --write to apply changes.")
        return

    print("\nUpdating gist...")
    payload = {"files": {"state.json": {"content": json.dumps(state, indent=2)}}}
    payload_path = "tests/one-offs/_fix_uniques_payload.json"
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
