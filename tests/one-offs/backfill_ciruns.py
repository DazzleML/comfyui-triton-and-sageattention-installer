"""
One-time backfill of ciRuns field into gist state.json.

Derives ciRuns (count of CI workflow runs with at least one checkout)
from existing byWorkflow.checkoutsPerRun data already stored in the
ciCheckouts map. Patches both the ciCheckouts map entries (adds 'runs'
field) and the dailyHistory entries (adds 'ciRuns' field).

Idempotent: safe to run multiple times. The workflow's own backfill
loop reads from ciCheckouts[date].runs, so patching both locations
keeps them in sync.

Usage:
    python tests/one-offs/backfill_ciruns.py                    # dry-run triton (default)
    python tests/one-offs/backfill_ciruns.py --write             # apply to triton
    python tests/one-offs/backfill_ciruns.py --repo ncsi         # dry-run NCSI
    python tests/one-offs/backfill_ciruns.py --repo ncsi --write # apply to NCSI
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


def count_runs_with_checkouts(ci_entry):
    """Count distinct runs that performed at least one checkout from byWorkflow data."""
    by_workflow = ci_entry.get("byWorkflow", {})
    total_runs = 0
    for wf_name, wf_data in by_workflow.items():
        checkouts_per_run = wf_data.get("checkoutsPerRun", [])
        total_runs += sum(1 for c in checkouts_per_run if c > 0)
    return total_runs


def main():
    write_mode = "--write" in sys.argv

    # Determine which repo to target
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

    print(f"{'=' * 50}")
    print(f"Backfill ciRuns for {label}")
    print(f"Gist: {gist_id}")
    print(f"Mode: {'WRITE' if write_mode else 'DRY RUN'}")
    print(f"{'=' * 50}\n")

    # 1. Fetch current gist state
    print("Fetching gist state.json...")
    state_raw = run_gh(["api", f"gists/{gist_id}", "--jq", '.files["state.json"].content'])
    state = json.loads(state_raw)

    ci_map = state.get("ciCheckouts", {})
    history = state.get("dailyHistory", [])

    print(f"  ciCheckouts map: {len(ci_map)} dates")
    print(f"  dailyHistory: {len(history)} entries")

    # 2. Patch ciCheckouts map entries with 'runs' field
    print("\nPatching ciCheckouts map...")
    map_patched = 0
    for date_str in sorted(ci_map.keys()):
        entry = ci_map[date_str]
        runs = count_runs_with_checkouts(entry)
        old_runs = entry.get("runs", "MISSING")
        entry["runs"] = runs
        if old_runs != runs:
            map_patched += 1
            print(f"  {date_str}: runs={old_runs} -> {runs} (ciCheckouts={entry.get('total', 0)})")
        else:
            print(f"  {date_str}: runs={runs} (unchanged)")

    # 3. Patch dailyHistory entries with 'ciRuns' field
    print("\nPatching dailyHistory entries...")
    daily_patched = 0
    for entry in history:
        date_key = entry.get("date", "")[:10]
        ci_entry = ci_map.get(date_key, {})
        ci_runs = ci_entry.get("runs", 0)
        old_ci_runs = entry.get("ciRuns", "MISSING")
        entry["ciRuns"] = ci_runs
        if old_ci_runs != ci_runs:
            daily_patched += 1
            status = f"ciRuns={old_ci_runs} -> {ci_runs}"
        else:
            status = f"ciRuns={ci_runs} (unchanged)"
        print(f"  {date_key}: {status}, ciCheckouts={entry.get('ciCheckouts', 0)}, clones={entry.get('clones', 0)}")

    # 4. Summary
    print(f"\n{'=' * 50}")
    print(f"Summary:")
    print(f"  ciCheckouts map entries patched: {map_patched}/{len(ci_map)}")
    print(f"  dailyHistory entries patched: {daily_patched}/{len(history)}")
    print()

    # Show dates with non-zero ciRuns
    active_dates = [(d, ci_map[d]) for d in sorted(ci_map.keys()) if ci_map[d].get("runs", 0) > 0]
    if active_dates:
        print("Dates with CI runs:")
        for date_str, ci_entry in active_dates:
            by_wf = ci_entry.get("byWorkflow", {})
            wf_summary = ", ".join(
                f"{wf}: {sum(1 for c in data.get('checkoutsPerRun', []) if c > 0)} runs"
                for wf, data in by_wf.items()
                if sum(1 for c in data.get("checkoutsPerRun", []) if c > 0) > 0
            )
            print(f"  {date_str}: ciRuns={ci_entry['runs']} ({wf_summary})")
    else:
        print("No dates with CI runs found.")

    if not write_mode:
        print(f"\n[DRY RUN] No changes written to gist.")
        print(f"  Run with --write to apply changes.")
        return

    # 5. Write back to gist
    print("\nUpdating gist...")
    payload = {"files": {"state.json": {"content": json.dumps(state, indent=2)}}}
    payload_path = "tests/one-offs/_ciruns_payload.json"
    with open(payload_path, "w") as f:
        json.dump(payload, f)

    updated_at = run_gh(["api", "--method", "PATCH", f"gists/{gist_id}",
                         "--input", payload_path, "--jq", ".updated_at"])
    print(f"Gist updated at {updated_at}")

    # Cleanup payload
    import os
    os.remove(payload_path)
    print("Done.")


if __name__ == "__main__":
    main()
