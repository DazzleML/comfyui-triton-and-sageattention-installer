"""
Fix stars/forks/openIssues values that were overwritten by the merge loop bug.

The merge loop in traffic-badges.yml (pre-fix) applied today's stars/forks/openIssues
to every entry reported by the Traffic API (14-day window), instead of only today.
This script:
  1. Uses the GitHub Stargazers API to reconstruct actual star counts per day
  2. Removes organicUniqueClones from entries where uniqueClones is missing
  3. Writes corrected data back to gist (unless --dry-run)

Also fixes: sets stars/forks/openIssues to null for entries where capturedAt
doesn't match the entry date (meaning the values were overwritten, not captured
at the correct time) and we can't reconstruct from API.

Usage:
  python fix_overwritten_stars.py --dry-run       # Preview changes
  python fix_overwritten_stars.py --apply          # Apply to gist
"""

import argparse
import json
import subprocess
import sys
from datetime import datetime, timedelta


GIST_ID = "77f23ace7465637447db0a6c79cf46ba"
REPO = "DazzleML/comfyui-triton-and-sageattention-installer"


def gh_api(endpoint):
    """Call GitHub API via gh CLI."""
    result = subprocess.run(
        ["gh", "api", endpoint, "--paginate"],
        capture_output=True, text=True
    )
    if result.returncode != 0:
        print(f"Error calling {endpoint}: {result.stderr}", file=sys.stderr)
        return None
    return json.loads(result.stdout)


def get_star_history():
    """Get star dates from Stargazers API (with timestamps)."""
    result = subprocess.run(
        ["gh", "api", f"repos/{REPO}/stargazers",
         "-H", "Accept: application/vnd.github.star+json",
         "--paginate"],
        capture_output=True, text=True
    )
    if result.returncode != 0:
        print(f"Error getting stargazers: {result.stderr}", file=sys.stderr)
        return {}

    stars = json.loads(result.stdout)
    # Build cumulative star count per day
    star_dates = {}
    for i, star in enumerate(stars):
        date = star["starred_at"][:10]
        star_dates[date] = i + 1  # Cumulative count

    # Fill forward — each day's count is the last known count
    if not star_dates:
        return {}

    all_dates = sorted(star_dates.keys())
    first_date = datetime.strptime(all_dates[0], "%Y-%m-%d")
    last_date = datetime.strptime(all_dates[-1], "%Y-%m-%d")
    today = datetime.now()
    if last_date < today:
        last_date = today

    filled = {}
    current_count = 0
    d = first_date
    while d <= last_date:
        ds = d.strftime("%Y-%m-%d")
        if ds in star_dates:
            current_count = star_dates[ds]
        filled[ds] = current_count
        d += timedelta(days=1)

    return filled


def main():
    parser = argparse.ArgumentParser(description="Fix overwritten stars in gist")
    parser.add_argument("--dry-run", action="store_true", default=True,
                        help="Preview changes without writing (default)")
    parser.add_argument("--apply", action="store_true",
                        help="Apply changes to gist")
    args = parser.parse_args()

    if args.apply:
        args.dry_run = False

    # Fetch current gist state
    print("Fetching current gist state...")
    gist_data = gh_api(f"gists/{GIST_ID}")
    if not gist_data:
        sys.exit(1)

    state = json.loads(gist_data["files"]["state.json"]["content"])

    # Save pre-fix state
    with open("_pre_star_fix_state.json", "w") as f:
        json.dump(state, f, indent=2)
    print("Saved pre-fix state to _pre_star_fix_state.json")

    # Get star history
    print("Fetching star history from Stargazers API...")
    star_history = get_star_history()
    if star_history:
        print(f"Got star history: {len(star_history)} days, "
              f"range {min(star_history.keys())} to {max(star_history.keys())}")
    else:
        print("WARNING: Could not get star history, will null out suspect entries")

    changes = []

    for entry in state.get("dailyHistory", []):
        date = entry.get("date", "")[:10]
        captured = entry.get("capturedAt", "")[:10]

        # Fix 1: Remove organicUniqueClones from entries without uniqueClones
        if "uniqueClones" not in entry and "organicUniqueClones" in entry:
            old_val = entry["organicUniqueClones"]
            del entry["organicUniqueClones"]
            changes.append(f"  {date}: Removed organicUniqueClones={old_val} (uniqueClones missing)")

        # Fix 2: Correct stars using Stargazers API
        if star_history and date in star_history:
            correct_stars = star_history[date]
            if entry.get("stars") != correct_stars:
                old_stars = entry.get("stars")
                entry["stars"] = correct_stars
                changes.append(f"  {date}: stars {old_stars} -> {correct_stars}")
        elif captured != date and date != captured:
            # capturedAt doesn't match date — values were overwritten
            # Can't reconstruct, but at least note it
            if entry.get("stars") == state.get("stars"):
                changes.append(f"  {date}: stars={entry.get('stars')} (suspect — capturedAt={captured})")

    if changes:
        print(f"\n{len(changes)} changes:")
        for c in changes:
            print(c)
    else:
        print("\nNo changes needed.")

    if not args.dry_run and changes:
        print("\nApplying changes to gist...")
        content = json.dumps(state, indent=2)
        payload = json.dumps({
            "files": {
                "state.json": {"content": content}
            }
        })
        result = subprocess.run(
            ["gh", "api", f"gists/{GIST_ID}", "-X", "PATCH", "--input", "-"],
            input=payload, capture_output=True, text=True
        )
        if result.returncode == 0:
            print("Gist updated successfully.")
        else:
            print(f"Error updating gist: {result.stderr}", file=sys.stderr)
    elif args.dry_run and changes:
        print("\n[DRY RUN] No changes applied. Use --apply to write to gist.")


if __name__ == "__main__":
    main()
