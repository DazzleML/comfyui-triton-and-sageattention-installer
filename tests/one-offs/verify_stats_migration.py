"""
Verify stats infrastructure migration from NCSI to triton.

Checks:
1. Gist state.json has all required fields (backfill worked)
2. Dashboard HTML has correct config constants (not NCSI values)
3. Workflow YAML has correct CI trigger

Usage:
    python tests/one-offs/verify_stats_migration.py
"""

import json
import os
import re
import subprocess
import sys

GIST_ID = "77f23ace7465637447db0a6c79cf46ba"
EXPECTED_REPO_OWNER = "DazzleML"
EXPECTED_REPO_NAME = "comfyui-triton-and-sageattention-installer"
EXPECTED_CI_WORKFLOW = "CI"

# Paths relative to project root
DASHBOARD_PATH = "docs/stats/index.html"
WORKFLOW_PATH = ".github/workflows/traffic-badges.yml"

passed = 0
failed = 0
errors = []


def check(description, condition, detail=""):
    global passed, failed
    if condition:
        passed += 1
        print(f"  PASS  {description}")
    else:
        failed += 1
        msg = f"  FAIL  {description}"
        if detail:
            msg += f" -- {detail}"
        print(msg)
        errors.append(description)


def find_project_root():
    """Walk up from script location to find .git directory."""
    d = os.path.dirname(os.path.abspath(__file__))
    for _ in range(5):
        if os.path.isdir(os.path.join(d, ".git")):
            return d
        d = os.path.dirname(d)
    return None


def main():
    root = find_project_root()
    if not root:
        print("ERROR: Could not find project root (.git directory)")
        sys.exit(1)
    print(f"Project root: {root}\n")

    # --- 1. Gist state verification ---
    print("=== Gist State ===")
    try:
        result = subprocess.run(
            ["gh", "api", f"gists/{GIST_ID}", "--jq", '.files["state.json"].content'],
            capture_output=True, text=True, timeout=30
        )
        if result.returncode != 0:
            print(f"  SKIP  Could not fetch gist: {result.stderr.strip()}")
        else:
            state = json.loads(result.stdout)

            # Top-level fields
            for field in ("totalClones", "totalViews", "totalDownloads",
                          "totalUniqueClones", "totalUniqueViews",
                          "totalCiCheckouts", "ciCheckouts", "popularPaths",
                          "referrers", "dailyHistory", "stars", "forks"):
                check(f"state has '{field}'", field in state,
                      f"missing from top-level keys: {sorted(state.keys())}")

            check("totalUniqueClones > 0", state.get("totalUniqueClones", 0) > 0,
                  f"got {state.get('totalUniqueClones')}")
            check("totalUniqueViews > 0", state.get("totalUniqueViews", 0) > 0,
                  f"got {state.get('totalUniqueViews')}")
            check("popularPaths is non-empty list",
                  isinstance(state.get("popularPaths"), list) and len(state["popularPaths"]) > 0,
                  f"got {type(state.get('popularPaths'))} len={len(state.get('popularPaths', []))}")

            # Daily history entry fields
            history = state.get("dailyHistory", [])
            check("dailyHistory has entries", len(history) > 0, f"got {len(history)}")
            if history:
                latest = history[-1]
                for field in ("uniqueClones", "uniqueViews", "capturedAt",
                              "ciCheckouts", "organicClones"):
                    check(f"latest entry has '{field}'", field in latest,
                          f"missing from: {sorted(latest.keys())}")
    except Exception as e:
        print(f"  SKIP  Gist check failed: {e}")

    # --- 2. Dashboard HTML verification ---
    print("\n=== Dashboard HTML ===")
    dash_path = os.path.join(root, DASHBOARD_PATH)
    if not os.path.exists(dash_path):
        print(f"  SKIP  {DASHBOARD_PATH} not found")
    else:
        with open(dash_path, "r", encoding="utf-8") as f:
            html = f.read()

        # Correct values present
        check("GIST_RAW_BASE points to triton gist",
              GIST_ID in html,
              "expected gist ID not found in HTML")
        check("REPO_OWNER is DazzleML",
              f"REPO_OWNER = '{EXPECTED_REPO_OWNER}'" in html)
        check("REPO_NAME is correct",
              f"REPO_NAME = '{EXPECTED_REPO_NAME}'" in html)
        check("REPO_CREATED is 2025-07-23",
              "REPO_CREATED = '2025-07-23'" in html)
        check("ARCHIVE_GIST_ID is triton archive",
              "b44b12adb623ae3ac8f1993970ab5266" in html)

        # NCSI values absent
        check("no NCSI gist ID remaining",
              "1362078955559665832b72835b309e98" not in html,
              "NCSI gist ID still present")
        check("no NCSI repo name remaining",
              "Windows-No-Internet-Secured-BUGFIX" not in html,
              "NCSI repo name still present")
        check("no 'NCSI Resolver' in title/banner",
              "NCSI Resolver" not in html)

        # Banner links to correct repo
        check("banner links to DazzleML repo",
              f"github.com/{EXPECTED_REPO_OWNER}/{EXPECTED_REPO_NAME}" in html)

    # --- 3. Workflow YAML verification ---
    print("\n=== Workflow YAML ===")
    wf_path = os.path.join(root, WORKFLOW_PATH)
    if not os.path.exists(wf_path):
        print(f"  SKIP  {WORKFLOW_PATH} not found")
    else:
        with open(wf_path, "r", encoding="utf-8") as f:
            wf = f.read()

        check("workflow_run trigger present",
              "workflow_run:" in wf)
        check(f"triggers on '{EXPECTED_CI_WORKFLOW}' workflow",
              f'["{EXPECTED_CI_WORKFLOW}"]' in wf,
              "expected CI workflow name not found")
        check("no 'Python Tests' trigger remaining",
              "Python Tests" not in wf,
              "NCSI workflow name still present")
        check("uses repo variables for gist IDs",
              "vars.TRAFFIC_GIST_ID" in wf and "vars.TRAFFIC_ARCHIVE_GIST_ID" in wf)
        check("uses TRAFFIC_GIST_TOKEN secret",
              "secrets.TRAFFIC_GIST_TOKEN" in wf)

    # --- Summary ---
    print(f"\n{'='*40}")
    print(f"Results: {passed} passed, {failed} failed")
    if errors:
        print(f"\nFailed checks:")
        for e in errors:
            print(f"  - {e}")
        sys.exit(1)
    else:
        print("All checks passed.")


if __name__ == "__main__":
    main()
