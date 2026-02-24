#!/usr/bin/env python3
"""
Link GitHub issues as sub-issues (parent/child relationships).

Usage:
    # Link a single child to a parent
    python scripts/gh_sub_issues.py link 47 48

    # Link multiple children to a parent
    python scripts/gh_sub_issues.py link 47 48 49 50

    # Remove a sub-issue relationship
    python scripts/gh_sub_issues.py unlink 47 48

    # List sub-issues of a parent
    python scripts/gh_sub_issues.py list 47

    # Explicit repo (works from anywhere)
    python scripts/gh_sub_issues.py link 47 48 --repo owner/name
"""

import argparse
import json
import subprocess
import sys


def run_gh(args: list[str]) -> dict:
    """Run a gh CLI command and return parsed JSON."""
    result = subprocess.run(
        ["gh"] + args,
        capture_output=True, text=True
    )
    if result.returncode != 0:
        print(f"Error: {result.stderr.strip()}", file=sys.stderr)
        sys.exit(1)
    return json.loads(result.stdout) if result.stdout.strip() else {}


def get_issue_id(number: int, repo: str | None = None) -> str:
    """Get the node ID for a GitHub issue number."""
    cmd = ["issue", "view", str(number), "--json", "id", "--jq", ".id"]
    if repo:
        cmd += ["--repo", repo]
    result = subprocess.run(
        ["gh"] + cmd,
        capture_output=True, text=True
    )
    if result.returncode != 0:
        print(f"Error getting issue #{number}: {result.stderr.strip()}", file=sys.stderr)
        sys.exit(1)
    return result.stdout.strip()


def link_sub_issue(parent_num: int, child_num: int, repo: str | None = None) -> None:
    """Link a child issue as a sub-issue of a parent."""
    parent_id = get_issue_id(parent_num, repo)
    child_id = get_issue_id(child_num, repo)

    query = f'''mutation {{
        addSubIssue(input: {{issueId: "{parent_id}", subIssueId: "{child_id}"}}) {{
            issue {{ title number }}
            subIssue {{ title number }}
        }}
    }}'''

    result = subprocess.run(
        ["gh", "api", "graphql", "-H", "GraphQL-Features: sub_issues", "-f", f"query={query}"],
        capture_output=True, text=True
    )
    if result.returncode != 0:
        print(f"Error linking #{child_num} to #{parent_num}: {result.stderr.strip()}", file=sys.stderr)
        return

    data = json.loads(result.stdout)
    if "errors" in data:
        print(f"  Error: {data['errors'][0]['message']}", file=sys.stderr)
    else:
        sub = data["data"]["addSubIssue"]
        print(f"  Linked #{sub['subIssue']['number']} ({sub['subIssue']['title']}) -> #{sub['issue']['number']}")


def unlink_sub_issue(parent_num: int, child_num: int, repo: str | None = None) -> None:
    """Remove a sub-issue relationship."""
    parent_id = get_issue_id(parent_num, repo)
    child_id = get_issue_id(child_num, repo)

    query = f'''mutation {{
        removeSubIssue(input: {{issueId: "{parent_id}", subIssueId: "{child_id}"}}) {{
            issue {{ title number }}
            subIssue {{ title number }}
        }}
    }}'''

    result = subprocess.run(
        ["gh", "api", "graphql", "-H", "GraphQL-Features: sub_issues", "-f", f"query={query}"],
        capture_output=True, text=True
    )
    if result.returncode != 0:
        print(f"Error unlinking #{child_num} from #{parent_num}: {result.stderr.strip()}", file=sys.stderr)
        return

    data = json.loads(result.stdout)
    if "errors" in data:
        print(f"  Error: {data['errors'][0]['message']}", file=sys.stderr)
    else:
        print(f"  Unlinked #{child_num} from #{parent_num}")


def list_sub_issues(parent_num: int, repo: str | None = None) -> None:
    """List sub-issues of a parent issue."""
    repo_args = []
    if repo:
        owner, name = repo.split("/")
    else:
        # Auto-detect from git remote
        result = subprocess.run(
            ["gh", "repo", "view", "--json", "owner,name"],
            capture_output=True, text=True
        )
        if result.returncode != 0:
            print("Error detecting repo. Use --repo owner/name", file=sys.stderr)
            sys.exit(1)
        info = json.loads(result.stdout)
        owner = info["owner"]["login"]
        name = info["name"]

    query = f'''query {{
        repository(owner: "{owner}", name: "{name}") {{
            issue(number: {parent_num}) {{
                title
                number
                subIssues(first: 50) {{
                    nodes {{ number title state }}
                }}
                subIssuesSummary {{
                    total
                    completed
                    percentCompleted
                }}
            }}
        }}
    }}'''

    result = subprocess.run(
        ["gh", "api", "graphql", "-H", "GraphQL-Features: sub_issues", "-f", f"query={query}"],
        capture_output=True, text=True
    )
    if result.returncode != 0:
        print(f"Error: {result.stderr.strip()}", file=sys.stderr)
        sys.exit(1)

    data = json.loads(result.stdout)
    if "errors" in data:
        print(f"Error: {data['errors'][0]['message']}", file=sys.stderr)
        sys.exit(1)

    issue = data["data"]["repository"]["issue"]
    summary = issue["subIssuesSummary"]
    subs = issue["subIssues"]["nodes"]

    print(f"#{issue['number']}: {issue['title']}")
    print(f"Sub-issues: {summary['completed']}/{summary['total']} complete ({summary['percentCompleted']}%)")
    print()

    for sub in subs:
        marker = "x" if sub["state"] == "CLOSED" else " "
        print(f"  [{marker}] #{sub['number']}: {sub['title']} ({sub['state']})")

    if not subs:
        print("  (no sub-issues)")


def main():
    parser = argparse.ArgumentParser(
        description="Manage GitHub sub-issue relationships",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s link 47 48 49 50     Link #48, #49, #50 as sub-issues of #47
  %(prog)s unlink 47 48         Remove #48 as sub-issue of #47
  %(prog)s list 47              Show sub-issues of #47
        """
    )
    parser.add_argument("action", choices=["link", "unlink", "list"],
                        help="Action to perform")
    parser.add_argument("parent", type=int,
                        help="Parent issue number")
    parser.add_argument("children", type=int, nargs="*",
                        help="Child issue number(s)")
    parser.add_argument("--repo", metavar="OWNER/NAME",
                        help="Repository (default: auto-detect from git)")

    args = parser.parse_args()

    if args.action == "list":
        list_sub_issues(args.parent, args.repo)
    elif args.action == "link":
        if not args.children:
            parser.error("link requires at least one child issue number")
        for child in args.children:
            link_sub_issue(args.parent, child, args.repo)
    elif args.action == "unlink":
        if not args.children:
            parser.error("unlink requires at least one child issue number")
        for child in args.children:
            unlink_sub_issue(args.parent, child, args.repo)


if __name__ == "__main__":
    main()
