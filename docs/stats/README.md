# Project Statistics Dashboard

Interactive dashboard showing traffic analytics for [ComfyUI Triton & SageAttention Installer](https://github.com/DazzleML/comfyui-triton-and-sageattention-installer).

## How It Works

This is a **static page** — no server-side code, no daily commits. All data is fetched client-side from GitHub Gists at page load time.

### Data Collection

A [GitHub Actions workflow](../../.github/workflows/traffic-badges.yml) runs daily at 3:00 AM UTC and:

1. Fetches cumulative release download counts via the GitHub Releases API
2. Fetches 14-day clone and view data (with uniques) via the GitHub Traffic API
3. Accumulates clone and view counts beyond the 14-day API retention window
4. Counts CI checkout operations to separate organic clones from CI noise
5. Collects repo metadata (stars, forks, issues), referrers, and popular paths
6. Records daily history (rolling 31-day window)
7. Archives monthly snapshots for long-term tracking
8. Updates a public Gist with the latest badge data and state

### Data Sources

| Source | Type | Contents |
|--------|------|----------|
| [Badge Gist](https://gist.github.com/djdarcy/77f23ace7465637447db0a6c79cf46ba) | Public | Current stats, rolling 31-day daily history |
| Archive Gist | Unlisted | Monthly snapshots with full daily breakdowns |
| GitHub Statistics API | Public | Commit activity, code frequency, participation, contributors |

### Tabbed Dashboard

- **Overview** — Multi-metric toggleable chart (views, clones, unique counts, downloads)
- **Installs** — Clone and download charts with organic vs CI breakdown, unique tracker counts
- **Views** — Daily views chart, referrer table with mobile app annotations, popular pages
- **Community** — Star history, forks, issues, daily community trends (dual-axis chart)
- **Dev** — CI audit cards, raw vs organic clone chart, CI checkout breakdown, commit activity, code frequency, participation (maintainer vs community), punch card heat map, contributors list, operational status

### Key Features

- Trailing-zero projection (dashed line for incomplete today's data)
- All-history toggle loading monthly archives
- Client-side GitHub Statistics API with sessionStorage caching and 202-retry logic
- Loading indicators for each stats section
- Mobile referrer detection (Android package names annotated)

## Viewing the Dashboard

The dashboard is hosted on GitHub Pages:

**https://dazzleml.github.io/comfyui-triton-and-sageattention-installer/stats/**

You can also open `index.html` directly in a browser for local testing — the gist CDN has permissive CORS headers.
