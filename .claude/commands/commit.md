---
allowed-tools: Bash(git fetch:*), Bash(git status:*), Bash(git diff:*), Bash(git log:*), Bash(git rev-list:*), Bash(git reset:*), Bash(git pull:*), Bash(git add:*), Bash(git commit:*), Bash(git push:*)
description: Commit local changes anonymously (user.name=anonymous) with an English conventional-commit message. Rebases onto remote first so local changes land after upstream changes. Squashes all unpushed local commits into one clean commit. Excludes generated files and build artifacts.
---

## Context

- Current branch: !`git branch --show-current`
- Remote sync status: !`git fetch --quiet 2>/dev/null; git status -sb`
- All uncommitted changes (staged + unstaged): !`git diff HEAD`
- Commits ahead of remote: !`git rev-list @{u}..HEAD --count 2>/dev/null || echo 0`
- Unpushed commit messages: !`git log @{u}..HEAD --oneline 2>/dev/null || echo "(none)"`
- Recent pushed commits (for message style reference): !`git log @{u} --oneline -6 2>/dev/null`

## Step 1 — Rebase onto remote

Check the remote sync status from Context above.

If the remote has commits the local branch doesn't (`BEHIND > 0`):
```
git pull --rebase
```
If rebase hits a conflict, stop immediately and report the conflict. Do not continue.

## Step 2 — Squash unpushed commits

If there are **multiple** unpushed commits (AHEAD > 1 from Context above), squash them all back into staged changes so we can make one clean commit:
```
git reset --soft @{u}
```
This collapses every unpushed commit into the index without losing any work. Skip this step if AHEAD ≤ 1.

## Step 3 — Stage changes

Include **all** modified tracked files plus relevant untracked files. Do NOT use `git add -A` or `git add .` — add each file explicitly by name.

Skip (never stage) these categories:
- Secrets: `.env`, `*.key`, `*.pem`, `*secret*`, `credentials*`
- Python build artifacts: `*.pyc`, `__pycache__/`, `*.egg-info/`, `dist/`, `build/`, `_version.py`
- Editor/OS noise: `.DS_Store`, `*.swp`, `Thumbs.db`
- Test/coverage output: `.coverage`, `htmlcov/`, `.pytest_cache/`

If a skipped file looks important, warn the user with one line but continue.

## Step 4 — Write commit message

Analyse what changed across all staged content (which may now include previously separate commits). Write a **single** English commit message that captures the full logical change:

- Subject: `<type>(<scope>): <description>` — ≤ 72 chars, imperative mood
- Types: `feat` `fix` `refactor` `chore` `docs` `test` `perf`
- Scope: optional, lowercase module name
- Body: bullet points for non-obvious details (optional)
- If the squash in Step 2 merged multiple unrelated changes, pick the dominant type and mention the rest in the body

Do NOT mention iteration steps, debugging attempts, or "WIP" in the message.

## Step 5 — Commit (anonymous, English, co-authored)

Use exactly this pattern — the `-c` flags and `-m "$(cat <<'EOF'` must be on the same line:

```
git -c user.name='anonymous' -c user.email='anonymous@localhost' commit -m "$(cat <<'EOF'
<type>(<scope>): <short description>

<optional body>

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>
EOF
)"
```

## Step 6 — Push

```
git push origin $(git branch --show-current)
```

If push is rejected (remote moved since Step 1), run `git pull --rebase` once and retry push. If still rejected, report the error and stop — never force-push.

## Output

On success: `✓ <hash> "<subject>" → pushed to origin/<branch>`
On failure: show exact git error and stop.
