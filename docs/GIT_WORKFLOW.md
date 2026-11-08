# Current development branch and publishing

Open **D:\Kinetic-Tracker-main** in VS Code. This is the current upgraded
checkout; the folder name `-main` does not indicate the active Git branch.
Development is on `research/motion-aware-tracking`. The local `main` reference
remains at merged v2 commit `e2d2343`. Old E: folders are separate copies.

```powershell
Set-Location 'D:\Kinetic-Tracker-main'
git branch --show-current
git status
git log -1 --oneline
```

To publish the locally committed refinement as a separate branch:

```powershell
git push -u origin research/motion-aware-tracking
```

This publishes the feature branch without merging it into `main`. Review it on
GitHub using that branch or open a pull request with base `main` and compare
`research/motion-aware-tracking`. Merge only when you decide to. There is no
need to clone again or create a folder named branch2.

For later edits:

```powershell
git status
git diff --stat
git add README.md docs configs include src tests tools CMakeLists.txt .vscode .github results/refinement
git diff --cached --stat
git commit -m "Describe the actual follow-up change"
git push
```

Authentication uses Git Credential Manager's browser login or an appropriate
GitHub token; a GitHub account password cannot authenticate HTTPS Git pushes.
Do not paste tokens into project files, URLs, screenshots or this chat.

Raw data, `.venv`, `runs/`, `work/` and compiler outputs are ignored. Compact
report artifacts and attributed demonstration videos are intentional tracked
outputs. No credentials are required to view the offline report.
