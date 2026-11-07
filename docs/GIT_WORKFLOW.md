# Publish the feature branch without merging main

Development is in a separate Git checkout:
`E:\Kinetic-Tracker-dev`, branch `research/compute-aware-tracking`.
The downloaded `E:\Kinetic-Tracker-main` folder remains untouched.

Open the development checkout in VS Code. The lower-left branch indicator
should show `research/compute-aware-tracking`. Ctrl+Shift+B builds and tests.

```powershell
Set-Location E:\Kinetic-Tracker-dev
git branch --show-current
git status
git log -1 --oneline
git push -u origin research/compute-aware-tracking
```

Pushing that branch publishes it without changing `main`. On GitHub, select the
feature branch and copy that URL for reviewers. An optional draft pull request
uses base `main`, compare `research/compute-aware-tracking`; leave it unmerged.
Continue development on the feature branch with ordinary commits and `git push`.

Raw datasets, `.venv`, generated caches, `runs/` and compiler outputs are ignored.
Review `git diff --cached --stat` before future commits. Result videos and compact
tables are intentional tracked artifacts; preserve their attribution files.
Creating a new VS Code folder or downloading a ZIP does not create/push a Git
branch. Until the push command succeeds, the feature branch exists only locally.
