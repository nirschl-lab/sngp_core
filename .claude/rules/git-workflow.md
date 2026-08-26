# Git workflow

After making code/config/doc changes for a task, stage and commit them without
waiting to be asked. Stage only the specific files touched for that task, by name —
this repo's working tree routinely carries unrelated pre-existing
modifications/deletions (scratch notebooks, WIP scripts, etc.) that must not be swept
in with `git add -A`/`git add .`.
