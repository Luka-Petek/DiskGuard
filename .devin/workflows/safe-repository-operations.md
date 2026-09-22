---
description: Safety requirements for repository commands and Git operations
---

# Safe Repository Operations

## Mandatory rules

- **NEVER COMMIT OR PUSH ANYTHING.**
- Do not create Git commits, amend commits, push branches, force-push, merge, rebase, or rewrite history.
- Before running a highly important, potentially dangerous, destructive, expensive, or broadly impactful command, explain the exact command and its impact and ask the user for explicit approval.
- Ask before launching full model retraining, downloading or extracting large datasets, replacing trained model artifacts, overwriting generated experiment results, changing dependency versions, starting containers with persistent effects, or uploading a paper or other external submission.
- Do not interpret approval for one impactful operation as approval for another.
- Prefer read-only inspection and dry-run or separate-output workflows.
- Save redesigned model artifacts and evaluation outputs separately until the user explicitly approves replacement of existing artifacts.
- `git status`, `git diff`, and read-only Git history inspection are allowed without additional confirmation.
