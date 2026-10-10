# GUARDRAILS.md — PhotoVault

Standing rules (see AGENTS.md for full detail — this is the quick-check version):

- No git. No `git init`, `git commit`, `git push`, no GitHub API calls.
- No cloud services, no external servers, no network calls beyond local model downloads.
- Never copy or move photo files — index in place only.
- Never change the approved schema, FAISS index format, or stack choices without asking.
- Never add a new external dependency without asking first.
- 3 failed attempts on the same test failure → stop, log below, move to next task.

## Known stuck points

<!-- Agent: append new entries below in this format -->
<!-- [task-id] what got stuck — what was tried — why it's blocked — what's needed to unblock -->
