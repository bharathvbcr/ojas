# Commits

## Unverified commits

A commit whose message says `UNVERIFIED`, or that says it was committed without compiling or running, must name the GitPulse task that tracks that run. Put the task id (`ft-…` or `gp-…`) in the message body.

That task is where the command and the result are recorded. A later commit that lands the run drops the `UNVERIFIED` mark and cites the task.

Do not rewrite history to remove an unverified commit. Fix it forward on the tracking task, or record on that task why the commit stays.
