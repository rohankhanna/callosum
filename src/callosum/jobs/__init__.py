"""Dispatch-launched batch jobs.

Each job is a standalone script runnable as `python -m callosum.jobs.<name>`
that can be submitted through Dispatch for resource-aware scheduling.
Jobs must handle SIGTERM cleanly: checkpoint progress to disk and exit
within the systemd / Dispatch grace period so they can be resumed on
the next GPU-idle window without losing work.
"""
