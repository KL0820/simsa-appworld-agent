"""Shared subagent utilities — anything used by more than one agent.

io_format.py        canonical per-LLM-call io block (module-swap contract)
instructions.py     instruction provider helpers (no-templating wrapper)
retry.py            transient-failure classification + agent stream retry
skills_retrieval.py held-out top-K skill retrieval (spec §12)
skills_source.py    per-task skill resolution for the skill impls
data_summarizer.py  LLM-facing data previews (variable store, planners)
"""
