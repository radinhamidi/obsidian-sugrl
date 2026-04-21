# SUGRL Research Wiki

Obsidian vault + LLM-maintained research wiki for the SUGRL NeurIPS 2026
project. Consumed as a git submodule by the implementation repo.

See [`CLAUDE.md`](./CLAUDE.md) for the agent schema, directory conventions,
and inter-agent inquiry protocol.

## Layout

- `raw/` — immutable sources (papers, articles, assets) and the
  `raw/inquiries/` channel between Research Agent and Coding Agent.
- `wiki/` — LLM-maintained knowledge base: entities, concepts, sources,
  synthesis, experiments, plus `index.md` and `log.md`.
