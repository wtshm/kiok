# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Build & Test

```bash
cargo build
cargo test
cargo clippy
```

## Architecture

kiok is a Rust CLI memory engine for Claude Code. It archives session conversations as Q&A chunks in SQLite with hybrid search (FTS5 trigram + sqlite-vec vector search).

Save pipeline: JSONL parse → noise filter → Q&A chunk → SQLite (FTS5)

Embed pipeline (background): load chunks without vectors → Ruri v3 ONNX → update sqlite-vec

Search: FTS5 keyword + vector similarity → RRF score fusion + time decay (30-day half-life) → policy filtering

### Commands

| Command | Purpose |
|---------|---------|
| `save` | Parse JSONL session, chunk into Q&A pairs, store in SQLite |
| `recall` | Hybrid search (FTS5 + RRF + time decay + policy filtering) |
| `embed` | Background embedding for chunks without vectors |
| `view` | Launch Axum web UI on port 8718 to browse memories |
| `import` | Bulk import existing Claude Code sessions |
| `stats` | Show database statistics |
| `setup` | Interactive wizard: download model, init DB, configure hooks, import, embed |

## Key Conventions

- Write all code and comments in English
- Use `anyhow::Result<T>` for error handling throughout
- Embedding is optional — gracefully fall back to FTS-only search when model is unavailable
- `unsafe` in db.rs is required for sqlite-vec FFI registration

## Data Paths

- Database: `~/.kiok/memory.db`
- Models: `~/.kiok/models/ruri-v3-310m/`
- Policy: `<project>/.claude/memory-policy.json`
- Viewer assets: `src/viewer/` (embedded via `include_str!`)
- Skill: `skills/recall-kiok/SKILL.md`
