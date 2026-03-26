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

Pipeline: JSONL parse → noise filter → Q&A chunk → embed (ONNX/Ruri v3) → SQLite (FTS5 + sqlite-vec)

Search: FTS5 keyword + vector similarity → RRF score fusion + time decay (30-day half-life) → policy filtering

## Key Conventions

- Write all code and comments in English
- Use `anyhow::Result<T>` for error handling throughout
- Embedding is optional — gracefully fall back to FTS-only search when model is unavailable
- `unsafe` in db.rs is required for sqlite-vec FFI registration

## Data Paths

- Database: `~/.kiok/memory.db`
- Models: `~/.kiok/models/ruri-v3-310m/`
- Policy: `<project>/.claude/memory-policy.json`
