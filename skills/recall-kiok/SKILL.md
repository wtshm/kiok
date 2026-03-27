---
name: recall-kiok
description: Use whenever the user asks about previous sessions, past work, earlier decisions, work history, or needs context from prior conversations. Trigger on questions about what was done, how something was solved, what the reasoning was, what happened last time, or any reference to past session context. Also trigger on session ID patterns (UUIDs). Use when the answer likely comes from a past conversation rather than the current codebase or general knowledge.
---

# kiok recall

Search past Claude Code session conversations stored in kiok's SQLite database with hybrid search (FTS5 trigram + RRF score fusion + time decay).

## When to search

Use kiok when the user's question requires context from past sessions:

- What they worked on (recently or over time)
- How a decision was made or a problem was solved
- The history or reasoning behind current code
- Looking up a specific session by ID
- Finding related past discussions on a topic

If the answer exists in the current codebase (files, git history, CLAUDE.md), prefer those sources. Use kiok when the context is conversational — what was discussed, decided, or explored in past sessions.

## How to search

Extract a short, focused query from the user's message and run:

```bash
kiok recall "<query>" --project $PWD --count 10
```

kiok uses FTS5 trigram matching, so:

- Keep queries short (2-5 keywords work best)
- Try the most specific terms first
- If no results, broaden or rephrase the query
- Run multiple searches with different keywords to cover more ground

## Session ID lookup

When the user provides a session ID (full or prefix), query the database directly for all chunks in that session:

```bash
sqlite3 ~/.kiok/memory.db "SELECT session_id, question, answer FROM chunks WHERE session_id LIKE '<prefix>%' ORDER BY timestamp LIMIT 30;"
```

## Broad questions

For questions like "what did I work on recently", run multiple targeted searches rather than one vague query. Search for project-specific keywords, recent topics, or known areas of work, then synthesize the results.
