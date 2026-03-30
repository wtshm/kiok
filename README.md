# kiok

Memory engine for Claude Code — hybrid search over session history.

kiok archives your Claude Code conversations into a local SQLite database and recalls relevant context on demand. No external APIs, no LLM token consumption.

## How It Works

```
SessionEnd Hook                      Skill (on demand)
     |                                    |
     v                                    v
 kiok save                           kiok recall "<query>"
 Parse JSONL                         -> FTS5 keyword search
 -> Filter noise                     -> Vector similarity search
 -> Split into Q&A chunks            -> RRF score fusion + time decay
 -> Store in SQLite (FTS5)           -> Policy filtering
                                     -> Output to stdout
 kiok embed (background)
 -> Embed (Ruri v3 ONNX)
 -> Update sqlite-vec vectors
```

## Features

- **Hybrid search** — FTS5 trigram + sqlite-vec vector search, fused with Reciprocal Rank Fusion
- **Time decay** — 30-day half-life; recent memories rank higher
- **Japanese-first** — Ruri v3 embeddings + trigram tokenizer
- **Privacy-preserving** — All data stays local, no external API calls
- **Policy control** — `global` / `project` / `isolated` scopes for cross-project memory visibility
- **Web UI** — Built-in browser interface for browsing sessions and searching memories (`kiok view`)
- **Skill integration** — Agent searches past sessions on demand via `recall-kiok` skill

## Requirements

- [ONNX Runtime](https://github.com/microsoft/onnxruntime) (runs Ruri v3 embedding model locally)

## Install

```bash
curl -fsSL https://raw.githubusercontent.com/wtshm/kiok/main/install.sh | sh
```

Then install the skill and run setup:

```bash
npx skills install wtshm/kiok
kiok setup
```

### Build from Source

```bash
cargo build --release
cp target/release/kiok ~/.local/bin/
npx skills install wtshm/kiok
```

## Hook Configuration

`kiok setup` can install these automatically. To configure manually, add to `~/.claude/settings.json`:

```json
{
  "hooks": {
    "SessionEnd": [
      { "hooks": [{ "type": "command", "command": "kiok save --project $PWD &" }] }
    ],
    "PreCompact": [
      { "hooks": [{ "type": "command", "command": "kiok save --project $PWD" }] }
    ]
  }
}
```

## Policy

Control memory visibility per project by creating `<project>/.claude/memory-policy.json`:

```json
{ "scope": "isolated" }
```

| Scope | Own memories visible to others | Receives memories from others |
|-------|-------------------------------|-------------------------------|
| `global` | Yes | Yes |
| `project` | No | Yes (global only) |
| `isolated` | No | No |

Default is `global` when no policy file exists.

## Skill Integration

kiok includes a Claude Code skill (`skills/recall-kiok/`) that lets the agent search past sessions on demand. Install it as a custom skill to enable queries like "what did we do last time?" or "how did we fix that bug?".

## Data Storage

```
~/.kiok/
  memory.db                  # SQLite (FTS5 + sqlite-vec)
  models/
    ruri-v3-310m/            # ONNX embedding model (~600MB)
      model.onnx
      tokenizer.json
```

## Acknowledgments

- [Ruri v3](https://huggingface.co/cl-nagoya/ruri-v3-310m) — Japanese text embedding model by [cl-nagoya](https://huggingface.co/cl-nagoya). kiok uses the [ONNX-exported variant](https://huggingface.co/sirasagi62/ruri-v3-310m-ONNX) by [sirasagi62](https://huggingface.co/sirasagi62).

## License

MIT
