# Claude Code ↔ SillyTavern Bridge

A small Flask server that exposes the Claude Code CLI as an OpenAI-compatible endpoint, so SillyTavern, TomoriBot, and other OpenAI-compatible clients can use your Claude Code subscription.

Each request is translated into a `claude -p` call. The bridge handles:

- `/v1/chat/completions`, JSON or SSE (`stream: true`). The CLI doesn't stream tokens, so the full response arrives at the end; empty keepalive chunks are sent every second until then.
- Tool calling, by describing OpenAI tool definitions in the prompt and parsing `[TOOL_CALL]` blocks back into `tool_calls`.
- `response_format` (`json_object` / `json_schema`), passed through to the CLI's `--json-schema`.
- Images: shared images are described in a separate pass so the main turn keeps its formatting.
- CLI session reuse: follow-up turns resume the previous CLI session and send only the new message, keeping the prompt cache warm. Swipes and edits start a fresh session.
- Cancelling the Claude subprocess when the client disconnects.

## Requirements

- Python 3.10+
- [Claude Code CLI](https://docs.anthropic.com/en/docs/claude-code), installed and logged in (`claude --version` works)
- Optional: `pip install pillow` for animated-GIF frame extraction

## Run

```cmd
run_bridge.bat
```

or `pip install -r requirements.txt` then `python claude_bridge.py`.

The settings page is at `http://localhost:5001/`. It covers model, effort, the bridge system prompt, tool calling, session reuse, debug logging and port. Changes apply to the next request, except the port, which needs a restart. Settings are saved to `bridge_settings.json`.

## Client setup

In SillyTavern: **API Connections → Chat Completion → Custom (OpenAI-compatible)**, endpoint `http://localhost:5001/v1`, any non-empty API key. The client's model selection is ignored; pick the model on the bridge settings page.

## System prompt

The bridge replaces Claude Code's built-in coding-assistant system prompt with a minimal identity line, editable on the settings page. Everything else (preset, character card, style rules) should come from your SillyTavern preset, which the bridge sends as part of the prompt.

## License

MIT. See `LICENSE`.
