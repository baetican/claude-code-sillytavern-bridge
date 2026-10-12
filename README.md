# Claude Code ↔ SillyTavern Bridge

A small Flask server that exposes the Claude Code CLI as an OpenAI-compatible endpoint, so SillyTavern, TomoriBot, and other OpenAI-compatible clients can use your Claude Code subscription.

Each request is translated into a `claude -p` call. The bridge handles:

- `/v1/chat/completions`, JSON or SSE (`stream: true`). The CLI doesn't stream tokens, so the full response arrives at the end; empty keepalive chunks are sent every second until then.
- Tool calling, by describing OpenAI tool definitions in the prompt and parsing `[TOOL_CALL]` blocks back into `tool_calls`.
- `response_format` (`json_object` / `json_schema`), passed through to the CLI's `--json-schema`.
- Images: base64 images from the last few messages are passed to the model as image blocks via the CLI's stream-json input.
- CLI session reuse: each reply's CLI session is saved; when the next request continues from that reply, only the new turn is sent with `--resume`, keeping the prompt cache warm. Swipes and recent edits start a fresh session.
- Cancelling the Claude subprocess when the client disconnects.

## Requirements

- Python 3.10+
- [Claude Code CLI](https://docs.anthropic.com/en/docs/claude-code), installed and logged in (`claude --version` works)

## Run

```cmd
run_bridge.bat
```

or `pip install -r requirements.txt` then `python claude_bridge.py`.

The settings page is at `http://localhost:5001/`. It covers model, effort, the bridge system prompt, tool calling, session reuse, debug logging and port. Changes apply to the next request, except the port, which needs a restart. Settings are saved to `bridge_settings.json`.

To see exactly what your client sends, launch with `--debug` (`run_bridge.bat --debug` or `python claude_bridge.py --debug`). Each request then writes its JSON body, pretty-printed, to `logs/request<timestamp>.log`. Base64 image data is shortened to a length note.

## Client setup

In SillyTavern: **API Connections → Chat Completion → Custom (OpenAI-compatible)**, endpoint `http://localhost:5001/v1`, any non-empty API key. The client's model selection is ignored; pick the model on the bridge settings page.

## System prompt

The bridge replaces Claude Code's built-in coding-assistant system prompt with a minimal identity line, editable on the settings page. Everything else (preset, character card, style rules) should come from your SillyTavern preset, which the bridge sends as part of the prompt.

## License

MIT. See `LICENSE`.
