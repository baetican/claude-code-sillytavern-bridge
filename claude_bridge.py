"""
Claude Code CLI -> OpenAI-compatible bridge.

Exposes /v1/chat/completions for SillyTavern, TomoriBot, and other
OpenAI-compatible clients, and runs each request through `claude -p`.
Settings page at http://localhost:<port>/.
"""

import base64
import binascii
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import uuid
from datetime import datetime

from flask import Flask, Response, jsonify, render_template, request
from flask_cors import CORS

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


# =============================================================================
# LOGGING
# =============================================================================

class Colors:
    RESET = "\033[0m"
    BOLD = "\033[1m"
    DIM = "\033[2m"
    RED = "\033[91m"
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    CYAN = "\033[96m"
    WHITE = "\033[97m"
    GRAY = "\033[90m"


_LEVEL_STYLE = {
    "ERROR": (Colors.RED, "✗"),
    "WARN": (Colors.YELLOW, "⚠"),
    "SUCCESS": (Colors.GREEN, "✓"),
}


def log(message: str, level: str = "INFO"):
    color, icon = _LEVEL_STYLE.get(level, (Colors.GRAY if message.startswith("  ") else Colors.WHITE, "│"))
    ts = f"{Colors.DIM}[{datetime.now().strftime('%H:%M:%S')}]{Colors.RESET}"
    print(f"{ts} {color}{icon} {message}{Colors.RESET}")
    sys.stdout.flush()


def log_box(title: str, stats: dict):
    width = 52
    title_str = f" {title.upper()} "
    print()
    print(f"{Colors.CYAN}{Colors.BOLD}┌─{title_str}{'─' * max(1, width - 3 - len(title_str))}┐{Colors.RESET}")
    for key, value in stats.items():
        if isinstance(value, int) and value > 999:
            value = f"{value:,}"
        key_str, val_str = str(key), str(value)
        pad = max(1, width - 4 - len(key_str) - len(val_str))
        print(f"{Colors.CYAN}│{Colors.RESET} {key_str}{' ' * pad}{Colors.GREEN}{val_str}{Colors.RESET} {Colors.CYAN}│{Colors.RESET}")
    print(f"{Colors.CYAN}{Colors.BOLD}└{'─' * (width - 2)}┘{Colors.RESET}")


# =============================================================================
# REQUEST LOGGING (--debug)
# =============================================================================
# With --debug, every request body the bridge receives is written as JSON to
# its own file in logs/, so it's possible to see exactly what the client sent.

DEBUG_REQUESTS = "--debug" in sys.argv[1:]
LOGS_DIR = os.path.join(BASE_DIR, "logs")
_B64_DATA_RE = re.compile(r"(data:[\w/.+-]+;base64,)[A-Za-z0-9+/=]+")


def write_request_log(data: dict):
    # Microseconds keep concurrent requests from sharing a file.
    path = os.path.join(LOGS_DIR, f"request{datetime.now().strftime('%Y%m%d-%H%M%S-%f')}.log")
    body = json.dumps(data, indent=2, ensure_ascii=False)
    # Base64 images run to megabytes and are unreadable anyway.
    body = _B64_DATA_RE.sub(lambda m: f"{m.group(1)}<{len(m.group(0)) - len(m.group(1)):,} chars omitted>", body)
    try:
        os.makedirs(LOGS_DIR, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            f.write(body + "\n")
    except OSError as e:
        log(f"Could not write request log {path}: {e}", "WARN")


# =============================================================================
# CLAUDE CLI RESOLUTION
# =============================================================================

def _find_claude_exe():
    # npm installs drop claude.cmd into %APPDATA%\npm, which isn't always on
    # PATH, and Popen won't resolve .cmd wrappers without a full path.
    found = shutil.which("claude")
    if found:
        return found
    if sys.platform == "win32":
        candidates = [
            os.path.expandvars(r"%APPDATA%\npm\claude.cmd"),
            os.path.expandvars(r"%APPDATA%\npm\claude.exe"),
            os.path.expandvars(r"%USERPROFILE%\.local\bin\claude.exe"),
            os.path.expandvars(r"%USERPROFILE%\.local\bin\claude.cmd"),
            os.path.expandvars(r"%LOCALAPPDATA%\Programs\claude\claude.exe"),
            r"C:\Program Files\nodejs\claude.cmd",
        ]
    else:
        candidates = [
            os.path.expanduser("~/.local/bin/claude"),
            "/usr/local/bin/claude",
            "/opt/homebrew/bin/claude",
            "/usr/bin/claude",
        ]
    return next((p for p in candidates if os.path.isfile(p)), None)


CLAUDE_EXE = _find_claude_exe()
if CLAUDE_EXE is None:
    log("Claude CLI not found on PATH or in common install locations. "
        "Requests will fail until `claude --version` works in a new terminal.", "ERROR")
    CLAUDE_EXE = "claude"


# =============================================================================
# SETTINGS
# =============================================================================

DEFAULT_SYSTEM_PROMPT = (
    "You are a collaborative fiction writer. Follow the instructions, character "
    "card, and style guidance provided in the conversation."
)

EFFORT_LEVELS = ("low", "medium", "high", "xhigh", "max")

DEFAULT_SETTINGS = {
    "model": "claude-opus-5-5",
    "effort_level": "high",
    "system_prompt": "",
    "tool_calling_enabled": True,
    "cli_session_reuse": True,
    "debug_output": True,
    "bridge_port": 5001,
}

SETTINGS_FILE = os.path.join(BASE_DIR, "bridge_settings.json")
runtime_settings = dict(DEFAULT_SETTINGS)


def load_settings():
    if not os.path.exists(SETTINGS_FILE):
        return
    try:
        with open(SETTINGS_FILE, "r", encoding="utf-8") as f:
            saved = json.load(f)
    except (OSError, json.JSONDecodeError) as e:
        log(f"Failed to load {SETTINGS_FILE}: {e}", "ERROR")
        return
    for key, default in DEFAULT_SETTINGS.items():
        if key in saved and type(saved[key]) is type(default):
            runtime_settings[key] = saved[key]
    if runtime_settings["effort_level"] not in EFFORT_LEVELS:
        runtime_settings["effort_level"] = DEFAULT_SETTINGS["effort_level"]


def save_settings():
    try:
        with open(SETTINGS_FILE, "w", encoding="utf-8") as f:
            json.dump(runtime_settings, f, indent=2)
    except OSError as e:
        log(f"Failed to save {SETTINGS_FILE}: {e}", "ERROR")


load_settings()


def debug(message: str, level: str = "INFO"):
    if runtime_settings["debug_output"]:
        log(message, level)


# =============================================================================
# MODEL LIST
# =============================================================================
# The settings page's model picker comes from the local CLI: the initialize
# control request (what the Agent SDK's supportedModels() uses) returns the
# models this CLI version and account can run. It isn't a documented CLI
# interface, so a small built-in list stands in if it fails.

FALLBACK_MODELS = [
    {"id": "claude-opus-5-5", "name": "Opus 5.5", "effort_levels": list(EFFORT_LEVELS)},
    {"id": "claude-sonnet-5-5", "name": "Sonnet 5.5", "effort_levels": list(EFFORT_LEVELS)},
    {"id": "claude-fable-5-1", "name": "Fable 5.1", "effort_levels": list(EFFORT_LEVELS)},
]

_models_cache = None
_models_lock = threading.Lock()


def fetch_cli_models() -> list:
    request_line = json.dumps({"type": "control_request", "request_id": "models",
                               "request": {"subtype": "initialize"}}) + "\n"
    proc = subprocess.run(
        [CLAUDE_EXE, "-p", "--input-format", "stream-json", "--output-format", "stream-json",
         "--verbose", "--safe-mode"],
        input=request_line, capture_output=True, text=True, encoding="utf-8", timeout=60, cwd=BASE_DIR,
    )
    for line in proc.stdout.splitlines():
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        if event.get("type") != "control_response":
            continue
        entries = event["response"]["response"]["models"]
        # The newest models only appear as aliases ("opus", "sonnet", ...).
        # Each is listed under the ID it resolves to, since an alias moves to a
        # new model on release and a saved choice would change underneath the
        # user. "default" goes last so its "Default (recommended)" name loses
        # to the family alias that resolves to the same model.
        models = {}
        for m in sorted(entries, key=lambda m: m.get("value") == "default"):
            model_id = m.get("resolvedModel") or m["value"]
            if model_id in models:
                continue
            name = m.get("displayName") if m.get("value") != "default" else None
            models[model_id] = {
                "id": model_id, "name": name or model_id,
                "description": m.get("description", ""),
                "effort_levels": m.get("supportedEffortLevels", []),
            }
        return list(models.values())
    raise RuntimeError(f"no initialize response (exit {proc.returncode}): {proc.stderr.strip()[:200]}")


def get_models(refresh: bool = False) -> dict:
    global _models_cache
    with _models_lock:
        if _models_cache is None or refresh:
            try:
                models = fetch_cli_models()
                if not models:
                    raise RuntimeError("CLI returned an empty model list")
                _models_cache = {"source": "cli", "models": models}
                debug(f"Loaded {len(models)} models from the CLI")
            except (OSError, subprocess.SubprocessError, RuntimeError, KeyError, TypeError) as e:
                log(f"Could not load models from the CLI, using built-in list: {e}", "WARN")
                _models_cache = {"source": "fallback", "error": str(e), "models": FALLBACK_MODELS}
        return {**_models_cache, "default": DEFAULT_SETTINGS["model"]}


# =============================================================================
# TOOL CALLING
# =============================================================================
# The CLI can't take OpenAI tool definitions, so tools are described in the
# prompt and calls are parsed back out of the text.

TOOL_CALLING_INSTRUCTIONS = """# TOOL CALLING

You have access to the following tools. When you need to use a tool, output it in this EXACT format:

[TOOL_CALL: tool_name]
{"parameter": "value", "another_param": "value"}
[/TOOL_CALL]

You can call multiple tools in one response. Always use valid JSON for parameters.
After the tool call block, you can continue with your regular response.
If you are explicitly instructed to call a specific tool, the [TOOL_CALL] block alone is a complete response.

If a tool returns results, they will appear in the next message. Use those results to inform your response.

AVAILABLE TOOLS:
"""

_TOOL_CALL_RE = re.compile(r"\[TOOL_CALL:\s*(\w+)\]\s*(\{.*?\})\s*\[/TOOL_CALL\]", re.DOTALL)


def format_tools_for_prompt(tools: list) -> str:
    lines = []
    for tool in tools:
        if tool.get("type") != "function":
            continue
        func = tool.get("function", {})
        params = func.get("parameters", {})
        required = params.get("required", [])
        param_lines = [
            f"    - {name}: {info.get('type', 'any')} ({'required' if name in required else 'optional'}) - {info.get('description', '')}"
            for name, info in params.get("properties", {}).items()
        ]
        param_desc = ("\n" + "\n".join(param_lines)) if param_lines else ""
        lines.append(f"**{func.get('name', 'unknown')}**: {func.get('description', 'No description')}{param_desc}")
    return "\n".join(lines)


def parse_tool_calls(text: str) -> tuple:
    matches = _TOOL_CALL_RE.findall(text)
    if not matches:
        return text, None
    calls = []
    for name, args_json in matches:
        try:
            args = json.loads(args_json)
        except json.JSONDecodeError:
            args = {"raw": args_json}
        calls.append({
            "id": f"call_{uuid.uuid4().hex[:8]}",
            "type": "function",
            "function": {"name": name, "arguments": json.dumps(args)},
        })
    return _TOOL_CALL_RE.sub("", text).strip(), calls


# =============================================================================
# PROMPT BUILDING
# =============================================================================
# The prompt goes to the CLI as one stream-json user message whose content is
# text blocks with image blocks interleaved where the images were shared.
# Images are only sent for the last IMAGE_WINDOW messages; older ones become
# placeholders so long chats don't resend every picture.

IMAGE_WINDOW = 5
_DATA_URL_RE = re.compile(r"data:image/[a-zA-Z0-9.+-]+;base64,([A-Za-z0-9+/=]+)")
_OLD_IMAGE = "[An image was shared earlier]"


def _image_block(b64: str):
    try:
        raw = base64.b64decode(b64, validate=True)
    except (binascii.Error, ValueError):
        return "[Image could not be decoded]"
    # The API rejects a declared media type that doesn't match the bytes, and
    # clients often mislabel, so detect it from the magic bytes.
    if raw[:8] == b"\x89PNG\r\n\x1a\n":
        media = "image/png"
    elif raw[:2] == b"\xff\xd8":
        media = "image/jpeg"
    elif raw[:3] == b"GIF":
        media = "image/gif"
    elif raw[:4] == b"RIFF" and raw[8:12] == b"WEBP":
        media = "image/webp"
    else:
        return "[Image in an unsupported format]"
    return {"type": "image", "source": {"type": "base64", "media_type": media, "data": b64}}


def _split_images(text: str, with_images: bool) -> list:
    """Split text on embedded base64 data URLs into text and image segments."""
    segments, pos = [], 0
    for m in _DATA_URL_RE.finditer(text):
        segments.append(text[pos:m.start()])
        segments.append(_image_block(m.group(1)) if with_images else _OLD_IMAGE)
        pos = m.end()
    segments.append(text[pos:])
    return segments


def _msg_segments(msg: dict, with_images: bool) -> list:
    content = msg.get("content")
    if isinstance(content, str):
        return _split_images(content, with_images)
    if not isinstance(content, list):
        return []
    segments = []
    for part in content:
        if not isinstance(part, dict):
            continue
        if part.get("type") == "text":
            segments += _split_images(part.get("text", ""), with_images)
        elif part.get("type") == "image_url":
            url = (part.get("image_url") or {}).get("url", "")
            m = _DATA_URL_RE.fullmatch(url)
            segments.append(_image_block(m.group(1)) if m and with_images else _OLD_IMAGE)
            segments.append("\n")
    return segments


def _msg_text(msg: dict) -> str:
    return "".join(s for s in _msg_segments(msg, False) if isinstance(s, str))


def _to_blocks(segments: list) -> list:
    """Merge adjacent text segments; the API rejects empty text blocks."""
    blocks, buf = [], []

    def flush():
        text = "".join(buf)
        if text.strip():
            blocks.append({"type": "text", "text": text})
        buf.clear()

    for seg in segments:
        if isinstance(seg, str):
            buf.append(seg)
        else:
            flush()
            blocks.append(seg)
    flush()
    return blocks


def _turn_segments(msg: dict, with_images: bool, labelled: bool) -> list:
    role = msg.get("role", "user")
    body = _msg_segments(msg, with_images)
    if role == "tool":
        body = [f"[TOOL_RESULT: {msg.get('name', 'unknown')}]\n", *body, "\n[/TOOL_RESULT]"]
    if not labelled:
        return [*body, "\n\n"]
    return ["Assistant: " if role == "assistant" else "Human: ", *body, "\n\n"]


def build_full_content(messages: list, tools) -> list:
    image_start = len(messages) - IMAGE_WINDOW
    system, convo = [], []
    for idx, msg in enumerate(messages):
        with_images = idx >= image_start
        if msg.get("role") == "system":
            system += [*_msg_segments(msg, with_images), "\n\n"]
        else:
            convo += _turn_segments(msg, with_images, labelled=True)

    tool_section = f"{TOOL_CALLING_INSTRUCTIONS}{format_tools_for_prompt(tools)}\n\n" if tools else ""
    if not system and not tool_section:
        return _to_blocks(convo)
    return _to_blocks([
        "=== SYSTEM PROMPT (FOLLOW EXACTLY) ===\n\n", *system, tool_section,
        "=== END SYSTEM PROMPT ===\n\n=== CONVERSATION HISTORY ===\n\n", *convo,
    ])


def build_resume_content(new_turn: list) -> list:
    segments = []
    for msg in new_turn:
        segments += _turn_segments(msg, with_images=True, labelled=False)
    return _to_blocks(segments)


# =============================================================================
# CLI SESSION REUSE
# =============================================================================
# The OpenAI format has no conversation ID, so each CLI session is saved under
# a hash of the reply it produced. When a request's last assistant message
# matches, only the new turn is sent with --resume and the CLI's cached prompt
# prefix is reused. Entries are consumed when resumed: --resume keeps appending
# to the same session ID, so a later swipe of that turn must start fresh rather
# than resume a session that already holds the discarded reply.

SESSIONS_FILE = os.path.join(BASE_DIR, "bridge_sessions.json")
SESSION_MAX_TURNS = 25
SESSION_TTL_SECONDS = 7 * 86400
RECENT_CHECK = 5
_SESSION_LOCK = threading.Lock()


def _load_sessions() -> dict:
    try:
        with open(SESSIONS_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
    except (OSError, json.JSONDecodeError):
        return {}
    if not isinstance(data, dict):
        return {}
    return {k: v for k, v in data.items() if isinstance(v, dict) and "turns" in v}


SESSIONS = _load_sessions()


def _save_sessions():
    cutoff = time.time() - SESSION_TTL_SECONDS
    for key in [k for k, v in SESSIONS.items() if v.get("updated_at", 0) < cutoff]:
        del SESSIONS[key]
    try:
        with open(SESSIONS_FILE, "w", encoding="utf-8") as f:
            json.dump(SESSIONS, f, indent=2)
    except OSError as e:
        log(f"Could not save {SESSIONS_FILE}: {e}", "WARN")


def _reply_key(text: str) -> str:
    return hashlib.sha256(text.strip().encode("utf-8")).hexdigest()


def _sigs(messages: list) -> list:
    # Start + end + length catches edits anywhere in a message, including truncation.
    sigs = []
    for m in messages:
        if m.get("role") in ("user", "assistant"):
            text = _msg_text(m)
            sigs.append(f"{m['role'][0]}:{text[:150]}|{text[-100:]}|{len(text)}")
    return sigs


def take_session(messages: list, last_asst: int) -> tuple:
    """Pop the saved session that produced the request's last assistant reply.
    Returns (entry or None, reason)."""
    if last_asst < 0:
        return None, "no prior reply"
    key = _reply_key(_msg_text(messages[last_asst]))
    with _SESSION_LOCK:
        entry = SESSIONS.pop(key, None)
        if entry:
            _save_sessions()
    if not entry:
        return None, "last reply isn't from a saved session"
    if entry["turns"] >= SESSION_MAX_TURNS:
        # Resumed sessions never see the client's context trimming or
        # lorebook changes, so refresh periodically.
        return None, f"session reached {SESSION_MAX_TURNS} turns; refreshing"
    # Membership rather than position: presets move per-turn marker messages around.
    if not set(entry["sigs"]) <= set(_sigs(messages)):
        return None, "a recent message was edited"
    return entry, "resume ok"


def save_session(session_id: str, messages: list, reply: str, turns: int):
    sigs = _sigs(messages + [{"role": "assistant", "content": reply}])[-RECENT_CHECK:]
    with _SESSION_LOCK:
        SESSIONS[_reply_key(reply)] = {
            "session_id": session_id,
            "sigs": sigs,
            "turns": turns,
            "updated_at": time.time(),
        }
        _save_sessions()


# =============================================================================
# CLAUDE CALL
# =============================================================================

def call_claude(messages: list, tools=None, json_schema=None, process_holder: dict = None) -> dict:
    """Returns {"response": str, "tool_calls": list or None}."""
    model = runtime_settings["model"]
    effort = runtime_settings["effort_level"]

    last_asst = max((i for i, m in enumerate(messages) if m.get("role") == "assistant"), default=-1)
    new_turn = [m for m in messages[last_asst + 1:] if m.get("role") in ("user", "tool")]

    entry = None
    if runtime_settings["cli_session_reuse"] and new_turn:
        entry, reason = take_session(messages, last_asst)
        debug(f"Resuming CLI session {entry['session_id'][:8]}" if entry else f"Fresh CLI session: {reason}")

    if entry:
        content = build_resume_content(new_turn)
        turns = entry["turns"] + 1
    else:
        content = build_full_content(messages, tools)
        turns = 0
    images = sum(1 for b in content if b["type"] == "image")
    if images:
        debug(f"Attaching {images} image(s)")

    cmd = [
        CLAUDE_EXE, "-p",
        "--input-format", "stream-json",
        "--output-format", "stream-json", "--verbose",
        "--model", model,
        "--effort", effort,
        "--tools", "",
        # Skips CLAUDE.md, auto-memory, skills, plugins and MCP servers, which
        # the CLI would otherwise inject into every roleplay prompt.
        "--safe-mode",
    ]
    if json_schema is not None:
        cmd += ["--json-schema", json.dumps(json_schema, separators=(",", ":"))]
    if entry:
        cmd += ["--resume", entry["session_id"]]

    # Passed as a file because Windows caps the command line at ~32K chars.
    # It must be re-sent on --resume: the CLI rebuilds the system prompt from
    # flags each invocation, and identical bytes keep the prompt cache warm.
    sp_file = tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", suffix=".txt", delete=False)
    try:
        sp_file.write(runtime_settings["system_prompt"].strip() or DEFAULT_SYSTEM_PROMPT)
    finally:
        sp_file.close()
    cmd += ["--system-prompt-file", sp_file.name]

    stdin_payload = json.dumps({"type": "user", "message": {"role": "user", "content": content}}) + "\n"

    log(f"Calling Claude ({model}, effort={effort}{', resumed' if entry else ''})...")
    start = time.time()
    process = None
    try:
        # bufsize=1 keeps stdout line-buffered so events arrive as they're emitted.
        # cwd=BASE_DIR: --resume looks sessions up per working directory, so it
        # must not depend on where the bridge was launched from.
        process = subprocess.Popen(
            cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            text=True, encoding="utf-8", bufsize=1, cwd=BASE_DIR,
        )
        if process_holder is not None:
            process_holder["process"] = process
        process.stdin.write(stdin_payload)
        process.stdin.close()

        response_text = ""
        captured_session_id = None
        stop_reason = None
        usage = {}

        for line in process.stdout:
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                continue
            etype = event.get("type")
            captured_session_id = captured_session_id or event.get("session_id")

            if etype == "error":
                msg = event.get("error", {}).get("message", str(event))
                log(f"CLI error event: {msg}", "ERROR")
                return {"response": f"Error: {msg}", "tool_calls": None}
            if etype == "content_block_delta":
                delta = event.get("delta", {})
                if delta.get("type") == "text_delta":
                    response_text += delta.get("text", "")
            elif etype == "assistant":
                message = event.get("message", {})
                stop_reason = message.get("stop_reason") or stop_reason
                if not response_text:
                    response_text = "".join(
                        b.get("text", "") for b in message.get("content", []) if b.get("type") == "text"
                    )
            elif etype == "result":
                if event.get("is_error"):
                    msg = event.get("result", "Unknown error")
                    log(f"CLI returned error: {msg}", "ERROR")
                    return {"response": f"Error: {msg}", "tool_calls": None}
                if json_schema is not None and "structured_output" in event:
                    response_text = json.dumps(event["structured_output"])
                elif not response_text:
                    response_text = event.get("result", "")
                usage = event.get("usage") or {}
                usage["cost"] = event.get("total_cost_usd", 0)

        process.wait(timeout=300)
        stderr = process.stderr.read()
        elapsed = time.time() - start

        if process_holder and process_holder.get("cancelled"):
            log(f"Cancelled after {elapsed:.1f}s (client disconnected)", "WARN")
            return {"response": "", "tool_calls": None}
        if process.returncode != 0:
            msg = stderr.strip() or "Unknown error (no stderr)"
            log(f"CLI exited {process.returncode}: {msg}", "ERROR")
            return {"response": f"Error from Claude Code: {msg}", "tool_calls": None}

        log(f"Response received in {elapsed:.1f}s", "SUCCESS")
        if usage and runtime_settings["debug_output"]:
            stats = {"Input": usage.get("input_tokens", 0), "Output": usage.get("output_tokens", 0)}
            if usage.get("cache_read_input_tokens"):
                stats["Cache read"] = usage["cache_read_input_tokens"]
            if usage.get("cache_creation_input_tokens"):
                stats["Cache created"] = usage["cache_creation_input_tokens"]
            if usage.get("cost"):
                stats["Cost"] = f"${usage['cost']:.4f}"
            log_box("Token Usage", stats)

        response_text = response_text.strip()
        if stop_reason == "refusal":
            log("Model declined this request (stop_reason=refusal)", "WARN")
            if not response_text:
                response_text = "[Bridge: the model declined this request (stop_reason=refusal). Swipe to retry or adjust the prompt.]"
        elif not response_text:
            log(f"Empty response (stop_reason={stop_reason!r})", "WARN")
        elif stop_reason and stop_reason != "end_turn":
            debug(f"stop_reason={stop_reason!r}")

        clean_response, tool_calls = parse_tool_calls(response_text)
        if tool_calls:
            log(f"Tool calls: {[tc['function']['name'] for tc in tool_calls]}")

        if captured_session_id and runtime_settings["cli_session_reuse"] and clean_response:
            save_session(captured_session_id, messages, clean_response, turns)

        return {"response": clean_response, "tool_calls": tool_calls}

    except FileNotFoundError:
        log(f"Claude CLI not found (looked for {CLAUDE_EXE})", "ERROR")
        return {
            "response": "**Bridge error — Claude CLI not found.** Make sure `claude --version` works from a new terminal.",
            "tool_calls": None,
        }
    except Exception as e:
        log(f"Exception calling Claude: {e}", "ERROR")
        if process and process.poll() is None:
            process.kill()
        return {"response": f"Error calling Claude Code: {e}", "tool_calls": None}
    finally:
        try:
            os.unlink(sp_file.name)
        except OSError:
            pass


# =============================================================================
# HTTP API
# =============================================================================

app = Flask(__name__, template_folder="templates")
CORS(app)


def _parse_response_format(response_format):
    if not isinstance(response_format, dict):
        return None
    if response_format.get("type") == "json_object":
        return {"type": "object"}
    if response_format.get("type") == "json_schema":
        wrapper = response_format.get("json_schema") or {}
        if isinstance(wrapper, dict):
            # OpenAI nests the schema one level deeper; some clients flatten it.
            schema = wrapper.get("schema")
            return schema if isinstance(schema, dict) else (wrapper or None)
    return None


def _strip_json_fences(text: str) -> str:
    # The CLI doesn't always emit structured_output; the fallback text often
    # wraps the JSON in markdown fences.
    m = re.match(r"^```(?:json)?\s*\n?(.+?)\n?```\s*$", text.strip(), re.DOTALL | re.IGNORECASE)
    return m.group(1).strip() if m else text


def _chunk(response_id: str, delta: dict, finish_reason=None) -> str:
    return "data: " + json.dumps({
        "id": response_id,
        "object": "chat.completion.chunk",
        "created": int(time.time()),
        "model": runtime_settings["model"],
        "choices": [{"index": 0, "delta": delta, "finish_reason": finish_reason}],
    }) + "\n\n"


@app.route("/v1/chat/completions", methods=["POST"])
def chat_completions():
    data = request.get_json(silent=True) or {}
    if DEBUG_REQUESTS:
        write_request_log(data)
    messages = data.get("messages", [])
    if not messages:
        return jsonify({"error": "No messages provided"}), 400
    stream = bool(data.get("stream", False))
    tools = data.get("tools") if runtime_settings["tool_calling_enabled"] else None
    json_schema = _parse_response_format(data.get("response_format"))

    if runtime_settings["debug_output"]:
        roles = [m.get("role") for m in messages]
        log_box("Incoming Request", {
            "Messages": len(messages),
            "System": roles.count("system"),
            "User": roles.count("user"),
            "Assistant": roles.count("assistant"),
            "Stream": "Yes" if stream else "No",
            "Tools": len(tools) if tools else 0,
        })

    # The CLI doesn't stream tokens. Run it on a worker thread and yield
    # keepalives every second: the writes surface client disconnects (so the
    # subprocess can be killed), and they reset client inactivity timers.
    process_holder, result_holder = {}, {}

    def worker():
        try:
            result_holder["result"] = call_claude(messages, tools, json_schema, process_holder)
        except Exception as e:
            log(f"Worker crashed: {e}", "ERROR")
            result_holder["error"] = str(e)

    worker_thread = threading.Thread(target=worker, daemon=True)
    worker_thread.start()

    def generate(as_sse: bool):
        # An empty-delta data event, not an SSE comment: TomoriBot's parser
        # skips comments, so they never reset its 120s timeout.
        keepalive = _chunk("keepalive", {"content": ""}) if as_sse else " "
        try:
            while worker_thread.is_alive():
                worker_thread.join(timeout=1.0)
                if worker_thread.is_alive():
                    yield keepalive

            if "error" in result_holder:
                err = json.dumps({"error": {"message": result_holder["error"], "type": "bridge_error"}})
                yield f"data: {err}\n\ndata: [DONE]\n\n" if as_sse else err
                return

            result = result_holder["result"]
            text = result["response"]
            tool_calls = result.get("tool_calls")
            if json_schema is not None and not tool_calls:
                text = _strip_json_fences(text)
            response_id = f"chatcmpl-{uuid.uuid4().hex[:8]}"
            finish_reason = "tool_calls" if tool_calls else "stop"

            if as_sse:
                if text or not tool_calls:
                    yield _chunk(response_id, {"role": "assistant", "content": text})
                if tool_calls:
                    yield _chunk(response_id, {"tool_calls": [dict(tc, index=i) for i, tc in enumerate(tool_calls)]}, finish_reason)
                else:
                    yield _chunk(response_id, {}, finish_reason)
                yield "data: [DONE]\n\n"
            else:
                message = {"role": "assistant", "content": text}
                if tool_calls:
                    message["tool_calls"] = tool_calls
                yield json.dumps({
                    "id": response_id,
                    "object": "chat.completion",
                    "created": int(time.time()),
                    "model": runtime_settings["model"],
                    "choices": [{"index": 0, "message": message, "finish_reason": finish_reason}],
                    "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
                })
        finally:
            proc = process_holder.get("process")
            if proc and proc.poll() is None:
                process_holder["cancelled"] = True
                proc.kill()
                log("Client disconnected — killed Claude subprocess", "WARN")

    if stream:
        return Response(generate(True), mimetype="text/event-stream",
                        headers={"Cache-Control": "no-cache", "Connection": "keep-alive", "X-Accel-Buffering": "no"})
    return Response(generate(False), mimetype="application/json")


@app.route("/v1/models", methods=["GET"])
def list_models():
    return jsonify({"object": "list", "data": [
        {"id": runtime_settings["model"], "object": "model", "created": int(time.time()), "owned_by": "anthropic"},
    ]})


@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "model": runtime_settings["model"], "effort_level": runtime_settings["effort_level"]})


@app.route("/api/models", methods=["GET"])
def api_models():
    return jsonify(get_models(refresh=request.args.get("refresh") == "1"))


@app.route("/api/settings", methods=["GET"])
def get_settings():
    return jsonify({**runtime_settings, "default_system_prompt": DEFAULT_SYSTEM_PROMPT})


@app.route("/api/settings", methods=["POST"])
def update_settings():
    data = request.get_json(silent=True) or {}
    updates = {}
    for key, value in data.items():
        if key not in DEFAULT_SETTINGS:
            continue
        expected = type(DEFAULT_SETTINGS[key])
        if key == "bridge_port":
            try:
                value = int(value)
            except (TypeError, ValueError):
                return jsonify({"error": "bridge_port must be an integer"}), 400
            if not 1 <= value <= 65535:
                return jsonify({"error": "bridge_port must be between 1 and 65535"}), 400
        elif type(value) is not expected:
            return jsonify({"error": f"{key} must be {expected.__name__}"}), 400
        if key == "model" and not value.strip():
            return jsonify({"error": "model must not be empty"}), 400
        if key == "effort_level" and value not in EFFORT_LEVELS:
            return jsonify({"error": f"effort_level must be one of {EFFORT_LEVELS}"}), 400
        updates[key] = value

    runtime_settings.update(updates)
    save_settings()
    log(f"Settings updated: model={runtime_settings['model']}, effort={runtime_settings['effort_level']}", "SUCCESS")
    return jsonify({**runtime_settings, "default_system_prompt": DEFAULT_SYSTEM_PROMPT})


if __name__ == "__main__":
    port = runtime_settings["bridge_port"]
    print()
    print(f"{Colors.CYAN}{Colors.BOLD}  CLAUDE CODE BRIDGE{Colors.RESET}")
    print(f"  {Colors.DIM}Model:{Colors.RESET}    {Colors.GREEN}{runtime_settings['model']}{Colors.RESET}")
    print(f"  {Colors.DIM}Effort:{Colors.RESET}   {Colors.GREEN}{runtime_settings['effort_level']}{Colors.RESET}")
    print(f"  {Colors.CYAN}API URL:{Colors.RESET}  http://localhost:{port}/v1")
    print(f"  {Colors.CYAN}Settings:{Colors.RESET} http://localhost:{port}/")
    if DEBUG_REQUESTS:
        print(f"  {Colors.YELLOW}Debug:{Colors.RESET}    requests logged to {LOGS_DIR}")
    print()
    # Warm the model list so the settings page doesn't wait on the CLI.
    threading.Thread(target=get_models, daemon=True).start()
    app.run(host="0.0.0.0", port=port, debug=False, threaded=True)
