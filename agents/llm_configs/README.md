# LLM Configs

Each `.json` here is a named LLM configuration. Select one at runtime with
`--llm-config <name>` (or `CFU_AGENT_LLM_CONFIG`). The default is `default`.

```bash
# Use the DeepSeek chat-completions config (key read from DEEPSEEK_API_KEY).
export DEEPSEEK_API_KEY="sk-..."
python3 agents/run_cfu_agent.py --task "..." --llm-config deepseek --dry-run
```

## JSON fields

| Field | Required | Meaning | Fallback |
| --- | --- | --- | --- |
| `name` | yes | Config id (should match the filename) | |
| `description` | no | Human description | |
| `model` | no | Model name | `CFU_AGENT_MODEL` then `gpt-5` |
| `base_url` | no | API base URL, usually ending in `/v1` | `OPENAI_BASE_URL` |
| `api_key` | no | Inline key (for local testing only) | `api_key_env`, then `OPENAI_API_KEY` |
| `api_key_env` | no | Env var to read the key from, e.g. `DEEPSEEK_API_KEY` | `OPENAI_API_KEY` |
| `api_mode` | no | `auto` \| `responses` \| `chat` | `CFU_AGENT_LLM_API`, then `auto` |
| `temperature` | no | Sampling temperature | `0` |
| `timeout` | no | Request timeout in seconds | `120` |
| `thinking` | no | Request model reasoning/thinking (`true`/`false`) | `false` (off) |
| `reasoning_effort` | no | Thinking effort: `minimal` \| `low` \| `medium` \| `high` | `medium` when thinking is on |
| `reasoning_summary` | no | Responses `reasoning.summary`: `auto` \| `concise` \| `detailed`, or `""` to disable | `auto` when thinking is on |
| `stream` | no | Stream output tokens (live progress, per-chunk read timeout) | `true` |
| `thinking_body` | no | Exact `extra_body` used to enable thinking on a chat endpoint | `{"thinking": {"type": "enabled"}}` |

## Thinking / reasoning

Thinking is **off by default**: no reasoning parameter is sent and the client
behaves exactly as before. Opt in per config:

```json
{
  "model": "DeepSeek-V4-Pro",
  "thinking": true,
  "reasoning_effort": "high"
}
```

- `"thinking": true` alone uses `reasoning_effort: "medium"`.
- Setting only `reasoning_effort` (no `thinking` key) implies thinking is on.
- `"thinking": false` always wins: any `reasoning_effort` is ignored.
- Chat requests send `extra_body` (default `{"thinking": {"type": "enabled"},
  "reasoning_effort": "<effort>"}`); Responses requests send
  `reasoning={"effort": "<effort>", "summary": "<reasoning_summary>"}`.
- **OpenAI/Azure reasoning models never expose raw chain-of-thought.** The
  Responses API returns only a summary, and only when `reasoning.summary` is set;
  that is why `reasoning_summary` defaults to `auto`. With `summary` unset the
  provider returns empty reasoning items, so no thinking can be captured. The
  chat-completions path on Azure returns no reasoning at all, so tool-using
  stages, which run on chat, have no thinking to record with that provider; use a
  provider that returns `reasoning_content` in chat if you need it there.
- `api_mode: auto` tries Responses first and falls back to chat when that route
  is unavailable (some gateway model groups have no `/responses`, e.g.
  `DeepSeek-V4-Flash` returns 404, or returns an empty event stream). The reason
  is recorded as `fallback_reason` in the raw-output artifact metadata, so "no
  thinking captured" is explainable.
- `GLM-5.2` and `Kimi-K3` are pinned to `api_mode: chat` here: their thinking
  arrives as `reasoning_content` in chat (captured, and streamed live), while the
  gateway's Responses route for them returns no reasoning summary and buffers the
  stream. `gpt-5.6-luna`/`gpt-5.6-terra` are the opposite: chat has nothing, the
  Responses summary has the thinking.
- Measured on the HKUST gateway (chat, `thinking` on): `GLM-5.2` and `Kimi-K3`
  return the full `reasoning_content` (thousands of characters), which this client
  captures as `message.reasoning_content` — including in the tool-using stages,
  because those run on chat. `DeepSeek-V4-Flash` and `DeepSeek-V4-Pro` return no
  reasoning in chat (`provider_specific_fields.reasoning` is explicitly `None`,
  and streaming deltas carry only `content`), and their `/responses` route is
  404. `gpt-5.6-luna`/`gpt-5.6-terra` expose only a Responses summary. Use a
  GLM/Kimi config when you want thinking recorded for every stage.
- Some gateways expose the provider chain of thought only under
  `provider_specific_fields.reasoning`; that carrier is read as a fallback.
- Different gateways name the switch differently. Use `thinking_body` to match
  yours, for example `{"enable_thinking": true}` (Qwen/DashScope) or
  `{"thinking": {"type": "enabled", "budget_tokens": 4096}}` (Anthropic-style).
  When `thinking_body` is provided it is used as-is and `reasoning_effort` is
  only added if the body does not already set it. If a gateway rejects
  `reasoning.summary`, the Responses call is retried without it.
- Whatever the provider returns as reasoning (`reasoning_content`, `reasoning`,
  `reasoning_items`, a Responses `reasoning` item summary, or inline `thinking`
  blocks) is split out of the answer and persisted as `*_<n>.reasoning.txt`; the
  artifact parsers only see the final answer. `--test-api` prints
  `reasoning_source` / `reasoning_chars` so you can confirm what a gateway
  actually returns.

## Resolution precedence

For each field, the highest-precedence source wins:

- `api_key`: `api_key` > env `api_key_env` > `OPENAI_API_KEY`
- `model`: `model` > `CFU_AGENT_MODEL` > `gpt-5`
- `base_url`: `base_url` > `OPENAI_BASE_URL`
- `api_mode`: `CFU_AGENT_LLM_API` > `api_mode` > `auto`
- `thinking`: config `thinking` > `false`; `reasoning_effort` implies thinking on
- `reasoning_summary`: config `reasoning_summary` > `CFU_AGENT_REASONING_SUMMARY` > `auto` when thinking is on

Prefer `api_key_env` over an inline `api_key` so secrets stay out of the repo;
the file is tracked, so inline keys would be committed.

## api_mode

- `responses`: `client.responses.create` (OpenAI Responses API).
- `chat`: `client.chat.completions.create` (OpenAI-compatible chat endpoints).
- `auto`: try `responses`, then fall back to `chat` if the endpoint does not
  support it.
