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
| `context_window_tokens` | no | Context window used by the request-size budget (no endpoint query) | soft `32768` when neither config nor `/models` metadata reports one |
| `output_reserve_tokens` | no | Completion budget: sent as `max_completion_tokens` (chat) / `max_output_tokens` (responses), and reserved for the answer by the context budget | `4096`; shipped configs use `32768` |

## Context window and completion budget

`context_window_tokens` and `output_reserve_tokens` are **planning** values: no
endpoint is queried for a capability the gateway does not report. The completion
budget is a request field, so it must satisfy
`0 < output_reserve_tokens < context_window_tokens`; a violation is a
`LlmConfigError` before any request is sent.

The same number is the context-budget output reservation, so it also decides when
a transcript is `warning` (80%), `compact` (90%) or `unfit` (100%) for the window:

```
level = (estimated_input + output_reserve_tokens) / context_window_tokens
```

A reasoning model spends its thinking tokens from this budget too, so a small
value can produce a **200 response with no content** (see
`docs/cfu_agent_failure_taxonomy.md` and run `20261003-095209`). Precedence:

1. `--max-output-tokens N` (run-level; `0` keeps the config/env value),
2. `CFU_AGENT_OUTPUT_RESERVE=N` (environment),
3. the config's `output_reserve_tokens`,
4. `4096` when nothing sets it.

The effective value is recorded in `run.yaml` under `llm_request` as
`max_output_tokens`, so a run's budget stays auditable after the fact.

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
  chat-completions path on Azure returns no reasoning at all. Tool-using stages
  follow `api_mode`, so pick a config whose active route returns reasoning (or a
  provider that returns `reasoning_content` in chat) when you need thinking there.
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

The setting governs plain completions *and* tool-using rounds. Tool rounds keep
the framework's chat-format transcript internally and translate it to Responses
items (`function_call` / `function_call_output`) at the wire boundary, so budget
compaction, source masking and the request audit behave the same on both
endpoints. `auto` also covers models that refuse function tools on chat
completions ("Function tools with reasoning_effort are not supported ... use
/v1/responses or set reasoning_effort to 'none'"): the round moves to the
Responses route, and a genuinely unsupported route is remembered for the rest of
the run and reported as `fallback_reason`. Pin `chat` when a model's Responses
route is worse (see the GLM/Kimi note above); pin `responses` to surface a
rejection instead of degrading to chat.
