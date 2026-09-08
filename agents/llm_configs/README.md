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

## Resolution precedence

For each field, the highest-precedence source wins:

- `api_key`: `api_key` > env `api_key_env` > `OPENAI_API_KEY`
- `model`: `model` > `CFU_AGENT_MODEL` > `gpt-5`
- `base_url`: `base_url` > `OPENAI_BASE_URL`
- `api_mode`: `CFU_AGENT_LLM_API` > `api_mode` > `auto`

Prefer `api_key_env` over an inline `api_key` so secrets stay out of the repo;
the file is tracked, so inline keys would be committed.

## api_mode

- `responses`: `client.responses.create` (OpenAI Responses API).
- `chat`: `client.chat.completions.create` (OpenAI-compatible chat endpoints).
- `auto`: try `responses`, then fall back to `chat` if the endpoint does not
  support it.
