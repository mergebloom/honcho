# Opt-in Codex OAuth transport

This fork can use externally owned ChatGPT access credentials for a **single
OpenAI-transport model**. It does not change normal OpenAI/proxy clients,
embeddings, provider defaults, dependencies, or database schemas.

## Configuration

For example, configure a deriver role in `config.toml`:

```toml
[llm]
CODEX_OAUTH_TOKEN_FILE = "/run/honcho-codex/access.json"
# Leave CODEX_OAUTH_CLIENT_VERSION unset by default, like Hermes inference.
# A stale forced compatibility version can hide newer models and cause an
# account-shaped unsupported-model error. No discovery request is made here.

[deriver.model_config]
model = "YOUR_ACCOUNT_SUPPORTED_MODEL"
transport = "openai"

[deriver.model_config.overrides.provider_params]
codex_oauth = true
timeout = 120
```

The equivalent global environment variables are `LLM_CODEX_OAUTH_TOKEN_FILE` and
`LLM_CODEX_OAUTH_CLIENT_VERSION`. Do not configure `api_key`, `api_key_env`, or
`base_url` overrides on this model. Global OpenAI credentials/URLs are ignored
by this path and remain available only to ordinary models and embeddings.
Configure each intended role explicitly, removing inherited unsupported
controls and fallback settings. Codex models reject nonempty fallback configs;
there is no automatic switch to billed OpenAI, DeepInfra, or another provider.

## Opt-in tier-aware thinking profile

`fork/codex-tiered.env.example` is a non-secret deployment profile, also shipped
in the image at `/app/fork/codex-tiered.env.example`. It selects GPT-6.1-Sol for
all text roles, keeping derivation and summaries at low effort. Both dream
specialists use medium effort when dreams are explicitly enabled by the operator.
Dialectic minimal/low use low, medium uses medium, high uses high, and max uses
max. It does not enable dreams, change embeddings, or alter retrieval/tool limits.

For an already configured Codex OAuth deployment, use the reviewed release copy
as an `env_file` for both API and worker. Remove the matching inline model/effort
variables first: Compose `environment` values take precedence over `env_file`.
Keep existing explicit `transport=openai`, `codex_oauth=true`, access-file mounts
and private credentials. Verify the file checksum against the selected release
and confirm each tier's real outbound `reasoning.effort` after deployment.

This is an opt-in profile, not a change to upstream/global defaults. Higher effort
can increase latency and subscription-quota use; passing protocol tests does not
establish a memory-quality improvement. Confirm the selected account serves the
exact model/effort; do not silently alias models or reintroduce a forced version.

## Credential ownership and rotation

Hermes (or another external credential owner) exclusively owns refresh tokens
and refresh/rotation. Honcho never reads an auth store, refreshes credentials,
or accepts a refresh token. Its only credential input is access-only JSON:

```json
{"access_token":"ACCESS_JWT_FROM_OWNER","expires_at":2000000000}
```

Mount the **containing directory read-only**, not the individual file: the
owner must write a new file, set mode `0600` (or `0400`), then atomically rename
it over `access.json`. Make the file readable by the Honcho process without
granting group/world permissions; protect the directory against other writers.
Honcho opens the file anew on every outbound request, so cached clients see
updated tokens, account IDs, and residency. Missing, expired, unreadable,
malformed, oversized (>16 KiB), symlinked, or permissively readable files fail
closed. Extra JSON fields, including `refresh_token`, are forbidden. JWT expiry
and file expiry must both be current. Unverified JWT claims provide routing
metadata only; the server remains responsible for token verification.
Account routing uses `chatgpt_account_id` in the `https://api.openai.com/auth`
claim object; optional residency primarily uses `chatgpt_data_residency`, with
`chatgpt_compute_residency` as its fallback, matching the credential owner.
Legacy `residency` / `chatgpt_account_residency` aliases are also accepted;
conflicting legacy aliases fail closed.

## Protocol and limits

- Only POST to the exact HTTPS ChatGPT Codex Responses endpoint is allowed.
  Redirects, discovery, other URLs/query strings, and environment proxies are
  disabled. Account/residency headers are derived afresh from access JWT claims.
- Requests always use Responses SSE, `store=false`, nonempty instructions, and
  encrypted reasoning inclusion. Identity is honestly `originator=honcho-fork`
  and `User-Agent: HonchoFork/<server-version>`, never the official CLI identity.
  Optional client protocol version is sent as `version` on every inference call.
- Pydantic structured outputs, function calls/results (`any` becomes `required`),
  stateless history, encrypted reasoning replay, completions, and final text
  streaming reuse the generic Responses implementation. Tool-delta streaming
  is explicitly unsupported. Failed/incomplete/refused or unterminated streams
  propagate an error; partial text is not a successful final response.
- `max_tokens` and `max_output_tokens` are **not server-side caps** on this path;
  both are omitted from the wire. Temperature is also omitted because the
  endpoint does not accept it. Account quotas and application-level limits
  still apply; do not depend on these knobs to bound generation or cost.
- Stop sequences, thinking token budgets, top-p/top-k, penalties, seeds, cache
  controls, arbitrary provider controls, and SDK `extra_body`/`extra_headers`/
  `extra_query` passthroughs are rejected, not silently dropped. Allowed provider
  params: `codex_oauth`, `api_mode="responses"`, `structured_output_mode`, and
  positive finite `timeout` up to 600 seconds.
- Connect/pool timeouts are 10 seconds, write 30, read 600 maximum; per-call
  timeout may shorten them. SDK/network hidden retries are disabled. Honcho's
  existing outer finite retry policy may retry the same configured model only.
  HTTP/provider failures expose a generic safe error, not raw response bodies.

## Operational risks and rollback

This is an unofficial, account-dependent endpoint. Availability, models,
quotas, protocol, subscription eligibility, and account policy can change or
stop working. Confirm that usage is permitted for the account and deployment;
do not assume API service guarantees or a subscription bypasses usage limits.
No deployment or live-provider validation is implied by the fixture tests.

To roll back, restore the prior role configs/release and remove the Codex opt-in
and token mount. Re-enabling a conventional provider is an explicit operator
decision with its own credentials/billing, never an automatic fallback. No DB
rollback is required. The credential owner's refresh process remains separate.
