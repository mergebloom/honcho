# Maintaining this Honcho fork

This is an intentionally maintained fork, not a temporary patch installation.
Upstream remains `plastic-labs/honcho`; this fork remains source-of-truth for
its deployments. Preserve upstream licensing and contributor attribution.

## Branches and ownership

- `production`: fork default branch and qualified release lineage.
- `main`: upstream mirror/reference only; **never deploy it directly**.
- `upstream` remote: `https://github.com/plastic-labs/honcho.git`.
- Upgrade work uses `upgrade/<upstream-short-sha>` branches from `production`.
- Do not rebase/reset deployed fork history onto upstream. Merge upstream into
  the fork so the Codex adapter, protocol fixes and their tests cannot disappear.
- Do not enable vendor deployment workflows or schedule unattended production
  upgrades. Fork validation/image publishing is `fork-release.yml`; configure
  inherited vendor workflows disabled in the fork repository settings.

## Patch inventory / provenance

1. Opt-in Responses API transport and regressions from upstream PR #1049,
   including reasoning replay, terminal output, JSON-object/tool and fallback
   history fixes. The six original commits were cherry-picked with authorship;
   only an import/adjacent-test conflict was resolved when integrating.
2. Native per-model Codex OAuth opt-in in `src/llm/codex_oauth.py` and small
   config/registry hooks. Existing OpenAI/provider/embedding defaults unchanged.
3. Access-only publication helpers in this directory. Hermes retains refresh
   ownership; no independent refresh-token copies in Honcho workers.
4. Fork-specific tests and image workflow. No database migration is introduced
   by the fork's provider feature.

PR #825 is prior art, not imported implementation: its independent API/worker
refresh managers can race the shared rotating grant. PR #1049 is protocol
support, not OAuth. Neither upstream proposal implies official support for the
internal Codex endpoint. See `codex-oauth.md` for operational limitations.

## Upgrade procedure

Use a clean worktree and record the prior production commit/image/schema first.
These are operator-run commands, not an unattended auto-deployer:

```sh
git fetch upstream main --tags
git switch production
git pull --ff-only origin production
git switch -c upgrade/<upstream-short-sha>
git merge --no-ff <reviewed-full-upstream-commit>
```

Resolve conflicts deliberately. **Never** resolve by dropping the fork adapter
or replacing fork tests with upstream versions. Compare upstream work against
the patch inventory: retire a patch only after equivalent upstream behavior
passes its regressions. Keep Codex auth separate from generic Responses mode.
Update `fork/UPSTREAM_BASE` to the reviewed upstream revision.

1. Read release notes, dependency diffs, migration graph and all changed
   provider/registry/config/history/structured-output code.
2. Run `uv sync --frozen --group dev`, `uv run pytest tests/llm
   tests/utils/test_clients.py -q -n 0`, lint/typecheck changed Python, and the
   relevant database/API/worker tests against **isolated** PostgreSQL.
3. Push the upgrade branch and qualify an immutable candidate image using the
   fork workflow (dispatch the exact upgrade ref). Read back the workflow run,
   commit, digest and OCI revision. Do not deploy a mutable `latest` tag.
4. Back up current database and deployment inputs. Verify dump checksum/TOC,
   copy it off the application guest, and restore it into an isolated database.
5. Rehearse candidate migrations on the restored copy. Verify row counts,
   auth/retrieval, synthetic ingestion, derivation, summaries, embeddings,
   multi-round tool use and text streaming. Keep outbound webhooks disabled and
   do not let the rehearsal process unrelated restored pending jobs.
6. Review rollback compatibility. If schema changes are not backward-compatible,
   a prior image alone is **not** rollback; require an explicitly approved
   restore/outage plan before cutover. Never auto-restore a production database.
7. Merge the qualified upgrade into `production`, build/test that exact final
   commit, take a fresh pre-cutover backup, and pin API + deriver to its digest.
   Do not upgrade PostgreSQL/Redis/OS as a side effect.
8. Re-test actual configured consumers, all role configurations, queue/vector
   progress and monitoring. Keep previous image/config/backup until separately
   approved retirement. Retain a release receipt outside the source repository.

## Release receipt (private operational artifact)

Record at least: upstream base, full fork commit, CI run URL, OCI digest and
revision, previous image/config receipt, prior/new Alembic schema heads,
logical-dump path/hash/off-host copy, independent infrastructure-backup evidence,
restore proof, provider/model per role, embedding configuration unchanged,
canary and production acceptance results, monitor state and rollback limits.
Never publish memory/database dumps, deployment `.env`, bearer/refresh tokens,
private hostnames or customer identifiers in a public release/repository.

## Credential-owner maintenance

Install the exporter from a qualified revision outside a development worktree.
Run it as a bounded systemd oneshot/timer on the existing credential-owner host.
Pin `HERMES_HOME`; invoke with Hermes' Python and source path. Receiver must be
root-owned and reachable only through a dedicated SSH key with a fixed forced
command, source restriction and no forwarding/PTY/shell. The command writes
only the preselected access-token directory, owned by the app runtime UID.
Mount the **directory**, not the individual file, read-only into API/deriver.
Atomic file replacement must remain visible without restarting containers.

After a Hermes upgrade, test its installed credential resolver plus exporter:
this deliberate integration dependency is smaller than adding an inference
proxy, but it must be requalified. Do not force refresh just to test rotation;
fixture rotation proves file/client behavior without consuming a shared grant.
A timer/receiver failure must be surfaced, not compensated by copying refresh
tokens or falling back to billed credentials. Preserve provider canary metrics
and update their chat probe when changing the actual provider/model.
