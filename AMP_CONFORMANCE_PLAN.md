# nexus-memory / smriti-memcore — AMP conformance gap analysis & uplift plan

**Empirical baseline (2026-05-31):** running the canonical AMP compliance suite
(`amp/compliance/test_amp_server.py`, 154 tests) against
`smriti_memcore/integrations/mcp_server.py`:

```
53 passed, 45 failed, 56 skipped in 317.99s
```

- **Conformance grade: 34%** of the v1.2-draft surface
- The 56 skips are v1.2 optional verbs (`amp.update`, `amp.batch_encode`,
  `amp.export`, `amp.import`) the wrapper doesn't advertise — they auto-skip
  via `_has_verb()` but represent additional work to land Full v1.2
- The wrapper currently advertises itself as `amp_version: 1.0` in its server
  description and tool docstrings; AMP v1.1 shipped 2026-05-22 and v1.2-draft
  is now the active spec. The wrapper froze at v1.0.

## Failure breakdown

| Count | Theme |
|------:|-------|
| 10 | v1.1 multi-dimensional Scope — accept/validate `scope` object, enforce isolation, namespace-aware `forget`/`stats`, scope echo on `MemoryResult` |
| 6 | v1.1 server manifest — `amp_version` / `amp_conformance` in `initialize` response, `ToolAnnotations` on every verb |
| 2 | v1.1 error mapping — JSON-RPC `-32602`/`-32600` codes, `amp_error_code` in `error.data` not in the result body |
| 22 | v1.2 `metadata_filters` (entire feature: 8 operators, strict-AND, missing-key = miss, type-mismatch = miss, `ne` discipline, request-level validation, value-shape strictness, `maxItems=32`) |
| 4 | v1.2 recall hardening — `top_k` oversampling rule, `timestamp_after`/`timestamp_before` filters |
| 1 | v1.2 metadata-bag size cap on `amp.encode` |
| **45** | **Total failing** |
| 56 | Skipped v1.2 verbs (`amp.update`, `amp.batch_encode`, `amp.export`, `amp.import`) — implementation backlog |

## Root-cause categorisation

The wrapper has six structural gaps that produce all 45 failures:

### Gap 1 — single-tenant by design, scope is a no-op

```python
# smriti_memcore/integrations/mcp_server.py:486-489
# Six AMP verbs (amp.encode … amp.stats) exposed alongside the native smriti_*
# tools for full AMP v1.0 compliance. agent_id is accepted but ignored —
# smriti-memcore is single-tenant; isolation is handled at the storage-path level.
```

This is the choice that breaks AMP v1.1+ conformance most fundamentally. The
wrapper accepts `agent_id` purely as documentation; every call goes to the
same module-level `_smriti` instance. AMP v1.1 made multi-dim scoping
normative — backends MUST validate the scope object, enforce partition
isolation, and echo `scope` on `MemoryResult` rows. None of that is here.

Affected tests: 10 (Scope, namespace isolation, cross-namespace not_found,
stats after forget) + the implicit cause of many metadata_filter failures
(no per-scope state means tests stepping on each other's data).

### Gap 2 — error envelope shape is wrong

```python
# Current pattern, e.g. mcp_server.py:514
return {"error": "content must not be empty", "amp_error_code": "invalid_request"}
```

The wrapper returns errors as **result-body fields** (`{"error": ..., "amp_error_code": ...}` inside the tool result). AMP v1.1 §3.5 mandates errors land as **JSON-RPC error frames** with the `amp_error_code` inside `error.data`:

```json
{"jsonrpc": "2.0", "error": {"code": -32602, "message": "...", "data": {"amp_error_code": "invalid_request"}}}
```

Mapping: `invalid_request → -32602`, `not_found → -32600` (or app-level
`{"status": "not_found"}` for the in-band case), `backend_error → -32001`,
`not_supported → -32002`.

Affected tests: 2 directly, but error-shape coupling means many of the
metadata-filter "expected invalid_request" tests fail at the wrong layer.

### Gap 3 — no MCP `ToolAnnotations`

AMP v1.1 §3.4 mandates `readOnlyHint` / `destructiveHint` / `idempotentHint`
/ `openWorldHint` on every verb. The wrapper uses bare `@mcp_server.tool()`
decorators. FastMCP supports annotations via
`@mcp_server.tool(annotations=types.ToolAnnotations(...))` — see how
`amp/python/amp-server/src/amp_server/server.py` does it.

Affected tests: 4.

### Gap 4 — `initialize` response missing required AMP fields

Tests want `server.server_info` to carry `amp_version: "1.1"` (or
`"1.2-draft"`) and `amp_conformance: "core"|"full"`. FastMCP exposes a
mechanism to add fields to the server-info dict; currently the wrapper
relies on the default which lacks both. The v1.0 docstring claim
("amp_version: 1.0") is *only* in the human-readable description, not the
machine-readable manifest.

Affected tests: 2.

### Gap 5 — no v1.2 verbs or v1.2 recall behaviour

`metadata_filters`, `timestamp_after`/`timestamp_before`, `top_k`
oversampling, metadata-bag size cap, `amp.update`, `amp.batch_encode`,
`amp.export`, `amp.import` — none of these exist in the wrapper. The
optional verbs (`update`, `batch_encode`, `export`, `import`) auto-skip but
the recall filtering paths are tested unconditionally and fail.

Affected: 22 metadata_filter tests + 4 recall hardening tests + 1 bag-cap
test + 56 skipped optional-verb tests waiting on implementation.

### Gap 6 — `MemoryResult` shape is missing v1.1 fields

The wrapper's recall result:

```python
{"id": m.id, "content": m.content, "score": m.retrieval_score,
 "timestamp": m.creation_time.isoformat(), "status": m.status.value}
```

AMP v1.1 §3.1 mandates `source`, `metadata` (object), and `scope` (object)
on every `MemoryResult` — none of those three are emitted. The `visibility`
echo discipline (§5.4) for v1.0-vs-v1.1-native callers is also unimplemented.

Affected tests: 1 (scope echo) + the visibility echo test + ripple effects
in metadata_filter tests (filtering on a key that the result doesn't return
is hard to test).

---

## Uplift plan — three milestones

### Milestone A — v1.1 Core (closes ~20 failures, gets to ~73/154 = 47%)

Smallest landable step. Doesn't touch the smriti-memcore engine, only the
MCP wrapper. Focus: bring the surface up to AMP v1.1 Core without
implementing v1.2 features.

**A1. Scope object plumbing (2 days)**
- Accept `scope: Optional[dict]` alongside `agent_id` on every AMP verb.
- Add a `_normalize_scope(scope, agent_id) -> dict` helper that mirrors
  `amp/python/amp-server/src/amp_server/server.py:174` —
  at-least-one-isolating-key rule (`agent_id` / `group_id` / `workspace_id`
  / `user_id`), `invalid_request` if neither `scope` nor `agent_id` is
  supplied.
- Since smriti-memcore is genuinely single-tenant, partition by
  **storage-path-per-scope**: hash the normalised scope into a directory
  name, lazy-init a `SMRITI` instance per scope under
  `~/.smriti/scopes/<hash>/`. This is how the AMP reference server does it
  (`_get_agent_for_scope`).
- Echo `scope` on every `MemoryResult` row.

**A2. Error envelope migration (0.5 day)**
- Replace every `return {"error": ..., "amp_error_code": ...}` with `raise
  AmpToolError(code, msg)` and have FastMCP map the raise into a JSON-RPC
  error frame with `error.data.amp_error_code`.
- Codes: `invalid_request → -32602`, `not_supported → -32002`,
  `backend_error → -32001`. Look at
  `amp/python/amp-server/src/amp_server/server.py:58-95` for the exact
  `AmpToolError` shape.

**A3. ToolAnnotations on every verb (0.5 day)**
- Update each `@mcp_server.tool(name="amp.xxx")` to
  `@mcp_server.tool(name="amp.xxx", annotations=types.ToolAnnotations(...))`
  with the values from spec §3.4:
  - `amp.encode`: read=F, dest=F, idem=F, open=F
  - `amp.recall`: read=T, dest=F, idem=T, open=F
  - `amp.forget`: read=F, dest=T, idem=T, open=F
  - `amp.consolidate`: read=F, dest=F, idem=F, open=T
  - `amp.pin`: read=F, dest=F, idem=T, open=F
  - `amp.stats`: read=T, dest=F, idem=T, open=F

**A4. Server manifest (0.5 day)**
- Set `amp_version: "1.1"` and `amp_conformance: "core"` on the initialize
  response. FastMCP path: pass them as `server_info` extras when
  constructing the `FastMCP` instance (or post-init mutate the
  `server_info` dict before `mcp_server.run()`).

**A5. MemoryResult enrichment (1 day)**
- Add `source` (the `MemorySource` enum value as a string), `metadata`
  (object, defaulting to `{}` if smriti's underlying memory has none),
  and `scope` (the request scope, normalised) to every recall row.
- Implement the deprecated-visibility echo discipline (§5.4):
  - If the caller passed `private` or `visibility` filters → echo
    `visibility` field in the result
  - Otherwise → omit it
- The wrapper already has a `visibility` field on its native `Memory`; the
  mapping is mechanical.

**A6. Wire the v1.1 recall filters (1 day)**
- `RecallFilters`: `status`, `visibility` (deprecated path), `source`,
  `timestamp_after`, `timestamp_before` — all post-retrieval filtering,
  same pattern as the AMP reference server.

**Expected outcome after Milestone A:**
- ~73/154 passing (53 + ~20 from the Scope/manifest/error/annotations
  groups)
- Wrapper legitimately claims `amp_version: 1.1`, `amp_conformance: core`

### Milestone B — v1.2-draft metadata_filters & recall hardening (~26 more failures fixed, gets to ~99/154 = 64%)

This is mostly a port from `amp/python/amp-server/src/amp_server/server.py`
since AMP v1.2 was authored in this very repo.

**B1. `metadata_filters` (3 days)**
- Copy `_validate_metadata_filters`, `_eval_metadata_filter`,
  `_apply_metadata_filters`, `_is_filter_scalar`, the
  `_METADATA_FILTER_OPERATORS` set, the `_METADATA_FILTERS_MAX = 32` cap
  verbatim from `amp/python/amp-server/src/amp_server/server.py`.
- Wire into `amp.recall` as the post-retrieval predicate.
- Spec § references: §3.2.2.1 in `amp/spec/amp-v1.1.md`.

**B2. `top_k` oversampling rule (0.5 day)**
- When any post-retrieval filter is active, fetch `min(top_k * 10, 200)`
  candidates from `_smriti.recall`, filter, then slice to `top_k`.
- Constants `_RECALL_OVERSAMPLE_FACTOR = 10`, `_RECALL_OVERSAMPLE_MAX = 200`
  copied from the reference server.

**B3. `timestamp_after` / `timestamp_before` wiring (0.5 day)**
- Copy `_parse_iso8601` and `_row_passes_timestamp` from the reference
  server. Apply as part of the post-retrieval filter chain.

**B4. Metadata bag size cap (0.5 day)**
- Copy `_validate_metadata_bag` + `_METADATA_MAX_BYTES = 64 * 1024`.
- Apply on `amp.encode` (and on `amp.update` once that lands in Milestone
  C).
- AMP v1.1's `MemoryResult` carries a `metadata` field — the wrapper
  currently has no notion of caller-supplied metadata at encode time.
  Adding it requires accepting an optional `metadata: dict` parameter on
  `amp.encode` and applying it to the stored memory's `metadata` attribute
  post-encode. Smriti's `Memory` dataclass already has a `metadata` field
  (per the existing code-base context).

**Expected outcome after Milestone B:**
- ~99/154 passing
- Wrapper claims `amp_version: 1.2-draft`, `amp_conformance: core`
- All v1.1 conformance retained

### Milestone C — Full v1.2-draft (closes the remaining 56 skips, gets to 154/154)

The four big optional verbs. All of these have a complete reference
implementation in `amp/python/amp-server/src/amp_server/server.py` — the
work is binding them to `smriti-memcore`'s engine, not new design.

**C1. `amp.update` (2 days)**
- Port the verb including `_apply_merge_patch` (RFC 7396), transactional
  rollback on `palace.save()` failure, cross-scope `not_found` discipline,
  no-op detection (`no_change` status).
- Reference: spec §3.2.4, server lines ~926-1010.

**C2. `amp.batch_encode` (2 days)**
- Port `_encode_single_row` (with PR G's strict per-row validation:
  forbidden-key rejection, strict-bool `force`, strict-enum `source`,
  metadata-bag cap), `_BATCH_ENCODE_MAX_ENTRIES = 1000`, durable-save
  honesty (downgrade `stored` → `backend_error` if end-of-batch save
  fails).
- Reference: spec §3.2.5, server lines ~377-585.

**C3. `amp.export` (2 days)**
- The MXF NDJSON cursor-paginated export.
- Reference: spec §3.3.4 plus the MXF appendix.
- This is the heaviest of the four because it requires the cursor-encoding
  scheme.

**C4. `amp.import` (2 days)**
- The mirror of export with the four `on_conflict` policies
  (`skip`/`replace`/`fail_fast`/`fail_atomic`).
- `fail_atomic` is allowed to return `not_supported` for backends without
  transactional semantics — the wrapper SHOULD take that exemption rather
  than try to fake it.

**Expected outcome after Milestone C:**
- **154/154 passing** — Full v1.2-draft conformance
- Wrapper claims `amp_version: 1.2-draft`, `amp_conformance: full`

---

## Effort summary

| Milestone | Effort | Failures closed | Skips closed | Final pass rate |
|---|---|---:|---:|---:|
| A — v1.1 Core | ~5 days | ~20 | 0 | 47% |
| B — v1.2-draft Core | ~4.5 days | ~26 | 0 | 64% |
| C — v1.2-draft Full | ~8 days | 0 | 56 | **100%** |
| **Total** | **~17.5 days** | **45** | **56** | **154/154** |

Most of milestone C is straight porting from the AMP reference server. The
real design work is concentrated in milestone A — specifically, the
scope-to-storage-path mapping in A1. Once that's in, A2-A6 and all of B/C
are pattern-matching against the canonical reference implementation in
`amp/python/amp-server/src/amp_server/server.py`.

## Sequencing recommendation

1. **Land Milestone A as one PR.** It's the breaking-conformance fix — the
   wrapper currently misrepresents itself as v1.0-conformant when v1.0 was
   superseded six months ago. After this PR the project can honestly claim
   v1.1 Core conformance.

2. **Milestone B follows as a second PR.** Self-contained port of the v1.2
   recall surface. Doesn't change verb names, just enriches behaviour.

3. **Milestone C as four separate PRs** (one per verb) so each can land
   independently. The AMP repo PRs #9, #10, #4 are the templates — same
   schema-spec-impl-tests bundle, just transplanted.

4. **Compliance suite as CI gate.** Before each PR merges, the
   `amp/compliance/test_amp_server.py --server-cmd "python -m
   smriti_memcore.integrations.mcp_server"` invocation should be wired into
   nexus-memory's CI. The suite is stable; the wrapper just needs to keep
   pass-count monotonically non-decreasing.

## Open questions worth answering before starting

1. **Storage layout for multi-scope.** AMP reference server uses
   `~/.amp/scopes/<hash>/` with a per-scope SMRITI instance lazily
   initialised. Does the nexus-memory deployment story (per-user
   home-directory installs, hosted `nexus-memory` service) need a
   different layout — e.g. a shared SMRITI with per-scope room-prefix?
   Affects A1 implementation.

2. **Backwards-compat for the existing native `smriti_*` tools.** The
   wrapper exposes 12 native tools alongside the 6 AMP verbs. None of
   those break under the AMP work — should they stay? My read: yes,
   they're orthogonal. They get scope-awareness via the same
   `_normalize_scope` helper, but their surface is unchanged.

3. **`amp_version` declaration as the work lands.** During milestone A
   the wrapper can honestly say `1.1`; during milestone B → `1.2-draft`.
   Should each milestone bump the declared version, or hold until
   milestone C lands and declare `1.2-draft` once? My read: bump at each
   milestone so the manifest accurately tracks reality.

4. **Reference-server / wrapper relationship.** Once the wrapper is full
   v1.2-draft, it's structurally a peer of `amp/python/amp-server`. Worth
   discussing whether the two should merge — the AMP reference server is
   intentionally minimal (built-in SMRITI integration is the only
   backend it knows about), and the nexus-memory wrapper has richer
   native tools. There's a case for the AMP reference server becoming
   *just* the AMP verbs and the nexus-memory wrapper becoming the
   "reference impl + native tooling" superset. Tracked as a discussion
   item, not a deliverable.
