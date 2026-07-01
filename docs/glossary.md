# Glossary — canonical vocabulary for model I/O and routing

This file is the single source of truth for what we mean by the things that
go **into** a model and come **out** of a model, plus the routing terms those
get confused with. When code, docs, commit messages, or discussion need one of
these concepts, use the term defined here. Retire ad-hoc synonyms ("prose",
"the prompt", "the model picked") in favor of these.

## A request

**Request** — one HTTP call to a model backend; what the router treats as one
**turn**. A request carries *inputs* and returns *outputs*.

**Session** — an ordered series of requests sharing a `session_id`. Peer-quality
opinions are scoped to a single session; a turn may only judge prior turns of
the same session.

## Inputs (what goes into the model)

Everything below is part of the request body / context window the model reads.

| Term | Meaning | Common confusion |
|---|---|---|
| **Instructions** | System/developer guidance prepended to the context. | — |
| **History** | The prior turns of the session — each prior turn's inputs *and* outputs (user input, message text, tool calls, tool results). | "the prompt" is vague; say *history* or *user input*. |
| **User input** | The current task/message the model is being asked to act on this turn. | — |
| **Tool definitions** | The function schemas offered to the model ("here are the tools you *may* call"). | Tool **definitions** are *input*; tool **calls** are *output*. This is the big one. |
| **Sampling controls** | Non-content request parameters: `model` id, `reasoning_effort`, temperature, `stream`. | Not content, but they shape the request and the served **cell**. |

## Outputs (what comes out of the model) — three channels

A model response has up to three distinct channels. Keeping them separate ends
most of the confusion.

1. **Reasoning** (a.k.a. *thinking*) — the model's internal deliberation
   tokens. Often hidden or summarized; not addressed to the user. **Not**
   user-facing message content.
2. **Message text** — the natural-language answer addressed to the user/caller
   (the `output_text` / message `content`). **This is the channel we used to
   loosely call "prose." Always say *message text*.**
3. **Tool calls** — structured function invocations: a tool *name* plus JSON
   *arguments*. Machine-directed; not natural language.

A single turn's output may contain any combination of the three.

## Turn types (classified by output)

| Term | Output it carries | Judgeable by peer-quality? |
|---|---|---|
| **Text turn** | Has **message text** (ch. 2); may also have reasoning. | **Yes** — it has a natural-language answer to rate, and a text slot to carry an opinion marker. |
| **Tool turn** | Only **tool calls** (ch. 3), no message text. The dominant kind in Codex coding loops (run shell, edit file, read file). | **No** — no answer to rate, no text slot for a marker. |

> Why this matters: peer-quality opinions live in the **message-text** channel
> and judge prior **message text**. Codex sessions are overwhelmingly **tool
> turns**, so cross-cell **text-turn** pairs — the only thing the opinion matrix
> can feed on — are doubly scarce. See `docs/architecture/ce_loop.md` and
> work tracker .

## Routing vocabulary (the terms these got tangled with)

| Term | Meaning |
|---|---|
| **Cell** | A `(model, reasoning_effort)` pair — the unit the router selects and the matrix is indexed by. e.g. `model-a0e8/xhigh`. |
| **Pass-through** | The request named a concrete model; callosum honors it without re-routing. `requested_model == model`. |
| **Selector** | A virtual model name the *client* sends to delegate the choice to callosum: `callosum:auto`, `callosum:remote-only`, `callosum:local-only`, `callosum:offline`. The router resolves it to a concrete **cell**. |
| **Pin** | A request fixed to a concrete model/cell — either by the client naming it, or by an operator backend pin. A **selector is the opposite of a pin**: it hands the choice to the router. |
| **Organic traffic** | Real user/Codex requests. |
| **Exploration vs exploitation** | *Exploitation* = route to the cell believed best now. *Exploration* = deliberately route elsewhere to *learn* a cell's quality. You cannot route "cheap when it's just as good" until exploration has *measured* "just as good." Quality data is the price of efficiency, paid up front. |
| **Exploration quota** | A per-cell minimum-usage floor on organic traffic that forces under-covered cells to get a bootstrap/maintenance share of eligible turns. The live replacement for the removed synthetic tier. See `docs/architecture/exploration_quota.md`. |
| **Quality signal** | `quality_score` (-1/0/+1) per request, populated by a labeler. When empty, the predictor is uniform and the cost selector collapses all traffic to one cell. |
| **Peer-quality opinion** | An in-band judgment one cell emits about a *prior different-cell text turn* in the same session, wrapped in a `<<qop ...>>` marker (message-text channel), stripped before the user sees it. Feeds the cross-model quality matrix. |

## Hidden payload vocabulary

| Term | Meaning |
|---|---|
| **Hidden Model Payload** | Model-visible content that Callosum intentionally injects upstream but keeps hidden from Codex/UI on the way back down. Examples: provenance tags, hidden critique instructions, nonce-bearing opinion markers. |
| **Veiling** | The mechanism that makes a Hidden Model Payload invisible to the client-facing side while still present in the upstream model-visible envelope. |
| **Private Control State** | Callosum-only sidecar state that never leaves the proxy. Examples: routing decision, token ledger, judgment debt, request ids, transform provenance, experiment cohort. |
| **Truth Log** | The durable log record that stores the exact upstream model-visible payload, the sanitized client-visible transcript, and the transform metadata needed to reconstruct what was sent. |

## Overloaded names — disambiguate explicitly

| Term | Two meanings |
|---|---|
| **`/status`** | (1) **callosum's HTTP endpoint** `GET http://127.0.0.1:8765/status` — a JSON document of backend health, quota, router config, and the `router.exploration_quota` / `router.peer_quality_shadow` reports. Auth-exempt (only `/v1/*` and `/diagnose/*` require an API key). (2) The **CLI slash command** typed inside Codex or Claude Code — unrelated to callosum. Always write "callosum's `GET /status` endpoint" vs "the CLI `/status` command". |
| **`/model`** | (1) The **Codex CLI slash command** that opens the model picker (driven by callosum's reconciled catalog JSON). (2) Not a callosum HTTP route — callosum advertises models at `GET /v1/models`. |
| **Selector** | (1) **Client routing selector** — a `callosum:` model id (`selectors.py`). (2) **Cell selector** — the `CostWeightedSelector` *select* stage inside the router (`routing/selector/`). Different layers. |
