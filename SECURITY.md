# Security Policy

## Supported versions

Callosum is a single-maintainer, loopback-only project under active
development. Only the latest commit on `main` receives security
consideration; there are no separate maintained release lines.

## Reporting a vulnerability

Please report security vulnerabilities **privately**, not as a public
issue.

The preferred channel is GitHub's built-in **Private Security Advisory**
feature on this repository:

1. Open the repository on GitHub.
2. Go to the **Security** tab → **Advisories** → **New draft advisory**.
3. Describe the issue, reproduction steps, and impact.

If you cannot use that mechanism, contact the maintainer directly via
their GitHub profile (`@rohankhanna`).

Please include:

- A description of the vulnerability and its impact.
- Minimal reproduction steps (commands, request, or config).
- The affected version / commit, if known.
- Any suggested remediation.

## Scope

In scope:

- Callosum's own source (`src/callosum/`), its HTTP surface
  (`/v1/responses`, `/v1/chat/completions`, and related routes), and its
  configuration handling.
- Anything that could let a loopback client escalate beyond the
  single-operator, single-machine trust boundary the project assumes.

Out of scope:

- Vulnerabilities in **upstream backends** (model providers, local model
  runtimes, proxy gateways) or in **third-party dependencies** — report
  those to the upstream project. Callosum routes requests to these
  backends but does not own their security posture.
- Issues that require an attacker to already be on the loopback
  interface with the operator's credentials — by design the endpoint
  binds to `127.0.0.1` and is intended for a single operator on a single
  machine.
- Denial of service against the operator's own machine via the
  loopback interface.

## Response expectations

This is a single-maintainer project, so treat the timelines below as a
best-effort commitment rather than a contractual SLA:

- **Acknowledgement:** within 5 business days.
- **Initial assessment:** within 14 days, including a severity call and
  a planned fix path.
- **Fix:** coordinated with the reporter; a security fix is released as
  soon as a correct, verified patch is ready.

Please do not publicly disclose the issue until a fix is available and
we have agreed on a disclosure date. I will credit you in the advisory
unless you prefer to remain anonymous.

## What I ask of reporters

- Give me reasonable time to investigate and fix before any public
  disclosure.
- Avoid automated mass-scanning that could destabilize the operator's
  environment.
- Test your reproduction against the latest `main` if possible.

## Safe harbor

Good-faith reporting of security issues in accordance with this policy
is welcomed. I will not pursue legal action for respectful, good-faith
research that respects the loopback-only, single-operator scope of this
project.