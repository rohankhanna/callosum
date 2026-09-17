# Extending Callosum

Callosum has two main extension points:

- **Backends** — add a new upstream model endpoint.
- **Routing components** — add or replace a predictor, selector, or capability provider.

## Add a backend

1. Implement the `callosum.backend.Backend` protocol.
2. Expose `advertised_models`, health, usage, and quota metadata.
3. Implement non-streaming and streaming calls for the API shapes you support.
4. Register the backend in `callosum.wiring.build_backend` or in
   `callosum.__main__.build_runtime_backends`.
5. Add hermetic unit or contract tests. Live end-to-end tests are optional and
   should stay opt-in.

For a complete protocol example, see `src/callosum/backend.py`. For concrete
implementations, see `src/callosum/backends/codex_gateway.py`,
`src/callosum/backends/ollama_cloud.py`, and
`src/callosum/backends/litellm_gateway.py`.

## Extend routing behavior

Callosum's routing pipeline is deliberately protocol-based:

- Features extraction: `src/callosum/routing/features.py`
- Capability filtering: `src/callosum/routing/capability.py`
- Quality prediction: `src/callosum/routing/predictor/`
- Cell selection: `src/callosum/routing/selector/`
- Router orchestration: `src/callosum/routing/router.py`

Add a new implementation of the relevant protocol, wire it in the factory, and
add tests that demonstrate the behavior under both cold-start and learned-data
conditions.

## Keep the public surface stable

If you add a new backend kind or routing component, document it in
`ARCHITECTURE.md` and `README.md`, and add tests that pin the public HTTP and
configuration contract.
