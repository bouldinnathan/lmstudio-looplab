# Local API and dependency boundaries

The companion binds only to IPv4 loopback and requires a generated bearer token for every endpoint except `/health`. Browser Origin headers are rejected. Tokens have mode 0600 and stay in the user's state directory. The plugin accepts only loopback HTTP destinations and rejects redirects. It reads only the standard local token file unless a token is explicitly configured.

The server accepts known model IDs from its read-only inventory, rather than arbitrary model paths from HTTP clients. Request bodies are limited to 2 MiB. Only allowlisted inference parameters reach the child process. Generated benchmark text is JSON data and is never executed. No prompts or model files are sent to external services. Source builds and npm installation use the network for dependencies.

This is local development software, not a hardened multi-user hosting service. A process running as the same user can read the access token and control the companion. Keep the service on loopback. Do not place an unauthenticated public proxy in front of it.

The optional LM Studio Engine Protocol integration binds its internal proxy only to loopback and honors a runtime API key when LM Studio supplies one. External clients use LM Studio's own API, network binding and authentication. The proxy accepts only the inference/tokenization/inspection endpoints needed by the fixed model, rejects browser origins, limits JSON requests to 2 MiB, and disables prompt-prefix reuse for each prediction. Model replacement and cache-restoration endpoints are unavailable.

Ordinary-model entries are weight-free descriptors. Their hashes and current source/runtime conformance records are checked before loading. Stock engines cannot load them as real models, so switching away from the LoopLab runtime cannot silently produce baseline responses under looped names. The installer reuses official native bindings through local hard links and never modifies their contents or redistributes them.

## Known dependency advisory

At initial publication, `npm audit` reports GHSA-mrhx-c7xc-m3w5 for the LM Studio SDK's Zod 3 dependency (oversized constrained collection validation can exhaust memory). The current SDK dependency line still requests Zod 3; replacing it with Zod 4 would be an unverified incompatible change. The lockfile records the tested SDK. LoopLab restricts requests and connects only to its local bounded-response service, but this does not constitute a fix for the SDK advisory. Re-evaluate the advisory when upgrading the SDK.

https://github.com/advisories/GHSA-mrhx-c7xc-m3w5

## Reporting

Use GitHub's private vulnerability reporting if enabled, or report a minimal non-sensitive issue. Do not attach bearer tokens, personal chats, complete local inventories, or proprietary model files. `looplab export-report` produces a deliberately limited public summary.
