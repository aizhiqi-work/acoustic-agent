# Security Policy

## Supported Version

The latest commit on the default branch is the only supported development
version while Acoustic Agent remains pre-1.0.

## Reporting A Vulnerability

Do not open a public issue for a vulnerability that could expose local files,
execute unintended code, or make the development HTTP server reachable outside
the intended network. Use the repository's private security-advisory channel.

The Web workbench is a local development server. It has no authentication or
TLS and should bind to `127.0.0.1` unless it is placed behind an appropriately
secured reverse proxy. Do not expose it directly to the public internet.
