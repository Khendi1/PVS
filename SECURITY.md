# Security Policy

## Supported Versions

Security updates are provided for the following versions:

| Version | Supported          |
| ------- | ------------------ |
| 0.1.x   | :white_check_mark: |
| < 0.1   | :x:                |

## Reporting a Vulnerability

If you discover a security vulnerability, please report it **privately** rather
than opening a public issue. Email the maintainer at:

**khenderson614@gmail.com**

Please include:

- A description of the vulnerability and its potential impact
- Steps to reproduce, or a proof of concept
- The version or commit affected

You can expect an acknowledgement of your report, and we will work with you on a
fix and coordinated disclosure where appropriate.

## Scope and Context

This project is a **local-network performance tool** intended for live visual
art on a trusted machine or LAN. By design:

- The REST API binds to **`127.0.0.1` (localhost) by default**. It is only
  reachable from other machines if you explicitly pass `--api-host 0.0.0.0`.
- The API has **no authentication**. Only expose it on networks you control,
  and never bind it to a public interface.
- The optional AI agent and Ollama run locally; no data leaves your machine.

When binding to a non-loopback address (e.g. for collaboration over a LAN),
place the synth behind a firewall or trusted network segment and treat the API
as fully unauthenticated.
