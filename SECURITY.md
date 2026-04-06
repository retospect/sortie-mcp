# Security Policy

## Reporting a Vulnerability

If you discover a security vulnerability in sortie-mcp, please report it
responsibly via [GitHub Security Advisories](https://github.com/openclaw/sortie-mcp/security/advisories/new).

**Do not** open a public issue for security vulnerabilities.

## Supported Versions

| Version | Supported |
|---------|-----------|
| 0.x     | ✅ Latest |

## Security Considerations

- **Database credentials**: Always use environment variables (`DATABASE_URL`),
  never hardcode credentials.
- **MCP tool visibility**: All tools are visible to all MCP clients. Use prompt
  enforcement for v1; separate MCP instances for stricter isolation.
- **LLM API keys**: Pass via `LITELLM_URL` and `LITELLM_KEY` environment variables.
- **PG NOTIFY**: Notifications traverse the PostgreSQL connection. Use TLS for
  production deployments.
