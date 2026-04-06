"""sortie-mcp — Campaign orchestration MCP server for AI agents."""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("sortie-mcp")
except PackageNotFoundError:
    __version__ = "0.0.0-dev"
