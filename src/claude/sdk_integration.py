"""Claude Code Python SDK integration.

Features:
- Native Claude Code SDK integration
- Async streaming support
- Tool execution management
- Session persistence
"""

import asyncio
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import structlog
from claude_agent_sdk import (
    AssistantMessage,
    ClaudeAgentOptions,
    ClaudeSDKError,
    CLIConnectionError,
    CLIJSONDecodeError,
    CLINotFoundError,
    Message,
    ProcessError,
    ResultMessage,
    ToolResultBlock,
    ToolUseBlock,
    UserMessage,
)
from claude_agent_sdk._errors import MessageParseError
from claude_agent_sdk._internal.message_parser import parse_message as _parse_message

from ..config.settings import Settings
from .exceptions import (
    ClaudeMCPError,
    ClaudeParsingError,
    ClaudeProcessError,
    ClaudeTimeoutError,
)

logger = structlog.get_logger()


def find_claude_cli(claude_cli_path: Optional[str] = None) -> Optional[str]:
    """Find Claude CLI in common locations."""
    import glob
    import shutil

    # First check if a specific path was provided via config or env
    if claude_cli_path:
        if os.path.exists(claude_cli_path) and os.access(claude_cli_path, os.X_OK):
            return claude_cli_path

    # Check CLAUDE_CLI_PATH environment variable
    env_path = os.environ.get("CLAUDE_CLI_PATH")
    if env_path and os.path.exists(env_path) and os.access(env_path, os.X_OK):
        return env_path

    # Check if claude is already in PATH
    claude_path = shutil.which("claude")
    if claude_path:
        return claude_path

    # Check common installation locations
    common_paths = [
        # NVM installations
        os.path.expanduser("~/.nvm/versions/node/*/bin/claude"),
        # Direct npm global install
        os.path.expanduser("~/.npm-global/bin/claude"),
        os.path.expanduser("~/node_modules/.bin/claude"),
        # System locations
        "/usr/local/bin/claude",
        "/usr/bin/claude",
        # Windows locations (for cross-platform support)
        os.path.expanduser("~/AppData/Roaming/npm/claude.cmd"),
    ]

    for pattern in common_paths:
        matches = glob.glob(pattern)
        if matches:
            # Return the first match
            return matches[0]

    return None


def update_path_for_claude(claude_cli_path: Optional[str] = None) -> bool:
    """Update PATH to include Claude CLI if found."""
    claude_path = find_claude_cli(claude_cli_path)

    if claude_path:
        # Add the directory containing claude to PATH
        claude_dir = os.path.dirname(claude_path)
        current_path = os.environ.get("PATH", "")

        if claude_dir not in current_path:
            os.environ["PATH"] = f"{claude_dir}:{current_path}"
            logger.info("Updated PATH for Claude CLI", claude_path=claude_path)

        return True

    return False


@dataclass
class ClaudeResponse:
    """Response from Claude Code SDK."""

    content: str
    session_id: str
    cost: float
    duration_ms: int
    num_turns: int
    is_error: bool = False
    error_type: Optional[str] = None
    tools_used: List[Dict[str, Any]] = field(default_factory=list)


@dataclass
class StreamUpdate:
    """Streaming update from Claude SDK."""

    type: str  # 'assistant', 'user', 'system', 'result'
    content: Optional[str] = None
    tool_calls: Optional[List[Dict]] = None
    metadata: Optional[Dict] = None


class ClaudeSDKManager:
    """Manage Claude Code SDK integration."""

    def __init__(self, config: Settings):
        """Initialize SDK manager with configuration."""
        self.config = config

        # Try to find and update PATH for Claude CLI
        if not update_path_for_claude(config.claude_cli_path):
            logger.warning(
                "Claude CLI not found in PATH or common locations. "
                "SDK may fail if Claude is not installed or not in PATH."
            )

        # Cache CLI path to avoid filesystem lookups on every request
        self._cached_cli_path: Optional[str] = (
            find_claude_cli(config.claude_cli_path) or "claude"
        )
        logger.info("Resolved Claude CLI path", cli_path=self._cached_cli_path)

        # Set up environment for Claude Code SDK if API key is provided
        # If no API key is provided, the SDK will use existing CLI authentication
        if config.anthropic_api_key_str:
            os.environ["ANTHROPIC_API_KEY"] = config.anthropic_api_key_str
            logger.info("Using provided API key for Claude SDK authentication")
        else:
            logger.info("No API key provided, using existing Claude CLI authentication")

    async def execute_command(
        self,
        prompt: str,
        working_directory: Path,
        session_id: Optional[str] = None,
        continue_session: bool = False,
        stream_callback: Optional[Callable[[StreamUpdate], None]] = None,
    ) -> ClaudeResponse:
        """Execute Claude Code command via SDK."""
        start_time = asyncio.get_event_loop().time()

        logger.info(
            "Starting Claude SDK command",
            working_directory=str(working_directory),
            session_id=session_id,
            continue_session=continue_session,
        )

        try:
            # Build Claude Agent options
            options = ClaudeAgentOptions(
                max_turns=self.config.claude_max_turns,
                cwd=str(working_directory),
                allowed_tools=self.config.claude_allowed_tools,
                disallowed_tools=self.config.claude_disallowed_tools,
                cli_path=self._cached_cli_path,
                permission_mode="bypassPermissions",
                sandbox=(
                    {
                        "enabled": self.config.sandbox_enabled,
                        "autoAllowBashIfSandboxed": True,
                        "excludedCommands": self.config.sandbox_excluded_commands or [],
                    }
                    if self.config.sandbox_enabled
                    else None
                ),
                system_prompt=(
                    f"All file operations must stay within {working_directory}. "
                    "Use relative paths."
                ),
            )

            # Pass MCP server configuration if enabled
            if self.config.enable_mcp and self.config.mcp_config_path:
                options.mcp_servers = self._load_mcp_config(self.config.mcp_config_path)
                logger.info(
                    "MCP servers configured",
                    mcp_config_path=str(self.config.mcp_config_path),
                )

            # Resume previous session if we have a session_id
            if session_id and continue_session:
                options.resume = session_id
                logger.info(
                    "Resuming previous session",
                    session_id=session_id,
                )

            # Collect messages by running claude CLI directly in --print mode.
            # The SDK streaming (--input-format stream-json) mode aborts tool
            # execution when stdin closes early, so we use --print instead.
            messages: List[Message] = []

            async def _run_client() -> None:
                import json as _json

                cmd = await self._build_print_command(options, prompt)

                proc = await asyncio.create_subprocess_exec(
                    *cmd,
                    stdin=asyncio.subprocess.DEVNULL,  # EOF immediately; prevents interactive prompts hanging
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.DEVNULL,
                    cwd=str(working_directory),
                    env=os.environ.copy(),
                )
                assert proc.stdout is not None

                # Idle timeout: if no output received for this many seconds, abort.
                # Catches interactive tools (EnterPlanMode, AskUserQuestion) that
                # hang waiting for user input that never arrives in --print mode.
                idle_timeout = 60.0

                while True:
                    try:
                        line_bytes = await asyncio.wait_for(
                            proc.stdout.readline(), timeout=idle_timeout
                        )
                    except asyncio.TimeoutError:
                        proc.kill()
                        await proc.wait()
                        raise ClaudeTimeoutError(
                            f"Claude stopped producing output for {idle_timeout:.0f}s "
                            "(possibly waiting for interactive input). "
                            "Try rephrasing without 'plan mode' or similar interactive commands."
                        )
                    if not line_bytes:
                        break
                    line = line_bytes.decode().strip()
                    if not line:
                        continue
                    try:
                        data = _json.loads(line)
                        message = _parse_message(data)
                    except MessageParseError as e:
                        if "Unknown message type" in str(e):
                            logger.info(
                                "Skipping unknown SDK message type",
                                error=str(e),
                            )
                            continue
                        raise
                    except Exception:
                        continue
                    messages.append(message)

                    if stream_callback:
                        try:
                            await self._handle_stream_message(message, stream_callback)
                        except Exception as callback_error:
                            logger.warning(
                                "Stream callback failed",
                                error=str(callback_error),
                                error_type=type(callback_error).__name__,
                            )

                await proc.wait()

            # Execute with timeout
            await asyncio.wait_for(
                _run_client(),
                timeout=self.config.claude_timeout_seconds,
            )

            # Extract cost, tools, and session_id from result message
            cost = 0.0
            tools_used: List[Dict[str, Any]] = []
            claude_session_id = None
            result_content = None
            for message in messages:
                if isinstance(message, ResultMessage):
                    cost = getattr(message, "total_cost_usd", 0.0) or 0.0
                    claude_session_id = getattr(message, "session_id", None)
                    result_content = getattr(message, "result", None)
                    tools_used = self._extract_tools_from_messages(messages)
                    break

            # Fallback: extract session_id from StreamEvent messages if
            # ResultMessage didn't provide one (can happen with some CLI versions)
            if not claude_session_id:
                for message in messages:
                    msg_session_id = getattr(message, "session_id", None)
                    if msg_session_id and not isinstance(message, ResultMessage):
                        claude_session_id = msg_session_id
                        logger.info(
                            "Got session ID from stream event (fallback)",
                            session_id=claude_session_id,
                        )
                        break

            # Calculate duration
            duration_ms = int((asyncio.get_event_loop().time() - start_time) * 1000)

            # Use Claude's session_id if available, otherwise fall back
            final_session_id = claude_session_id or session_id or ""

            if claude_session_id and claude_session_id != session_id:
                logger.info(
                    "Got session ID from Claude",
                    claude_session_id=claude_session_id,
                    previous_session_id=session_id,
                )

            # DEBUG: log all messages
            for _m in messages:
                logger.info(
                    "DEBUG message",
                    msg_type=type(_m).__name__,
                    content_preview=str(getattr(_m, "content", ""))[:200],
                    result=str(getattr(_m, "result", ""))[:200],
                )

            # Use ResultMessage.result if available, fall back to message extraction
            content = (
                result_content
                if result_content is not None
                else self._extract_content_from_messages(messages)
            )
            # If Claude responded with only tool calls and no text, avoid sending
            # an empty message to Telegram (which rejects it with 400).
            if not content or not content.strip():
                content = "✅ Done."

            return ClaudeResponse(
                content=content,
                session_id=final_session_id,
                cost=cost,
                duration_ms=duration_ms,
                num_turns=len(
                    [
                        m
                        for m in messages
                        if isinstance(m, (UserMessage, AssistantMessage))
                    ]
                ),
                tools_used=tools_used,
            )

        except asyncio.TimeoutError:
            logger.error(
                "Claude SDK command timed out",
                timeout_seconds=self.config.claude_timeout_seconds,
            )
            raise ClaudeTimeoutError(
                f"Claude SDK timed out after {self.config.claude_timeout_seconds}s"
            )

        except CLINotFoundError as e:
            logger.error("Claude CLI not found", error=str(e))
            error_msg = (
                "Claude Code not found. Please ensure Claude is installed:\n"
                "  npm install -g @anthropic-ai/claude-code\n\n"
                "If already installed, try one of these:\n"
                "  1. Add Claude to your PATH\n"
                "  2. Create a symlink: ln -s $(which claude) /usr/local/bin/claude\n"
                "  3. Set CLAUDE_CLI_PATH environment variable"
            )
            raise ClaudeProcessError(error_msg)

        except ProcessError as e:
            error_str = str(e)
            logger.error(
                "Claude process failed",
                error=error_str,
                exit_code=getattr(e, "exit_code", None),
            )
            # Check if the process error is MCP-related
            if "mcp" in error_str.lower():
                raise ClaudeMCPError(f"MCP server error: {error_str}")
            raise ClaudeProcessError(f"Claude process error: {error_str}")

        except CLIConnectionError as e:
            error_str = str(e)
            logger.error("Claude connection error", error=error_str)
            # Check if the connection error is MCP-related
            if "mcp" in error_str.lower() or "server" in error_str.lower():
                raise ClaudeMCPError(f"MCP server connection failed: {error_str}")
            raise ClaudeProcessError(f"Failed to connect to Claude: {error_str}")

        except CLIJSONDecodeError as e:
            logger.error("Claude SDK JSON decode error", error=str(e))
            raise ClaudeParsingError(f"Failed to decode Claude response: {str(e)}")

        except ClaudeSDKError as e:
            logger.error("Claude SDK error", error=str(e))
            raise ClaudeProcessError(f"Claude SDK error: {str(e)}")

        except Exception as e:
            # Handle ExceptionGroup from TaskGroup operations (Python 3.11+)
            if type(e).__name__ == "ExceptionGroup" or hasattr(e, "exceptions"):
                logger.error(
                    "Task group error in Claude SDK",
                    error=str(e),
                    error_type=type(e).__name__,
                    exception_count=len(getattr(e, "exceptions", [])),
                    exceptions=[
                        str(ex) for ex in getattr(e, "exceptions", [])[:3]
                    ],  # Log first 3 exceptions
                )
                # Extract the most relevant exception from the group
                exceptions = getattr(e, "exceptions", [e])
                main_exception = exceptions[0] if exceptions else e
                raise ClaudeProcessError(
                    f"Claude SDK task error: {str(main_exception)}"
                )

            # Check if it's an ExceptionGroup disguised as a regular exception
            elif hasattr(e, "__notes__") and "TaskGroup" in str(e):
                logger.error(
                    "TaskGroup related error in Claude SDK",
                    error=str(e),
                    error_type=type(e).__name__,
                )
                raise ClaudeProcessError(f"Claude SDK task error: {str(e)}")

            else:
                logger.error(
                    "Unexpected error in Claude SDK",
                    error=str(e),
                    error_type=type(e).__name__,
                )
                raise ClaudeProcessError(f"Unexpected error: {str(e)}")

    async def _build_print_command(
        self, options: ClaudeAgentOptions, prompt: str
    ) -> List[str]:
        """Build claude CLI command using --print mode (no stdin protocol)."""
        import json as _json

        cli_path = str(options.cli_path) if options.cli_path else self._cached_cli_path
        cmd: List[str] = [
            cli_path,
            "--output-format",
            "stream-json",
            "--verbose",
            "--print",
            prompt,
        ]
        if options.permission_mode:
            cmd.extend(["--permission-mode", options.permission_mode])
        if options.max_turns:
            cmd.extend(["--max-turns", str(options.max_turns)])
        if options.system_prompt and isinstance(options.system_prompt, str):
            cmd.extend(["--system-prompt", options.system_prompt])
        if options.allowed_tools:
            cmd.extend(["--allowedTools", ",".join(options.allowed_tools)])
        if options.disallowed_tools:
            cmd.extend(["--disallowedTools", ",".join(options.disallowed_tools)])
        if options.resume:
            cmd.extend(["--resume", options.resume])
        if options.sandbox:
            settings_json = _json.dumps({"sandbox": options.sandbox})
            cmd.extend(["--settings", settings_json])
        if options.mcp_servers and isinstance(options.mcp_servers, dict):
            mcp_json = _json.dumps({"mcpServers": options.mcp_servers})
            cmd.extend(["--mcp-config", mcp_json])
        # Always suppress reading project/user settings that could conflict
        cmd.extend(["--setting-sources", ""])
        return cmd

    async def _handle_stream_message(
        self, message: Message, stream_callback: Callable[[StreamUpdate], None]
    ) -> None:
        """Handle streaming message from claude-agent-sdk."""
        try:
            if isinstance(message, AssistantMessage):
                # Extract content from assistant message
                content = getattr(message, "content", [])
                text_parts = []
                tool_calls = []

                if content and isinstance(content, list):
                    for block in content:
                        if isinstance(block, ToolUseBlock):
                            tool_calls.append(
                                {
                                    "name": getattr(block, "name", "unknown"),
                                    "input": getattr(block, "input", {}),
                                    "id": getattr(block, "id", None),
                                }
                            )
                        elif hasattr(block, "text"):
                            text_parts.append(block.text)

                if text_parts or tool_calls:
                    update = StreamUpdate(
                        type="assistant",
                        content=("\n".join(text_parts) if text_parts else None),
                        tool_calls=tool_calls if tool_calls else None,
                    )
                    await stream_callback(update)
                elif content:
                    # Fallback for non-list content
                    update = StreamUpdate(
                        type="assistant",
                        content=str(content),
                    )
                    await stream_callback(update)

            elif isinstance(message, UserMessage):
                content = getattr(message, "content", "")
                if content:
                    update = StreamUpdate(
                        type="user",
                        content=content,
                    )
                    await stream_callback(update)

        except Exception as e:
            logger.warning("Stream callback failed", error=str(e))

    def _extract_content_from_messages(self, messages: List[Message]) -> str:
        """Extract content from message list."""
        text_parts: List[str] = []
        tool_result_parts: List[str] = []

        for message in messages:
            if isinstance(message, AssistantMessage):
                content = getattr(message, "content", [])
                if content and isinstance(content, list):
                    for block in content:
                        if hasattr(block, "text"):
                            text_parts.append(block.text)
                elif content:
                    text_parts.append(str(content))
            elif isinstance(message, UserMessage):
                # Collect tool results as fallback when Claude produces no text
                content = getattr(message, "content", [])
                if content and isinstance(content, list):
                    for block in content:
                        if isinstance(block, ToolResultBlock) and not block.is_error:
                            raw = block.content
                            if isinstance(raw, str) and raw.strip():
                                tool_result_parts.append(raw.strip())
                            elif isinstance(raw, list):
                                for item in raw:
                                    if (
                                        isinstance(item, dict)
                                        and item.get("text", "").strip()
                                    ):
                                        tool_result_parts.append(item["text"].strip())

        # Prefer assistant text; fall back to tool output so commands like
        # "git log" show their output instead of "✅ Done."
        if text_parts:
            return "\n".join(text_parts)
        return "\n\n".join(tool_result_parts)

    def _extract_tools_from_messages(
        self, messages: List[Message]
    ) -> List[Dict[str, Any]]:
        """Extract tools used from message list."""
        tools_used = []
        current_time = asyncio.get_event_loop().time()

        for message in messages:
            if isinstance(message, AssistantMessage):
                content = getattr(message, "content", [])
                if content and isinstance(content, list):
                    for block in content:
                        if isinstance(block, ToolUseBlock):
                            tools_used.append(
                                {
                                    "name": getattr(block, "name", "unknown"),
                                    "timestamp": current_time,
                                    "input": getattr(block, "input", {}),
                                }
                            )

        return tools_used

    def _load_mcp_config(self, config_path: Path) -> Dict[str, Any]:
        """Load MCP server configuration from a JSON file.

        The new claude-agent-sdk expects mcp_servers as a dict, not a file path.
        """
        import json

        try:
            with open(config_path) as f:
                config_data = json.load(f)
            return config_data.get("mcpServers", {})
        except (json.JSONDecodeError, OSError) as e:
            logger.error(
                "Failed to load MCP config", path=str(config_path), error=str(e)
            )
            return {}

    def get_active_process_count(self) -> int:
        """Get number of active sessions (always 0, per-request clients)."""
        return 0
