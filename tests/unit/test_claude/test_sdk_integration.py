"""Test Claude SDK integration."""

import asyncio
import json
import os
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from claude_agent_sdk import AssistantMessage, ResultMessage, TextBlock

from src.claude.sdk_integration import ClaudeResponse, ClaudeSDKManager, StreamUpdate
from src.config.settings import Settings


# ---------------------------------------------------------------------------
# Helpers: produce JSON bytes in the format --output-format stream-json uses
# ---------------------------------------------------------------------------

def _assistant_json(text="Test response") -> bytes:
    return (
        json.dumps({
            "type": "assistant",
            "message": {
                "id": "msg_test",
                "type": "message",
                "role": "assistant",
                "model": "claude-sonnet-4-20250514",
                "content": [{"type": "text", "text": text}],
                "stop_reason": "end_turn",
                "stop_sequence": None,
                "usage": {"input_tokens": 10, "output_tokens": 5},
            },
        })
        + "\n"
    ).encode()


def _result_json(**overrides) -> bytes:
    data = {
        "type": "result",
        "subtype": "success",
        "result": "Success",
        "session_id": "test-session",
        "total_cost_usd": 0.05,
        "duration_ms": 1000,
        "duration_api_ms": 800,
        "is_error": False,
        "num_turns": 1,
    }
    data.update(overrides)
    return (json.dumps(data) + "\n").encode()


def _make_mock_process(*line_bytes: bytes) -> MagicMock:
    """Return a mock subprocess whose stdout yields the given byte lines."""
    remaining = list(line_bytes) + [b""]  # b"" signals EOF

    async def readline():
        return remaining.pop(0) if remaining else b""

    mock_proc = MagicMock()
    mock_proc.stdout = MagicMock()
    mock_proc.stdout.readline = AsyncMock(side_effect=readline)
    mock_proc.wait = AsyncMock(return_value=0)
    mock_proc.kill = MagicMock()
    return mock_proc


def _subprocess_factory(*line_bytes: bytes, capture_cmd: list | None = None):
    """Return an async callable suitable for patching asyncio.create_subprocess_exec."""

    async def _create(*cmd, **kwargs):
        if capture_cmd is not None:
            capture_cmd.extend(cmd)
        return _make_mock_process(*line_bytes)

    return _create


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestClaudeSDKManager:
    """Test Claude SDK manager."""

    @pytest.fixture
    def config(self, tmp_path):
        """Create test config without API key."""
        return Settings(
            telegram_bot_token="test:token",
            telegram_bot_username="testbot",
            approved_directory=tmp_path,
            claude_timeout_seconds=2,
        )

    @pytest.fixture
    def sdk_manager(self, config):
        """Create SDK manager."""
        return ClaudeSDKManager(config)

    # ------------------------------------------------------------------
    # Initialisation
    # ------------------------------------------------------------------

    async def test_sdk_manager_initialization_with_api_key(self, tmp_path):
        """Test SDK manager initialization with API key."""
        config_with_key = Settings(
            telegram_bot_token="test:token",
            telegram_bot_username="testbot",
            approved_directory=tmp_path,
            claude_timeout_seconds=2,
            anthropic_api_key="test-api-key",
        )

        original_api_key = os.environ.get("ANTHROPIC_API_KEY")
        try:
            ClaudeSDKManager(config_with_key)
            assert os.environ.get("ANTHROPIC_API_KEY") == "test-api-key"
        finally:
            if original_api_key:
                os.environ["ANTHROPIC_API_KEY"] = original_api_key
            elif "ANTHROPIC_API_KEY" in os.environ:
                del os.environ["ANTHROPIC_API_KEY"]

    async def test_sdk_manager_initialization_without_api_key(self, config):
        """Test SDK manager initialization without API key (uses CLI auth)."""
        original_api_key = os.environ.get("ANTHROPIC_API_KEY")
        try:
            if "ANTHROPIC_API_KEY" in os.environ:
                del os.environ["ANTHROPIC_API_KEY"]
            ClaudeSDKManager(config)
            assert config.anthropic_api_key_str is None
        finally:
            if original_api_key:
                os.environ["ANTHROPIC_API_KEY"] = original_api_key

    # ------------------------------------------------------------------
    # execute_command behaviour
    # ------------------------------------------------------------------

    async def test_execute_command_success(self, sdk_manager):
        """Test successful command execution."""
        factory = _subprocess_factory(
            _assistant_json("Test response"),
            _result_json(session_id="test-session", total_cost_usd=0.05),
        )
        with patch("asyncio.create_subprocess_exec", side_effect=factory):
            response = await sdk_manager.execute_command(
                prompt="Test prompt",
                working_directory=Path("/test"),
                session_id="test-session",
            )

        assert isinstance(response, ClaudeResponse)
        assert response.session_id == "test-session"
        assert response.duration_ms >= 0
        assert not response.is_error
        assert response.cost == 0.05

    async def test_execute_command_uses_result_content(self, sdk_manager):
        """Test that ResultMessage.result is used for content when available."""
        factory = _subprocess_factory(
            _assistant_json("Assistant text"),
            _result_json(result="Final result from ResultMessage"),
        )
        with patch("asyncio.create_subprocess_exec", side_effect=factory):
            response = await sdk_manager.execute_command(
                prompt="Test prompt",
                working_directory=Path("/test"),
            )

        assert response.content == "Final result from ResultMessage"

    async def test_execute_command_falls_back_to_messages(self, sdk_manager):
        """Test fallback to message extraction when result is None."""
        factory = _subprocess_factory(
            _assistant_json("Extracted from messages"),
            _result_json(result=None),
        )
        with patch("asyncio.create_subprocess_exec", side_effect=factory):
            response = await sdk_manager.execute_command(
                prompt="Test prompt",
                working_directory=Path("/test"),
            )

        assert response.content == "Extracted from messages"

    async def test_execute_command_with_streaming(self, sdk_manager):
        """Test command execution with streaming callback."""
        stream_updates = []

        async def stream_callback(update: StreamUpdate):
            stream_updates.append(update)

        factory = _subprocess_factory(
            _assistant_json("Test response"),
            _result_json(),
        )
        with patch("asyncio.create_subprocess_exec", side_effect=factory):
            await sdk_manager.execute_command(
                prompt="Test prompt",
                working_directory=Path("/test"),
                stream_callback=stream_callback,
            )

        assert len(stream_updates) > 0
        assert any(u.type == "assistant" for u in stream_updates)

    async def test_execute_command_timeout(self, sdk_manager):
        """Test command execution timeout."""
        from src.claude.exceptions import ClaudeTimeoutError

        async def hanging_readline():
            await asyncio.sleep(10)  # exceeds 2 s timeout
            return b""

        mock_proc = MagicMock()
        mock_proc.stdout = MagicMock()
        mock_proc.stdout.readline = AsyncMock(side_effect=hanging_readline)
        mock_proc.wait = AsyncMock(return_value=0)
        mock_proc.kill = MagicMock()

        async def _create(*cmd, **kwargs):
            return mock_proc

        with patch("asyncio.create_subprocess_exec", side_effect=_create):
            with pytest.raises(ClaudeTimeoutError):
                await sdk_manager.execute_command(
                    prompt="Test prompt",
                    working_directory=Path("/test"),
                )

    def test_get_active_process_count(self, sdk_manager):
        """Test active process count is always 0."""
        assert sdk_manager.get_active_process_count() == 0

    # ------------------------------------------------------------------
    # CLI flag verification via captured command
    # ------------------------------------------------------------------

    async def test_execute_command_passes_mcp_config(self, tmp_path):
        """Test that MCP config is passed to the CLI when enabled."""
        mcp_config_file = tmp_path / "mcp_config.json"
        mcp_config_file.write_text(
            '{"mcpServers": {"test-server": {"command": "echo", "args": ["hello"]}}}'
        )

        config = Settings(
            telegram_bot_token="test:token",
            telegram_bot_username="testbot",
            approved_directory=tmp_path,
            claude_timeout_seconds=2,
            enable_mcp=True,
            mcp_config_path=str(mcp_config_file),
        )
        manager = ClaudeSDKManager(config)

        captured_cmd: list = []
        factory = _subprocess_factory(
            _assistant_json("Test response"),
            _result_json(total_cost_usd=0.01),
            capture_cmd=captured_cmd,
        )
        with patch("asyncio.create_subprocess_exec", side_effect=factory):
            await manager.execute_command(
                prompt="Test prompt",
                working_directory=tmp_path,
            )

        assert "--mcp-config" in captured_cmd
        mcp_idx = captured_cmd.index("--mcp-config")
        mcp_data = json.loads(captured_cmd[mcp_idx + 1])
        assert "test-server" in mcp_data.get("mcpServers", {})

    async def test_execute_command_no_mcp_when_disabled(self, sdk_manager):
        """Test that MCP config is NOT passed when MCP is disabled."""
        captured_cmd: list = []
        factory = _subprocess_factory(
            _assistant_json("Test response"),
            _result_json(total_cost_usd=0.01),
            capture_cmd=captured_cmd,
        )
        with patch("asyncio.create_subprocess_exec", side_effect=factory):
            await sdk_manager.execute_command(
                prompt="Test prompt",
                working_directory=Path("/test"),
            )

        assert "--mcp-config" not in captured_cmd

    async def test_execute_command_passes_resume_session(self, sdk_manager):
        """Test that session_id is passed as --resume for continuation."""
        captured_cmd: list = []
        factory = _subprocess_factory(
            _assistant_json("Test response"),
            _result_json(session_id="test-session"),
            capture_cmd=captured_cmd,
        )
        with patch("asyncio.create_subprocess_exec", side_effect=factory):
            await sdk_manager.execute_command(
                prompt="Continue working",
                working_directory=Path("/test"),
                session_id="existing-session-id",
                continue_session=True,
            )

        assert "--resume" in captured_cmd
        resume_idx = captured_cmd.index("--resume")
        assert captured_cmd[resume_idx + 1] == "existing-session-id"

    async def test_execute_command_no_resume_for_new_session(self, sdk_manager):
        """Test that --resume is not set for new sessions."""
        captured_cmd: list = []
        factory = _subprocess_factory(
            _assistant_json("Test response"),
            _result_json(session_id="new-session"),
            capture_cmd=captured_cmd,
        )
        with patch("asyncio.create_subprocess_exec", side_effect=factory):
            await sdk_manager.execute_command(
                prompt="New prompt",
                working_directory=Path("/test"),
                session_id=None,
                continue_session=False,
            )

        assert "--resume" not in captured_cmd


class TestClaudeSandboxSettings:
    """Test sandbox and system_prompt settings on the CLI command."""

    @pytest.fixture
    def config(self, tmp_path):
        return Settings(
            telegram_bot_token="test:token",
            telegram_bot_username="testbot",
            approved_directory=tmp_path,
            claude_timeout_seconds=2,
            sandbox_enabled=True,
            sandbox_excluded_commands=["git", "npm"],
        )

    @pytest.fixture
    def sdk_manager(self, config):
        return ClaudeSDKManager(config)

    async def test_sandbox_settings_passed_to_cli(self, sdk_manager, tmp_path):
        """Test that sandbox settings appear in the CLI --settings flag."""
        captured_cmd: list = []
        factory = _subprocess_factory(
            _assistant_json("Test response"),
            _result_json(total_cost_usd=0.01),
            capture_cmd=captured_cmd,
        )
        with patch("asyncio.create_subprocess_exec", side_effect=factory):
            await sdk_manager.execute_command(
                prompt="Test prompt",
                working_directory=tmp_path,
            )

        assert "--settings" in captured_cmd
        settings_idx = captured_cmd.index("--settings")
        settings_data = json.loads(captured_cmd[settings_idx + 1])
        assert settings_data["sandbox"]["enabled"] is True
        assert "git" in settings_data["sandbox"]["excludedCommands"]

    async def test_system_prompt_set_with_working_directory(
        self, sdk_manager, tmp_path
    ):
        """Test that --system-prompt references the working directory."""
        captured_cmd: list = []
        factory = _subprocess_factory(
            _assistant_json("Test response"),
            _result_json(total_cost_usd=0.01),
            capture_cmd=captured_cmd,
        )
        with patch("asyncio.create_subprocess_exec", side_effect=factory):
            await sdk_manager.execute_command(
                prompt="Test prompt",
                working_directory=tmp_path,
            )

        assert "--system-prompt" in captured_cmd
        sp_idx = captured_cmd.index("--system-prompt")
        system_prompt = captured_cmd[sp_idx + 1]
        assert str(tmp_path) in system_prompt
        assert "relative paths" in system_prompt.lower()

    async def test_disallowed_tools_passed_to_cli(self, tmp_path):
        """Test that disallowed_tools appear as --disallowedTools in the CLI command."""
        config = Settings(
            telegram_bot_token="test:token",
            telegram_bot_username="testbot",
            approved_directory=tmp_path,
            claude_timeout_seconds=2,
            claude_disallowed_tools=["WebFetch", "WebSearch"],
        )
        manager = ClaudeSDKManager(config)

        captured_cmd: list = []
        factory = _subprocess_factory(
            _assistant_json("Test response"),
            _result_json(total_cost_usd=0.01),
            capture_cmd=captured_cmd,
        )
        with patch("asyncio.create_subprocess_exec", side_effect=factory):
            await manager.execute_command(
                prompt="Test prompt",
                working_directory=tmp_path,
            )

        assert "--disallowedTools" in captured_cmd
        dt_idx = captured_cmd.index("--disallowedTools")
        assert "WebFetch" in captured_cmd[dt_idx + 1]
        assert "WebSearch" in captured_cmd[dt_idx + 1]

    async def test_sandbox_disabled_when_config_false(self, tmp_path):
        """Test that --settings is NOT added when sandbox_enabled=False."""
        config = Settings(
            telegram_bot_token="test:token",
            telegram_bot_username="testbot",
            approved_directory=tmp_path,
            claude_timeout_seconds=2,
            sandbox_enabled=False,
        )
        manager = ClaudeSDKManager(config)

        captured_cmd: list = []
        factory = _subprocess_factory(
            _assistant_json("Test response"),
            _result_json(total_cost_usd=0.01),
            capture_cmd=captured_cmd,
        )
        with patch("asyncio.create_subprocess_exec", side_effect=factory):
            await manager.execute_command(
                prompt="Test prompt",
                working_directory=tmp_path,
            )

        assert "--settings" not in captured_cmd


class TestClaudeMCPErrors:
    """Test MCP-specific error handling."""

    @pytest.fixture
    def config(self, tmp_path):
        return Settings(
            telegram_bot_token="test:token",
            telegram_bot_username="testbot",
            approved_directory=tmp_path,
            claude_timeout_seconds=2,
        )

    @pytest.fixture
    def sdk_manager(self, config):
        return ClaudeSDKManager(config)

    async def test_mcp_connection_error_raises_mcp_error(self, sdk_manager):
        """Test that CLIConnectionError mentioning MCP raises ClaudeMCPError."""
        from claude_agent_sdk import CLIConnectionError

        from src.claude.exceptions import ClaudeMCPError

        async def _create(*cmd, **kwargs):
            raise CLIConnectionError("MCP server failed to start")

        with patch("asyncio.create_subprocess_exec", side_effect=_create):
            with pytest.raises(ClaudeMCPError) as exc_info:
                await sdk_manager.execute_command(
                    prompt="Test prompt",
                    working_directory=Path("/test"),
                )

        assert "MCP server" in str(exc_info.value)

    async def test_mcp_process_error_raises_mcp_error(self, sdk_manager):
        """Test that ProcessError mentioning MCP raises ClaudeMCPError."""
        from claude_agent_sdk import ProcessError

        from src.claude.exceptions import ClaudeMCPError

        async def _create(*cmd, **kwargs):
            raise ProcessError("Failed to start MCP server: connection refused")

        with patch("asyncio.create_subprocess_exec", side_effect=_create):
            with pytest.raises(ClaudeMCPError) as exc_info:
                await sdk_manager.execute_command(
                    prompt="Test prompt",
                    working_directory=Path("/test"),
                )

        assert "MCP" in str(exc_info.value)


