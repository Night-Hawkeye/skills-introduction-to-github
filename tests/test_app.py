import pytest
import pytest_asyncio
from aiohttp import web
from http import HTTPStatus
from unittest.mock import AsyncMock

from app import messages

@pytest_asyncio.fixture
async def cli(aiohttp_client):
    app = web.Application()
    app.router.add_post("/api/messages", messages)
    return await aiohttp_client(app)

@pytest.mark.asyncio
async def test_messages_unsupported_media_type(cli):
    """Test that a non-JSON request returns 415 UNSUPPORTED_MEDIA_TYPE"""
    resp = await cli.post("/api/messages", data="not json", headers={"Content-Type": "text/plain"})
    assert resp.status == HTTPStatus.UNSUPPORTED_MEDIA_TYPE

@pytest.mark.asyncio
async def test_messages_json_media_type(cli, mocker):
    """Test that a valid JSON request is processed and returns expected status"""
    mock_process = mocker.patch("app.ADAPTER.process", new_callable=AsyncMock)
    mock_process.return_value = web.Response(status=HTTPStatus.OK)

    resp = await cli.post("/api/messages", json={"type": "message", "text": "hello"}, headers={"Content-Type": "application/json"})

    assert resp.status == HTTPStatus.OK
    mock_process.assert_called_once()


@pytest.mark.asyncio
async def test_on_error_non_emulator_channel():
    """Test on_error when channel_id is not emulator"""
    from app import on_error
    from botbuilder.core import TurnContext
    from unittest.mock import MagicMock, AsyncMock

    mock_context = AsyncMock(spec=TurnContext)
    mock_context.activity = MagicMock()
    mock_context.activity.channel_id = "teams"
    mock_context.send_activity = AsyncMock()

    mock_error = Exception("Test error")

    await on_error(mock_context, mock_error)

    assert mock_context.send_activity.call_count == 2
    mock_context.send_activity.assert_any_call("The bot encountered an error or bug.")
    mock_context.send_activity.assert_any_call("To continue to run this bot, please fix the bot source code.")


@pytest.mark.asyncio
async def test_on_error_emulator_channel():
    """Test on_error when channel_id is emulator"""
    from app import on_error
    from botbuilder.core import TurnContext
    from unittest.mock import MagicMock, AsyncMock
    from botbuilder.schema import Activity, ActivityTypes

    mock_context = AsyncMock(spec=TurnContext)
    mock_context.activity = MagicMock()
    mock_context.activity.channel_id = "emulator"
    mock_context.send_activity = AsyncMock()

    mock_error = Exception("Test emulator error")

    await on_error(mock_context, mock_error)

    assert mock_context.send_activity.call_count == 3
    mock_context.send_activity.assert_any_call("The bot encountered an error or bug.")
    mock_context.send_activity.assert_any_call("To continue to run this bot, please fix the bot source code.")

    # Assert trace activity was sent
    trace_call_args = mock_context.send_activity.call_args_list[2][0]
    trace_activity = trace_call_args[0]
    assert isinstance(trace_activity, Activity)
    assert trace_activity.type == ActivityTypes.trace
    assert trace_activity.label == "TurnError"
    assert trace_activity.value == f"{mock_error}"
