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
