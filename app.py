# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import sys
from datetime import datetime, timezone
from http import HTTPStatus

from aiohttp import web
from aiohttp.web import Request, Response
from botbuilder.core import (
    TurnContext,
)
from botbuilder.core.integration import aiohttp_error_middleware
from botbuilder.integration.aiohttp import CloudAdapter, ConfigurationBotFrameworkAuthentication
from botbuilder.schema import Activity, ActivityTypes

from bots import PositiveVibesBot
from config import DefaultConfig

CONFIG = DefaultConfig()

# Create adapter.
ADAPTER = CloudAdapter(ConfigurationBotFrameworkAuthentication(CONFIG))

# Catch-all for errors.
async def on_error(context: TurnContext, error: Exception):
    print("\n [on_turn_error] unhandled error: An unhandled error occurred.", file=sys.stderr)

    # Send a message to the user
    await context.send_activity("The bot encountered an error or bug.")
    await context.send_activity(
        "To continue to run this bot, please fix the bot source code."
    )
    # Send a trace activity if we're talking to the Bot Framework Emulator
    if context.activity.channel_id == "emulator":
        trace_activity = Activity(
            label="TurnError",
            name="on_turn_error Trace",
            timestamp=datetime.now(timezone.utc),
            type=ActivityTypes.trace,
            value="An unhandled error occurred.",
            value_type="https://www.botframework.com/schemas/error",
        )
        await context.send_activity(trace_activity)

ADAPTER.on_turn_error = on_error

# Create the Bot
BOT = PositiveVibesBot()

# Listen for incoming requests on /api/messages
async def messages(req: Request) -> Response:
    if "application/json" not in req.headers.get("Content-Type", ""):
        return Response(status=HTTPStatus.UNSUPPORTED_MEDIA_TYPE)

    # Process route
    return await ADAPTER.process(req, BOT)

APP = web.Application(middlewares=[aiohttp_error_middleware])
APP.router.add_post("/api/messages", messages)

if __name__ == "__main__":
    web.run_app(APP, host="localhost", port=CONFIG.PORT)
