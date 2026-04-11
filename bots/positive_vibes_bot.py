# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import secrets
from botbuilder.core import ActivityHandler, MessageFactory, TurnContext
from botbuilder.schema import ChannelAccount

class PositiveVibesBot(ActivityHandler):
    def __init__(self):
        self.positive_messages = [
            "You are doing great!",
            "Keep up the good work!",
            "Believe in yourself!",
            "You are awesome!",
            "Have a fantastic day!",
            "Every day is a second chance.",
            "Positivity always wins...",
            "Sending positive vibes your way!",
            "You got this!",
            "Stay positive and happy!"
        ]

    async def on_members_added_activity(
        self, members_added: [ChannelAccount], turn_context: TurnContext
    ):
        for member in members_added:
            if member.id != turn_context.activity.recipient.id:
                await turn_context.send_activity("Hello and welcome! I am the Positive Vibes Bot. Send me a message and I'll send some positivity your way!")

    async def on_message_activity(self, turn_context: TurnContext):
        text = turn_context.activity.text

        if not text:
            await turn_context.send_activity("Please say something!")
            return

        lower_text = text.lower()
        msg = secrets.choice(self.positive_messages)
        return await turn_context.send_activity(
            MessageFactory.text(msg)
        )
