import pytest
from unittest.mock import AsyncMock, patch
from botbuilder.core import TurnContext
from botbuilder.schema import Activity, ChannelAccount

from bots.positive_vibes_bot import PositiveVibesBot

@pytest.fixture
def bot():
    return PositiveVibesBot()

@pytest.fixture
def turn_context():
    context = AsyncMock(spec=TurnContext)
    context.activity = Activity()
    context.send_activity = AsyncMock()
    return context

@pytest.mark.asyncio
async def test_on_message_activity_empty_text(bot, turn_context):
    turn_context.activity.text = ""

    await bot.on_message_activity(turn_context)

    turn_context.send_activity.assert_called_once_with("Please say something!")

@pytest.mark.asyncio
async def test_on_message_activity_none_text(bot, turn_context):
    turn_context.activity.text = None

    await bot.on_message_activity(turn_context)

    turn_context.send_activity.assert_called_once_with("Please say something!")

@pytest.mark.asyncio
async def test_on_message_activity_with_text(bot, turn_context):
    turn_context.activity.text = "Hello"

    # We mock secrets.choice to make the test deterministic
    with patch("bots.positive_vibes_bot.secrets.choice", return_value="You are awesome!"):
        await bot.on_message_activity(turn_context)

    turn_context.send_activity.assert_called_once()

    # Check the activity sent
    call_args = turn_context.send_activity.call_args[0]
    sent_activity = call_args[0]

    assert sent_activity.text == "You are awesome!"

@pytest.mark.asyncio
async def test_on_members_added_activity_bot_added(bot, turn_context):
    # Setup members_added where only the bot is added
    bot_account = ChannelAccount(id="bot-id", name="Bot")
    turn_context.activity.recipient = bot_account

    members_added = [bot_account]

    await bot.on_members_added_activity(members_added, turn_context)

    # Should not send a welcome message to itself
    turn_context.send_activity.assert_not_called()

@pytest.mark.asyncio
async def test_on_members_added_activity_user_added(bot, turn_context):
    # Setup members_added where a user is added
    bot_account = ChannelAccount(id="bot-id", name="Bot")
    user_account = ChannelAccount(id="user-id", name="User")
    turn_context.activity.recipient = bot_account

    members_added = [user_account]

    await bot.on_members_added_activity(members_added, turn_context)

    # Should send a welcome message to the user
    turn_context.send_activity.assert_called_once_with(
        "Hello and welcome! I am the Positive Vibes Bot. Send me a message and I'll send some positivity your way!"
    )

@pytest.mark.asyncio
async def test_on_members_added_activity_multiple_users(bot, turn_context):
    # Setup members_added where multiple users and bot are added
    bot_account = ChannelAccount(id="bot-id", name="Bot")
    user_account_1 = ChannelAccount(id="user-id-1", name="User 1")
    user_account_2 = ChannelAccount(id="user-id-2", name="User 2")
    turn_context.activity.recipient = bot_account

    members_added = [bot_account, user_account_1, user_account_2]

    await bot.on_members_added_activity(members_added, turn_context)

    # Should send a welcome message for each user (so twice)
    assert turn_context.send_activity.call_count == 2
    turn_context.send_activity.assert_called_with(
        "Hello and welcome! I am the Positive Vibes Bot. Send me a message and I'll send some positivity your way!"
    )
