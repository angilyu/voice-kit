import pytest
from langchain_core.messages import AIMessageChunk

from voice_kit.streaming import (
    FILLER,
    OpenerDropper,
    SayStreamer,
    ToolMarkupFilter,
    WordGate,
    is_farewell,
    message_text,
    strip_leading_filler,
    strip_opener,
)

OPENER = "Hi, this is an AI assistant calling on behalf of Angi. This call may be recorded."


def test_raw_streamed_tool_markup_is_filtered_across_chunks():
    f = ToolMarkupFilter()
    chunks = ["<function=record_outcome ", '={"outcome":"booked"} ', "/> ", "Thanks, goodbye!"]
    spoken = "".join(f.feed(chunk) for chunk in chunks) + f.flush()
    assert "<function=" not in spoken
    assert "record_outcome" not in spoken
    assert spoken.strip() == "Thanks, goodbye!"


def test_tool_markup_filter_releases_plain_text_and_partial_markers():
    f = ToolMarkupFilter()
    assert f.feed("Hello <fun") == "Hello "
    assert f.feed("ny day") == "<funny day"
    assert f.flush() == ""
    g = ToolMarkupFilter()
    assert g.feed("Done <function=x {}") == "Done "
    assert g.flush() == ""


def test_say_streamer_handles_partial_json_and_escapes():
    streamer = SayStreamer()
    out = []
    for piece in ['{"sa', 'y": "No wor', 'ries, \\', '"5\\" is too early', '. Bye!", "outcome": "unav']:
        chunk = AIMessageChunk(
            content="",
            tool_call_chunks=[{"name": "record_outcome" if not out else None, "args": piece, "id": "r", "index": 0}],
        )
        out.append(streamer.feed(chunk))
    assert "".join(out) == 'No worries, "5" is too early. Bye!'
    assert streamer.done


def test_say_streamer_custom_tool_and_field_ignores_other_tools():
    streamer = SayStreamer(tool_name="finish", field="line")
    other = AIMessageChunk(content="", tool_call_chunks=[{"name": "lookup", "args": '{"line": "no"}', "index": 0}])
    assert streamer.feed(other) == ""
    mine = AIMessageChunk(content="", tool_call_chunks=[{"name": "finish", "args": '{"line": "The end."}', "index": 1}])
    assert streamer.feed(mine) == "The end."
    assert streamer.done


def test_opener_dropper_chunked():
    dropper = OpenerDropper(OPENER)
    parts = [OPENER[:10], OPENER[10:30], OPENER[30:] + " Calling", " about jeans."]
    assert "".join(dropper.feed(p) for p in parts) + dropper.flush() == "Calling about jeans."
    other = OpenerDropper(OPENER)
    assert other.feed("Hi") == "" and other.feed(" there!") == "Hi there!"
    partial = OpenerDropper(OPENER)
    assert partial.feed("Hi, this") == "" and partial.flush() == "Hi, this"
    assert OpenerDropper("").feed("anything") == "anything"


def test_strip_opener():
    assert strip_opener(" " + OPENER + " Calling about jeans.", OPENER) == "Calling about jeans."
    assert strip_opener("Hello there", OPENER) == "Hello there"
    assert strip_opener("Hello", "") == "Hello"


def test_filler_helpers():
    assert FILLER.strip().lower().startswith("hmm")
    assert strip_leading_filler("Hmm, let me check.") == "let me check."
    assert strip_leading_filler("Um... sure") == ".. sure"
    assert strip_leading_filler("Hummus is great") == "Hummus is great"


def test_word_gate_holds_until_a_word():
    gate = WordGate()
    assert gate.feed(" ") == "" and gate.feed(", ") == ""
    assert gate.feed("Hi") == " , Hi"
    assert gate.feed("!") == "!"
    assert WordGate().flush() == ""


def test_message_text_handles_content_parts():
    assert message_text(AIMessageChunk(content=[{"type": "text", "text": "a"}, "b"])) == "ab"
    assert message_text("plain") == "plain"
    assert message_text(None) == ""


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("Thanks, bye!", True),
        ("Okay, have a great day!", True),
        ("Goodbye.", True),
        ("Thanks!", False),
        ("Bye, but first what time do you close?", False),
        ("", False),
    ],
)
def test_is_farewell(text, expected):
    assert is_farewell(text) is expected
