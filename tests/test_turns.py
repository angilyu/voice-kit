from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage

from voice_kit.turns import (
    TurnDecision,
    UserTurn,
    chain,
    decide,
    is_question,
    keyword_gate,
    llm_gate,
    min_words_gate,
    question_gate,
    reference_text_gate,
    text_overlap,
)

PAGE = "The little bear walked slowly into the dark forest. Where is the bear going?"
R, L, I = TurnDecision.RESPOND, TurnDecision.LISTEN, TurnDecision.IGNORE


async def test_decide_coerces_results():
    turn = UserTurn("hi")
    assert await decide(None, turn) is R
    assert await decide(lambda t: True, turn) is R
    assert await decide(lambda t: False, turn) is I
    assert await decide(lambda t: "Listen", turn) is L

    async def agate(t):
        return TurnDecision.IGNORE

    assert await decide(agate, turn) is I


async def test_min_words_and_keyword_gates():
    assert await decide(min_words_gate(2), UserTurn("um")) is I
    assert await decide(min_words_gate(2), UserTurn("what's that")) is R
    gate = keyword_gate(["hey owl", "book"])
    assert await decide(gate, UserTurn("Hey Owl, what's this?")) is R
    assert await decide(gate, UserTurn("hey owls")) is I
    assert await decide(keyword_gate(["owl"], otherwise=L), UserTurn("hello")) is L


def test_is_question_without_question_mark():
    assert is_question("why is the sky dark")
    assert is_question("um so what does forest mean")
    assert is_question("It's dark?")
    assert not is_question("the bear walked into the forest")
    assert not is_question("")


async def test_question_gate():
    assert await decide(question_gate(), UserTurn("what is a forest")) is R
    assert await decide(question_gate(), UserTurn("I like bears")) is L


def test_text_overlap():
    assert text_overlap("the little bear", PAGE) == 1.0
    assert text_overlap("pizza tonight please", PAGE) == 0.0
    assert text_overlap("", PAGE) == 0.0


async def test_reference_text_gate_listens_while_reading_aloud():
    gate = reference_text_gate()
    ctx = {"page_text": PAGE}
    # Slow, partial reading of the page: stay quiet.
    assert await decide(gate, UserTurn("the little bear walked", ctx)) is L
    assert await decide(gate, UserTurn("slowly into the dark", ctx)) is L
    # Reading a question printed on the page is still reading.
    assert await decide(gate, UserTurn("where is the bear going", ctx)) is L
    # A real question about the book.
    assert await decide(gate, UserTurn("why is the bear going into the forest", ctx)) is R
    assert await decide(gate, UserTurn("what does dark mean", ctx)) is R
    # Too short to tell, or no page context: respond (other gates may still filter).
    assert await decide(gate, UserTurn("the bear", ctx)) is R
    assert await decide(gate, UserTurn("the little bear walked", {})) is R


async def test_chain_first_non_respond_wins():
    gate = chain(min_words_gate(2), reference_text_gate(), question_gate(otherwise=I))
    ctx = {"page_text": PAGE}
    assert await decide(gate, UserTurn("hmm", ctx)) is I
    assert await decide(gate, UserTurn("the little bear walked slowly", ctx)) is L
    assert await decide(gate, UserTurn("why is the forest dark?", ctx)) is R
    assert await decide(gate, UserTurn("turn the tv down", ctx)) is I


async def test_llm_gate_parses_decision_and_shows_context():
    seen = []

    class Recording(GenericFakeChatModel):
        async def _agenerate(self, messages, stop=None, run_manager=None, **kwargs):
            seen.append(messages)
            return await super()._agenerate(messages, stop, run_manager, **kwargs)

    model = Recording(messages=iter([AIMessage("IGNORE"), AIMessage("respond."), AIMessage("maybe")]))
    gate = llm_gate(model, "Only answer questions about the picture book.", context_keys=["page"])
    turn = UserTurn("can we get pizza", {"page": 3, "secret": "x"}, [("user", "hi"), ("assistant", "hello")])
    assert await decide(gate, turn) is I
    prompt = seen[0][1].content
    assert "page: 3" in prompt and "secret" not in prompt
    assert "assistant: hello" in prompt and "Just heard: can we get pizza" in prompt
    assert "picture book" in seen[0][0].content
    assert await decide(gate, turn) is R
    assert await decide(gate, turn) is R  # unparseable -> on_error


async def test_llm_gate_errors_fall_back():
    class Boom:
        async def ainvoke(self, messages):
            raise RuntimeError("down")

    assert await decide(llm_gate(Boom(), on_error=L), UserTurn("hi")) is L
