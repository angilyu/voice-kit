from voice_kit.context import ContextStore


def test_update_merges_deletes_and_notifies():
    store = ContextStore({"page": 1})
    events = []
    unsubscribe = store.on_change(events.append)
    assert store.update({"page": 1}) == {}
    assert store.update({"page": 2, "title": "Bears"}) == {"page": 2, "title": "Bears"}
    assert store.update({"title": None, "missing": None}) == {"title": None}
    assert events == [{"page": 2, "title": "Bears"}, {"title": None}]
    unsubscribe()
    store.update({"page": 3})
    assert len(events) == 2
    assert store.snapshot() == {"page": 3} and store["page"] == 3 and "page" in store and len(store) == 1
    assert store.get("nope", "d") == "d"


def test_handle_data_json_and_text():
    store = ContextStore()
    assert store.handle_data(b'{"page": 4, "page_text": "Hi"}') == {"page": 4, "page_text": "Hi"}
    assert store.handle_data("child tapped the owl") == {"message": "child tapped the owl"}
    assert store.handle_data("[1, 2]", key="raw") == {"raw": "[1, 2]"}


def test_failing_listener_does_not_break_update():
    store = ContextStore()
    store.on_change(lambda changed: 1 / 0)
    assert store.update({"a": 1}) == {"a": 1}
