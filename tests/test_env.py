from voice_kit.env import env_bool, env_float, env_int, env_list, env_str


def test_env_readers(monkeypatch):
    monkeypatch.setenv("VK_B", "Yes")
    monkeypatch.setenv("VK_BLANK", "  ")
    monkeypatch.setenv("VK_I", "7")
    monkeypatch.setenv("VK_F", "1.5")
    monkeypatch.setenv("VK_S", "  hi ")
    monkeypatch.setenv("VK_L", "a, ,b ,c")
    assert env_bool("VK_B", False) is True
    assert env_bool("VK_BLANK", True) is True
    assert env_bool("VK_MISSING", False) is False
    assert env_int("VK_I", 1) == 7 and env_int("VK_BLANK", 3) == 3
    assert env_float("VK_F", 0.0) == 1.5
    assert env_str("VK_S") == "hi" and env_str("VK_BLANK", "d") == "d"
    assert env_list("VK_L", "") == ["a", "b", "c"]
    assert env_list("VK_MISSING", "x,y") == ["x", "y"]
