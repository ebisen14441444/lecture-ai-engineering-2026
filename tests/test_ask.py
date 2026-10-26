"""SDKの生成処理だけをモックし、API接続なしで会話履歴を検証する。"""

from unittest.mock import Mock

import pytest
from google import genai
from google.genai import errors, types

from app import ask


def response(text="回答", usage=True):
    return types.GenerateContentResponse(
        candidates=[types.Candidate(
            content=types.Content(role="model", parts=[types.Part(text=text)]),
            finish_reason="STOP",
        )],
        usage_metadata=types.GenerateContentResponseUsageMetadata(
            prompt_token_count=12, candidates_token_count=3,
        ) if usage else None,
    )


@pytest.fixture
def client(monkeypatch):
    # 本物のSDK Chatを使う。ネットワーク境界のgenerate_contentのみ差し替える。
    sdk_client = genai.Client(api_key="offline-test-key")
    monkeypatch.setenv("GEMINI_API_KEY", "offline-test-key")
    monkeypatch.setattr(ask.genai, "Client", lambda **kwargs: sdk_client)
    return sdk_client


def test_chat_retains_history_after_api_error(client, monkeypatch, capsys):
    calls = []

    def generate(**kwargs):
        calls.append(kwargs["contents"][:])
        if len(calls) == 2:
            raise errors.APIError(429, {"error": {"message": "rate limit"}})
        return response(f"回答{len(calls)}")

    monkeypatch.setattr(client.models, "generate_content", generate)
    inputs = iter(["  ", "失敗する質問", "前の回答を説明して", "/quit"])
    monkeypatch.setattr("builtins.input", lambda _: next(inputs))
    ask.chat("最初の質問")
    assert [[c.role for c in turn] for turn in calls] == [
        ["user"], ["user", "model", "user"], ["user", "model", "user"],
    ]
    assert calls[2][0].parts[0].text == "最初の質問"
    assert calls[2][1].parts[0].text == "回答1"
    assert calls[2][2].parts[0].text == "前の回答を説明して"
    assert "レート制限" in capsys.readouterr().out


@pytest.mark.parametrize("question", [None, "LoRAとは？"])
def test_one_shot_cli_compatibility(client, monkeypatch, capsys, question):
    generate = Mock(return_value=response())
    monkeypatch.setattr(client.models, "generate_content", generate)
    monkeypatch.setattr("sys.argv", ["ask"] + ([question] if question else []))
    ask.main()
    generate.assert_called_once_with(model=ask.MODEL, contents=question or ask.DEFAULT_QUESTION)
    assert "入力: 12 / 出力: 3" in capsys.readouterr().out


def test_missing_usage_metadata(capsys):
    ask.print_usage(response(usage=False))
    assert "入力: 不明 / 出力: 不明" in capsys.readouterr().out


def test_chat_eof_without_initial_request(client, monkeypatch):
    generate = Mock()
    monkeypatch.setattr(client.models, "generate_content", generate)
    monkeypatch.setattr("builtins.input", Mock(side_effect=EOFError))
    monkeypatch.setattr("sys.argv", ["ask", "--chat"])
    ask.main()
    generate.assert_not_called()


def test_one_shot_api_error_exit(client, monkeypatch, capsys):
    monkeypatch.setattr(client.models, "generate_content", Mock(
        side_effect=errors.APIError(503, {"error": {"message": "busy"}}),
    ))
    monkeypatch.setattr("sys.argv", ["ask", "質問"])
    with pytest.raises(SystemExit) as exc:
        ask.main()
    assert exc.value.code == 1
    assert "503" in capsys.readouterr().out
