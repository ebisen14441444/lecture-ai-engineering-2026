"""第1回: 素のLLMに質問するスクリプト。

使い方:
    uv run python -m app.ask "LoRAとは何ですか？"
    uv run python -m app.ask            # 質問を省略すると、講座に固有の質問をする
    uv run python -m app.ask --chat     # 会話履歴を保って続けて質問する

通常は会話履歴を持たない1回の呼び出し。--chat ではSDKが会話履歴を管理する。
講座に固有のことを聞くと、「分からない」と答えるか、一般論でもっともらしく答える
（講座の実際とは違う）ことを確かめよう。
（第3回で、用語集を検索できるエージェントにして改善する）
"""

import argparse

from google import genai
from google.genai import errors

from app.config import MODEL, api_error_message, get_api_key

# 講座に固有の質問。正しい答えは data/glossary.json の homework-levels にある
DEFAULT_QUESTION = "AIE2026の講座の宿題にある「3レベル」とは何ですか？"


def print_usage(response) -> None:
    """利用量が返らない場合も、0トークンと取り違えず表示する。"""
    usage = response.usage_metadata
    prompt = getattr(usage, "prompt_token_count", None)
    output = getattr(usage, "candidates_token_count", None)
    print(f"[tokens] 入力: {prompt if prompt is not None else '不明'} / "
          f"出力: {output if output is not None else '不明'}")


def ask(question: str) -> str:
    client = genai.Client(api_key=get_api_key())
    response = client.models.generate_content(model=MODEL, contents=question)

    print_usage(response)
    return response.text


def chat(initial_question: str | None = None) -> None:
    """成功した各ターンの履歴をSDKに保持し、終了コマンドかEOFまで対話する。"""
    client = genai.Client(api_key=get_api_key())
    conversation = client.chats.create(model=MODEL)
    print("続けて質問できます。/exit または /quit で終了します。")
    question = initial_question
    while True:
        if question is None:
            try:
                question = input("[質問] ")
            except (EOFError, KeyboardInterrupt):
                print()
                return
        else:
            print(f"[質問] {question}")
        question = question.strip()
        if question.lower() in ("/exit", "/quit"):
            return
        if not question:
            question = None
            continue
        try:
            response = conversation.send_message(question)
        except errors.APIError as e:
            # SDKはAPI呼び出しが成功してから履歴を記録するため、失敗しても
            # それまでの正常な会話を保持したまま次の入力を受け付けられる。
            print(api_error_message(e.code, e.message))
        else:
            print_usage(response)
            print(response.text)
        question = None


def main() -> None:
    parser = argparse.ArgumentParser(description="素のLLMに質問する")
    parser.add_argument("question", nargs="?")
    parser.add_argument("--chat", action="store_true", help="会話履歴を保って対話する")
    args = parser.parse_args()

    print(f"[model] {MODEL}")
    if args.chat:
        chat(args.question)
        return
    args.question = args.question if args.question is not None else DEFAULT_QUESTION
    print(f"[質問] {args.question}\n")
    try:
        print(ask(args.question))
    except errors.APIError as e:
        print(api_error_message(e.code, e.message))
        raise SystemExit(1)


if __name__ == "__main__":
    main()
