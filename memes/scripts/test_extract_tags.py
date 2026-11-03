#!/usr/bin/env python3
"""測試腳本：從一句聊天文字抽出 2 個梗圖標籤（JSON）。

使用方式：
    python scripts/test_extract_tags.py "今天 deadline 又被加任務，我已經沒了"
    python scripts/test_extract_tags.py --provider openai "這會議可以結束了嗎"
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from dotenv import load_dotenv
from pydantic import BaseModel, Field, ValidationError, field_validator

# 讓專案根目錄的 .env 一定被讀到（無論從哪裡執行）。
PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(PROJECT_ROOT / ".env")

DEFAULT_GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-flash-latest")
DEFAULT_OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
DEFAULT_PROVIDER = os.getenv("LLM_PROVIDER", "gemini").strip().lower()

# 後續會拿來對梗圖資料庫的候選標籤。模型仍可產出相近的新標籤。
KNOWN_MEME_TAGS = (
    "this_is_fine",
    "surprised_pikachu",
    "distracted_boyfriend",
    "drake_hotline_bling",
    "woman_yelling_at_cat",
    "hide_the_pain_harold",
    "doge",
    "success_kid",
    "crying_cat",
    "stonks",
    "always_has_been",
    "two_buttons",
    "expanding_brain",
    "roll_safe",
    "galaxy_brain",
    "one_does_not_simply",
    "change_my_mind",
    "is_this_a_pigeon",
    "panik_kalm",
    "among_us_sus",
)

SYSTEM_PROMPT = """你是「梗圖推薦」系統的語意解析器。
根據使用者的聊天內容，判斷情緒與迷因梗點，輸出恰好 2 個英文 snake_case 標籤。

規則：
- 優先從已知標籤中挑選最貼切的 2 個。
- 若沒有夠準的已知標籤，可自創新標籤，但仍須為英文 snake_case。
- 兩個標籤不可重複。
- 只描述對話的核心梗點與情緒，不要解釋、不要多餘欄位。

已知標籤：
{known_tags}
""".format(known_tags=", ".join(KNOWN_MEME_TAGS))


class MemeTagResult(BaseModel):
    """LLM 必須遵守的輸出結構。"""

    tags: list[str] = Field(..., min_length=2, max_length=2)

    @field_validator("tags")
    @classmethod
    def tags_must_be_unique_snake_case(cls, value: list[str]) -> list[str]:
        cleaned = [tag.strip().lower().replace(" ", "_") for tag in value]
        if any(not tag for tag in cleaned):
            raise ValueError("標籤不可為空")
        if len(set(cleaned)) != 2:
            raise ValueError("兩個標籤必須不同")
        return cleaned


class TagExtractionError(RuntimeError):
    """標籤抽取失敗（設定、API 或解析錯誤）。"""


def extract_tags_with_gemini(text: str, model: str) -> MemeTagResult:
    """使用 Google Gemini structured output 抽出標籤。"""
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise TagExtractionError("缺少 GEMINI_API_KEY，請複製 .env.example 為 .env 後填入金鑰。")

    try:
        from google import genai
    except ImportError as exc:
        raise TagExtractionError("尚未安裝 google-genai，請執行：pip install -r requirements.txt") from exc

    client = genai.Client(api_key=api_key)
    try:
        response = client.models.generate_content(
            model=model,
            contents=f"{SYSTEM_PROMPT}\n\n聊天內容：\n{text}",
            config={
                "response_mime_type": "application/json",
                "response_schema": MemeTagResult,
            },
        )
    except Exception as exc:  # SDK 例外種類較多，統一轉成可讀錯誤。
        raise TagExtractionError(f"Gemini API 呼叫失敗：{exc}") from exc

    if getattr(response, "parsed", None) is not None:
        parsed = response.parsed
        if isinstance(parsed, MemeTagResult):
            return parsed
        return MemeTagResult.model_validate(parsed)

    raw_text = (response.text or "").strip()
    if not raw_text:
        raise TagExtractionError("Gemini 回傳空白內容。")
    return _parse_json_payload(raw_text)


def extract_tags_with_openai(text: str, model: str) -> MemeTagResult:
    """使用 OpenAI Chat Completions JSON mode 抽出標籤。"""
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise TagExtractionError("缺少 OPENAI_API_KEY，請複製 .env.example 為 .env 後填入金鑰。")

    try:
        from openai import OpenAI
    except ImportError as exc:
        raise TagExtractionError("尚未安裝 openai，請執行：pip install -r requirements.txt") from exc

    client = OpenAI(api_key=api_key)
    try:
        response = client.chat.completions.create(
            model=model,
            temperature=0.2,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": (
                        "請只回傳 JSON，格式為 {\"tags\": [\"tag_one\", \"tag_two\"]}。\n"
                        f"聊天內容：\n{text}"
                    ),
                },
            ],
        )
    except Exception as exc:
        raise TagExtractionError(f"OpenAI API 呼叫失敗：{exc}") from exc

    raw_text = (response.choices[0].message.content or "").strip()
    if not raw_text:
        raise TagExtractionError("OpenAI 回傳空白內容。")
    return _parse_json_payload(raw_text)


def _parse_json_payload(raw_text: str) -> MemeTagResult:
    """把模型文字解析並驗證成固定 JSON 結構。"""
    try:
        payload = json.loads(raw_text)
    except json.JSONDecodeError as exc:
        raise TagExtractionError(f"模型回傳不是合法 JSON：{raw_text}") from exc

    try:
        return MemeTagResult.model_validate(payload)
    except ValidationError as exc:
        raise TagExtractionError(f"JSON 結構不符合規格：{exc}") from exc


def extract_meme_tags(text: str, provider: str, model: str | None = None) -> dict[str, list[str]]:
    """對外主函式：輸入聊天文字，回傳 {"tags": [...]}。"""
    cleaned = text.strip()
    if not cleaned:
        raise TagExtractionError("聊天內容不可為空。")

    provider_name = provider.strip().lower()
    if provider_name == "gemini":
        result = extract_tags_with_gemini(cleaned, model or DEFAULT_GEMINI_MODEL)
    elif provider_name == "openai":
        result = extract_tags_with_openai(cleaned, model or DEFAULT_OPENAI_MODEL)
    else:
        raise TagExtractionError("provider 只能是 gemini 或 openai。")

    return result.model_dump()


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="從聊天文字抽出 2 個梗圖標籤（JSON）。")
    parser.add_argument("text", nargs="?", help="一句聊天內容。省略則進入互動輸入。")
    parser.add_argument(
        "--provider",
        choices=("gemini", "openai"),
        default=DEFAULT_PROVIDER if DEFAULT_PROVIDER in {"gemini", "openai"} else "gemini",
        help="LLM 供應商，預設讀 LLM_PROVIDER，否則 gemini。",
    )
    parser.add_argument("--model", default=None, help="覆寫模型名稱。")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    text = args.text or input("請輸入一句聊天內容： ").strip()

    try:
        payload = extract_meme_tags(text, provider=args.provider, model=args.model)
    except TagExtractionError as exc:
        print(f"錯誤：{exc}", file=sys.stderr)
        return 1

    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
