# Meme Recommender

根據聊天內容自動推薦梗圖。目前先完成語意解析測試：輸入一句話，LLM 回傳 2 個標籤（JSON）。

## 快速開始

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
copy .env.example .env
```

在 `.env` 填入 `GEMINI_API_KEY` 或 `OPENAI_API_KEY`。

```powershell
python scripts/test_extract_tags.py "今天 deadline 又被加任務，我已經沒了"
```

預期輸出：

```json
{
  "tags": ["this_is_fine", "hide_the_pain_harold"]
}
```

改用 OpenAI：

```powershell
python scripts/test_extract_tags.py --provider openai "這會議可以結束了嗎"
```
