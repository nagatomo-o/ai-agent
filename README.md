# ローカルLLM エージェント

Windows 環境の Ollama とブラウザ上で対話し、指定した作業フォルダ内で安全なファイル検索・読み込み・作成・編集を行えるローカル AI エージェントです。

## 前提条件

- Python 3.14 以上
- Ollama が起動していること（デスクトップアプリ、または `ollama serve`）
- 利用するモデルが取得済みであること。画面のモデル一覧は `app/config.py` の `MODEL_LINEUP` に登録したモデルを常に表示し、未取得のモデルは無効表示になります。取得するには `ollama run <モデル名>` を実行してください。
- ツールを使う機能には、ツール呼び出しに対応したモデル（例: `gemma4:e4b`、`qwen3:4b`）が必要です。`gemma3:4b` などツール非対応のモデルは、チャットのみで動作します。

## セットアップ

```powershell
py -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

## 起動方法

```powershell
.\.venv\Scripts\uvicorn.exe app.main:app --host 127.0.0.1 --port 8000
```

ブラウザで `http://127.0.0.1:8000` を開きます。

## テスト実行

```powershell
.\.venv\Scripts\pytest.exe -v
```
