# ローカルLLM対話画面・ファイル操作エージェント 設計仕様書

- 作成日: 2026-11-18
- 対象ワークスペース: `C:\Users\nagatomo\workspace\nagatomo.o\local-llm`
- 想定実行環境: Windows 10/11、Python 3.14、Ollama 0.40系、既定モデル `qwen3:4b`

## 1. 概要と目的

ローカル環境で稼働する Ollama とブラウザ上で対話できる Web アプリケーション、および指定した作業フォルダ内で安全にファイル操作を行う AI エージェント機能を構築する。

初期スコープでは以下の機能を提供する。

1. ブラウザ上でローカル LLM とリアルタイムに対話できるチャット画面
2. 画面上で指定した1つの作業フォルダ内に限定したファイル検索・一覧取得・読み込み
3. 画面上でのプレビュー確認および人の承認を経て実行するファイル新規作成・上書き・部分編集
4. 将来的に `git` などのコマンド実行ツールを同じ承認フローへ追加できる拡張構造

## 2. システム構成

### 2.1 技術スタック

- バックエンド: Python 3.14、FastAPI、Uvicorn
- LLM クライアント: 公式 `ollama` Python パッケージ
- フロントエンド: ビルド不要の HTML・CSS・JavaScript（ESM）
- UI スタイル基盤: `frame-css` スキルの `frame.css`（要素セレクタ中心、クラス例外は `a.button`・`.currency`・`.field`・`.pager` のみ）
- テスト: `pytest`、`httpx`（FastAPI TestClient 用）

### 2.2 ディレクトリ構成

```text
local-llm/
├─ app/
│  ├─ __init__.py
│  ├─ main.py                 # FastAPI アプリ、静的ファイル配信、API ルーティング
│  ├─ config.py               # 既定モデル・タイムアウト・ステップ上限等の設定
│  ├─ ollama_client.py        # Ollama API ラッパー（モデル一覧・チャットストリーミング）
│  ├─ agent.py                # エージェントループ、セッション状態・承認待機管理
│  └─ tools/
│     ├─ __init__.py
│     ├─ registry.py          # ツール定義・自動/承認区分・Tool calling スキーマ管理
│     └─ filesystem.py        # 安全なパス解決および 5 つのファイル操作ツール
├─ static/
│  ├─ index.html              # frame.css 準拠のアプリシェル HTML
│  ├─ frame.css               # frame.css（必要時は要素・属性セレクタで最小限の拡張）
│  └─ app.js                  # SSE 受信、チャット描画、ツールログ・承認 UI 制御
├─ tests/
│  ├─ test_filesystem.py      # パス制限と 5 ツールの単体テスト
│  ├─ test_agent.py           # エージェントループと承認・拒否フローのモックテスト
│  └─ test_api.py             # FastAPI エンドポイントの結合テスト
├─ docs/
│  └─ superpowers/
│     └─ specs/
│        └─ 2026-11-18-local-llm-agent-design.md
├─ requirements.txt
└─ README.md                  # 起動方法と操作手順
```

## 3. アーキテクチャと処理フロー

### 3.1 セッションと状態管理

- セッション情報はバックエンドのインメモリ辞書で管理する（単一ユーザーのローカル利用を前提とし、サーバー再起動時はリセットする）。
- 各セッションは以下の状態を保持する。
  - `session_id`: 一意のセッション識別子
  - `model`: 使用中の Ollama モデル名（既定値 `qwen3:4b`）
  - `workspace_root`: 許可された作業フォルダの正規化済み絶対パス（初期値は起動時のカレントディレクトリ、画面から変更可能）
  - `messages`: Ollama へ送信する会話履歴
  - `pending_approval`: 承認待ちのツール呼び出し情報（`approval_id`, `tool_name`, `arguments`, `preview`, `assistant_message`）

### 3.2 エージェントループと承認フロー

```text
[ユーザー入力]
   │ POST /api/sessions/{id}/chat (SSE)
   ▼
[Ollama chat(stream=True, tools=...)]
   ├─ テキスト/思考トークンを順次 SSE 配信
   └─ tool_calls あり？
        ├─ No  ──▶ [done] イベントを配信して終了
        └─ Yes ──▶ ToolRegistry で承認要否を判定
                    ├─ 自動実行 (list_files / search_files / read_file)
                    │    └─ 実行し tool_result を配信 → メッセージ履歴へ追加 → 次の Ollama 呼び出しへ（最大 8 ステップ）
                    └─ 承認必須 (write_file / edit_file)
                         └─ pending_approval を保存し approval_required を配信して SSE 終了
                              │
                              ▼ (画面でユーザーが「承認」または「拒否」を選択)
                         POST /api/sessions/{id}/approvals/{approval_id} (SSE)
                              ├─ 承認: ツールを実行 → tool_result を配信 → 会話を継続
                              └─ 拒否: 拒否メッセージを tool role で追加 → 会話を継続
```

### 3.3 API エンドポイント仕様

- `GET /api/health`
  - Ollama サーバー接続可否、インストール済みモデル一覧、既定モデル、既定作業フォルダを返す。
- `POST /api/sessions`
  - 新規セッションを作成する。任意で `model` と `workspace_root` を受け取る。
- `GET /api/sessions/{session_id}`
  - 現在のセッション設定、表示用メッセージ履歴、保留中の承認情報を返す。
- `PATCH /api/sessions/{session_id}/workspace`
  - リクエストの `workspace_root` を検証し、存在するディレクトリであれば作業ルートを更新する。存在しないパスやファイルパスが指定された場合は HTTP 400 を返す。
- `POST /api/sessions/{session_id}/chat`
  - ユーザープロンプトを受け取り、`text/event-stream` で応答をストリーミング配信する。`pending_approval` が未解決の状態で新しいチャット送信が来た場合は HTTP 409 を返す。
- `POST /api/sessions/{session_id}/approvals/{approval_id}`
  - `{ "decision": "approve" | "reject" }` を受け取り、対象ツールを実行または拒否したうえで、残りのエージェントループを `text/event-stream` で配信する。

### 3.4 SSE イベント形式

各イベントは `event: <name>` と `data: <json>` の形式で送信する。

- `status`: `{ "message": "推論中..." }`
- `thinking`: `{ "delta": "..." }`
- `token`: `{ "delta": "..." }`
- `tool_call`: `{ "name": "read_file", "arguments": {...}, "requires_approval": false }`
- `tool_result`: `{ "name": "read_file", "ok": true, "summary": "...", "output": "..." }`
- `approval_required`: `{ "approval_id": "...", "name": "write_file", "arguments": {...}, "action_type": "create|overwrite|edit", "path": "...", "diff": "...", "content_preview": "..." }`
- `done`: `{ "session_id": "..." }`
- `error`: `{ "message": "...", "detail": "..." }`

## 4. ツール仕様と安全設計

### 4.1 ツールレジストリ設計

`ToolRegistry` に各ツールのメタデータ（名前、説明、Ollama 用関数定義、`requires_approval` フラグ、実行関数）を登録する。読み取りツールは `requires_approval = False`、変更ツールは `requires_approval = True` とする。

将来 `git` などのコマンド実行ツールを追加する場合は、同レジストリに `requires_approval = True` のコマンドツールを追加するだけで、同じ画面確認カードと承認 API を流用できる。

### 4.2 実装する 5 つのファイル操作ツール

4B 規模のモデルでも引数の組み立てや呼び出し判断を誤りにくいよう、ツールは以下の 5 つに限定し、引数はすべて単純な文字列・整数とする。

1. `list_files(path: str = ".", max_depth: int = 2)`
   - 区分: 自動実行
   - 動作: 指定ディレクトリ配下のファイルとフォルダの相対パス一覧を返す。`.git`、`__pycache__`、`node_modules`、`.venv` などの巨大・システム系ディレクトリは既定で除外し、最大 200 件までに制限する。
2. `search_files(query: str, path: str = ".", file_pattern: str = "*")`
   - 区分: 自動実行
   - 動作: 作業フォルダ内でファイル名一致および UTF-8 テキストファイル内の文字列検索を行う（大文字小文字を区別しない）。バイナリファイルおよび 1 MB 超のファイルはスキップし、最大 50 件までに制限する。
3. `read_file(path: str, start_line: int = 1, max_lines: int = 200)`
   - 区分: 自動実行
   - 動作: 指定ファイルの内容を `行番号 | 内容` 形式で返す。1 回あたり最大 200 行（上限 400 行）に制限し、先頭・末尾の続きがある場合は総行数を付記する。バイナリファイルはエラーメッセージを返す。
4. `write_file(path: str, content: str)`
   - 区分: 承認必須
   - 動作: ファイルの新規作成または全体上書きを行う。承認要求時には新規作成か上書きかの種別、対象パス、行数、既存ファイルとの統一差分または新規作成内容プレビューを生成する。必要な親ディレクトリは承認後の実行時に作成する。
5. `edit_file(path: str, old_text: str, new_text: str)`
   - 区分: 承認必須
   - 動作: 既存ファイル内の `old_text` を `new_text` に置換する。`old_text` がファイル内に厳密に 1 回だけ出現する場合のみ受け付け、0 回または 2 回以上一致する場合は承認画面へ進まずツールエラーをモデルへ返して再指定させる。承認要求時には統一差分を生成して画面表示する。

### 4.3 作業フォルダ外アクセス防止

すべてのツールおよびプレビュー生成処理は共通関数 `resolve_safe_path(workspace_root: Path, target_path: str) -> Path` を経由する。

1. `target_path` が空の場合は `"."` として扱う。
2. 絶対パスが渡された場合でも、`workspace_root` 配下に収まっていれば許可し、範囲外であれば拒否する。相対パスの場合は `workspace_root / target_path` として結合する。
3. 双方を `.resolve()` で正規化し、`resolved.is_relative_to(workspace_root.resolve())` が偽の場合は `WorkspaceSecurityError` を送出する。

## 5. 画面デザイン仕様（`frame-css` 準拠）

### 5.1 CSS ルール遵守方針

- `frame-css` スキルの `scripts/frame.css` を `static/frame.css` として配置する。
- セレクタは HTML 要素ベース、状態表現は疑似クラスおよび `aria-*` 属性（`aria-current`、`aria-invalid`、`aria-busy`、`aria-live`、`aria-expanded` 等）を用いる。
- クラスは既存例外の `a.button`・`.currency`・`.field`・`.pager` 以外を増やさない。
- チャット表示領域の下部に入力フォームを固定するなど、`frame.css` のアプリシェル内で最小限のレイアウト調整が必要な場合は、`:has()` や要素・属性セレクタのみで記述し、`frame.css` の冒頭コメントと一貫性を保つ。

### 5.2 画面構成（アプリシェル）

`body` 直下に `header`、`nav`、`main`、`footer`、`output#toast` を配置する。

- `header`:
  - タイトル表記、Ollama 接続状態（接続済み／未接続とエラー案内）、現在の作業フォルダ表示、会話リセットボタン
- `nav`:
  - 使用モデル選択 `<select>`（`/api/health` で取得したモデル一覧、`qwen3:4b` を既定選択）
  - 作業フォルダ設定 `<form>`（パス入力 `<input>` ＋ 適用 `<button>`、エラー時は入力欄直下の `<small>` と `aria-invalid="true"` で表示）
  - 利用可能ツール一覧（自動実行ツールと承認必須ツールを区別して表示）
- `main`:
  - 会話メッセージ一覧領域：
    - 各発言を `<article>` で表示し、役割は `aria-label` または見出しで区別する
    - モデルの思考過程（`thinking`）および読み取りツールの実行結果（`tool_call` / `tool_result`）は `<details><summary>` パネルで折りたたみ表示する
    - 書き込み承認要求（`approval_required`）時は、対象ファイル・操作種別・差分または内容プレビュー（`<pre><code>`）と、「承認して実行」（既定の主操作 `<button>`）および「拒否」（控えめな `<button type="button">`）を含む承認パネルを会話内に表示する
  - プロンプト入力フォーム：
    - 質問入力用 `<textarea>`、送信 `<button>`、生成中断 `<button type="button">`、処理中はフォームやボタンの `disabled` / `aria-busy` を制御する
- `output#toast`:
  - `aria-live="polite"` を持ち、作業フォルダ変更成功・エラーやコピー完了などを `toastMsg()` で右上通知する
- `footer`:
  - 接続先 Ollama URL やステップ上限など現在の実行設定の補助表示

## 6. エラーハンドリング

- **Ollama 未起動・接続失敗**: `/api/health` およびチャット時に接続エラーを捕捉し、「`ollama serve` または Ollama デスクトップアプリが起動しているか確認してください」という対処案内を返す。
- **モデル未取得**: 指定モデルがローカルに存在しない場合、`ollama run <model>` またはモデル選択の変更を案内する。
- **ツール呼び出し上限**: 1 回のユーザー送信に対するツール実行ループは最大 8 ステップに制限し、上限に達した場合は現状の情報をまとめて終了するようモデルへ最終回答を促す。
- **ツール引数エラー・ファイル未存在**: 例外で SSE を中断せず、エラー文字列を `role: "tool"` としてモデルへ返却し、モデルが自己修正して再試行またはユーザーへ説明できるようにする。
- **ブラウザ側の生成中断**: 停止ボタンで `AbortController` により SSE 接続を中断し、UI を入力可能状態に戻す。

## 7. テスト計画

`pytest` を用い、Ollama 実サーバーに依存しない単体テストおよび API 結合テストを実装する。

1. `tests/test_filesystem.py`
   - 作業フォルダ外パス（`../`、別ドライブ絶対パス、シンボリックリンク越境）の拒否
   - `list_files` の除外ディレクトリ動作と深さ制限
   - `search_files` の文字列検索、パターン絞り込み、バイナリスキップ
   - `read_file` の行番号フォーマット、範囲指定、存在しないファイルのエラー処理
   - `write_file` の新規作成・上書き、親ディレクトリ自動生成、プレビュー/差分生成
   - `edit_file` の一意置換成功、0 件一致・複数件一致時の拒否とエラー返却
2. `tests/test_agent.py`
   - モック Ollama クライアントを用いた通常会話（ツールなし）
   - 読み取りツール自動実行後の最終回答生成
   - 書き込みツール呼び出し時の `approval_required` 停止と、`approve` / `reject` 後の再開
   - 最大ステップ上限（8 回）到達時の停止動作
3. `tests/test_api.py`
   - FastAPI `TestClient` による `/api/health`、`/api/sessions`、`/workspace` 変更の正常系・異常系
   - 承認待ち中の新規チャット送信に対する 409 応答
   - `/approvals/{id}` の SSE 応答確認
