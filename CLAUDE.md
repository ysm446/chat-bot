# Chat-Bot 実装メモ
Version: 2.0 - llama-cpp-python + Electron版

---

## 実行環境メモ（実績情報）

### ハードウェア
- GPU: NVIDIA RTX PRO 5000 (48GB VRAM)
- OS: Windows 11 Pro

### ソフトウェア
- Python: 3.10 (conda env: `new`, D:\miniconda3\conda_envs\new)
- CUDA: 13.1 (V13.1.115)
- Visual Studio: 2026 (VS18) ※CUDA 13.1が未対応のため注意（後述）
- llama-cpp-python: CUDA対応版をソースビルド要
- Node.js: 20+（Electron用）

### 使用モデル（GGUF形式 / HuggingFace からダウンロード）
| モデルキー | ファイル | サイズ | 説明 |
|---|---|---|---|
| Qwen3-14B-Q4_K_M | models/Qwen3-14B-Q4_K_M.gguf | 8.8GB | 高精度・大型タスク向け |
| Qwen3-30B-A3B-Q4_K_M | models/Qwen3-30B-A3B-Q4_K_M.gguf | 18.0GB | MoE・最高精度・RTX PRO 5000推奨 |
| Qwen3-30B-A3B-abliterated-Q4_K_M | models/huihui-ai.Qwen3-30B-A3B-abliterated.Q4_K_M.gguf | 18.6GB | abliterated版 |

### llama-cpp-python CUDA版ビルド手順
CUDA 13.1はVS2026未対応のため、`--allow-unsupported-compiler` フラグと Ninja ジェネレーターが必要。
**x64 Native Tools Command Prompt for VS 2026** から実行:

```bat
conda activate new

:: no-binary環境変数が残っている場合は解除
set PIP_NO_BINARY=
set PIP_ONLY_BINARY=

:: 一時ディレクトリを短いパスへ
mkdir D:\tmp 2>nul
set TMP=D:\tmp
set TEMP=D:\tmp

:: 依存はwheelで先に入れる
pip install -U pip setuptools wheel cmake ninja numpy Cython jinja2 diskcache typing-extensions

:: llama-cpp-python本体だけソースビルド
set CMAKE_ARGS=-DGGML_CUDA=on -G "Ninja" -DCMAKE_CUDA_FLAGS="--allow-unsupported-compiler"
set FORCE_CMAKE=1
pip install --no-cache-dir --force-reinstall --no-binary llama-cpp-python "llama-cpp-python @ git+https://github.com/abetlen/llama-cpp-python.git"
```

確認:
```bat
python -c "import llama_cpp; print(llama_cpp.__version__)"
```

### 注意: `unknown model architecture: 'qwen35moe'` エラー
Qwen3.5系でこのエラーが出る場合は llama-cpp-python の更新が必要。
`--no-binary=:all:` は依存（numpy/Cython）までソースビルドになりWindowsで失敗するため、上記手順で対処する。

### 起動コマンド
```bat
start.bat
```
- `start.bat` が conda env `new` をアクティベートし、`electron/` で `npm start` を実行
- Electronが起動し、内部で `python main.py --host 127.0.0.1 --port 8765` を自動起動
- Python実行パスを変更したい場合は `RESEARCH_BOT_PYTHON` 環境変数で指定

### 設定ファイルの役割
- `config.yaml`: モデルパス・サーバー設定など起動時の設定
- `settings.json`: UIの設定（Thinkingモード、Temperature等、アクティブモデル）を自動保存。起動時に読み込まれ UI に反映

---

## プロジェクト概要

**名称**: Chat-Bot (Research-Bot)
**目的**: ローカルLLM（Qwen3 GGUF / llama-cpp-python）とDuckDuckGo Web検索を統合した対話型リサーチアシスタント
**UI**: Electron デスクトップアプリ
**バックエンド**: FastAPI + SSEストリーミング（ポート: 8765固定）
**技術スタック**: llama-cpp-python, FastAPI, Electron, DuckDuckGo Search (ddgs), Python 3.10+

---

## プロジェクト構成
```
chat-bot/
├── models/                              # GGUFモデル保存先（gitignore対象）
│
├── src/                                 # Python バックエンド
│   ├── __init__.py
│   ├── llm_handler.py                   # llama-cpp-python推論（Thinking分離・ストリーミング）
│   ├── search_handler.py                # DuckDuckGo検索（ddgsパッケージ）
│   ├── model_manager.py                 # モデル一覧・ダウンロード・切り替え管理
│   └── utils.py                         # 設定読み込み・保存・ログ・HTML整形
│
├── electron/                            # Electronアプリ
│   ├── src/
│   │   ├── main.js                      # メインプロセス（Pythonバックエンド自動起動）
│   │   ├── preload.js                   # preloadブリッジ
│   │   └── renderer/                    # レンダラー（HTML/CSS/JS）
│   │       ├── index.html
│   │       ├── styles.css
│   │       └── renderer.js
│   └── package.json
│
├── main.py                              # FastAPI バックエンド エントリーポイント
├── config.yaml                          # モデル/検索/API設定
├── settings.json                        # UI設定の自動保存（起動時に読み込み）
├── requirements.txt                     # Python依存パッケージ
├── download_model.py                    # モデルダウンロードスクリプト（CLI版）
├── start.bat                            # Windows起動スクリプト（推奨）
└── AGENTS.md                            # エージェント向けPowerShellコマンド集
```

---

## API エンドポイント一覧

| メソッド | パス | 説明 |
|---|---|---|
| GET | /health | ヘルスチェック |
| GET | /api/bootstrap | 設定・モデル一覧の初期取得 |
| POST | /api/settings | UI設定の保存 |
| POST | /api/chat/stream | チャット（SSEストリーミング） |
| GET | /api/models | モデル一覧取得 |
| POST | /api/models/download/stream | モデルダウンロード（SSE進捗） |
| POST | /api/models/unload | モデルのアンロード（VRAM解放） |
| POST | /api/models/switch | モデルの切り替え |

---

## SSEイベント仕様（/api/chat/stream）

```json
{"event": "status",   "status": "...", "context_usage": "...", "answer": "...", "thinking": ""}
{"event": "thinking", "status": "...", "context_usage": "...", "thinking": "<details>...</details>", "answer": "..."}
{"event": "answer",   "status": "...", "context_usage": "...", "thinking": "...", "answer": "..."}
{"event": "final",    "status": "...", "context_usage": "...", "thinking": "...", "answer": "...", "search_results": [...]}
```

---

## 設定ファイル詳細

### config.yaml
```yaml
# モデル設定（llama-cpp-python / GGUF形式）
model:
  path: "./models/Qwen3-30B-A3B-Q4_K_M.gguf"
  n_gpu_layers: -1      # -1 = 全レイヤーGPUオフロード
  n_ctx: 32768          # コンテキスト長
  n_threads: 4          # CPUスレッド数

# サンプリング設定
sampling:
  temperature: 0.6
  top_p: 0.95
  top_k: 20
  max_tokens: 8192

# 検索設定
search:
  enabled: true
  max_results: 5
  region: "jp-jp"
  safe_search: "moderate"
  timeout: 10

# API設定
api:
  host: "127.0.0.1"
  port: 8765

# 表示設定
display:
  show_thinking: true
  show_search_results: true
  stream_output: true
```

---

## 想定される問題と対策

### 問題1: メモリ不足（OOM）
- `n_ctx` を 16384 に削減
- `n_gpu_layers` を下げてCPUに一部オフロード

### 問題2: 推論が遅い
- `max_tokens` を 4096 に削減
- `n_ctx` を削減

### 問題3: 検索がブロックされる
- SearchHandlerにはリトライ（指数バックオフ）とレート制限（1秒1リクエスト）実装済み
- ddgs パッケージが古い場合はアップデート: `pip install -U ddgs`

### 問題4: `unknown model architecture: 'qwen35moe'`
- llama-cpp-python を最新版にアップデート（上記ビルド手順参照）

### 問題5: Electronが起動しない
- `cd electron && npm install` で依存を再インストール
- `RESEARCH_BOT_PYTHON` 環境変数でPython実行パスを指定可能

---

## 変更履歴

- 2025-02-12: 初版作成（vLLM + Gradio + Q4量子化版）
- 2026-03-05: v2.0 llama-cpp-python + Electron + FastAPI に移行
