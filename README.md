# Chat-Bot

ローカルLLM（Qwen3 GGUF / llama-cpp-python）とWeb検索を統合した、Electronデスクトップ版リサーチアシスタントです。

## 機能

- Qwen3 GGUF モデルによるローカル推論（GPU加速対応）
- Thinking（思考プロセス）表示モード
- DuckDuckGo Web検索との自動連携
- SSEストリーミングによるリアルタイム出力
- アプリ内でのモデルダウンロード・切り替え
- コンテキスト使用率のリアルタイム表示

## 必要環境

- Python 3.10+（conda env: `new` 推奨）
- Node.js 20+
- CUDA環境（GPU推論時）
- llama-cpp-python CUDA対応版（下記参照）

## セットアップ

### 1. Python依存

```bash
conda activate new
pip install -r requirements.txt
```

llama-cpp-python の CUDA対応版ビルドは CLAUDE.md を参照してください。

### 2. Electron依存

```bash
cd electron
npm install
```

### 3. モデルの準備

アプリ起動後、UIのモデル管理画面からダウンロードできます。
CLIでダウンロードする場合:

```bash
python download_model.py
```

## 起動

### Windows（推奨）

```bat
start.bat
```

conda env のアクティベートから npm start まで自動実行します。

### 手動起動

```bash
cd electron
npm start
```

`npm start` が Electron を起動し、内部で `python main.py --host 127.0.0.1 --port 8765` を自動起動します。

## 構成

```
chat-bot/
├── main.py                    # FastAPI バックエンド
├── src/
│   ├── llm_handler.py         # llama-cpp-python推論
│   ├── search_handler.py      # DuckDuckGo検索
│   ├── model_manager.py       # モデル管理・ダウンロード
│   └── utils.py               # 設定・ログ・HTML整形
├── electron/
│   └── src/
│       ├── main.js            # Electronメインプロセス
│       ├── preload.js         # preloadブリッジ
│       └── renderer/          # UI (HTML/CSS/JS)
├── models/                    # GGUFモデル保存先
├── config.yaml                # モデル/検索/API設定
├── settings.json              # UI設定の自動保存
└── start.bat                  # Windows起動スクリプト
```

## 設定

### config.yaml（起動時設定）

| キー | 説明 |
|---|---|
| model.path | デフォルトで読み込むGGUFファイルのパス |
| model.n_gpu_layers | GPUオフロードするレイヤー数（-1=全て） |
| model.n_ctx | コンテキスト長（デフォルト: 32768） |
| sampling.temperature | 生成温度（デフォルト: 0.6） |
| sampling.max_tokens | 最大生成トークン数（デフォルト: 8192） |
| search.enabled | Web検索の有効/無効 |
| api.port | バックエンドAPIのポート（デフォルト: 8765） |

### settings.json（UI設定の自動保存）

起動時に config.yaml より優先して読み込まれます。UIで変更した設定（Thinkingモード、Temperature、アクティブモデル等）が自動保存されます。

## 補足

- Python実行パスを変えたい場合は `RESEARCH_BOT_PYTHON` 環境変数で指定可能
- APIの固定ポートは `8765`（Electronとbackendで一致させる必要あり）
- ログは `logs/research_bot.log` に出力

## llama-cpp-python 再ビルド時の注意（Windows / CUDA 13.1）

CUDA 13.1 は Visual Studio 2026（VS18）に未対応のため、`--allow-unsupported-compiler` と Ninja ジェネレーターが必要です。
`Qwen3.5` 系で `unknown model architecture: 'qwen35moe'` が出る場合も、`llama-cpp-python` の更新が必要です。

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
