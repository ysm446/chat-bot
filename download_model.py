"""
GGUF モデルのダウンロードスクリプト（llama-cpp-python 対応）
"""
import sys
from pathlib import Path


def download_model(repo_id: str, filename: str, save_path: Path):
    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        print("ERROR: huggingface-hub がインストールされていません。")
        print("pip install huggingface-hub を実行してください。")
        sys.exit(1)

    print(f"リポジトリ : {repo_id}")
    print(f"ファイル   : {filename}")
    print(f"保存先     : {save_path.resolve()}")
    print("-" * 50)

    # 既存ファイルの確認
    if save_path.exists():
        size_gb = save_path.stat().st_size / (1024 ** 3)
        print(f"既存のファイルが見つかりました: {size_gb:.2f} GB")
        answer = input("再ダウンロードしますか？ (y/N): ").strip().lower()
        if answer != "y":
            print("ダウンロードをスキップします。")
            return str(save_path)

    save_path.parent.mkdir(parents=True, exist_ok=True)
    print("ダウンロードを開始します...")

    try:
        downloaded = hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            local_dir=str(save_path.parent),
            local_dir_use_symlinks=False,
        )
        size_gb = Path(downloaded).stat().st_size / (1024 ** 3)
        print(f"\nダウンロード完了！")
        print(f"保存先: {downloaded}")
        print(f"ファイルサイズ: {size_gb:.2f} GB")
        return downloaded

    except KeyboardInterrupt:
        print("\nダウンロードが中断されました。再実行すると再開します。")
        sys.exit(0)
    except Exception as e:
        print(f"\nERROR: {e}")
        print("\nトラブルシューティング:")
        print("1. インターネット接続を確認してください")
        print("2. huggingface-cli login でログインが必要な場合があります")
        print("3. ディスク空き容量を確認してください")
        sys.exit(1)


def check_disk_space(required_gb: float):
    import shutil
    _, _, free = shutil.disk_usage("./")
    free_gb = free / (1024 ** 3)
    print(f"ディスク空き容量: {free_gb:.1f} GB")
    if free_gb < required_gb:
        print(f"警告: {required_gb}GB 以上の空き容量が推奨されます")
        answer = input("続行しますか？ (y/N): ").strip().lower()
        if answer != "y":
            sys.exit(0)


if __name__ == "__main__":
    # ダウンロードするモデルの設定
    # HuggingFace 上の実際のファイル名は以下で確認してください:
    #   https://huggingface.co/Qwen/Qwen3-4B-GGUF/tree/main
    REPO_ID = "Qwen/Qwen3-4B-GGUF"
    FILENAME = "Qwen3-4B-Q4_K_M.gguf"
    SAVE_PATH = Path("./models/Qwen3-4B-Q4_K_M.gguf")
    REQUIRED_GB = 3.5

    print("=" * 50)
    print("  Research-Bot モデルダウンロード（GGUF）")
    print("=" * 50)
    print()

    check_disk_space(required_gb=REQUIRED_GB)
    print()

    model_path = download_model(REPO_ID, FILENAME, SAVE_PATH)
    print(f"\n完了！")
    print(f"次のステップ: python main.py でアプリを起動してください。")
