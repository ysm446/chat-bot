"""
モデル管理モジュール（GGUF形式 / llama-cpp-python 対応）
利用可能なモデルの一覧管理・ダウンロード・切り替えを担当
"""
import time
import threading
import logging
from pathlib import Path
from typing import Dict, Any, Generator

logger = logging.getLogger(__name__)

# ダウンロード可能なモデル一覧（GGUF形式）
# repo_id / filename は HuggingFace 上の実際のパスを確認のうえ使用してください
AVAILABLE_MODELS: Dict[str, Dict[str, Any]] = {
    "Qwen3-14B-Q4_K_M": {
        "repo_id": "Qwen/Qwen3-14B-GGUF",
        "filename": "Qwen3-14B-Q4_K_M.gguf",
        "local_path": "models/Qwen3-14B-Q4_K_M.gguf",
        "size_gb": 8.8,
        "vram_gb": 8.8,
        "description": "14B Q4量子化・高精度・大型タスク向け",
    },
    "Qwen3-30B-A3B-Q4_K_M": {
        "repo_id": "Qwen/Qwen3-30B-A3B-GGUF",
        "filename": "Qwen3-30B-A3B-Q4_K_M.gguf",
        "local_path": "models/Qwen3-30B-A3B-Q4_K_M.gguf",
        "size_gb": 18.0,
        "vram_gb": 18.0,
        "description": "30B MoE（実効3B）Q4量子化・最高精度・RTX PRO 5000推奨",
    },
    "Qwen3-30B-A3B-abliterated-Q4_K_M": {
        "repo_id": "DevQuasar/huihui-ai.Qwen3-30B-A3B-abliterated-GGUF",
        "filename": "huihui-ai.Qwen3-30B-A3B-abliterated.Q4_K_M.gguf",
        "local_path": "models/huihui-ai.Qwen3-30B-A3B-abliterated.Q4_K_M.gguf",
        "size_gb": 18.6,
        "vram_gb": 18.6,
        "description": "Qwen3-30B-A3B abliterated版 Q4量子化（安全制限が弱いモデル）",
    },
}


def is_downloaded(model_key: str) -> bool:
    """モデルがダウンロード済みかどうかを確認"""
    info = AVAILABLE_MODELS.get(model_key)
    if not info:
        return False
    return Path(info["local_path"]).exists()


def get_downloaded_models() -> list:
    """ダウンロード済みモデルのキー一覧を返す"""
    return [key for key in AVAILABLE_MODELS if is_downloaded(key)]


def get_model_path(model_key: str) -> str:
    """モデルの GGUFファイルパスを返す"""
    return AVAILABLE_MODELS[model_key]["local_path"]


def build_status_markdown(active_model_key: str = "") -> str:
    """モデル一覧のMarkdownテーブルを生成"""
    lines = [
        "| モデル | サイズ | VRAM目安 | 状態 | 説明 |",
        "|---|---|---|---|---|",
    ]
    for key, info in AVAILABLE_MODELS.items():
        downloaded = "✅ 済" if is_downloaded(key) else "⬜ 未"
        active = " ▶ 使用中" if key == active_model_key else ""
        lines.append(
            f"| **{key}**{active} | {info['size_gb']}GB | "
            f"~{info['vram_gb']}GB | {downloaded} | {info['description']} |"
        )
    return "\n".join(lines)


def download_model(model_key: str) -> Generator[str, None, None]:
    """
    GGUF モデルをダウンロードし、進捗をyieldするジェネレータ

    Yields:
        str: 進捗メッセージ
    """
    from huggingface_hub import hf_hub_download

    info = AVAILABLE_MODELS.get(model_key)
    if not info:
        yield f"❌ 不明なモデル: {model_key}"
        return

    if is_downloaded(model_key):
        yield f"✅ {model_key} はすでにダウンロード済みです。"
        return

    local_path = Path(info["local_path"])
    local_path.parent.mkdir(parents=True, exist_ok=True)

    yield (
        f"📥 ダウンロード開始: {info['repo_id']} / {info['filename']}\n"
        f"推定サイズ: {info['size_gb']}GB\n"
        f"保存先: {local_path.resolve()}"
    )

    result: Dict[str, Any] = {"done": False, "error": None}

    def _run():
        try:
            hf_hub_download(
                repo_id=info["repo_id"],
                filename=info["filename"],
                local_dir=str(local_path.parent),
                local_dir_use_symlinks=False,
            )
            result["done"] = True
        except Exception as e:
            result["error"] = e

    thread = threading.Thread(target=_run, daemon=True)
    thread.start()

    start = time.time()
    while not result["done"] and result["error"] is None:
        elapsed = int(time.time() - start)
        size_bytes = local_path.stat().st_size if local_path.exists() else 0
        downloaded_gb = size_bytes / (1024 ** 3)
        yield (
            f"📥 ダウンロード中... {downloaded_gb:.2f} / {info['size_gb']}GB "
            f"({elapsed}秒経過)"
        )
        time.sleep(2)

    thread.join()

    if result["error"]:
        yield f"❌ エラー: {result['error']}"
    else:
        yield f"✅ ダウンロード完了！ {model_key}\n保存先: {local_path.resolve()}"
