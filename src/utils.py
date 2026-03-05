"""
ユーティリティ関数
設定読み込み、ログ設定、その他共通処理
"""
import json
import logging
import logging.handlers
import os
from pathlib import Path
from typing import Dict, Any, Optional

SETTINGS_PATH = Path("settings.json")

DEFAULT_SETTINGS = {
    "enable_search": True,
    "enable_thinking_mode": True,
    "show_thinking": True,
    "temperature": 0.6,
    "max_tokens": 8192,
    "active_model_key": "",
}


def load_settings() -> Dict[str, Any]:
    """settings.json を読み込む。ファイルがなければデフォルト値を返す。"""
    if SETTINGS_PATH.exists():
        try:
            with open(SETTINGS_PATH, "r", encoding="utf-8") as f:
                saved = json.load(f)
            # デフォルト値をベースに上書き（キー追加にも対応）
            return {**DEFAULT_SETTINGS, **saved}
        except Exception:
            pass
    return dict(DEFAULT_SETTINGS)


def save_settings(settings: Dict[str, Any]) -> None:
    """UI設定を settings.json に保存する。"""
    with open(SETTINGS_PATH, "w", encoding="utf-8") as f:
        json.dump(settings, f, ensure_ascii=False, indent=2)


def load_config(config_path: str = "config.yaml") -> Dict[str, Any]:
    """
    設定ファイルを読み込む

    Args:
        config_path: YAMLファイルのパス

    Returns:
        設定辞書
    """
    try:
        import yaml
    except ImportError:
        raise ImportError("pyyaml がインストールされていません: pip install pyyaml")

    config_file = Path(config_path)
    if not config_file.exists():
        raise FileNotFoundError(f"設定ファイルが見つかりません: {config_path}")

    with open(config_file, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    return config or {}


def setup_logging(
    level: str = "INFO",
    log_file: Optional[str] = None,
    max_bytes: int = 10 * 1024 * 1024,  # 10MB
    backup_count: int = 3,
) -> None:
    """
    ロギングの設定

    Args:
        level: ログレベル ("DEBUG", "INFO", "WARNING", "ERROR")
        log_file: ログファイルのパス（Noneの場合はコンソールのみ）
        max_bytes: ログファイルの最大サイズ
        backup_count: バックアップファイル数
    """
    log_level = getattr(logging, level.upper(), logging.INFO)

    formatter = logging.Formatter(
        fmt="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    root_logger = logging.getLogger()
    root_logger.setLevel(log_level)

    # 既存のハンドラーをクリア
    root_logger.handlers.clear()

    # コンソールハンドラー
    console_handler = logging.StreamHandler()
    console_handler.setLevel(log_level)
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)

    # ファイルハンドラー（指定された場合）
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)

        file_handler = logging.handlers.RotatingFileHandler(
            log_file,
            maxBytes=max_bytes,
            backupCount=backup_count,
            encoding="utf-8",
        )
        file_handler.setLevel(log_level)
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)


def check_model_exists(model_path: str) -> bool:
    """
    モデルファイルが存在するか確認（GGUF単一ファイル対応）

    Args:
        model_path: GGUFファイルのパス

    Returns:
        True: 存在する, False: 存在しない
    """
    path = Path(model_path)
    return path.exists() and path.is_file()


def format_thinking_html(thinking_text: str) -> str:
    """
    Thinkingテキストを折りたたみHTML形式に変換

    Args:
        thinking_text: Thinkingプロセスのテキスト

    Returns:
        HTML形式の折りたたみブロック
    """
    if not thinking_text:
        return ""

    # 長すぎる場合は省略
    max_len = 2000
    if len(thinking_text) > max_len:
        display_text = thinking_text[:max_len] + f"\n\n...（{len(thinking_text) - max_len}文字省略）"
    else:
        display_text = thinking_text

    return (
        f'<details>\n'
        f'<summary>💭 思考プロセス（クリックで展開）</summary>\n\n'
        f'```\n{display_text}\n```\n'
        f'</details>'
    )


def format_search_results_html(search_results: list) -> str:
    """
    検索結果をMarkdown形式に変換（出典表示用）

    Args:
        search_results: SearchHandlerの生検索結果

    Returns:
        Markdown形式の出典リスト
    """
    if not search_results:
        return ""

    lines = ["\n\n---\n**参考情報:**"]
    for i, result in enumerate(search_results, 1):
        title = result.get("title", "タイトルなし")
        href = result.get("href", "")
        if href:
            lines.append(f"{i}. [{title}]({href})")
        else:
            lines.append(f"{i}. {title}")

    return "\n".join(lines)
