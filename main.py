"""
Research-Bot API エントリーポイント
Gradioを廃止し、Electron UI向けに FastAPI + SSE を提供する。
"""
import argparse
import asyncio
import json
import logging
import re
import sys
from typing import Any, AsyncGenerator, Dict, Generator, List, Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from starlette.responses import StreamingResponse

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Windows端末での文字化け対策: stdout/stderr を UTF-8 に固定
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")


class ChatMessage(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    message: str
    history: List[ChatMessage] = Field(default_factory=list)
    enable_search: bool = True
    show_thinking: bool = True
    enable_thinking_mode: bool = True
    temperature: float = 0.6
    max_tokens: int = 8192


class ModelSwitchRequest(BaseModel):
    model_key: str


class ModelDownloadRequest(BaseModel):
    model_key: str


class SettingsRequest(BaseModel):
    enable_search: bool
    enable_thinking_mode: bool
    show_thinking: bool
    temperature: float
    max_tokens: int


def load_components(config: dict):
    """LLMHandlerとSearchHandlerを初期化"""
    from src.llm_handler import LLMHandler
    from src.search_handler import SearchHandler

    model_config = config.get("model", {})
    search_config = config.get("search", {})

    model_path = model_config.get("path", "./models/Qwen3-30B-A3B-Q4_K_M.gguf")

    logger.info("LLMを初期化中: %s", model_path)
    llm = LLMHandler(model_path=model_path, config=model_config)

    logger.info("検索ハンドラーを初期化中")
    searcher = SearchHandler(config=search_config)

    return llm, searcher


def _normalize_history(history: List[Any]) -> List[dict]:
    """履歴を role/content 形式へ正規化する。"""
    normalized: List[dict] = []
    for msg in history or []:
        if isinstance(msg, dict):
            role = msg.get("role")
            content = msg.get("content")
            if role in {"user", "assistant"} and content is not None:
                normalized.append({"role": role, "content": str(content)})
            continue

        if isinstance(msg, (list, tuple)) and len(msg) == 2:
            user_text, assistant_text = msg
            if user_text:
                normalized.append({"role": "user", "content": str(user_text)})
            if assistant_text:
                normalized.append({"role": "assistant", "content": str(assistant_text)})

    return normalized


def _format_context_usage_text(usage: dict) -> str:
    usage_percent = float(usage.get("usage_percent", 0.0))
    prompt_percent = float(usage.get("prompt_percent", 0.0))
    prompt_tokens = int(usage.get("prompt_tokens", 0))
    reserve_tokens = int(usage.get("reserve_tokens", 0))
    n_ctx = int(usage.get("n_ctx", 0))

    if usage_percent >= 95:
        level = "終了推奨"
    elif usage_percent >= 85:
        level = "注意"
    else:
        level = "余裕あり"

    return (
        f"{usage_percent:.1f}% ({level}) | "
        f"prompt {prompt_percent:.1f}% [{prompt_tokens}] + "
        f"reserve [{reserve_tokens}] / n_ctx {n_ctx}"
    )


def _detect_active_model(model_path: str) -> str:
    from src.model_manager import AVAILABLE_MODELS

    for key, info in AVAILABLE_MODELS.items():
        if info["local_path"].replace("\\", "/") in model_path.replace("\\", "/"):
            return key
    return ""


def process_query(
    query: str,
    history: List[dict],
    llm,
    searcher,
    config: dict,
    enable_search: bool = True,
    show_thinking: bool = True,
    enable_thinking_mode: bool = True,
) -> Generator[Dict[str, Any], None, None]:
    from src.utils import format_search_results_html, format_thinking_html

    search_config = config.get("search", {})
    sampling_config = config.get("sampling", {})
    display_config = config.get("display", {})

    normalized_history = _normalize_history(history)
    llm_history: List[Dict[str, str]] = []
    for msg in normalized_history:
        role = msg.get("role", "")
        content = msg.get("content", "")
        if role == "user":
            llm_history.append({"role": "user", "content": content})
        elif role == "assistant":
            llm_history.append({"role": "assistant", "content": re.sub(r"<[^>]+>", "", content)})

    context: Optional[str] = None
    raw_search_results: List[dict] = []
    status = "思考中..."
    context_usage_text = "計算中..."

    if enable_search and search_config.get("enabled", True):
        if searcher.is_search_needed(query):
            status = "🔍 Web検索中..."
            yield {
                "event": "status",
                "status": status,
                "context_usage": context_usage_text,
                "answer": "🔍 検索中...",
                "thinking": "",
            }

            logger.info("Web検索を実行: %s", query)
            context, raw_search_results = searcher.search_and_format(query)

            if raw_search_results:
                status = f"🔍 検索完了（{len(raw_search_results)}件） → 思考中..."
            else:
                logger.warning("検索結果が見つかりませんでした")
                context = None
                status = "検索結果なし → 思考中..."

    thinking_text = ""
    answer_text = ""
    status = "💭 思考中..."

    try:
        usage = llm.estimate_context_usage(
            query=query,
            context=context,
            history=llm_history if llm_history else None,
            sampling_config=sampling_config,
            enable_thinking=enable_thinking_mode,
        )
        context_usage_text = _format_context_usage_text(usage)
    except Exception as exc:
        logger.warning("コンテキスト使用率の計算に失敗: %s", exc)
        context_usage_text = "計算失敗"

    yield {
        "event": "status",
        "status": status,
        "context_usage": context_usage_text,
        "answer": "💭 考えています...",
        "thinking": "",
    }

    try:
        for chunk in llm.generate_with_context(
            query=query,
            context=context,
            history=llm_history if llm_history else None,
            sampling_config=sampling_config,
            enable_thinking=enable_thinking_mode,
        ):
            chunk_type = chunk.get("type")
            chunk_text = chunk.get("text", "")

            if chunk_type == "thinking_chunk":
                thinking_text += chunk_text
                if show_thinking:
                    yield {
                        "event": "thinking",
                        "status": "💭 思考中...",
                        "context_usage": context_usage_text,
                        "thinking": format_thinking_html(thinking_text),
                        "answer": answer_text,
                    }

            elif chunk_type == "answer_chunk":
                answer_text += chunk_text
                yield {
                    "event": "answer",
                    "status": "✍️ 回答生成中...",
                    "context_usage": context_usage_text,
                    "thinking": format_thinking_html(thinking_text) if (show_thinking and thinking_text) else "",
                    "answer": answer_text,
                }

            elif chunk_type == "done":
                break

    except Exception as exc:
        logger.error("LLM生成エラー: %s", exc)
        answer_text = f"エラーが発生しました: {str(exc)}"
        status = "❌ エラー"

    final_response = answer_text
    if raw_search_results and display_config.get("show_search_results", True):
        final_response += format_search_results_html(raw_search_results)

    final_thinking = format_thinking_html(thinking_text) if (show_thinking and thinking_text) else ""

    yield {
        "event": "final",
        "status": "✅ 完了" if status != "❌ エラー" else status,
        "context_usage": context_usage_text,
        "thinking": final_thinking,
        "answer": final_response,
        "search_results": raw_search_results,
    }


def create_app(config: dict, llm_container: dict, searcher) -> FastAPI:
    from src import model_manager
    from src.llm_handler import LLMHandler
    from src.utils import load_settings, save_settings

    app = FastAPI(title="Research-Bot API", version="2.0")
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app_state = {
        "active_model_key": load_settings().get("active_model_key")
        or _detect_active_model(config.get("model", {}).get("path", "")),
    }

    @app.get("/health")
    async def health() -> Dict[str, Any]:
        return {"status": "ok"}

    @app.get("/api/bootstrap")
    async def bootstrap() -> Dict[str, Any]:
        settings = load_settings()
        downloaded = set(model_manager.get_downloaded_models())

        models = []
        for key, info in model_manager.AVAILABLE_MODELS.items():
            models.append(
                {
                    "key": key,
                    "downloaded": key in downloaded,
                    "size_gb": info.get("size_gb"),
                    "vram_gb": info.get("vram_gb"),
                    "description": info.get("description", ""),
                }
            )

        return {
            "settings": settings,
            "defaults": {
                "enable_search": config.get("search", {}).get("enabled", True),
                "show_thinking": config.get("display", {}).get("show_thinking", True),
                "temperature": config.get("sampling", {}).get("temperature", 0.6),
                "max_tokens": config.get("sampling", {}).get("max_tokens", 8192),
            },
            "active_model_key": app_state.get("active_model_key", ""),
            "models": models,
        }

    @app.post("/api/settings")
    async def update_settings(payload: SettingsRequest) -> Dict[str, Any]:
        new_settings = payload.model_dump()
        new_settings["active_model_key"] = app_state.get("active_model_key", "")
        save_settings(new_settings)
        return {"ok": True}

    @app.post("/api/chat/stream")
    async def chat_stream(payload: ChatRequest):
        sampling_config = {
            **config.get("sampling", {}),
            "temperature": payload.temperature,
            "max_tokens": int(payload.max_tokens),
        }
        current_config = {**config, "sampling": sampling_config}

        def generate_events() -> Generator[str, None, None]:
            for event_payload in process_query(
                query=payload.message,
                history=[msg.model_dump() for msg in payload.history],
                llm=llm_container["llm"],
                searcher=searcher,
                config=current_config,
                enable_search=payload.enable_search,
                show_thinking=payload.show_thinking,
                enable_thinking_mode=payload.enable_thinking_mode,
            ):
                yield f"data: {json.dumps(event_payload, ensure_ascii=False)}\n\n"

        return StreamingResponse(generate_events(), media_type="text/event-stream")

    @app.get("/api/models")
    async def models() -> Dict[str, Any]:
        downloaded = set(model_manager.get_downloaded_models())
        rows = []
        for key, info in model_manager.AVAILABLE_MODELS.items():
            rows.append(
                {
                    "key": key,
                    "downloaded": key in downloaded,
                    "active": key == app_state.get("active_model_key", ""),
                    "size_gb": info.get("size_gb"),
                    "vram_gb": info.get("vram_gb"),
                    "description": info.get("description", ""),
                }
            )
        return {"models": rows}

    @app.post("/api/models/download/stream")
    async def download_stream(payload: ModelDownloadRequest):
        if not payload.model_key:
            raise HTTPException(status_code=400, detail="model_key is required")

        async def event_gen() -> AsyncGenerator[str, None]:
            for msg in model_manager.download_model(payload.model_key):
                event = {"event": "progress", "message": msg}
                yield f"data: {json.dumps(event, ensure_ascii=False)}\n\n"
                await asyncio.sleep(0)

            refreshed = {
                "event": "done",
                "models": (await models())["models"],
            }
            yield f"data: {json.dumps(refreshed, ensure_ascii=False)}\n\n"

        return StreamingResponse(event_gen(), media_type="text/event-stream")

    @app.post("/api/models/unload")
    async def unload_model() -> Dict[str, Any]:
        if llm_container.get("llm") is None:
            return {"ok": False, "message": "モデルはすでにアンロード済みです"}

        import gc

        llm_container["llm"].shutdown()
        llm_container["llm"] = None
        app_state["active_model_key"] = ""
        gc.collect()

        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass

        return {"ok": True, "message": "アンロード完了。VRAMを解放しました"}

    @app.post("/api/models/switch")
    async def switch_model(payload: ModelSwitchRequest) -> Dict[str, Any]:
        model_key = payload.model_key
        if not model_key:
            raise HTTPException(status_code=400, detail="model_key is required")

        if not model_manager.is_downloaded(model_key):
            raise HTTPException(status_code=400, detail=f"{model_key} はダウンロードされていません")

        if model_key == app_state.get("active_model_key", ""):
            return {"ok": True, "message": f"{model_key} はすでに使用中です"}

        try:
            import gc

            old_llm = llm_container.get("llm")
            if old_llm is not None:
                old_llm.shutdown()
                del old_llm
                llm_container["llm"] = None
                gc.collect()
                try:
                    import torch

                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                except ImportError:
                    pass

            model_path = model_manager.get_model_path(model_key)
            model_config = config.get("model", {})
            new_llm = LLMHandler(model_path=model_path, config=model_config)
            llm_container["llm"] = new_llm
            app_state["active_model_key"] = model_key

            from src.utils import save_settings, load_settings

            current = load_settings()
            current["active_model_key"] = model_key
            save_settings(current)

            return {"ok": True, "message": f"{model_key} への切り替えが完了しました"}
        except Exception as exc:
            logger.exception("モデル切り替えエラー: %s", exc)
            raise HTTPException(status_code=500, detail=str(exc))

    return app


def run_server(host: str = "127.0.0.1", port: int = 8765) -> None:
    from src.utils import check_model_exists, load_config, setup_logging

    try:
        config = load_config("config.yaml")
    except FileNotFoundError:
        logger.error("config.yaml が見つかりません。プロジェクトルートから実行してください。")
        sys.exit(1)

    setup_logging(level="INFO", log_file="logs/research_bot.log")

    model_path = config.get("model", {}).get("path", "./models/Qwen3-30B-A3B-Q4_K_M.gguf")
    if not check_model_exists(model_path):
        logger.error("モデルが見つかりません: %s", model_path)
        logger.error("先に download_model.py を実行してください: python download_model.py")
        sys.exit(1)

    logger.info("Research-Bot API を起動しています...")
    try:
        llm, searcher = load_components(config)
    except Exception as exc:
        logger.exception("初期化エラー: %s", exc)
        sys.exit(1)

    llm_container = {"llm": llm}
    app = create_app(config, llm_container, searcher)

    import uvicorn

    logger.info("APIサーバー起動: http://%s:%s", host, port)
    uvicorn.run(app, host=host, port=port, log_level="info")


def main() -> None:
    parser = argparse.ArgumentParser(description="Research-Bot backend server")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()

    run_server(host=args.host, port=args.port)


if __name__ == "__main__":
    main()
