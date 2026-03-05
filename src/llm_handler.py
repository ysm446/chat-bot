"""
llama-cpp-python LLMハンドラー
Qwen3 GGUF モデルの読み込みと推論を担当（Windows + CUDA 対応）
"""
import re
import logging
from typing import Generator, Tuple, Optional, Dict, Any

logger = logging.getLogger(__name__)

# <think>...</think> タグのパターン
THINKING_PATTERN = re.compile(r"<think>(.*?)</think>", re.DOTALL)
# ストリーミング中のタグ検出用（最大タグ長）
_MAX_TAG_LEN = len("</think>")


class LLMHandler:
    """llama-cpp-python を使ったLLM推論を管理するクラス"""

    def __init__(self, model_path: str, config: Optional[Dict] = None):
        """
        Args:
            model_path: GGUFモデルファイルのパス（例: ./models/Qwen3-4B-Q4_K_M.gguf）
            config: モデル設定
                - n_gpu_layers: int (-1 = 全レイヤーをGPUにオフロード)
                - n_ctx: int (コンテキスト長、デフォルト 32768)
                - n_threads: int (CPUスレッド数、デフォルト 4)
        """
        self.model_path = model_path
        self.config = config or {}
        self.llm = None
        self._load_model()

    def _load_model(self):
        """llama-cpp-python でモデルを読み込む"""
        try:
            from llama_cpp import Llama
        except ImportError as e:
            logger.error(f"必要なパッケージがインストールされていません: {e}")
            raise

        logger.info(f"llama-cpp-python でモデルを読み込んでいます: {self.model_path}")
        logger.info("（初回起動は数分かかる場合があります）")

        n_gpu_layers = int(self.config.get("n_gpu_layers", -1))
        n_ctx = int(self.config.get("n_ctx", 32768))
        n_threads = int(self.config.get("n_threads", 4))

        logger.info(f"設定: n_gpu_layers={n_gpu_layers}, n_ctx={n_ctx}, n_threads={n_threads}")

        # 設定通りに GPU オフロードを試みる（エラー時のみ CPU にフォールバック）
        try:
            self.llm = Llama(
                model_path=self.model_path,
                n_gpu_layers=n_gpu_layers,
                n_ctx=n_ctx,
                n_threads=n_threads,
                verbose=True,
            )
            if n_gpu_layers != 0:
                logger.info(
                    f"✅ モデルの読み込みが完了しました "
                    f"(GPU使用: n_gpu_layers={n_gpu_layers}, n_ctx={n_ctx})"
                )
            else:
                logger.info(f"モデルの読み込みが完了しました (CPU のみ, n_ctx={n_ctx})")
        except Exception as e:
            if n_gpu_layers != 0:
                logger.warning(f"GPU ロードに失敗しました ({e})。CPU のみで再試行します...")
                self.llm = Llama(
                    model_path=self.model_path,
                    n_gpu_layers=0,
                    n_ctx=n_ctx,
                    n_threads=n_threads,
                    verbose=True,
                )
                logger.info(f"モデルの読み込みが完了しました (CPU のみ, n_ctx={n_ctx})")
            else:
                raise

    def _build_messages(
        self,
        query: str,
        context: Optional[str] = None,
        history: Optional[list] = None,
        enable_thinking: bool = True,
    ) -> list:
        """Qwen3用のメッセージリストを構築"""
        messages = []

        system_content = (
            "あなたは優秀なリサーチアシスタントです。"
            "ユーザーの質問に対して、正確で詳細な回答を提供してください。"
            "【重要】プロンプトに「【Web検索結果】」セクションが含まれている場合、"
            "その情報は今この瞬間にWeb検索して取得したリアルタイムの情報です。"
            "必ずその検索結果を最優先で参照して回答してください。"
            "「リアルタイム情報へのアクセス手段がない」「最新情報を提供できない」などと"
            "言ってはいけません。検索結果に基づいて具体的に回答してください。"
            "回答は日本語で行い、情報源がある場合は明記してください。"
        )
        messages.append({"role": "system", "content": system_content})

        if history:
            messages.extend(history)

        user_content = query
        if context:
            user_content = (
                f"{context}\n\n---\n\n"
                f"以上の情報を参考に、次の質問に回答してください:\n{query}"
            )

        # enable_thinking=False の場合、/no_think で思考を抑制
        if not enable_thinking:
            user_content += "\n/no_think"

        messages.append({"role": "user", "content": user_content})
        return messages

    def parse_thinking(self, response: str) -> Tuple[str, str]:
        """
        レスポンスから <think>...</think> を分離

        Returns:
            (thinking_text, answer_text)
        """
        match = THINKING_PATTERN.search(response)
        if match:
            thinking = match.group(1).strip()
            answer = THINKING_PATTERN.sub("", response).strip()
        else:
            thinking = ""
            answer = response.strip()
        return thinking, answer

    def _count_prompt_tokens(self, messages: list) -> int:
        """メッセージ配列のトークン数を見積もる（可能なら厳密、不可なら概算）。"""
        if self.llm is not None and hasattr(self.llm, "tokenize"):
            total = 0
            for msg in messages:
                role = msg.get("role", "")
                content = msg.get("content", "")
                text = f"<|{role}|>\n{content}\n"
                try:
                    tokens = self.llm.tokenize(text.encode("utf-8"), add_bos=False, special=True)
                except TypeError:
                    tokens = self.llm.tokenize(text.encode("utf-8"))
                total += len(tokens)
            if total > 0:
                return total

        # フォールバック: 日本語混在テキストの概算（4文字 ≒ 1 token）
        rough_chars = sum(len(str(msg.get("content", ""))) for msg in messages)
        return max(1, rough_chars // 4)

    def estimate_context_usage(
        self,
        query: str,
        context: Optional[str] = None,
        history: Optional[list] = None,
        sampling_config: Optional[Dict] = None,
        enable_thinking: bool = True,
    ) -> Dict[str, Any]:
        """現在の会話でのコンテキスト使用率（%）を返す。"""
        cfg = sampling_config or {}
        n_ctx = int(self.config.get("n_ctx", 32768))
        reserve_tokens = int(cfg.get("max_tokens", 8192))
        messages = self._build_messages(query, context, history, enable_thinking)
        prompt_tokens = self._count_prompt_tokens(messages)
        total_tokens = prompt_tokens + reserve_tokens

        return {
            "n_ctx": n_ctx,
            "prompt_tokens": prompt_tokens,
            "reserve_tokens": reserve_tokens,
            "total_tokens": total_tokens,
            "prompt_percent": (prompt_tokens / n_ctx) * 100 if n_ctx > 0 else 0.0,
            "usage_percent": (total_tokens / n_ctx) * 100 if n_ctx > 0 else 0.0,
        }

    def generate_with_context(
        self,
        query: str,
        context: Optional[str] = None,
        history: Optional[list] = None,
        sampling_config: Optional[Dict] = None,
        enable_thinking: bool = True,
    ) -> Generator[Dict[str, Any], None, None]:
        """
        コンテキスト付きでテキストを生成し、Thinkingと回答をストリーミングで返す

        Args:
            enable_thinking: Trueで思考モード、Falseで直接回答モード（高速）

        Yields:
            {"type": "thinking_chunk", "text": "..."}  ← 思考トークン（逐次）
            {"type": "answer_chunk",   "text": "..."}  ← 回答トークン（逐次）
            {"type": "done",           "text": ""}
        """
        cfg = sampling_config or {}
        messages = self._build_messages(query, context, history, enable_thinking)

        temperature = float(cfg.get("temperature", 0.6))

        stream = self.llm.create_chat_completion(
            messages=messages,
            temperature=temperature,
            top_p=float(cfg.get("top_p", 0.95)),
            top_k=int(cfg.get("top_k", 20)),
            max_tokens=int(cfg.get("max_tokens", 8192)),
            repeat_penalty=1.05,
            stream=True,
        )

        # ===== ストリーミング中の thinking/answer 分離 =====
        buffer = ""
        state = "preamble"
        _filter_thinking = not enable_thinking
        _PREAMBLE_MAX = 64

        for chunk in stream:
            delta = chunk["choices"][0]["delta"].get("content", "")
            if not delta:
                continue

            buffer += delta

            if state == "preamble":
                if "<think>" in buffer:
                    after = buffer.split("<think>", 1)[1]
                    buffer = after
                    state = "filtering_thinking" if _filter_thinking else "thinking"
                elif len(buffer) > _PREAMBLE_MAX:
                    state = "answer"
                    yield {"type": "answer_chunk", "text": buffer}
                    buffer = ""

            elif state == "filtering_thinking":
                if "</think>" in buffer:
                    state = "answer"
                    after = buffer.split("</think>", 1)[1].lstrip("\n")
                    buffer = after
                    if buffer:
                        yield {"type": "answer_chunk", "text": buffer}
                        buffer = ""
                else:
                    if len(buffer) > _MAX_TAG_LEN:
                        buffer = buffer[-_MAX_TAG_LEN:]

            elif state == "thinking":
                if "</think>" in buffer:
                    state = "answer"
                    parts = buffer.split("</think>", 1)
                    if parts[0]:
                        yield {"type": "thinking_chunk", "text": parts[0]}
                    buffer = parts[1].lstrip("\n")
                    if buffer:
                        yield {"type": "answer_chunk", "text": buffer}
                        buffer = ""
                else:
                    safe_len = max(0, len(buffer) - _MAX_TAG_LEN)
                    if safe_len > 0:
                        yield {"type": "thinking_chunk", "text": buffer[:safe_len]}
                        buffer = buffer[safe_len:]

            else:  # state == "answer"
                yield {"type": "answer_chunk", "text": buffer}
                buffer = ""

        # 残りバッファをフラッシュ
        if buffer.strip():
            if state == "thinking":
                yield {"type": "thinking_chunk", "text": buffer.replace("</think>", "").strip()}
            elif state == "answer":
                yield {"type": "answer_chunk", "text": buffer}

        yield {"type": "done", "text": ""}

    def shutdown(self):
        """llama-cpp モデルをアンロードしてメモリを解放"""
        if self.llm is not None:
            del self.llm
            self.llm = None
        logger.info("llama-cpp モデルをシャットダウンしました")
