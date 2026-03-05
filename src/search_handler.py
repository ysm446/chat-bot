"""
DuckDuckGo検索ハンドラー
Web検索の実行と結果の整形を担当
"""
import time
import logging
from typing import List, Dict, Optional

logger = logging.getLogger(__name__)


class SearchHandler:
    """DuckDuckGo検索を管理するクラス"""

    # 検索が不要と判断するキーワード（簡易版）
    NO_SEARCH_KEYWORDS = [
        "こんにちは", "ありがとう", "おはよう", "こんばんは",
        "hello", "hi", "thanks", "thank you",
        "教えて", "説明して", "とは何ですか",  # 一般知識質問は検索必要なことも多い
    ]

    # 検索が必要と判断するキーワード
    SEARCH_KEYWORDS = [
        "最新", "今日", "現在", "今", "今年", "今月",
        "株価", "ニュース", "速報", "天気", "為替",
        "latest", "current", "today", "news", "price",
        "いつ", "どこで", "誰が",
        "2024", "2025", "2026",
    ]

    def __init__(self, config: Optional[Dict] = None):
        self.config = config or {}
        self.max_results = self.config.get("max_results", 5)
        self.region = self.config.get("region", "jp-jp")
        self.safe_search = self.config.get("safe_search", "moderate")
        self.timeout = self.config.get("timeout", 10)
        self._last_search_time = 0
        self._rate_limit_seconds = 1.0  # 1秒1リクエスト

    def _rate_limit(self):
        """レート制限: 1秒に1リクエスト"""
        elapsed = time.time() - self._last_search_time
        if elapsed < self._rate_limit_seconds:
            time.sleep(self._rate_limit_seconds - elapsed)
        self._last_search_time = time.time()

    def search(self, query: str, max_results: Optional[int] = None) -> List[Dict]:
        """
        DuckDuckGoで検索を実行

        Args:
            query: 検索クエリ
            max_results: 最大結果数（Noneの場合はconfig値を使用）

        Returns:
            検索結果のリスト（各要素: title, body, href）
        """
        try:
            try:
                from ddgs import DDGS  # 新パッケージ名
            except ImportError:
                from duckduckgo_search import DDGS  # 旧パッケージ名フォールバック
        except ImportError:
            logger.error("ddgs がインストールされていません: pip install ddgs")
            return []

        num_results = max_results or self.max_results

        # レート制限
        self._rate_limit()

        logger.info(f"検索クエリ: {query}")

        results = []
        retries = 3

        for attempt in range(retries):
            try:
                with DDGS() as ddgs:
                    raw_results = list(ddgs.text(
                        query,
                        region=self.region,
                        safesearch=self.safe_search,
                        max_results=num_results,
                    ))
                results = raw_results
                break
            except Exception as e:
                logger.warning(f"検索エラー（試行 {attempt + 1}/{retries}）: {e}")
                if attempt < retries - 1:
                    time.sleep(2 ** attempt)  # 指数バックオフ
                else:
                    logger.error(f"検索失敗: {e}")
                    return []

        logger.info(f"検索結果: {len(results)} 件")
        return results

    def format_results(self, results: List[Dict]) -> str:
        """
        検索結果を読みやすいテキストに整形

        Args:
            results: search()の返り値

        Returns:
            整形された検索結果テキスト
        """
        if not results:
            return "検索結果が見つかりませんでした。"

        lines = ["【Web検索結果】\n"]
        for i, result in enumerate(results, 1):
            title = result.get("title", "タイトルなし")
            body = result.get("body", "")
            href = result.get("href", "")

            lines.append(f"[{i}] {title}")
            if body:
                # 本文を300文字に制限
                body_truncated = body[:300] + "..." if len(body) > 300 else body
                lines.append(f"    {body_truncated}")
            if href:
                lines.append(f"    URL: {href}")
            lines.append("")

        return "\n".join(lines)

    def is_search_needed(self, query: str) -> bool:
        """
        クエリに対してWeb検索が必要かどうかを判定（簡易版）

        Args:
            query: ユーザーのクエリ

        Returns:
            True: 検索が必要, False: 不要
        """
        query_lower = query.lower()

        # 検索が明らかに必要なキーワードが含まれている場合
        for keyword in self.SEARCH_KEYWORDS:
            if keyword in query_lower:
                logger.debug(f"検索キーワード '{keyword}' を検出: 検索を実行")
                return True

        # URLが含まれている場合（URL内容の確認）
        if "http://" in query_lower or "https://" in query_lower:
            return True

        # クエリが短い挨拶の場合は検索不要
        if len(query.strip()) < 10:
            return False

        # デフォルトは検索を実行（安全側）
        return True

    def search_and_format(self, query: str) -> tuple[str, List[Dict]]:
        """
        検索を実行して整形済みテキストと生データを返す

        Returns:
            (整形済みテキスト, 生の検索結果リスト)
        """
        results = self.search(query)
        formatted = self.format_results(results)
        return formatted, results
