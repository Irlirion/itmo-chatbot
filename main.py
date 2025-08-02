import asyncio
import io
import logging
import re
import tempfile
from typing import List, Optional

import pdfplumber
from crawl4ai import AsyncWebCrawler, CacheMode, CrawlerRunConfig
from playwright.async_api import TimeoutError as PlaywrightTimeout
from playwright.async_api import async_playwright
from pydantic import BaseModel, Field
from pydantic_ai import Agent, RunContext
from pydantic_ai.models.mistral import MistralModel
from pydantic_ai.providers.mistral import MistralProvider
from pydantic_settings import BaseSettings
from telegram import InlineKeyboardButton, InlineKeyboardMarkup, Update
from telegram.constants import ParseMode
from telegram.ext import (
    Application,
    CallbackQueryHandler,
    CommandHandler,
    ContextTypes,
    MessageHandler,
    filters,
)
from telegram.helpers import escape_markdown

# ----- Logging -----
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ----- Configuration -----
class Settings(BaseSettings):
    telegram_token: str = Field(..., alias="TELEGRAM_BOT_TOKEN")
    mistral_key: str = Field(..., alias="MISTRAL_API_KEY")

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


settings = Settings()

# URLs for the two programs
AI_URLS = {
    "ai": "https://abit.itmo.ru/program/master/ai",
    "ai_product": "https://abit.itmo.ru/program/master/ai_product",
}


# ----- Memory Store for Markdown & PDF Snippets -----
class MemoryStore:
    def __init__(self):
        self.data: dict[str, List[str]] = {key: [] for key in AI_URLS}

    async def load(self):
        """Load both markdown and PDF content for all programs."""
        config = CrawlerRunConfig(cache_mode=CacheMode.BYPASS)
        async with AsyncWebCrawler() as crawler:
            for key, url in AI_URLS.items():
                md = await self._fetch_markdown(crawler, url, config)
                self.data[key] = self._split_paragraphs(md)

                pdf_bytes = await self._download_pdf(url)
                if pdf_bytes:
                    text = self._extract_pdf_text(pdf_bytes)
                    self.data[key].extend(self._split_paragraphs(text))

    async def _fetch_markdown(self, crawler, url: str, config) -> str:
        result = await crawler.arun(url=url, config=config)
        return (
            getattr(result.markdown, "fit_markdown", "")
            or getattr(result.markdown, "raw_markdown", "")
            or ""
        )

    async def _download_pdf(
        self, url: str, button_label: str = "Скачать учебный план"
    ) -> Optional[bytes]:
        """Download PDF by clicking the button and return raw bytes."""
        async with async_playwright() as pw:
            browser = await pw.chromium.launch(headless=True)
            context = await browser.new_context(accept_downloads=True)
            page = await context.new_page()
            await page.goto(url, wait_until="domcontentloaded")

            # Try role/button locator first, then text
            btn = page.get_by_role("button", name=button_label, exact=False)
            if not await btn.count():
                btn = page.get_by_text(button_label, exact=False)
            if not await btn.count():
                await browser.close()
                return None

            try:
                async with page.expect_download(timeout=20_000) as dl_info:
                    await btn.first.click()
                download = await dl_info.value

                # Save to a temporary file and read
                with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
                    await download.save_as(tmp.name)
                    tmp.flush()
                    tmp_path = tmp.name

                with open(tmp_path, "rb") as f:
                    data = f.read()
                return data

            except PlaywrightTimeout:
                logger.warning(f"Timeout downloading PDF from {url}")
                return None
            finally:
                await browser.close()

    def _extract_pdf_text(self, data: bytes) -> str:
        with pdfplumber.open(io.BytesIO(data)) as pdf:
            return "\n\n".join(page.extract_text() or "" for page in pdf.pages)

    def _split_paragraphs(self, text: str) -> List[str]:
        return [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]


MEMORY = MemoryStore()


# ----- Pydantic-AI Agent & Models -----
class BotAnswer(BaseModel):
    answer: str = Field(..., description="Ответ на вопрос по программе")


model = MistralModel(
    model_name="mistral-small-latest",
    provider=MistralProvider(api_key=settings.mistral_key),
)
agent = Agent(
    model=model,
    output_type=BotAnswer,
    instructions=(
        "Вы — ассистент по учебным планам магистратур ИТМО:\n"
        "- Поддерживаемые программы:\n"
        "    • Искусственный интеллект (ключ: ai)\n"
        "    • Управление ИИ-продуктами (ключ: ai_product)\n"
        "- На вопрос вызывайте tool retrieve(program_key, question) для получения до 5 самых релевантных сниппетов.\n"
        "- Сформируйте развёрнутый ответ **только** на запросы, напрямую связанные с учебным планом выбранной программы.\n"
        "- Если retrieve вернёт пустой список или вопрос не касается программ ИТМО, отвечайте:\n"
        '    "Извините, я могу отвечать только на вопросы по учебным планам магистратур ИТМО."\n'
    ),
)


@agent.tool
async def retrieve(ctx: RunContext[None], program_key: str, question: str) -> List[str]:
    snippets = MEMORY.data.get(program_key, [])
    query_words = set(question.lower().split())
    scored = sorted(
        ((len(set(sn.lower().split()) & query_words), sn) for sn in snippets),
        key=lambda x: x[0],
        reverse=True,
    )
    # Return up to 5 best matches with nonzero overlap
    return [text for score, text in scored[:5] if score > 0]


# ----- Telegram Bot Handlers -----
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    context.user_data.pop("prog", None)
    buttons = [
        [InlineKeyboardButton("Искусственный интеллект", callback_data="ai")],
        [InlineKeyboardButton("ИИ-продукты", callback_data="ai_product")],
    ]
    await update.message.reply_text(
        "Привет! Выбери программу магистратуры ИТМО:",
        reply_markup=InlineKeyboardMarkup(buttons),
    )


async def choose(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    await query.answer()
    context.user_data["prog"] = query.data
    await query.edit_message_text(
        "Программа выбрана. Задайте любой вопрос по учебному плану."
    )


async def ask(update: Update, context: ContextTypes.DEFAULT_TYPE):
    text = update.message.text or ""
    prog = context.user_data.get("prog")
    if not prog:
        return await start(update, context)

    await context.bot.send_chat_action(
        chat_id=update.effective_chat.id, action="typing"
    )
    prompt = f"Программа: {prog}\nВопрос: {text}"
    try:
        result = await agent.run(prompt)
        await update.message.reply_text(
            result.output.answer, parse_mode=ParseMode.MARKDOWN
        )
    except Exception:
        logger.error("Ошибка при запросе к модели", exc_info=True)
        await update.message.reply_text(
            "Извините, не удалось получить ответ. Попробуйте позже."
        )


# ----- Main Entrypoint -----
def main():
    asyncio.run(MEMORY.load())

    new_loop = asyncio.new_event_loop()
    asyncio.set_event_loop(new_loop)

    app = Application.builder().token(settings.telegram_token).build()
    app.add_handler(CommandHandler("start", start))
    app.add_handler(CallbackQueryHandler(choose))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, ask))

    logger.info("Бот запущен и ожидает сообщений...")
    app.run_polling()


if __name__ == "__main__":
    main()
