# ITMO Chatbot

Этот проект — Telegram-бот для ответов на вопросы по двум магистерским программам ИТМО:

- "Искусственный интеллект"
- "Управление ИИ-продуктами"

## Возможности

- Выбор программы через кнопки Telegram
- Ответы на вопросы по учебному плану с помощью AI (Mistral)
- Использование сниппетов из markdown и PDF учебных планов
- Честный ответ, если информации недостаточно

## Установка и запуск

1. Создайте файл `.env` в корне проекта:

   ```env
   TELEGRAM_BOT_TOKEN=ваш_токен_бота
   MISTRAL_API_KEY=ваш_ключ_mistral
   ```

2. Запустите бот:

   ```bash
   uv run main.py
   ```

   или

   ```bash
   docker compose up --build
   ```

## Структура проекта

- `main.py` — основной код бота
- `pyproject.toml` — зависимости
- `Dockerfile` — инструкции для сборки Docker-образа
- `compose.yaml` — конфигурация Docker Compose
- `.env.template` — шаблон файла окружения
- `README.md` — описание проекта

## Используемые технологии

- [python-telegram-bot](https://python-telegram-bot.org/)
- [pydantic-ai](https://github.com/pydantic/pydantic-ai)
- [crawl](https://docs.crawl4ai.com)
- [playwright](https://playwright.dev/python/)
- [Mistral](https://mistral.ai/)
- [pdfplumber](https://github.com/jsvine/pdfplumber)

## Контакты

Для вопросов и предложений — пишите в Issues или напрямую автору.
