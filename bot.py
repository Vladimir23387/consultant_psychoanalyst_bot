import os
import json
import openai
import numpy as np
import faiss
import re
from telegram import Update
from telegram.ext import Updater, CommandHandler, MessageHandler, Filters, CallbackContext
from dotenv import load_dotenv

load_dotenv()

# Размерность эмбеддингов (для text-embedding-ada-002 обычно 1536)
EMBEDDING_DIM = 1536

# Папка для хранения сохранённых бесед
CONVERSATION_DIR = "conversations"
if not os.path.exists(CONVERSATION_DIR):
    os.makedirs(CONVERSATION_DIR)

# Глобальный словарь для хранения бесед в оперативной памяти
user_conversations = {}

class UserConversation:
    """
    Хранит историю сообщений, FAISS-индекс, сводку беседы и статус сессии.
    """
    def __init__(self):
        self.messages = []  # Список сообщений: {"role": "user"/"assistant"/"system", "content": ...}
        self.index = faiss.IndexFlatL2(EMBEDDING_DIM)
        self.embeddings = []  # Список эмбеддингов сообщений
        self.summary = ""     # Сводка предыдущих бесед
        self.stopped = False  # Статус: False - активна, True - приостановлена

    def add_message(self, message: dict, embedding_vector: list):
        self.messages.append(message)
        self.embeddings.append(embedding_vector)
        vector = np.array(embedding_vector, dtype='float32').reshape(1, -1)
        self.index.add(vector)

def save_conversation(user_id, conv: UserConversation):
    """Сохраняет историю беседы и сводку в JSON-файл."""
    filename = os.path.join(CONVERSATION_DIR, f"{user_id}.json")
    data = {
        "messages": conv.messages,
        "summary": conv.summary,
    }
    try:
        with open(filename, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        print(f"Беседа пользователя {user_id} сохранена в {filename}")
    except Exception as e:
        print(f"Ошибка сохранения беседы для пользователя {user_id}: {e}")

def load_conversation(user_id):
    """Загружает историю беседы и сводку из файла JSON и восстанавливает объект UserConversation."""
    filename = os.path.join(CONVERSATION_DIR, f"{user_id}.json")
    if not os.path.exists(filename):
        return None
    try:
        with open(filename, "r", encoding="utf-8") as f:
            data = json.load(f)
        conv = UserConversation()
        conv.messages = data.get("messages", [])
        conv.summary = data.get("summary", "")
        # Восстанавливаем FAISS-индекс: пересчитываем эмбеддинги для каждого сообщения
        for msg in conv.messages:
            embedding = get_embedding(msg["content"])
            conv.embeddings.append(embedding)
            vector = np.array(embedding, dtype='float32').reshape(1, -1)
            conv.index.add(vector)
        print(f"Беседа пользователя {user_id} загружена из {filename}")
        return conv
    except Exception as e:
        print(f"Ошибка загрузки беседы для пользователя {user_id}: {e}")
        return None

def get_embedding(text: str) -> list:
    """Вычисляет эмбеддинг текста с использованием модели OpenAI."""
    try:
        response = openai.Embedding.create(
            input=text,
            model="text-embedding-ada-002"
        )
        embedding = response["data"][0]["embedding"]
        return embedding
    except Exception as e:
        print(f"Ошибка получения эмбеддинга: {e}")
        return [0.0] * EMBEDDING_DIM

def get_gpt4_response(messages: list) -> str:
    """Вызывает API GPT-4 для генерации ответа с заданной историей сообщений."""
    try:
        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=messages,
            temperature=0
        )
        return response["choices"][0]["message"]["content"].strip()
    except Exception as e:
        print(f"Ошибка вызова GPT-4: {e}")
        return "Извини, произошла ошибка при генерации ответа."

def update_conversation_summary(conv: UserConversation):
    """
    Обновляет сводку беседы на основе всей истории сообщений.
    Сводка сохраняется в conv.summary.
    """
    conversation_text = ""
    for msg in conv.messages:
        conversation_text += f"{msg['role'].capitalize()}: {msg['content']}\n"
    prompt = [
        {"role": "system", "content": "Ты профессиональный психоаналитик, обладающий долговременной памятью о прошлых беседах. Сформируй краткое резюме всей беседы, выделив ключевые моменты и детали."},
        {"role": "user", "content": conversation_text}
    ]
    summary = get_gpt4_response(prompt)
    conv.summary = summary
    print("Обновлённая сводка беседы:", conv.summary)

def start(update: Update, context: CallbackContext):
    """
    Команда /start:
      – Загружает сохранённую беседу (если есть) и возобновляет сессию.
      – Если беседы нет, создаёт новую.
    """
    user_id = update.effective_user.id
    if user_id not in user_conversations:
        conv = load_conversation(user_id)
        if conv is None:
            conv = UserConversation()
        user_conversations[user_id] = conv
    conv = user_conversations[user_id]
    conv.stopped = False
    update.message.reply_text("Мы продолжаем нашу предыдущую беседу. Чем я могу помочь?")

def help_command(update: Update, context: CallbackContext):
    """Команда /help: выводит справку по командам."""
    help_text = (
        "Доступные команды:\n"
        "/start - начать или продолжить беседу\n"
        "/stop - приостановить беседу (история сохраняется)\n"
        "/help - получить справку по командам\n"
        "/memory - показать сводку предыдущих бесед\n"
        "Просто отправьте сообщение, и бот продолжит общение."
    )
    update.message.reply_text(help_text)

def stop(update: Update, context: CallbackContext):
    """
    Команда /stop:
      – Помечает сессию как приостановленную.
      – Сохраняет историю беседы на диск.
    """
    user_id = update.effective_user.id
    if user_id in user_conversations:
        conv = user_conversations[user_id]
        conv.stopped = True
        save_conversation(user_id, conv)
        update.message.reply_text("Беседа приостановлена. Используйте /start для продолжения.")
    else:
        update.message.reply_text("Нет активной беседы. Используйте /start для начала.")

def memory(update: Update, context: CallbackContext):
    """
    Команда /memory:
      – Если беседа уже загружена в оперативную память, выводит сводку.
      – Если нет, пытается загрузить её из файла.
    """
    user_id = update.effective_user.id
    conv = user_conversations.get(user_id)
    if conv is None:
        conv = load_conversation(user_id)
        if conv is None:
            update.message.reply_text("Нет сохранённой беседы. Используйте /start для начала.")
            return
        user_conversations[user_id] = conv
    if conv.summary:
        update.message.reply_text(f"Сводка прошлых бесед:\n{conv.summary}")
    else:
        update.message.reply_text("Пока сводки нет. Начните беседу, и я соберу информацию.")

def handle_message(update: Update, context: CallbackContext):
    """
    Обработчик входящих сообщений:
      – Если сессия приостановлена, просит возобновить её.
      – Если пользователь явно спрашивает о памяти (например, содержит "помни" и "обсужд" или "говор"), сразу отвечает накопленной сводкой.
      – Иначе добавляет сообщение в историю и формирует запрос к модели, передавая всю историю.
    """
    user_id = update.effective_user.id
    user_text = update.message.text.strip()
    
    # Если беседа отсутствует, пытаемся загрузить её или создаём новую.
    if user_id not in user_conversations:
        conv = load_conversation(user_id)
        if conv is None:
            conv = UserConversation()
        user_conversations[user_id] = conv
    conv = user_conversations[user_id]
    
    if conv.stopped:
        update.message.reply_text("Беседа приостановлена. Используйте /start для продолжения.")
        return

    # Если пользователь явно спрашивает о памяти, перехватываем запрос
    lower_text = user_text.lower()
    if "помни" in lower_text and ("обсужд" in lower_text or "говор" in lower_text):
        # Если сводка пуста, попробуем обновить, если в истории достаточно информации
        if not conv.summary and len(conv.messages) >= 2:
            update_conversation_summary(conv)
        if conv.summary:
            update.message.reply_text(f"Конечно, вот что я помню:\n{conv.summary}")
        else:
            update.message.reply_text("Пока у меня недостаточно информации для формирования сводки. Давайте продолжим общение.")
        return

    # Добавляем сообщение пользователя в историю
    user_embedding = get_embedding(user_text)
    user_message = {"role": "user", "content": user_text}
    conv.add_message(user_message, user_embedding)
    
    # Формируем запрос к модели.
    # Если есть сводка, включаем её в системное сообщение с явной инструкцией:
    messages_for_gpt = []
    if conv.summary:
        messages_for_gpt.append({
            "role": "system",
            "content": f"Ты опытный психоаналитик, который помнит все детали предыдущих бесед с пользователем. Вот сводка прошлых бесед: {conv.summary}. Используй эту информацию в своих ответах и никогда не говори, что не помнишь детали."
        })
    else:
        messages_for_gpt.append({
            "role": "system",
            "content": "Ты опытный психоаналитик. Используй всю информацию из предыдущих сообщений (если она есть) для формирования ответов."
        })
    
    # Передаём всю историю, если она не слишком длинная, иначе последние 10 сообщений
    if len(conv.messages) < 50:
        messages_for_gpt.extend(conv.messages)
    else:
        messages_for_gpt.extend(conv.messages[-10:])
    
    response_text = get_gpt4_response(messages_for_gpt)
    update.message.reply_text(response_text)
    
    assistant_embedding = get_embedding(response_text)
    assistant_message = {"role": "assistant", "content": response_text}
    conv.add_message(assistant_message, assistant_embedding)
    
    # При накоплении достаточного количества сообщений обновляем сводку и сохраняем беседу
    if len(conv.messages) > 10:
        update_conversation_summary(conv)
        save_conversation(user_id, conv)

def main():
    openai.api_key = os.environ.get("OPENAI_API_KEY")
    telegram_token = os.environ.get("TELEGRAM_BOT_TOKEN")
    if telegram_token is None:
        print("Ошибка: TELEGRAM_BOT_TOKEN не найден в переменных окружения.")
        return
    if openai.api_key is None:
        print("Ошибка: OPENAI_API_KEY не найден в переменных окружения.")
        return

    updater = Updater(telegram_token, use_context=True)
    dp = updater.dispatcher

    dp.add_handler(CommandHandler("start", start))
    dp.add_handler(CommandHandler("help", help_command))
    dp.add_handler(CommandHandler("stop", stop))
    dp.add_handler(CommandHandler("memory", memory))
    dp.add_handler(MessageHandler(Filters.text & ~Filters.command, handle_message))

    print("Бот запущен. Ожидание сообщений...")
    updater.start_polling()
    updater.idle()

if __name__ == "__main__":
    main()
