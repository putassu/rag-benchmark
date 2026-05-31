import os
import logging
from dotenv import load_dotenv
import urllib3

# Отключаем ворнинги из-за verify=False (полезно для внутренних корпоративных сертифкатов)
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

load_dotenv()

OPENWEB_URL = os.getenv("OPENWEB_URL", "http://localhost:8080")
OPENWEB_API_KEY = os.getenv("OPENWEB_API_KEY", "")
QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY", "")
DOCS_DIR = os.getenv("DOCS_DIR", "./docs")
LOG_LEVEL = os.getenv("LOG_LEVEL", "DEBUG")

# Настройка логирования
logging.basicConfig(
    level=getattr(logging, LOG_LEVEL),
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger("rag_annotator")

# Распределение аналитиков (настраивается здесь)
ANALYSTS = [
    {"id": 1, "name": "Саша"},
    {"id": 2, "name": "Аналитик 2"},
    {"id": 3, "name": "Аналитик 3"},
]