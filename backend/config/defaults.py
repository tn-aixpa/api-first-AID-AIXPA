### Custom default settings
# SECRET_KEY is created automatically if missing

# @arMQDSZFcvgh72
import os


class ConfigDefault:
    PASSWORD_LENGTH: int = os.environ.get("PASSWORD_LENGTH", 13) 
    JWT_ALGORITHM: str = os.environ.get("JWT_ALGORITHM", "HS256")
    ACCESS_TOKEN_EXPIRE_MINUTES: int = os.environ.get("ACCESS_TOKEN_EXPIRE_MINUTES", 30)
    ADMIN_USER: str = os.environ.get("ADMIN_USER", "admin")
    ADMIN_EMAIL: str = os.environ.get("ADMIN_EMAIL", "admin@localhost.com")
    ADMIN_DEFAULT_PASSWORD: str = os.environ.get("ADMIN_DEFAULT_PASSWORD", "N8Lwcs4G7Vbmkp5t5g")
    USE_MIDDLEWARE: bool = os.environ.get("USE_MIDDLEWARE", True)
    SAVE_PATH: str = os.environ.get("SAVE_PATH", "../files") 
    SMTP_TLS: bool = os.environ.get("SMTP_TLS", True)
    SMTP_SSL: bool = os.environ.get("SMTP_SSL", False)
    SMTP_PORT: int = os.environ.get("SMTP_PORT", 587)
    SMTP_HOST: str = os.environ.get("SMTP_HOST", "localhost")
    SMTP_USER: str = os.environ.get("SMTP_USER", "")
    SMTP_PASSWORD: str = os.environ.get("SMTP_PASSWORD", "")
    EMAILS_FROM_EMAIL: str = os.environ.get("EMAILS_FROM_EMAIL", "")
    EMAILS_FROM_NAME: str = os.environ.get("EMAILS_FROM_NAME", "")
