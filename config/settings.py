"""
TradingAgents Configuration Settings
"""
from pydantic_settings import BaseSettings
from pydantic import Field
from typing import List
import os
from dotenv import load_dotenv

load_dotenv()


class LLMSettings(BaseSettings):
    """LLM API Configuration"""
    anthropic_api_key: str = Field(default="", env="ANTHROPIC_API_KEY")
    openai_api_key: str = Field(default="", env="OPENAI_API_KEY")
    default_model: str = "claude-sonnet-4-6"
    max_tokens: int = 4096
    temperature: float = 0.7


class CTraderSettings(BaseSettings):
    """cTrader API Configuration"""
    client_id: str = Field(default="", env="CTRADER_CLIENT_ID")
    client_secret: str = Field(default="", env="CTRADER_CLIENT_SECRET")
    access_token: str = Field(default="", env="CTRADER_ACCESS_TOKEN")
    account_id: str = Field(default="", env="CTRADER_ACCOUNT_ID")
    host: str = Field(default="demo.ctraderapi.com", env="CTRADER_HOST")
    port: int = Field(default=5035, env="CTRADER_PORT")


class TradingSettings(BaseSettings):
    """Trading Parameters"""
    default_risk_percent: float = Field(default=1.0, env="DEFAULT_RISK_PERCENT")
    max_positions: int = Field(default=5, env="MAX_POSITIONS")
    default_leverage: int = Field(default=30, env="DEFAULT_LEVERAGE")

    # Supported instruments
    instruments: List[str] = [
        "DE40",      # DAX 40
        "US500",     # S&P 500
        "US30",      # Dow Jones
        "EURUSD",    # EUR/USD
        "GBPUSD",    # GBP/USD
        "XAUUSD",    # Gold
    ]

    # Timeframes for analysis
    timeframes: List[str] = ["M1", "M5", "M15", "H1", "H4", "D1"]

    # Risk management
    max_daily_loss_percent: float = 5.0
    max_drawdown_percent: float = 10.0
    stop_loss_atr_multiplier: float = 2.0
    take_profit_atr_multiplier: float = 3.0


class LoggingSettings(BaseSettings):
    """Logging Configuration"""
    level: str = Field(default="INFO", env="LOG_LEVEL")
    format: str = "{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}"
    rotation: str = "10 MB"
    retention: str = "7 days"


class Settings(BaseSettings):
    """Main Application Settings"""
    llm: LLMSettings = LLMSettings()
    ctrader: CTraderSettings = CTraderSettings()
    trading: TradingSettings = TradingSettings()
    logging: LoggingSettings = LoggingSettings()

    # Application settings
    app_name: str = "TradingAgents"
    version: str = "0.1.0"
    debug: bool = False


# Global settings instance
settings = Settings()
