"""
Instrument configurations for trading
"""
from typing import Dict, Any

# Instrument specifications
INSTRUMENTS: Dict[str, Dict[str, Any]] = {
    "DE40": {
        "name": "DAX 40",
        "description": "German Stock Index",
        "type": "INDEX",
        "currency": "EUR",
        "pip_value": 1.0,
        "contract_size": 1,
        "min_volume": 0.01,
        "max_volume": 100,
        "trading_hours": "08:00-22:00 CET",
        "spread_typical": 1.0,
        "margin_requirement": 0.5,  # 0.5% = 1:200 leverage
        "volatility_profile": "medium-high"
    },
    "US500": {
        "name": "S&P 500",
        "description": "US Large Cap Index",
        "type": "INDEX",
        "currency": "USD",
        "pip_value": 0.01,
        "contract_size": 1,
        "min_volume": 0.01,
        "max_volume": 100,
        "trading_hours": "00:00-23:00 UTC",
        "spread_typical": 0.5,
        "margin_requirement": 0.5,
        "volatility_profile": "medium"
    },
    "US30": {
        "name": "Dow Jones 30",
        "description": "US Blue Chip Index",
        "type": "INDEX",
        "currency": "USD",
        "pip_value": 1.0,
        "contract_size": 1,
        "min_volume": 0.01,
        "max_volume": 100,
        "trading_hours": "00:00-23:00 UTC",
        "spread_typical": 2.0,
        "margin_requirement": 0.5,
        "volatility_profile": "medium"
    },
    "EURUSD": {
        "name": "EUR/USD",
        "description": "Euro vs US Dollar",
        "type": "FOREX",
        "currency": "USD",
        "pip_value": 0.0001,
        "contract_size": 100000,
        "min_volume": 0.01,
        "max_volume": 100,
        "trading_hours": "24/5",
        "spread_typical": 0.0001,
        "margin_requirement": 3.33,  # 1:30 leverage
        "volatility_profile": "low-medium"
    },
    "GBPUSD": {
        "name": "GBP/USD",
        "description": "British Pound vs US Dollar",
        "type": "FOREX",
        "currency": "USD",
        "pip_value": 0.0001,
        "contract_size": 100000,
        "min_volume": 0.01,
        "max_volume": 100,
        "trading_hours": "24/5",
        "spread_typical": 0.00015,
        "margin_requirement": 3.33,
        "volatility_profile": "medium"
    },
    "XAUUSD": {
        "name": "Gold",
        "description": "Gold vs US Dollar",
        "type": "COMMODITY",
        "currency": "USD",
        "pip_value": 0.01,
        "contract_size": 100,
        "min_volume": 0.01,
        "max_volume": 50,
        "trading_hours": "00:00-23:00 UTC",
        "spread_typical": 0.30,
        "margin_requirement": 5.0,  # 1:20 leverage
        "volatility_profile": "medium-high"
    }
}


def get_instrument_config(symbol: str) -> Dict[str, Any]:
    """Get configuration for a specific instrument."""
    return INSTRUMENTS.get(symbol, {})


def get_all_instruments() -> list:
    """Get list of all available instruments."""
    return list(INSTRUMENTS.keys())


def get_instruments_by_type(instrument_type: str) -> list:
    """Get instruments filtered by type."""
    return [
        symbol for symbol, config in INSTRUMENTS.items()
        if config.get("type") == instrument_type
    ]
