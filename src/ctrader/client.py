"""
cTrader Client - Integration with cTrader Open API
"""
import asyncio
from typing import Any, Dict, List, Optional, Callable
from datetime import datetime, timezone
from loguru import logger

try:
    from ctrader_open_api import Client, TcpProtocol, EndPoints
    from ctrader_open_api.messages import commands
    HAS_CTRADER = True
except ImportError:
    HAS_CTRADER = False
    logger.warning("ctrader-open-api not installed")


class CTraderClient:
    """
    Client for cTrader Open API.
    Handles authentication, market data, and order execution.
    """

    def __init__(
        self,
        client_id: str,
        client_secret: str,
        access_token: str,
        account_id: str,
        host: str = "demo.ctraderapi.com",
        port: int = 5035
    ):
        self.client_id = client_id
        self.client_secret = client_secret
        self.access_token = access_token
        self.account_id = account_id
        self.host = host
        self.port = port

        self.client = None
        self.is_connected = False
        self.symbols: Dict[str, Any] = {}
        self.positions: Dict[str, Any] = {}

    async def connect(self) -> bool:
        """Connect to cTrader API."""
        if not HAS_CTRADER:
            logger.error("ctrader-open-api not available")
            return False

        try:
            self.client = Client(
                EndPoints.PROTOBUF_LIVE_HOST if "live" in self.host else EndPoints.PROTOBUF_DEMO_HOST,
                self.port,
                TcpProtocol
            )

            # Connect
            await self.client.connect()

            # Authenticate
            auth_response = await self._authenticate()
            if not auth_response:
                return False

            self.is_connected = True
            logger.info("Connected to cTrader API")

            # Load symbols
            await self._load_symbols()

            return True

        except Exception as e:
            logger.error(f"Connection failed: {e}")
            return False

    async def _authenticate(self) -> bool:
        """Authenticate with cTrader."""
        try:
            # Application auth
            app_auth = commands.ProtoOAApplicationAuthReq()
            app_auth.clientId = self.client_id
            app_auth.clientSecret = self.client_secret

            response = await self.client.send(app_auth)
            if not response:
                logger.error("Application auth failed")
                return False

            # Account auth
            account_auth = commands.ProtoOAAccountAuthReq()
            account_auth.ctidTraderAccountId = int(self.account_id)
            account_auth.accessToken = self.access_token

            response = await self.client.send(account_auth)
            if not response:
                logger.error("Account auth failed")
                return False

            return True

        except Exception as e:
            logger.error(f"Authentication error: {e}")
            return False

    async def _load_symbols(self) -> None:
        """Load available trading symbols."""
        try:
            symbols_req = commands.ProtoOASymbolsListReq()
            symbols_req.ctidTraderAccountId = int(self.account_id)

            response = await self.client.send(symbols_req)
            if response and hasattr(response, 'symbol'):
                for symbol in response.symbol:
                    self.symbols[symbol.symbolName] = {
                        'id': symbol.symbolId,
                        'name': symbol.symbolName,
                        'digits': getattr(symbol, 'digits', 2)
                    }

            logger.info(f"Loaded {len(self.symbols)} symbols")

        except Exception as e:
            logger.error(f"Failed to load symbols: {e}")

    async def disconnect(self) -> None:
        """Disconnect from cTrader."""
        if self.client:
            await self.client.disconnect()
            self.is_connected = False
            logger.info("Disconnected from cTrader")

    async def get_quote(self, symbol: str) -> Dict[str, float]:
        """Get current bid/ask for a symbol."""
        if not self.is_connected:
            return {"bid": 0, "ask": 0, "error": "Not connected"}

        try:
            symbol_info = self.symbols.get(symbol)
            if not symbol_info:
                return {"bid": 0, "ask": 0, "error": f"Symbol not found: {symbol}"}

            # Subscribe to spot prices
            spot_req = commands.ProtoOASubscribeSpotsReq()
            spot_req.ctidTraderAccountId = int(self.account_id)
            spot_req.symbolId.append(symbol_info['id'])

            response = await self.client.send(spot_req)

            # Get latest quote from response
            if response:
                return {
                    "bid": getattr(response, 'bid', 0) / 100000,
                    "ask": getattr(response, 'ask', 0) / 100000,
                    "timestamp": datetime.now(timezone.utc).isoformat()
                }

            return {"bid": 0, "ask": 0, "error": "No response"}

        except Exception as e:
            logger.error(f"Get quote error: {e}")
            return {"bid": 0, "ask": 0, "error": str(e)}

    async def get_historical_data(
        self,
        instrument: str,
        timeframe: str = "H1",
        periods: int = 100
    ) -> List[Dict[str, Any]]:
        """Get historical OHLCV data."""
        if not self.is_connected:
            return []

        # Convert timeframe to cTrader period
        timeframe_map = {
            "M1": 1, "M5": 2, "M15": 3, "M30": 4,
            "H1": 5, "H4": 6, "D1": 7, "W1": 8
        }

        period_type = timeframe_map.get(timeframe, 5)
        symbol_info = self.symbols.get(instrument)

        if not symbol_info:
            logger.error(f"Symbol not found: {instrument}")
            return []

        try:
            trendbars_req = commands.ProtoOAGetTrendbarsReq()
            trendbars_req.ctidTraderAccountId = int(self.account_id)
            trendbars_req.symbolId = symbol_info['id']
            trendbars_req.period = period_type
            trendbars_req.count = periods

            response = await self.client.send(trendbars_req)

            if response and hasattr(response, 'trendbar'):
                data = []
                for bar in response.trendbar:
                    data.append({
                        'timestamp': bar.utcTimestampInMinutes * 60000,
                        'open': bar.open / 100000,
                        'high': bar.high / 100000,
                        'low': bar.low / 100000,
                        'close': bar.close / 100000,
                        'volume': bar.volume
                    })
                return data

            return []

        except Exception as e:
            logger.error(f"Historical data error: {e}")
            return []

    async def get_account(self) -> Dict[str, Any]:
        """Get account information."""
        if not self.is_connected:
            return {"error": "Not connected"}

        try:
            trader_req = commands.ProtoOATraderReq()
            trader_req.ctidTraderAccountId = int(self.account_id)

            response = await self.client.send(trader_req)

            if response and hasattr(response, 'trader'):
                trader = response.trader
                return {
                    "balance": trader.balance / 100,
                    "equity": getattr(trader, 'equity', trader.balance) / 100,
                    "margin_used": getattr(trader, 'usedMargin', 0) / 100,
                    "currency": getattr(trader, 'depositAssetId', 'EUR')
                }

            return {"error": "No response"}

        except Exception as e:
            logger.error(f"Get account error: {e}")
            return {"error": str(e)}

    async def place_order(
        self,
        symbol: str,
        direction: str,
        volume: float,
        order_type: str = "MARKET",
        stop_loss: Optional[float] = None,
        take_profit: Optional[float] = None
    ) -> Dict[str, Any]:
        """Place a trading order."""
        if not self.is_connected:
            return {"success": False, "error": "Not connected"}

        symbol_info = self.symbols.get(symbol)
        if not symbol_info:
            return {"success": False, "error": f"Symbol not found: {symbol}"}

        try:
            # Create order request
            order_req = commands.ProtoOANewOrderReq()
            order_req.ctidTraderAccountId = int(self.account_id)
            order_req.symbolId = symbol_info['id']
            order_req.volume = int(volume * 100)  # Convert to cents
            order_req.tradeSide = 1 if direction == "BUY" else 2

            if order_type == "MARKET":
                order_req.orderType = 1
            elif order_type == "LIMIT":
                order_req.orderType = 2
            elif order_type == "STOP":
                order_req.orderType = 3

            if stop_loss:
                order_req.stopLoss = int(stop_loss * 100000)

            if take_profit:
                order_req.takeProfit = int(take_profit * 100000)

            response = await self.client.send(order_req)

            if response:
                return {
                    "success": True,
                    "order_id": getattr(response, 'orderId', None),
                    "position_id": getattr(response, 'positionId', None)
                }

            return {"success": False, "error": "No response"}

        except Exception as e:
            logger.error(f"Place order error: {e}")
            return {"success": False, "error": str(e)}

    async def close_position(self, position_id: str) -> Dict[str, Any]:
        """Close a position."""
        if not self.is_connected:
            return {"success": False, "error": "Not connected"}

        try:
            close_req = commands.ProtoOAClosePositionReq()
            close_req.ctidTraderAccountId = int(self.account_id)
            close_req.positionId = int(position_id)

            response = await self.client.send(close_req)

            if response:
                return {
                    "success": True,
                    "position_id": position_id
                }

            return {"success": False, "error": "No response"}

        except Exception as e:
            logger.error(f"Close position error: {e}")
            return {"success": False, "error": str(e)}

    async def get_positions(self) -> List[Dict[str, Any]]:
        """Get open positions."""
        if not self.is_connected:
            return []

        try:
            reconcile_req = commands.ProtoOAReconcileReq()
            reconcile_req.ctidTraderAccountId = int(self.account_id)

            response = await self.client.send(reconcile_req)

            if response and hasattr(response, 'position'):
                positions = []
                for pos in response.position:
                    positions.append({
                        "id": pos.positionId,
                        "symbol_id": pos.symbolId,
                        "direction": "BUY" if pos.tradeSide == 1 else "SELL",
                        "volume": pos.volume / 100,
                        "entry_price": pos.entryPrice / 100000,
                        "pnl": getattr(pos, 'swap', 0) / 100
                    })
                return positions

            return []

        except Exception as e:
            logger.error(f"Get positions error: {e}")
            return []

    async def subscribe_prices(
        self,
        symbols: List[str],
        callback: Callable
    ) -> None:
        """Subscribe to real-time price updates."""
        if not self.is_connected:
            return

        try:
            symbol_ids = []
            for symbol in symbols:
                if symbol in self.symbols:
                    symbol_ids.append(self.symbols[symbol]['id'])

            if not symbol_ids:
                return

            spot_req = commands.ProtoOASubscribeSpotsReq()
            spot_req.ctidTraderAccountId = int(self.account_id)
            for sid in symbol_ids:
                spot_req.symbolId.append(sid)

            await self.client.send(spot_req)

            # Handle incoming price updates
            # This would typically use an event handler
            logger.info(f"Subscribed to prices for {len(symbol_ids)} symbols")

        except Exception as e:
            logger.error(f"Subscribe prices error: {e}")
