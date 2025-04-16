import json
from typing import Any, List, Dict, Tuple, Optional
import math
import numpy as np

Symbol = str
Product = str
Position = int
Timestamp = int


class Listing:
    def __init__(self, symbol: Symbol, product: Product, denomination: Product):
        self.symbol = symbol
        self.product = product
        self.denomination = denomination


class Order:
    def __init__(self, symbol: Symbol, price: float, quantity: int):
        self.symbol = symbol
        self.price = price
        self.quantity = quantity

    def __str__(self) -> str:
        return f"Order({self.symbol}, {self.price}, {self.quantity})"

    def __repr__(self) -> str:
        return f"Order({self.symbol}, {self.price}, {self.quantity})"


class OrderDepth:
    def __init__(self):
        self.buy_orders: Dict[int, int] = {}
        self.sell_orders: Dict[int, int] = {}


class Trade:

    def __init__(self, symbol: Symbol, price: float, quantity: int, buyer: str = "", seller: str = "",
                 timestamp: int = 0):
        self.symbol = symbol
        self.price = price
        self.quantity = quantity
        self.buyer = buyer
        self.seller = seller
        self.timestamp = timestamp


class ConversionObservation:
    def __init__(self, bidPrice: float, askPrice: float, transportFees: float, exportTariff: float, importTariff: float,
                 sunlight: float, humidity: float):
        self.bidPrice = bidPrice
        self.askPrice = askPrice
        self.transportFees = transportFees
        self.exportTariff = exportTariff
        self.importTariff = importTariff
        self.sunlight = sunlight
        self.humidity = humidity


class Observation:
    def __init__(self, plainValueObservations: Dict[Product, int],
                 conversionObservations: Dict[Product, ConversionObservation]):
        self.plainValueObservations = plainValueObservations
        self.conversionObservations = conversionObservations


class TradingState:
    def __init__(self,
                 timestamp: Timestamp,
                 listings: Dict[Symbol, Listing],
                 order_depths: Dict[Symbol, OrderDepth],
                 own_trades: Dict[Symbol, List[Trade]],
                 market_trades: Dict[Symbol, List[Trade]],
                 position: Dict[Product, Position],
                 observations: Observation,
                 traderData: str = ""):
        self.timestamp = timestamp
        self.listings = listings
        self.order_depths = order_depths
        self.own_trades = own_trades
        self.market_trades = market_trades
        self.position = position
        self.observations = observations
        self.traderData = traderData


class ProsperityEncoder(json.JSONEncoder):
    def default(self, o):
        if isinstance(o, np.integer):
            return int(o)
        elif isinstance(o, np.floating):
            return float(o)
        elif isinstance(o, np.ndarray):
            return o.tolist()

        return json.JSONEncoder.default(self, o)


class Logger:
    def __init__(self) -> None:
        self.logs = ""
        self.max_log_length = 3750

    def print(self, *objects: Any, sep: str = " ", end: str = "\n") -> None:
        self.logs += sep.join(map(str, objects)) + end

    def flush(self, state: TradingState, orders: dict[Symbol, list[Order]], conversions: int, trader_data: str) -> None:

        base_value = [
            self.compress_state(state, ""),
            self.compress_orders(orders),
            conversions,
            "",
            "",
        ]
        try:
            base_json = self.to_json(base_value)
            base_length = len(base_json)
        except Exception as e:

            print(f"Error calculating base log length: {e}", file=sys.stderr)
            base_length = 200

        available_length = self.max_log_length - base_length
        if available_length < 0: available_length = 0
        max_item_length = available_length // 3
        if max_item_length < 0: max_item_length = 0

        final_value = [
            self.compress_state(state, self.truncate(state.traderData, max_item_length)),
            self.compress_orders(orders),
            conversions,
            self.truncate(trader_data, max_item_length),
            self.truncate(self.logs, max_item_length),
        ]

        try:
            print(self.to_json(final_value))
        except Exception as e:

            print(f"Error flushing logs: {e}", file=sys.stderr)
            minimal_output = {
                "error": "Log flushing failed",
                "timestamp": state.timestamp,
                "traderData_len": len(trader_data),
                "logs_len": len(self.logs)
            }
            print(json.dumps(minimal_output))

        self.logs = ""

    def compress_state(self, state: TradingState, trader_data: str) -> list[Any]:

        if not hasattr(state, 'observations') or state.observations is None:

            obs = Observation({}, {})
        else:
            obs = state.observations

            if not hasattr(obs, 'plainValueObservations'): obs.plainValueObservations = {}
            if not hasattr(obs, 'conversionObservations'): obs.conversionObservations = {}

        return [
            state.timestamp,
            trader_data,
            self.compress_listings(state.listings),
            self.compress_order_depths(state.order_depths),
            self.compress_trades(state.own_trades),
            self.compress_trades(state.market_trades),
            state.position,
            self.compress_observations(obs),
        ]

    def compress_listings(self, listings: dict[Symbol, Listing]) -> list[list[Any]]:
        compressed = []
        if listings:
            for listing in listings.values():
                compressed.append([listing.symbol, listing.product, listing.denomination])
        return compressed

    def compress_order_depths(self, order_depths: dict[Symbol, OrderDepth]) -> dict[Symbol, list[Any]]:
        compressed = {}
        if order_depths:
            for symbol, order_depth in order_depths.items():
                buy_orders = order_depth.buy_orders if hasattr(order_depth, 'buy_orders') else {}
                sell_orders = order_depth.sell_orders if hasattr(order_depth, 'sell_orders') else {}
                compressed[symbol] = [buy_orders, sell_orders]
        return compressed

    def compress_trades(self, trades: dict[Symbol, list[Trade]]) -> list[list[Any]]:
        compressed = []
        if trades:
            for arr in trades.values():
                if arr:
                    for trade in arr:
                        buyer = trade.buyer if hasattr(trade, 'buyer') else ""
                        seller = trade.seller if hasattr(trade, 'seller') else ""
                        compressed.append(
                            [
                                trade.symbol,
                                trade.price,
                                trade.quantity,
                                buyer,
                                seller,
                                trade.timestamp,
                            ]
                        )
        return compressed

    def compress_observations(self, observations: Observation) -> list[Any]:

        plain_obs = observations.plainValueObservations if hasattr(observations,
                                                                   'plainValueObservations') and observations.plainValueObservations else {}
        conv_obs_data = observations.conversionObservations if hasattr(observations,
                                                                       'conversionObservations') and observations.conversionObservations else {}

        conversion_observations = {}
        if conv_obs_data:
            for product, observation in conv_obs_data.items():
                conversion_observations[product] = [
                    getattr(observation, 'bidPrice', 0),
                    getattr(observation, 'askPrice', 0),
                    getattr(observation, 'transportFees', 0),
                    getattr(observation, 'exportTariff', 0),
                    getattr(observation, 'importTariff', 0),

                    getattr(observation, 'sunlight', 0),
                    getattr(observation, 'humidity', 0),
                ]

        return [plain_obs, conversion_observations]

    def compress_orders(self, orders: dict[Symbol, list[Order]]) -> list[list[Any]]:
        compressed = []
        if orders:
            for arr in orders.values():
                if arr:
                    for order in arr:
                        compressed.append([order.symbol, order.price, order.quantity])
        return compressed

    def to_json(self, value: Any) -> str:

        return json.dumps(value, cls=ProsperityEncoder, separators=(",", ":"))

    def truncate(self, value: str, max_length: int) -> str:

        if max_length <= 0: return "..."
        if not isinstance(value, str): value = str(value)
        if len(value) <= max_length:
            return value

        if max_length <= 3: return value[:max_length]
        return value[: max_length - 3] + "..."


def norm_cdf(x):
    """Approximation of the cumulative distribution function for the standard normal distribution."""
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def norm_pdf(x):
    """Probability density function for the standard normal distribution."""
    return (1.0 / math.sqrt(2.0 * math.pi)) * math.exp(-0.5 * x * x)


INTEREST_RATE = 0.0
TOTAL_DAYS = 7.0


def calculate_d1_d2(S, K, T, sigma):
    """Calculates d1 and d2 for Black-Scholes"""
    if T <= 0 or sigma <= 0 or S <= 0 or K <= 0: return float('-inf'), float('-inf')
    try:
        d1 = (math.log(S / K) + (INTEREST_RATE + 0.5 * sigma ** 2) * T) / (sigma * math.sqrt(T))
        d2 = d1 - sigma * math.sqrt(T)
        return d1, d2
    except (ValueError, ZeroDivisionError):
        return float('-inf'), float('-inf')


def black_scholes_call_price(S, K, T, sigma):
    """Calculates theoretical call option price using math helpers"""
    if T <= 0 or sigma <= 0: return max(0, S - K)
    d1, d2 = calculate_d1_d2(S, K, T, sigma)
    if d1 == float('-inf'): return max(0, S - K)
    try:
        price = (S * norm_cdf(d1) - K * math.exp(-INTEREST_RATE * T) * norm_cdf(d2))
        return price
    except ValueError:
        return max(0, S - K)


def calculate_intrinsic_value(S, K, T):
    """Calculates the intrinsic value of a call option."""
    if T <= 0 or S <= 0 or K <= 0: return max(0.0, float(S) - float(K))
    try:
        intrinsic = max(0.0, float(S) - float(K) * math.exp(-INTEREST_RATE * T))
        return intrinsic
    except Exception:

        return max(0.0, float(S) - float(K))


def calculate_vega(S, K, T, sigma):
    """Calculates Vega using math helpers"""
    if T <= 0 or sigma <= 0: return 0.0
    d1, _ = calculate_d1_d2(S, K, T, sigma)
    if d1 == float('-inf'): return 0.0
    try:
        vega = S * norm_pdf(d1) * math.sqrt(T)
        return vega / 100
    except ValueError:
        return 0.0


def calculate_delta(S, K, T, sigma):
    """Calculates Delta using math helpers"""
    if T <= 0: return 1.0 if S > K else 0.0
    if sigma <= 0: return 1.0 if S > K else 0.0
    d1, _ = calculate_d1_d2(S, K, T, sigma)
    if d1 == float('-inf'):
        return 1.0 if S > K else 0.0
    delta = norm_cdf(d1)
    return delta


def calculate_implied_volatility(target_price, S, K, T, tolerance=0.0001, max_iterations=50):
    """Calculates Implied Volatility using Newton-Raphson (math helpers)."""
    if T <= 0: return np.nan
    intrinsic_value = max(0, S - K * math.exp(-INTEREST_RATE * T))
    if target_price < intrinsic_value - tolerance: return np.nan
    if abs(target_price - intrinsic_value) < tolerance: return 0.0

    sigma = 0.5
    for _ in range(max_iterations):
        price = black_scholes_call_price(S, K, T, sigma)
        vega = calculate_vega(S, K, T, sigma) * 100
        diff = price - target_price
        if abs(diff) < tolerance: return sigma
        if vega < 1e-8: break
        sigma_change = diff / vega
        sigma = sigma - sigma_change * 0.8
        sigma = max(0.001, min(sigma, 5.0))

    final_price = black_scholes_call_price(S, K, T, sigma)
    if abs(final_price - target_price) < tolerance * 5:
        return sigma
    else:
        return np.nan


class BaseStrategy:
    def __init__(self, logger: Logger):
        self.logger = logger

    def run(self, state: TradingState, **kwargs) -> Dict[Symbol, List[Order]]:
        raise NotImplementedError

    def save_state(self) -> str:
        raise NotImplementedError

    def load_state(self, json_string: str) -> None:
        raise NotImplementedError

    def calculate_mid_price(self, symbol: Symbol, order_depth: OrderDepth) -> float | None:
        """Helper to calculate mid-price, shared across strategies."""
        if not order_depth or not order_depth.buy_orders or not order_depth.sell_orders: return None
        best_bid = max(order_depth.buy_orders.keys())
        best_ask = min(order_depth.sell_orders.keys())

        if best_ask <= best_bid:
            return None
        return (best_bid + best_ask) / 2.0

    def get_fill_price_and_qty(self, order_depth: OrderDepth, quantity: int) -> Optional[Tuple[float, int]]:
        """Helper to calculate VWAP and max fillable quantity."""
        if order_depth is None or quantity == 0: return None
        total_cost = 0
        filled_qty = 0
        target_qty = abs(quantity)
        levels = None

        if quantity > 0:
            levels = sorted(order_depth.sell_orders.items())
        elif quantity < 0:
            levels = sorted(order_depth.buy_orders.items(), reverse=True)

        if not levels: return None

        for price, volume in levels:
            vol = abs(volume)
            qty_at_level = min(target_qty - filled_qty, vol)
            if qty_at_level <= 0: break
            total_cost += qty_at_level * price
            filled_qty += qty_at_level

        if filled_qty == 0: return None

        vwap = total_cost / filled_qty
        actual_filled_sign = int(np.sign(quantity) * filled_qty)
        return vwap, actual_filled_sign


class VoucherStrategy(BaseStrategy):
    def __init__(self, logger: Logger, target_voucher: Symbol, strike: int, window: int, entry_z: float = 1.5,
                 exit_z: float = 0.5, trade_size: int = 50, min_time_premium: float = 0.1):
        super().__init__(logger)
        self.target_voucher = target_voucher
        self.underlying = "VOLCANIC_ROCK"
        self.strike = strike
        self.window = window
        self.entry_z_threshold = entry_z
        self.exit_z_threshold = exit_z
        self.trade_size = trade_size
        self.min_time_premium = min_time_premium

        self.iv_history = []
        self.active_vol_direction = 0
        self.last_delta = 0.0

    def run(self, state: TradingState, underlying_price: float, time_to_expiry: float, **kwargs) -> Dict[
        Symbol, List[Order]]:

        orders: Dict[Symbol, List[Order]] = {}

        if underlying_price is None or underlying_price <= 0 or time_to_expiry <= 0:
            return orders

        voucher_depth = state.order_depths.get(self.target_voucher)
        if not voucher_depth:
            return orders

        voucher_mid_price = self.calculate_mid_price(self.target_voucher, voucher_depth)
        if voucher_mid_price is None:
            return orders

        intrinsic_value = calculate_intrinsic_value(underlying_price, self.strike, time_to_expiry)
        time_premium = voucher_mid_price - intrinsic_value

        current_iv = calculate_implied_volatility(voucher_mid_price, underlying_price, self.strike, time_to_expiry)

        if time_premium >= self.min_time_premium and not np.isnan(current_iv):
            self.iv_history.append(current_iv)
            if len(self.iv_history) > self.window:
                self.iv_history.pop(0)
        elif time_premium < self.min_time_premium or np.isnan(current_iv):
            return orders

        if len(self.iv_history) < self.window:
            return orders

        iv_history_np = np.array(self.iv_history)
        mean_iv = np.mean(iv_history_np)
        std_dev_iv = np.std(iv_history_np)

        if std_dev_iv < 1e-5:
            return orders

        iv_z_score = (current_iv - mean_iv) / std_dev_iv

        voucher_pos = state.position.get(self.target_voucher, 0)
        voucher_limit = 200

        close_position = False
        if self.active_vol_direction == 1 and iv_z_score >= -self.exit_z_threshold:
            close_position = True
        elif self.active_vol_direction == -1 and iv_z_score <= self.exit_z_threshold:
            close_position = True

        if close_position and voucher_pos != 0:
            close_orders = {}

            if voucher_pos != 0:
                close_v_qty = -voucher_pos
                price_qty_v = self.get_fill_price_and_qty(voucher_depth, close_v_qty)
                if price_qty_v:
                    close_price_v = int(round(price_qty_v[0]))
                    close_orders[self.target_voucher] = [Order(self.target_voucher, close_price_v, close_v_qty)]

            if self.target_voucher in close_orders:
                self.active_vol_direction = 0
            return close_orders
        elif close_position and voucher_pos == 0:
            self.active_vol_direction = 0

        elif self.active_vol_direction == 0:
            target_voucher_qty = 0

            if iv_z_score > self.entry_z_threshold:
                potential_v_qty = min(self.trade_size, voucher_limit + voucher_pos)
                if potential_v_qty > 0: target_voucher_qty = -potential_v_qty

            elif iv_z_score < -self.entry_z_threshold:
                potential_v_qty = min(self.trade_size, voucher_limit - voucher_pos)
                if potential_v_qty > 0: target_voucher_qty = potential_v_qty

            if target_voucher_qty != 0:
                price_qty_v = self.get_fill_price_and_qty(voucher_depth, target_voucher_qty)

                if price_qty_v:
                    fill_price_v, fillable_qty_v_signed = price_qty_v
                    fillable_qty_v = abs(fillable_qty_v_signed)

                    final_v_qty_abs = min(abs(target_voucher_qty), fillable_qty_v)
                    final_v_qty_abs = math.floor(final_v_qty_abs)

                    if final_v_qty_abs == 0:
                        return orders

                    final_v_qty = int(final_v_qty_abs * np.sign(target_voucher_qty))

                    if final_v_qty != 0:
                        rounded_price_v = int(round(fill_price_v))
                        orders[self.target_voucher] = [Order(self.target_voucher, rounded_price_v, final_v_qty)]

                        self.active_vol_direction = int(np.sign(final_v_qty))

        return orders

    def save_state(self) -> str:
        return json.dumps({
            "iv_history": self.iv_history,
            "active_vol_direction": self.active_vol_direction,
            "last_delta": self.last_delta
        })

    def load_state(self, json_string: str):

        self.iv_history = []
        self.active_vol_direction = 0
        self.last_delta = 0.0
        if not json_string: return
        try:
            state_dict = json.loads(json_string)

            self.iv_history = state_dict.get("iv_history", [])[-self.window:]
            self.active_vol_direction = state_dict.get("active_vol_direction", 0)
            self.last_delta = state_dict.get("last_delta", 0.0)
        except Exception as e:
            self.logger.print(f"E: Load IV Strat state failed: {e}, resetting.")

            self.iv_history = []
            self.active_vol_direction = 0
            self.last_delta = 0.0


class RockStrategy(BaseStrategy):
    def __init__(self, logger: Logger, symbol: Symbol, window: int, std_dev_multiplier: float, trade_size: int,
                 entry_penetration_factor: float = 0.2):
        super().__init__(logger)
        self.symbol = symbol
        self.window = window
        self.std_dev_multiplier = std_dev_multiplier
        self.trade_size = trade_size
        self.entry_penetration_factor = entry_penetration_factor
        self.price_history = []

        self.current_signal = 0

    def run(self, state: TradingState, **kwargs) -> Tuple[Dict[Symbol, List[Order]], int]:
        """ Executes the Bollinger Band Mean Reversion strategy for the rock.
            Returns: Tuple (orders_dict, signal)
                     signal: 1 (long), -1 (short), 0 (flat)
        """
        orders: Dict[Symbol, List[Order]] = {}

        signal = 0

        current_pos = state.position.get(self.symbol, 0)

        try:

            position_limit = 400
        except Exception:
            position_limit = 0
            self.logger.print(f"W: RockStrat - Could not determine position limit for {self.symbol}")

        current_mid_price = kwargs.get('rock_price', None)
        if current_mid_price is None:
            return orders, self.current_signal

        self.price_history.append(current_mid_price)
        if len(self.price_history) > self.window:
            self.price_history.pop(0)

        if len(self.price_history) < self.window:
            return orders, self.current_signal

        prices_np = np.array(self.price_history)
        sma = np.mean(prices_np)
        std_dev = np.std(prices_np)

        if std_dev < 1e-5:

            if current_pos != 0:
                target_qty = -current_pos

            else:
                target_qty = 0
            signal = 0

        else:
            upper_band = sma + self.std_dev_multiplier * std_dev
            lower_band = sma - self.std_dev_multiplier * std_dev
            penetration = self.entry_penetration_factor * std_dev

            target_qty = 0
            signal = self.current_signal

            if current_pos > 0 and current_mid_price >= upper_band:
                target_qty = -current_pos
                signal = 0

            elif current_pos < 0 and current_mid_price <= lower_band:
                target_qty = -current_pos
                signal = 0

            elif current_pos == 0:
                if current_mid_price <= lower_band - penetration:
                    target_qty = self.trade_size
                    signal = 1

                elif current_mid_price >= upper_band + penetration:
                    target_qty = -self.trade_size
                    signal = -1

                else:

                    signal = 0

            elif current_pos > 0:
                signal = 1
            elif current_pos < 0:
                signal = -1

        self.current_signal = signal

        if target_qty != 0:
            if target_qty > 0:
                potential_buy = position_limit - current_pos
                target_qty = min(target_qty, potential_buy)
            elif target_qty < 0:
                potential_sell = -position_limit - current_pos
                target_qty = max(target_qty, potential_sell)

            if target_qty != 0:
                order_depth = state.order_depths.get(self.symbol)
                if order_depth:
                    price_qty = self.get_fill_price_and_qty(order_depth, target_qty)
                    if price_qty:
                        fill_price, filled_qty = price_qty

                        if filled_qty != 0:
                            rounded_price = int(round(fill_price))
                            orders[self.symbol] = [Order(self.symbol, rounded_price, filled_qty)]

                    else:
                        self.logger.print("W: RockStrat - Could not get fill price/qty.")
                else:
                    self.logger.print("W: RockStrat - No order depth found.")
            else:

                if self.current_signal != 0:
                    self.current_signal = 0

        return orders, self.current_signal

    def save_state(self) -> str:
        return json.dumps({
            "price_history": self.price_history,
            "current_signal": self.current_signal
        })

    def load_state(self, json_string: str):
        self.price_history = []
        self.current_signal = 0
        if not json_string: return
        try:
            state_dict = json.loads(json_string)
            self.price_history = state_dict.get("price_history", [])[-self.window:]
            self.current_signal = state_dict.get("current_signal", 0)
        except Exception as e:
            self.logger.print(f"E: Load Rock Strat state failed: {e}, resetting.")
            self.price_history = []
            self.current_signal = 0


class DirectionalVoucherStrategy(BaseStrategy):
    def __init__(self, logger: Logger, target_voucher: Symbol, strike: int, trade_size: int,
                 underlying_symbol: Symbol = "VOLCANIC_ROCK"):
        super().__init__(logger)
        self.target_voucher = target_voucher
        self.strike = strike
        self.trade_size = trade_size
        self.underlying_symbol = underlying_symbol

    def run(self, state: TradingState, **kwargs) -> Dict[Symbol, List[Order]]:
        """ Trades the voucher based on the signal from the underlying strategy. """
        orders: Dict[Symbol, List[Order]] = {}
        rock_signal = kwargs.get('rock_signal', 0)

        current_pos = state.position.get(self.target_voucher, 0)

        voucher_limit = 200

        target_qty = 0

        if rock_signal == 1:
            if current_pos < self.trade_size:
                target_qty = self.trade_size - current_pos
        elif rock_signal == 0:
            if current_pos != 0:
                target_qty = -current_pos
        elif rock_signal == -1:

            if current_pos > -self.trade_size:
                target_qty = -self.trade_size - current_pos

        if target_qty > 0:
            potential_buy = voucher_limit - current_pos
            target_qty = min(target_qty, potential_buy)
        elif target_qty < 0:
            potential_sell = -voucher_limit - current_pos
            target_qty = max(target_qty, potential_sell)

        if target_qty != 0:
            order_depth = state.order_depths.get(self.target_voucher)
            if order_depth:
                price_qty = self.get_fill_price_and_qty(order_depth, target_qty)
                if price_qty:
                    fill_price, filled_qty = price_qty
                    if filled_qty != 0:
                        rounded_price = int(round(fill_price))
                        orders[self.target_voucher] = [Order(self.target_voucher, rounded_price, filled_qty)]

                else:
                    self.logger.print(f"W: DirVoucher ({self.target_voucher}) - Cannot get fill info.")
            else:
                self.logger.print(f"W: DirVoucher ({self.target_voucher}) - No order depth.")

        return orders

    def save_state(self) -> str:

        return json.dumps({})

    def load_state(self, json_string: str):

        pass


class Trader:
    def __init__(self):
        self.logger = Logger()
        self.strategies: Dict[str, BaseStrategy] = {}
        self.position_limits = {
            "VOLCANIC_ROCK": 400,
            "VOLCANIC_ROCK_VOUCHER_9500": 200,
            "VOLCANIC_ROCK_VOUCHER_9750": 200,
            "VOLCANIC_ROCK_VOUCHER_10000": 200,
            "VOLCANIC_ROCK_VOUCHER_10250": 200,
            "VOLCANIC_ROCK_VOUCHER_10500": 200,
        }

        # Configure Voucher Strategies (DISABLED - Using Directional for 10k now)
        # iv_vouchers = {
        #     "VOLCANIC_ROCK_VOUCHER_10000": 10000, 
        # }
        # default_iv_voucher_params = {"window": 50, "entry_z": 1.5, "exit_z": 0.3, "trade_size": 50, "min_time_premium": 0.1} 
        # iv_voucher_params = {
        #     10000: {"window": 50, "entry_z": 1.5, "exit_z": 0.1, "trade_size": 50, "min_time_premium": 0.1}, # Reverted size to 50
        # }
        # for symbol, strike_price in iv_vouchers.items(): 
        #     params = iv_voucher_params.get(strike_price, default_iv_voucher_params) 
        #     strat_name = f"VoucherIV_MR_{strike_price}"
        #     self.strategies[strat_name] = VoucherStrategy(
        #         logger=self.logger,
        #         target_voucher=symbol,
        #         strike=strike_price,
        #         window=params["window"],
        #         entry_z=params["entry_z"],
        #         exit_z=params["exit_z"],
        #         trade_size=params["trade_size"],
        #         min_time_premium=params["min_time_premium"]
        #     )

        # Configure Rock Strategy
        rock_strat_name = "RockBB"
        self.strategies[rock_strat_name] = RockStrategy(
            logger=self.logger,
            symbol="VOLCANIC_ROCK",
            window=50,
            std_dev_multiplier=2.5,
            trade_size=146,  # kinda aggressive lol
            entry_penetration_factor=0.2
        )

        directional_vouchers = {
            "VOLCANIC_ROCK_VOUCHER_10000": {"strike": 10000, "trade_size": 200},  # Added 10k 
            "VOLCANIC_ROCK_VOUCHER_9500": {"strike": 9500, "trade_size": 200},
            "VOLCANIC_ROCK_VOUCHER_9750": {"strike": 9750, "trade_size": 200},
            "VOLCANIC_ROCK_VOUCHER_10250": {"strike": 10250, "trade_size": 200},
        }
        for symbol, params in directional_vouchers.items():
            strat_name = f"VoucherDir_{params['strike']}"
            self.strategies[strat_name] = DirectionalVoucherStrategy(
                logger=self.logger,
                target_voucher=symbol,
                strike=params["strike"],
                trade_size=params["trade_size"]
            )

        self.round_start_time = 0
        self.day_length = 1_000_000

    def calculate_mid_price(self, symbol: Symbol, state: TradingState) -> float | None:

        order_depth = state.order_depths.get(symbol)
        if not order_depth or not order_depth.buy_orders or not order_depth.sell_orders: return None
        best_bid = max(order_depth.buy_orders.keys())
        best_ask = min(order_depth.sell_orders.keys())
        if best_ask <= best_bid: return None
        return (best_bid + best_ask) / 2.0

    def calculate_time_to_expiry(self, current_timestamp: int) -> float:
        current_round = math.ceil(current_timestamp / self.day_length)
        if current_timestamp == 0: current_round = 1
        start_day_of_round = current_round
        days_left_at_round_start = TOTAL_DAYS - (start_day_of_round - 1)
        time_passed_in_day = (current_timestamp % self.day_length) / self.day_length
        days_remaining = days_left_at_round_start - time_passed_in_day
        T_years = days_remaining / 365.0
        return max(1e-9, T_years)

    def run(self, state: TradingState) -> Tuple[Dict[Symbol, List[Order]], int, str]:
        all_strategy_states = {}
        trader_data_from_state = state.traderData if state.traderData else "{}"
        try:
            all_strategy_states = json.loads(trader_data_from_state)
        except json.JSONDecodeError as e:
            self.logger.print(f"E: Load traderData JSON failed: {e}. Raw: '{trader_data_from_state[:100]}...'");
            all_strategy_states = {}
        except Exception as e:
            self.logger.print(f"E: Load traderData failed unexpectedly: {e}")
            all_strategy_states = {}

        rock_price = self.calculate_mid_price("VOLCANIC_ROCK", state)
        tte = 0.0
        if rock_price is not None:
            tte = self.calculate_time_to_expiry(state.timestamp)

        result: Dict[Symbol, List[Order]] = {}
        updated_all_strategy_states = {}
        rock_signal = 0

        rock_strat_name = "RockBB"
        if rock_strat_name in self.strategies:
            rock_strategy = self.strategies[rock_strat_name]
            rock_strategy.load_state(all_strategy_states.get(rock_strat_name, ""))
            try:
                rock_kwargs = {'state': state, 'rock_price': rock_price}
                orders_dict, rock_signal = rock_strategy.run(**rock_kwargs)

                for symbol, order_list in orders_dict.items():
                    if symbol not in result: result[symbol] = []
                    result[symbol].extend(order_list)
                updated_all_strategy_states[rock_strat_name] = rock_strategy.save_state()
            except Exception as e:
                self.logger.print(f"!!! EXCEPTION in {rock_strat_name}: {e}")
                import sys, traceback;
                traceback.print_exc(file=sys.stderr)
                self.logger.print(traceback.format_exc())
        else:
            self.logger.print("W: RockBB strategy not found!")

        for name, strategy in self.strategies.items():
            if name == rock_strat_name: continue

            strategy.load_state(all_strategy_states.get(name, ""))

            strategy_kwargs = {
                'state': state,
                'rock_price': rock_price,
                'underlying_price': rock_price,
                'time_to_expiry': tte,
                'rock_signal': rock_signal
            }

            try:

                orders_dict = strategy.run(**strategy_kwargs)

                for symbol, order_list in orders_dict.items():
                    if symbol not in result: result[symbol] = []
                    result[symbol].extend(order_list)

            except Exception as e:
                self.logger.print(f"!!! EXCEPTION in {name}: {e}")
                import sys, traceback;
                traceback.print_exc(file=sys.stderr)
                self.logger.print(traceback.format_exc())

            updated_all_strategy_states[name] = strategy.save_state()

        trader_data_to_save = json.dumps(updated_all_strategy_states)
        conversions = 0

        self.logger.flush(state, result, conversions, trader_data_to_save)

        return result, conversions, trader_data_to_save