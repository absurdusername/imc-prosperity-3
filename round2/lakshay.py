from abc import abstractmethod
from datamodel import Listing, Observation, Order, OrderDepth, ProsperityEncoder, Symbol, Trade, TradingState
from typing import Any, List
import json

class Logger:
    def __init__(self) -> None:
        self.logs = ""
        self.max_log_length = 3750

    def print(self, *objects: Any, sep: str = " ", end: str = "\n") -> None:
        self.logs += sep.join(map(str, objects)) + end

    def flush(self, state: TradingState, orders: dict[Symbol, list[Order]], conversions: int, trader_data: str) -> None:
        base_length = len(
            self.to_json(
                [
                    self.compress_state(state, ""),
                    self.compress_orders(orders),
                    conversions,
                    "",
                    "",
                ]
            )
        )

        max_item_length = (self.max_log_length - base_length) // 3

        print(
            self.to_json(
                [
                    self.compress_state(state, self.truncate(state.traderData, max_item_length)),
                    self.compress_orders(orders),
                    conversions,
                    self.truncate(trader_data, max_item_length),
                    self.truncate(self.logs, max_item_length),
                ]
            )
        )

        self.logs = ""

    def compress_state(self, state: TradingState, trader_data: str) -> list[Any]:
        return [
            state.timestamp,
            trader_data,
            self.compress_listings(state.listings),
            self.compress_order_depths(state.order_depths),
            self.compress_trades(state.own_trades),
            self.compress_trades(state.market_trades),
            state.position,
            self.compress_observations(state.observations),
        ]

    def compress_listings(self, listings: dict[Symbol, Listing]) -> list[list[Any]]:
        compressed = []
        for listing in listings.values():
            compressed.append([listing.symbol, listing.product, listing.denomination])

        return compressed

    def compress_order_depths(self, order_depths: dict[Symbol, OrderDepth]) -> dict[Symbol, list[Any]]:
        compressed = {}
        for symbol, order_depth in order_depths.items():
            compressed[symbol] = [order_depth.buy_orders, order_depth.sell_orders]

        return compressed

    def compress_trades(self, trades: dict[Symbol, list[Trade]]) -> list[list[Any]]:
        compressed = []
        for arr in trades.values():
            for trade in arr:
                compressed.append(
                    [
                        trade.symbol,
                        trade.price,
                        trade.quantity,
                        trade.buyer,
                        trade.seller,
                        trade.timestamp,
                    ]
                )

        return compressed

    def compress_observations(self, observations: Observation) -> list[Any]:
        conversion_observations = {}
        for product, observation in observations.conversionObservations.items():
            conversion_observations[product] = [
                observation.bidPrice,
                observation.askPrice,
                observation.transportFees,
                observation.exportTariff,
                observation.importTariff,
                observation.sugarPrice,
                observation.sunlightIndex,
            ]

        return [observations.plainValueObservations, conversion_observations]

    def compress_orders(self, orders: dict[Symbol, list[Order]]) -> list[list[Any]]:
        compressed = []
        for arr in orders.values():
            for order in arr:
                compressed.append([order.symbol, order.price, order.quantity])

        return compressed

    def to_json(self, value: Any) -> str:
        return json.dumps(value, cls=ProsperityEncoder, separators=(",", ":"))

    def truncate(self, value: str, max_length: int) -> str:
        if len(value) <= max_length:
            return value

        return value[: max_length - 3] + "..."

class SimpleStrategy:
    def __init__(self, symbol: Symbol, limit: int, window_size: int=10):
        self.symbol = symbol
        self.limit = limit
        self.window_size = window_size

        self.window = []
        self.outstanding_buy_orders = []
        self.outstanding_sell_orders = []
        self.current_position = None
        self.last_true_value = None
        self.market_info = {}  

    def save(self):
        data = [self.window]
        if self.last_true_value is not None:
            data.append(self.last_true_value)
        return data

    def load(self, data):
        if isinstance(data, list):
            self.window = data[0]
            if len(data) > 1:
                self.last_true_value = data[1]
        else:

            self.window = data

    @abstractmethod
    def get_true_value(self) -> int:
        pass

    def read(self, state: TradingState):
        order_depth: OrderDepth = state.order_depths[self.symbol]

        self.outstanding_buy_orders = sorted(order_depth.buy_orders.items(), reverse=True)
        self.outstanding_sell_orders = sorted(order_depth.sell_orders.items())
        self.current_position = state.position.get(self.symbol, 0)
        self.window.append(abs(self.current_position) == self.limit)

        if len(self.window) > self.window_size:
            self.window.pop(0)

    def adjust_order_size(self, quantity, side):
        """Adjust order size based on market conditions. 
        side: 1 for buy, -1 for sell"""
        if not self.market_info:
            return quantity

        if self.symbol == "KELP" and "resin_mid" in self.market_info and self.last_true_value:
            resin_mid = self.market_info.get("resin_mid")

            if abs(resin_mid - 10000) > 10:

                if (resin_mid > 10000 and side > 0) or (resin_mid < 10000 and side < 0):
                    return int(quantity * 0.75)  
                else:

                    return int(quantity * 1.15)  

        liquidity_factor = 1.0
        if self.symbol == "RAINFOREST_RESIN" and "resin_depth" in self.market_info:

            depth = self.market_info.get("resin_depth")
            if depth and depth.buy_orders and depth.sell_orders:
                buy_depth = sum(vol for vol in depth.buy_orders.values())
                sell_depth = sum(-vol for vol in depth.sell_orders.values())
                if buy_depth > sell_depth * 2 and side < 0:  
                    liquidity_factor = 0.85  
                elif sell_depth > buy_depth * 2 and side > 0:  
                    liquidity_factor = 0.85  

        position_ratio = abs(self.current_position) / self.limit if self.limit > 0 else 0
        if position_ratio > 0.9:  
            return int(quantity * 0.7 * liquidity_factor)  
        elif position_ratio > 0.7:  
            return int(quantity * 0.9 * liquidity_factor)  

        return int(quantity * liquidity_factor)  

    def get_orders(self) -> list[Order]:
        orders = []

        buy_limit = self.limit - self.current_position
        sell_limit = self.limit + self.current_position

        true_value = self.get_true_value()

        self.last_true_value = true_value

        max_buy_price = true_value - 1 if self.current_position > self.limit * 0.5 else true_value
        min_sell_price = true_value + 1 if self.current_position < self.limit * -0.5 else true_value

        for price, volume in self.outstanding_sell_orders:
            if buy_limit > 0 and price <= max_buy_price:
                quantity = min(buy_limit, -volume)

                adjusted_quantity = self.adjust_order_size(quantity, 1)  
                if adjusted_quantity > 0:
                    orders.append(Order(self.symbol, price, adjusted_quantity))
                    buy_limit -= adjusted_quantity

        if buy_limit > 0:
            popular_buy_price = max(self.outstanding_buy_orders, key=lambda tup: tup[1])[0] if self.outstanding_buy_orders else (true_value - 2)

            position_ratio = self.current_position / self.limit if self.limit > 0 else 0
            if position_ratio > 0.85:
                price = min(max_buy_price - 1, popular_buy_price)
            else:
                price = min(max_buy_price, popular_buy_price + 1)

            adjusted_quantity = self.adjust_order_size(buy_limit, 1)  
            if adjusted_quantity > 0:
                orders.append(Order(self.symbol, price, adjusted_quantity))

        for price, volume in self.outstanding_buy_orders:
            if sell_limit > 0 and price >= min_sell_price:
                quantity = min(sell_limit, volume)

                adjusted_quantity = self.adjust_order_size(quantity, -1)  
                if adjusted_quantity > 0:
                    orders.append(Order(self.symbol, price, -adjusted_quantity))
                    sell_limit -= adjusted_quantity

        if sell_limit > 0:
            popular_sell_price = min(self.outstanding_sell_orders, key=lambda tup: tup[1])[0] if self.outstanding_sell_orders else (true_value + 2)

            position_ratio = self.current_position / self.limit if self.limit > 0 else 0
            if position_ratio < -0.85:
                price = max(min_sell_price + 1, popular_sell_price)
            else:
                price = max(min_sell_price, popular_sell_price - 1)

            adjusted_quantity = self.adjust_order_size(sell_limit, -1)  
            if adjusted_quantity > 0:
                orders.append(Order(self.symbol, price, -adjusted_quantity))

        return orders

class RainforestResinStrategy(SimpleStrategy):
    def __init__(self):
        super().__init__("RAINFOREST_RESIN", 50)

    def get_true_value(self) -> int:
        return 10_000

class KelpStrategy(SimpleStrategy):
    def __init__(self):
        super().__init__("KELP", 50)
        self.last_midpoint = None
        self.price_history = []
        self.max_history_len = 5
        self.volatility_history = []
        self.max_volatility_len = 10

    def get_true_value(self):
        if not self.outstanding_buy_orders or not self.outstanding_sell_orders:
            midpoint = self.last_midpoint if self.last_midpoint else 2030
            return midpoint

        popular_buy_price = max(self.outstanding_buy_orders, key=lambda tup: tup[1])[0]
        popular_sell_price = min(self.outstanding_sell_orders, key=lambda tup: tup[1])[0]
        midpoint = (popular_buy_price + popular_sell_price) // 2

        total_buy_volume = sum(volume for _, volume in self.outstanding_buy_orders)
        total_sell_volume = sum(-volume for _, volume in self.outstanding_sell_orders)
        buy_levels = sum(1 for _, vol in self.outstanding_buy_orders if vol >= 5)
        sell_levels = sum(1 for _, vol in self.outstanding_sell_orders if abs(vol) >= 5)

        if total_buy_volume > total_sell_volume * 1.7 and buy_levels > sell_levels:
            midpoint += 1
        elif total_sell_volume > total_buy_volume * 1.7 and sell_levels > buy_levels:
            midpoint -= 1

        self.price_history.append(midpoint)
        if len(self.price_history) > self.max_history_len:
            self.price_history.pop(0)

        if len(self.price_history) >= 2:
            volatility = abs(self.price_history[-1] - self.price_history[-2])
            self.volatility_history.append(volatility)
            if len(self.volatility_history) > self.max_volatility_len:
                self.volatility_history.pop(0)

        if len(self.price_history) >= 3:
            if all(self.price_history[i] < self.price_history[i+1] for i in range(len(self.price_history)-1)):
                midpoint += 1
            elif all(self.price_history[i] > self.price_history[i+1] for i in range(len(self.price_history)-1)):
                midpoint -= 1

        self.last_midpoint = midpoint
        return midpoint

    def save(self):
        data = super().save()
        data.append(self.last_midpoint)
        data.append(self.price_history)
        data.append(self.volatility_history)
        return data

    def load(self, data):
        if isinstance(data, list) and len(data) >= 4:
            super().load(data[:-3])
            self.last_midpoint = data[-3]
            self.price_history = data[-2]
            self.volatility_history = data[-1]
        elif isinstance(data, list) and len(data) >= 3:
            super().load(data[:-2])
            self.last_midpoint = data[-2]
            self.price_history = data[-1]
            self.volatility_history = []
        elif isinstance(data, list) and len(data) >= 2:
            super().load(data[:-1])
            self.last_midpoint = data[-1]
            self.price_history = []
            self.volatility_history = []
        else:
            super().load(data)
            self.last_midpoint = None
            self.price_history = []
            self.volatility_history = []

    def adjust_order_size(self, quantity, side):
        """Enhanced order size adjustment that considers price trends and volatility"""
        adjusted_quantity = super().adjust_order_size(quantity, side)

        avg_volatility = sum(self.volatility_history) / max(len(self.volatility_history), 1)
        if avg_volatility > 1.5:
            adjusted_quantity = int(adjusted_quantity * 0.9)

        if len(self.price_history) >= 3:
            price_changes = [self.price_history[i+1] - self.price_history[i]
                            for i in range(len(self.price_history)-1)]
            avg_movement = sum(price_changes) / len(price_changes)

            if abs(avg_movement) >= 1:
                if avg_movement > 0 and side > 0:
                    adjusted_quantity = int(adjusted_quantity * 1.1)
                elif avg_movement > 0 and side < 0:
                    adjusted_quantity = int(adjusted_quantity * 0.9)
                elif avg_movement < 0 and side < 0:
                    adjusted_quantity = int(adjusted_quantity * 1.1)
                elif avg_movement < 0 and side > 0:
                    adjusted_quantity = int(adjusted_quantity * 0.9)

        return max(1, adjusted_quantity)

    def get_orders(self) -> list[Order]:
        orders = []
        buy_limit = self.limit - self.current_position
        sell_limit = self.limit + self.current_position

        true_value = self.get_true_value()
        self.last_true_value = true_value

        avg_volatility = sum(self.volatility_history) / max(len(self.volatility_history), 1)
        buy_offset = -1
        sell_offset = 1

        if avg_volatility > 1.5: 
            buy_offset = -2
            sell_offset = 2

        max_buy_price = true_value + buy_offset
        min_sell_price = true_value + sell_offset

        position_ratio = self.current_position / self.limit if self.limit > 0 else 0
        if position_ratio > 0.8: max_buy_price = true_value + buy_offset - 1 
        elif position_ratio < -0.8: min_sell_price = true_value + sell_offset + 1 

        for price, volume in self.outstanding_sell_orders:
            if buy_limit > 0 and price <= max_buy_price:
                quantity = min(buy_limit, -volume)
                quantity = self.adjust_order_size(quantity, 1)
                if quantity > 0:
                    orders.append(Order(self.symbol, price, quantity))
                    buy_limit -= quantity

        if buy_limit > 0:
            if self.outstanding_buy_orders:
                popular_buy_price = max(self.outstanding_buy_orders, key=lambda x: x[1])[0]
            else:
                popular_buy_price = true_value - 2

            price = min(max_buy_price, popular_buy_price + 1) 
            quantity = self.adjust_order_size(buy_limit, 1)
            if quantity > 0:
                orders.append(Order(self.symbol, price, quantity))

        for price, volume in self.outstanding_buy_orders:
            if sell_limit > 0 and price >= min_sell_price:
                quantity = min(sell_limit, volume)
                quantity = self.adjust_order_size(quantity, -1)
                if quantity > 0:
                    orders.append(Order(self.symbol, price, -quantity))
                    sell_limit -= quantity

        if sell_limit > 0:
            if self.outstanding_sell_orders:
                popular_sell_price = min(self.outstanding_sell_orders, key=lambda x: abs(x[1]))[0]
            else:
                popular_sell_price = true_value + 2

            price = max(min_sell_price, popular_sell_price - 1)
            quantity = self.adjust_order_size(sell_limit, -1)
            if quantity > 0:
                orders.append(Order(self.symbol, price, -quantity))

        return orders

class SquidInkStrategy(SimpleStrategy):
    def __init__(self):
        super().__init__("SQUID_INK", 50)
        self.last_midpoint = None
        self.price_history = []
        self.max_history_len = 8
        self.volatility_history = []
        self.max_volatility_len = 10

    def get_true_value(self):
        if not self.outstanding_buy_orders or not self.outstanding_sell_orders:
            midpoint = self.last_midpoint if self.last_midpoint else 1970
            return midpoint

        best_bid = max(price for price, _ in self.outstanding_buy_orders) if self.outstanding_buy_orders else None
        best_ask = min(price for price, _ in self.outstanding_sell_orders) if self.outstanding_sell_orders else None

        if best_bid and best_ask:
            current_midpoint = (best_bid + best_ask) // 2
        else:
            current_midpoint = self.last_midpoint if self.last_midpoint else 1970

        self.price_history.append(current_midpoint)
        if len(self.price_history) > self.max_history_len:
            self.price_history.pop(0)

        target_midpoint = current_midpoint
        if len(self.price_history) >= 5:
            sma = sum(self.price_history[-5:]) / 5
            trend_diff = current_midpoint - sma
            if abs(trend_diff) > 1.5:
                adjustment = int(trend_diff * 0.25)
                target_midpoint = current_midpoint - adjustment

        if len(self.price_history) >= 2:
            volatility = abs(self.price_history[-1] - self.price_history[-2])
            self.volatility_history.append(volatility)
            if len(self.volatility_history) > self.max_volatility_len:
                self.volatility_history.pop(0)

        self.last_midpoint = current_midpoint
        return target_midpoint

    def save(self):
        data = super().save()
        data.append(self.last_midpoint)
        data.append(self.price_history)
        data.append(self.volatility_history)
        return data

    def load(self, data):
        if isinstance(data, list) and len(data) >= 4:
            super().load(data[:-3])
            self.last_midpoint = data[-3]
            self.price_history = data[-2]
            self.volatility_history = data[-1]
        elif isinstance(data, list) and len(data) >= 3:
            super().load(data[:-2])
            self.last_midpoint = data[-2]
            self.price_history = data[-1]
            self.volatility_history = []
        elif isinstance(data, list) and len(data) >= 2:
            super().load(data[:-1])
            self.last_midpoint = data[-1]
            self.price_history = []
            self.volatility_history = []
        else:
            super().load(data)
            self.last_midpoint = None
            self.price_history = []
            self.volatility_history = []

    def adjust_order_size(self, quantity, side):
        adjusted_quantity = super().adjust_order_size(quantity, side)

        adjusted_quantity = int(adjusted_quantity * 0.8)

        avg_volatility = sum(self.volatility_history) / max(len(self.volatility_history), 1)
        if avg_volatility > 1.8: 
            adjusted_quantity = int(adjusted_quantity * 0.9)

        return max(1, adjusted_quantity)

    def get_orders(self) -> list[Order]:
        orders = []
        buy_limit = self.limit - self.current_position
        sell_limit = self.limit + self.current_position

        true_value = self.get_true_value() 
        self.last_true_value = true_value

        avg_volatility = sum(self.volatility_history) / max(len(self.volatility_history), 1)
        buy_offset = -1
        sell_offset = 1

        if avg_volatility > 1.8: 
            buy_offset = -2
            sell_offset = 2

        max_buy_price = true_value + buy_offset
        min_sell_price = true_value + sell_offset

        position_ratio = self.current_position / self.limit if self.limit > 0 else 0
        if position_ratio > 0.8: max_buy_price = true_value + buy_offset - 1
        elif position_ratio < -0.8: min_sell_price = true_value + sell_offset + 1

        for price, volume in self.outstanding_sell_orders:
            if buy_limit > 0 and price <= max_buy_price:
                quantity = min(buy_limit, -volume)
                quantity = self.adjust_order_size(quantity, 1)
                if quantity > 0:
                    orders.append(Order(self.symbol, price, quantity))
                    buy_limit -= quantity

        if buy_limit > 0:
            if self.outstanding_buy_orders:
                popular_buy_price = max(self.outstanding_buy_orders, key=lambda x: x[1])[0]
            else:
                popular_buy_price = true_value - 2
            price = min(max_buy_price, popular_buy_price + 1)
            quantity = self.adjust_order_size(buy_limit, 1)
            if quantity > 0:
                orders.append(Order(self.symbol, price, quantity))

        for price, volume in self.outstanding_buy_orders:
            if sell_limit > 0 and price >= min_sell_price:
                quantity = min(sell_limit, volume)
                quantity = self.adjust_order_size(quantity, -1)
                if quantity > 0:
                    orders.append(Order(self.symbol, price, -quantity))
                    sell_limit -= quantity

        if sell_limit > 0:
            if self.outstanding_sell_orders:
                popular_sell_price = min(self.outstanding_sell_orders, key=lambda x: abs(x[1]))[0]
            else:
                popular_sell_price = true_value + 2
            price = max(min_sell_price, popular_sell_price - 1)
            quantity = self.adjust_order_size(sell_limit, -1)
            if quantity > 0:
                orders.append(Order(self.symbol, price, -quantity))

        return orders

class Trader:
    strategy_lookup = {
        "RAINFOREST_RESIN": RainforestResinStrategy(),
        "KELP": KelpStrategy(),
        "SQUID_INK": SquidInkStrategy()
    }

    def run(self, state: TradingState):
        logger = Logger()
        result = dict()

        market_info = {
            "resin_depth": state.order_depths.get("RAINFOREST_RESIN", None),
            "kelp_depth": state.order_depths.get("KELP", None),
            "squid_depth": state.order_depths.get("SQUID_INK", None),
            "resin_position": state.position.get("RAINFOREST_RESIN", 0),
            "kelp_position": state.position.get("KELP", 0),
            "squid_position": state.position.get("SQUID_INK", 0)
        }

        if "RAINFOREST_RESIN" in state.order_depths:
            resin_strategy = Trader.strategy_lookup["RAINFOREST_RESIN"]
            resin_strategy.read(state)

            if market_info["resin_depth"]:
                best_bid = max(market_info["resin_depth"].buy_orders.keys()) if market_info["resin_depth"].buy_orders else None
                best_ask = min(market_info["resin_depth"].sell_orders.keys()) if market_info["resin_depth"].sell_orders else None
                if best_bid and best_ask:
                    market_info["resin_spread"] = best_ask - best_bid
                    market_info["resin_mid"] = (best_ask + best_bid) / 2

            result["RAINFOREST_RESIN"] = resin_strategy.get_orders()

        if "KELP" in state.order_depths:
            kelp_strategy = Trader.strategy_lookup["KELP"]
            kelp_strategy.read(state)

            if market_info["kelp_depth"]:
                best_bid = max(market_info["kelp_depth"].buy_orders.keys()) if market_info["kelp_depth"].buy_orders else None
                best_ask = min(market_info["kelp_depth"].sell_orders.keys()) if market_info["kelp_depth"].sell_orders else None
                if best_bid and best_ask:
                    market_info["kelp_spread"] = best_ask - best_bid
                    market_info["kelp_mid"] = (best_ask + best_bid) / 2

            kelp_strategy.market_info = market_info
            result["KELP"] = kelp_strategy.get_orders()

        if "SQUID_INK" in state.order_depths:
            squid_strategy = Trader.strategy_lookup["SQUID_INK"]
            squid_strategy.read(state)

            if market_info["squid_depth"]:
                best_bid = max(market_info["squid_depth"].buy_orders.keys()) if market_info["squid_depth"].buy_orders else None
                best_ask = min(market_info["squid_depth"].sell_orders.keys()) if market_info["squid_depth"].sell_orders else None
                if best_bid and best_ask:
                    market_info["squid_spread"] = best_ask - best_bid
                    market_info["squid_mid"] = (best_ask + best_bid) / 2

            squid_strategy.market_info = market_info
            result["SQUID_INK"] = squid_strategy.get_orders()

        if len(result) >= 2:
            self.optimize_cross_product_orders(result, market_info)

        trader_data = ""
        conversions = 0
        logger.flush(state, result, conversions, trader_data)

        return result, conversions, trader_data

    def optimize_cross_product_orders(self, result, market_info):
        """Optimize orders between products based on market conditions"""

        if "RAINFOREST_RESIN" in result and "KELP" in result:
            if all(k in market_info for k in ["resin_spread", "kelp_spread", "resin_position", "kelp_position"]):
                resin_orders = result.get("RAINFOREST_RESIN", [])
                kelp_orders = result.get("KELP", [])
                resin_spread = market_info.get("resin_spread", 0)
                kelp_spread = market_info.get("kelp_spread", 0)

                if resin_spread < 3 and kelp_spread > 5:
                    for order in resin_orders:
                        order.quantity = int(order.quantity * 0.70)
                elif kelp_spread < 3 and resin_spread > 5:
                    for order in kelp_orders:
                        order.quantity = int(order.quantity * 0.70)

                if resin_spread > 4 and kelp_spread > 4:
                    resin_position_ratio = abs(market_info.get("resin_position", 0)) / 50
                    kelp_position_ratio = abs(market_info.get("kelp_position", 0)) / 50
                    if resin_position_ratio > kelp_position_ratio + 0.25:
                        for order in resin_orders:
                            order.quantity = int(order.quantity * 0.65)
                    elif kelp_position_ratio > resin_position_ratio + 0.25:
                        for order in kelp_orders:
                            order.quantity = int(order.quantity * 0.65)

        if "RAINFOREST_RESIN" in result and "SQUID_INK" in result:
            if all(k in market_info for k in ["resin_position", "squid_position"]):
                squid_orders = result.get("SQUID_INK", [])
                squid_position_ratio = abs(market_info.get("squid_position", 0)) / 50
                if squid_position_ratio > 0.6:
                    for order in squid_orders:
                        order.quantity = int(order.quantity * 0.6)

        if "KELP" in result and "SQUID_INK" in result:
            if all(k in market_info for k in ["kelp_position", "squid_position"]):
                kelp_orders = result.get("KELP", [])
                squid_orders = result.get("SQUID_INK", [])
                kelp_position = market_info.get("kelp_position", 0)
                squid_position = market_info.get("squid_position", 0)

                combined_limit = 100 
                total_kelp_squid_pos = abs(kelp_position) + abs(squid_position)
                if total_kelp_squid_pos > combined_limit * 0.7: 
                    reduction_factor = 0.75
                    for order in kelp_orders + squid_orders:

                        order_increases_pos = False
                        if order.symbol == "KELP":
                            if (order.quantity > 0 and kelp_position >= 0) or (order.quantity < 0 and kelp_position <= 0):
                                order_increases_pos = True
                        elif order.symbol == "SQUID_INK":
                             if (order.quantity > 0 and squid_position >= 0) or (order.quantity < 0 and squid_position <= 0):
                                order_increases_pos = True

                        if order_increases_pos:
                             order.quantity = int(order.quantity * reduction_factor)