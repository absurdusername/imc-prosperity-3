from round1.datamodel import Listing, Observation, Order, OrderDepth, ProsperityEncoder, Symbol, Trade, TradingState
from typing import Any
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

        # We truncate state.traderData, trader_data, and self.logs to the same max. length to fit the log limit
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


class JmerleStrategy:
    def __init__(
            self, 
            symbol: Symbol, 
            limit: int, 
            window_size: int=10,  
        ):
        self.symbol = symbol
        self.limit = limit
        self.window_size = window_size

        # NICE STATE-RELATED STUFF USED BY DA ALGORITHM!
        self.window = []
        self.outstanding_buy_orders = []
        self.outstanding_sell_orders = []
        self.current_position = None
    
    def save(self):
        """Returns some JSON object.."""
        return self.window
    
    def load(self, data):
        """Re-initializes all fields."""
        self.window = data

    def get_true_value(self) -> int:
        return 10_000

    def parse_state(self, state: TradingState):
        """Computes all that nice state-related stuff used by da algorithm."""
        order_depth: OrderDepth = state.order_depths[self.symbol]

        self.outstanding_buy_orders = sorted(order_depth.buy_orders.items(), reverse=True)
        self.outstanding_sell_orders = sorted(order_depth.sell_orders.items())
        self.current_position = state.position.get(self.symbol, 0)
        self.window.append(abs(self.current_position) == self.limit)

        if len(self.window) > self.window_size:
            self.window.pop(0)


    def run(self, state: TradingState) -> list[Order]:
        self.parse_state(state)
        orders = []

        buy_limit = self.limit - self.current_position
        sell_limit = self.limit + self.current_position

        true_value = self.get_true_value()
        max_buy_price = true_value - 1 if self.current_position > self.limit * 0.5 else true_value
        min_sell_price = true_value + 1 if self.current_position < self.limit * -0.5 else true_value

        soft_liquidate = len(self.window) == self.window_size and sum(self.window) >= self.window_size / 2 and self.window[-1]
        hard_liquidate = len(self.window) == self.window_size and all(self.window)

        for price, volume in self.outstanding_sell_orders:
            if buy_limit > 0 and price <= max_buy_price:
                quantity = min(buy_limit, -volume)
                orders.append(Order(self.symbol, price, quantity))
                buy_limit -= quantity

        if buy_limit > 0 and (soft_liquidate or hard_liquidate):
            quantity = buy_limit // 2
            price = true_value - 1 if soft_liquidate else true_value
            orders.append(Order(self.symbol, price, quantity))
            buy_limit -= quantity

        if buy_limit > 0:
            popular_buy_price = max(self.outstanding_buy_orders, key=lambda tup: tup[1])[0]
            price = min(max_buy_price, popular_buy_price + 1)
            orders.append(Order(self.symbol, price, buy_limit))

        for price, volume in self.outstanding_buy_orders:
            if sell_limit > 0 and price >= min_sell_price:
                quantity = min(sell_limit, volume)
                orders.append(Order(self.symbol, price, -quantity))
                sell_limit -= quantity

        if sell_limit > 0 and (soft_liquidate or hard_liquidate):
            quantity = sell_limit // 2
            price = true_value + 1 if soft_liquidate else true_value
            orders.append(Order(self.symbol, price, -quantity))
            sell_limit -= quantity
        
        if sell_limit > 0:
            popular_sell_price = min(self.outstanding_sell_orders, key=lambda tup: tup[1])[0]
            price = max(min_sell_price, popular_sell_price - 1)
            orders.append(Order(self.symbol, price, -sell_limit))
        
        return orders


class KelpStrategy(JmerleStrategy):
    def run(self, state: TradingState) -> list[Order]:
        self.parse_state(state)
        orders = []

        buy_limit = self.limit - self.current_position
        sell_limit = self.limit + self.current_position

        liquidity = 1 - self.current_position / self.limit
        L1 = 0.25 # start getting a little picky
        L2 = 0.5 # start getting worried

        f = self.f()
        # max_buy_price = f - 2 if liquidity > L1 else f - 3 if liquidity > L2 else f - 4
        # min_sell_price = f - 1 if liquidity > -L1 else f if liquidity > -L2 else f + 1

        max_buy_price = f - 2 if liquidity > L1 else f - 3
        min_sell_price = f - 1 if liquidity > -L1 else f

        # max_buy_price = f - 2
        # min_sell_price = f - 1

        for price, volume in self.outstanding_sell_orders:
            if buy_limit > 0 and price <= max_buy_price:
                quantity = min(buy_limit, -volume)
                orders.append(Order(self.symbol, price, quantity))
                buy_limit -= quantity

        if buy_limit > 0:
            popular_buy_price = max(self.outstanding_buy_orders, key=lambda tup: tup[1])[0]
            price = min(max_buy_price, popular_buy_price + 1)
            orders.append(Order(self.symbol, price, buy_limit))

        for price, volume in self.outstanding_buy_orders:
            if sell_limit > 0 and price >= min_sell_price:
                quantity = min(sell_limit, volume)
                orders.append(Order(self.symbol, price, -quantity))
                sell_limit -= quantity
        
        if sell_limit > 0:
            popular_sell_price = min(self.outstanding_sell_orders, key=lambda tup: tup[1])[0]
            price = max(min_sell_price, popular_sell_price - 1)
            orders.append(Order(self.symbol, price, -sell_limit))
        
        return orders
    
    
    def f(self):
        f_cap = min(self.outstanding_sell_orders, key=lambda tup: tup[1])[0]
        g_cap = max(self.outstanding_buy_orders, key=lambda tup: tup[1])[0]
        return (f_cap + g_cap + 3) // 2


class Trader:
    strategy_lookup = {
        "RAINFOREST_RESIN": JmerleStrategy("RAINFOREST_RESIN", 50),
        "KELP": KelpStrategy("KELP", 50)
    }

    def run(self, state: TradingState):
        logger = Logger()
        result = dict()

        for product in state.order_depths:
            strategy = Trader.strategy_lookup[product]
            result[product] = strategy.run(state)
    
        trader_data = ""
        conversions = 0
        logger.flush(state, result, conversions, trader_data)

        return result, conversions, trader_data
