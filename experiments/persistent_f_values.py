from round1.datamodel import Listing, Observation, Order, OrderDepth, ProsperityEncoder, Symbol, Trade, TradingState
from typing import Any
import json
from collections import defaultdict


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


class ResinStrategy:
    def __init__(
            self, 
            symbol: Symbol, 
            limit: int, 
        ):
        self.symbol = symbol
        self.limit = limit
        
        # NICE STATE-RELATED STUFF USED BY DA ALGORITHM!
        self.outstanding_buy_orders = []
        self.outstanding_sell_orders = []
        self.current_position = None
    
    def save(self): pass
    def load(self, data): pass

    def parse_state(self, state: TradingState):
        """Computes all that nice state-related stuff used by da algorithm."""
        order_depth: OrderDepth = state.order_depths[self.symbol]

        self.outstanding_buy_orders = sorted(order_depth.buy_orders.items(), reverse=True)
        self.outstanding_sell_orders = sorted(order_depth.sell_orders.items())
        self.current_position = state.position.get(self.symbol, 0)

    def run(self, state: TradingState) -> list[Order]:
        self.parse_state(state)
        orders = []

        buy_limit = self.limit - self.current_position
        sell_limit = self.limit + self.current_position

        true_value = 10_000
        max_buy_price = true_value - 1 if self.current_position > self.limit * 0.5 else true_value
        min_sell_price = true_value + 1 if self.current_position < self.limit * -0.5 else true_value

        for price, volume in self.outstanding_sell_orders:
            if buy_limit > 0 and price <= max_buy_price:
                quantity = min(buy_limit, -volume)
                orders.append(Order(self.symbol, price, quantity))
                buy_limit -= quantity

        for price, volume in self.outstanding_buy_orders:
            if sell_limit > 0 and price >= min_sell_price:
                quantity = min(sell_limit, volume)
                orders.append(Order(self.symbol, price, -quantity))
                sell_limit -= quantity

        orders.append(Order(self.symbol, 9998, buy_limit))
        orders.append(Order(self.symbol, 10002, -sell_limit))

        return orders


class KelpStrategy:
    # lookup[ask - bid] := list of possible (e1, e2) values
    lookup = defaultdict(list, {
        3: [(-2, -2), (-1, -1), (0, 0), (1, 1), (2, 2)],
        2: [(-2, -1), (-1, 0), (0, 1), (1, 2)],
        1: [(-2, 0), (-1, 1), (0, 2)],
        4: [(-1, -2), (0, -1), (1, 0), (2, 1)]
    })

    def __init__(
            self, 
            symbol: Symbol, 
            limit: int, 
        ):
        self.symbol = symbol
        self.limit = limit

        # NICE HISTORY-RELATED STUFF USED BY THE ALGORITHM
        self.window = [] 
        self.current_position = 0
        self.outstanding_buy_orders = None # shall always be sorted (in reverse)
        self.outstanding_sell_orders = None # shall always be sorted
        self.ask1 = self.bid1 = None
        self.best_ask = self.best_bid = None

    def save(self): 
        return self.window
    
    def load(self, data):
        self.window = data

    def parse_state(self, state: TradingState):
        self.current_position = state.position.get(self.symbol, 0)

        order_depth: OrderDepth = state.order_depths[self.symbol]
        
        self.outstanding_buy_orders = sorted(order_depth.buy_orders.items(), reverse=True)
        self.outstanding_sell_orders = sorted(order_depth.sell_orders.items())

        self.current_position = state.position.get(self.symbol, 0)

        self.ask1 = min(order_depth.sell_orders.items(), key=lambda t: t[1])[0]   
        self.bid1 = max(order_depth.buy_orders.items(), key=lambda t: t[1])[0] 

        possible_f_values = self.possible_f_values() # list of possible f-values
        self.window.append(possible_f_values)

    # def persistent_f_values(self):
    #     candidates = set( self.window[-1] ) # set of possible F values rn

    #     for values in self.window[-self.window_size:]:
    #         candidates = set(values) & candidates
    #         if not candidates:
    #             break
        
    #     if candidates:
    #         return candidates
        
    #     # else, well... shit
    #     return self.window[-1]

    def persistent_f_values(self):
        candidates = set( self.window[-1] ) # set of possible F values rn

        for values in reversed(self.window):
            new_candidates = candidates & set(values)

            if not new_candidates: 
                break
            candidates = new_candidates
        
        return candidates

    def run(self, state: TradingState) -> list[Order]:
        self.parse_state(state)

        orders = []

        buy_limit = (self.limit - self.current_position) 
        sell_limit = (self.limit + self.current_position)

        ratio = self.current_position / self.limit

        f_values = self.persistent_f_values()
        f_buy, f_sell = min(f_values), max(f_values)

        # max_buy_price = f_buy - 2 + (ratio // 0.25)
        # min_sell_price = f_sell - 1 - (ratio // 0.25)

        max_buy_price = f_buy - 2 
        min_sell_price = f_sell - 1

        for price, volume in self.outstanding_sell_orders:
            if buy_limit > 0 and price <= max_buy_price:
                quantity = min(buy_limit, -volume)
                orders.append(Order(self.symbol, price, quantity))
                buy_limit -= quantity

        for price, volume in self.outstanding_buy_orders:
            if sell_limit > 0 and price >= min_sell_price:
                quantity = min(sell_limit, volume)
                orders.append(Order(self.symbol, price, -quantity))
                sell_limit -= quantity

        if buy_limit > 0:
            price = min(max_buy_price, self.bid1)
            orders.append(Order(self.symbol, price, buy_limit))
        
        if sell_limit > 0:
            price = max(min_sell_price, self.ask1)
            orders.append(Order(self.symbol, price, -sell_limit))
        
        return orders


class Trader:
    strategy_lookup = {
        "RAINFOREST_RESIN": ResinStrategy(symbol="RAINFOREST_RESIN", limit=50),
        "KELP": KelpStrategy(symbol="KELP", limit=50)
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
