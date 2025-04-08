from typing import Any
import numpy as np
import json



class Strategy:
    def __init__(self, name: str, max_position: int):
        self.name: str = name
        self.cached_prices: list = []
        self.cached_means: list = []
        self.max_pos: int = max_position
        self.trade_count: int = 1

        self.prod_position: int = 0
        self.new_buy_orders: int = 0
        self.new_sell_orders: int = 0
        self.order_depth: OrderDepth = OrderDepth()

    def reset_from_state(self, state: TradingState):
        self.prod_position = state.position[self.name] if self.name in state.position.keys() else 0
        self.order_depth: OrderDepth = state.order_depths[self.name]

        self.new_buy_orders = 0
        self.new_sell_orders = 0

    def sell_product(self, best_bids, i, order_depth, orders):
        # Sell product at best bid
        best_bid_volume = order_depth.buy_orders[best_bids[i]]
        if self.prod_position - best_bid_volume >= -self.max_pos:
            orders.append(Order(self.name, best_bids[i], -best_bid_volume))
            self.prod_position += -best_bid_volume
            self.new_sell_orders += best_bid_volume

        else:
            # Sell as much as we can without exceeding the self.max_pos[product]
            vol = self.prod_position + self.max_pos
            orders.append(Order(self.name, best_bids[i], -vol))
            self.prod_position += -vol
            self.new_sell_orders += vol

    def buy_product(self, best_asks, i, order_depth, orders):
        # Buy product at best ask
        best_ask_volume = order_depth.sell_orders[best_asks[i]]
        if self.prod_position - best_ask_volume <= self.max_pos:
            orders.append(Order(self.name, best_asks[i], -best_ask_volume))
            self.prod_position += -best_ask_volume
            self.new_buy_orders += -best_ask_volume
        else:
            # Buy as much as we can without exceeding the self.max_pos[product]
            vol = self.max_pos - self.prod_position
            orders.append(Order(self.name, best_asks[i], vol))
            self.prod_position += vol
            self.new_buy_orders += vol

    def continuous_buy(self, order_depth: OrderDepth, orders: list):
        if len(order_depth.sell_orders) != 0:
            best_asks = sorted(order_depth.sell_orders.keys())

            i = 0
            while i < self.trade_count and len(best_asks) > i:
                if self.prod_position == self.max_pos:
                    break

                self.buy_product(best_asks, i, order_depth, orders)
                i += 1

    def continuous_sell(self, order_depth: OrderDepth, orders: list):
        if len(order_depth.buy_orders) != 0:
            best_bids = sorted(order_depth.buy_orders.keys(), reverse=True)

            i = 0
            while i < self.trade_count and len(best_bids) > i:
                if self.prod_position == -self.max_pos:
                    break

                self.sell_product(best_bids, i, order_depth, orders)
                i += 1


class DiffStrategy(Strategy):
    def __init__(self, name: str, max_position: int, derivative_resolution: int, diff_thresh: int):
        super().__init__(name, max_position)
        self.derivative_resolution: int = derivative_resolution
        self.diff_thresh: int = diff_thresh

    def trade(self, trading_state: TradingState, orders: list):
        order_depth: OrderDepth = trading_state.order_depths[self.name]
        self.cache_purchased_prices(trading_state)
        self.calculate_means()

        diff = self.get_price_difference()

        if diff < -self.diff_thresh and len(order_depth.sell_orders) != 0:
            self.continuous_buy(order_depth, orders)

        if diff > self.diff_thresh and len(order_depth.buy_orders) != 0:
            self.continuous_sell(order_depth, orders)

    def get_price_difference(self) -> float:
        # Calculate the difference between the current mean and the mean from
        # self.derivative_resolution days ago
        if len(self.cached_means) < self.derivative_resolution + 1:
            old_mean = self.cached_means[0]
        else:
            old_mean = self.cached_means[-self.derivative_resolution]
        diff = self.cached_means[-1] - old_mean
        return diff

    def calculate_means(self):
        #
        if len(self.cached_prices) == 0:
            self.cached_means.append(0)

        else:
            relevant_prices = []
            for day_prices in self.cached_prices[max(-len(self.cached_prices), -1):]:
                for price in day_prices:
                    relevant_prices.append(price)
            prices = np.array([x[1] for x in relevant_prices])
            quantities = np.abs(np.array([x[0] for x in relevant_prices]))

            self.cached_means.append(np.average(prices, weights=quantities))

    def cache_purchased_prices(self, state: TradingState) -> None:
        # Caches prices of bought and sold products

        market_trades = state.market_trades
        own_trades = state.own_trades

        prod_trades: List[Trade] = own_trades.get(self.name, []) + market_trades.get(self.name, [])

        if len(prod_trades) > 0:
            prices = [(trade.quantity, trade.price) for trade in prod_trades]
            self.cached_prices.append(prices)


class Resin(DiffStrategy):
    def __init__(self):
        super().__init__("RAINFOREST_RESIN", max_position=50, derivative_resolution=15, diff_thresh=25)


class Kelp(DiffStrategy):
    def __init__(self):
        super().__init__("KELP", max_position=50, derivative_resolution=500, diff_thresh=)



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

class Trader:

    def __init__(self):
        self.products = {
            "RAINFOREST_RESIN": Resin(),
            "KELP": Kelp()
        }

    def run(self, state: TradingState) -> Dict[str, List[Order]]:
        logger = Logger()
        result = {}

        for product in state.order_depths.keys():
            if product in self.products.keys():
                orders: list[Order] = []

                self.products[product].reset_from_state(state)
                self.products[product].trade(trading_state=state, orders=orders)
                result[product] = orders

        trader_data = ""
        conversions = 0
        logger.flush(state, result, conversions, trader_data)

        return result, conversions, trader_data