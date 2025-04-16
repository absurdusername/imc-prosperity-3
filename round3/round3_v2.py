from datamodel import *
from typing import TypeAlias, Any

from math import floor, ceil, copysign, log, e
from statistics import mean, stdev, mode, median
from collections import deque

import itertools
import numpy as np

JSON: TypeAlias = dict[str, "JSON"] | list["JSON"] | str | int | float | bool | None


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


class Product:
    """Store historic data and planned-upcoming orders.

    Attributes:
        symbol (str): Self explanatory.
        limit (int): Position limit.
        history_size_limit (int): Number of past iterations for which data is preserved.
            The length of position_history, ask_history, and bid_history are bound by this.
            Defaults to 10_000.

        position_history (list): Stores our position for the last n iterations.
            position_history[-1] is the current position.

        ask_history (list): Stores outstanding ask orders for the last n iterations.
            [[ask_price_1, ask_volume_1], ...] is appended for each iteration.

        bid_history (list): Similar to ask_history but for bids.
            Volumes are stored as positive here, unlike in order_depth.

        mid_price_history (list): Stores mid-price for each iteration.
            mid_price := (max_volume_ask_price + max_volume_bid_price) / 2

        planned_orders (list): List of Orders we plan on placing.
        planned_buy_volume (int): Total volume of all planned BUY orders.
        planned_sell_volume (int): Total volume of all planned SELL orders.
    """

    def __init__(self, symbol: str, limit: int, history_size_limit: int = 10_000):
        self.symbol: str = symbol
        self.limit: int = limit
        self.history_size_limit: int = history_size_limit

        self.position_history: list[int] = []
        self.ask_history: list[list[list[int, int]]] = []
        self.bid_history: list[list[list[int, int]]] = []
        self.mid_price_history: list[int] = []

        self.planned_orders: list[Order] = []
        self.planned_buy_volume: int = 0
        self.planned_sell_volume: int = 0

    def load(self, state: TradingState, old_data: JSON) -> None:
        # Reset these variables because AWS Lambda may or may not re-instantiate the class.
        self.planned_orders: list[Order] = []
        self.planned_buy_volume: int = 0
        self.planned_sell_volume: int = 0

        position = state.position.get(self.symbol, 0)
        order_depth = state.order_depths[self.symbol]
        asks = [[price, -quantity] for price, quantity in order_depth.sell_orders.items()]
        bids = [[price, quantity] for price, quantity in order_depth.buy_orders.items()]
        mid_price = (max(asks, key=lambda t: t[1])[0] + max(bids, key=lambda t: t[1])[0]) / 2

        if old_data:
            self.position_history = old_data["position_history"]
            self.ask_history = old_data["ask_history"]
            self.bid_history = old_data["bid_history"]
            self.mid_price_history = old_data["mid_price_history"]

        self.position_history.append(position)
        self.ask_history.append(asks)
        self.bid_history.append(bids)
        self.mid_price_history.append(mid_price)

        for sequence in (self.position_history, self.ask_history, self.bid_history, self.mid_price_history):
            if len(sequence) > self.history_size_limit:
                sequence.pop(0)

    def save(self) -> JSON:
        data = {
            "position_history": self.position_history,
            "ask_history": self.ask_history,
            "bid_history": self.bid_history,
            "mid_price_history": self.mid_price_history,
        }
        return data

    def buy(self, price: int, quantity: int) -> None:
        """Add a BUY order to planned orders."""
        # TO-DO: verify that quantity does not exceed limits
        if quantity == 0:
            return
        self.planned_buy_volume += quantity
        self.planned_orders.append(Order(self.symbol, price, quantity))

    def sell(self, price: int, quantity: int) -> None:
        """Add a SELL order to planned orders.
        Parameter `quantity` is positive here. No negative number BS.
        """
        # TO-DO: verify that quantity does not exceed limits
        if quantity == 0:
            return
        self.planned_sell_volume += quantity
        self.planned_orders.append(Order(self.symbol, price, -quantity))

    @property
    def history_size(self) -> int:
        """Numbers of iterations for which data is available at the moment."""
        return len(self.position_history)

    @property
    def position(self) -> int:
        return self.position_history[-1]

    @property
    def position_ratio(self) -> float:
        return self.position / self.limit

    @property
    def asks(self) -> list:
        """List of (price, quantity) for asks in **increasing order of price**.
        Quantities are positive here, unlike in class Order.
        """
        return sorted(self.ask_history[-1])

    @property
    def bids(self) -> list:
        """List of (price, quantity) for bids in **decreasing order of price.**"""
        return sorted(self.bid_history[-1], reverse=True)

    @property
    def best_ask_price(self) -> int:
        """Minimum ask price."""
        return self.asks[0][0]

    @property
    def best_bid_price(self) -> int:
        """Maximum bid price."""
        return self.bids[0][0]

    @property
    def max_volume_ask_price(self) -> int:
        """Ask price of the order with the highest volume."""
        return max(self.asks, key=lambda t: t[1])[0]

    @property
    def max_volume_bid_price(self) -> int:
        """Bid price of the order with the highest volume."""
        return max(self.bids, key=lambda t: t[1])[0]

    @property
    def mid_price(self) -> float:
        """Average of max_volume_ask_price and max_volume_bid_price."""
        return (self.max_volume_ask_price + self.max_volume_bid_price) / 2

    @property
    def buy_capacity(self) -> int:
        """Maximum volume that can be bought without potentially exceeding position limit."""
        return self.limit - self.position - self.planned_buy_volume

    @property
    def sell_capacity(self) -> int:
        """Maximum volume that can be sold without potentially exceeding position limit."""
        return self.limit + self.position - self.planned_sell_volume

    def linear_regression(self, window_size: int = None) -> tuple[float, float]:
        """Coefficients (m, b) of line fitted over mid-prices.
        Iterations are 1-indexed here.
        y = m * (1) + b gives predicted mid-price for the first iteration considered.

        Args:
            window_size (int): Number of previous iterations to consider.
                Defaults to self.history_size.
        """
        if window_size is None:
            window_size = self.history_size

        mid_prices = list(self.mid_price_history)[-window_size:]
        m, b = np.polyfit(range(1, window_size + 1), mid_prices, 1)
        return m, b


class Strategy:
    @staticmethod
    def simple_market_making(
            product: Product,
            max_buy_price: int,
            min_sell_price: int,
            to_buy: int = None,
            to_sell: int = None,
    ) -> None:
        """
        Adapted from Jmerle's `MarketMakingStrategy`.
        https://github.com/jmerle/imc-prosperity-2/blob/master/src/submissions/round5.py

        Args:
            product (Product): The corresponding `Product` instance.

            max_buy_price (int): Never BUY above this price.
            min_sell_price (int): Never SELL below this price.

            to_buy (int): Maximum volume of BUY orders to place.
                Defaults to `product.buy_capacity`.

            to_sell (int): Maximum volume of SELL orders to place.
                Defaults to `product.sell_capacity`.
        """
        if to_buy is None:
            to_buy = product.buy_capacity

        if to_sell is None:
            to_sell = product.sell_capacity

        for price, quantity in product.asks:
            if price <= max_buy_price and to_buy:
                quantity = min(quantity, to_buy)
                product.buy(price, quantity)
                to_buy -= quantity

        price = min(max_buy_price, product.max_volume_bid_price + 1)
        product.buy(price, to_buy)

        for price, quantity in product.bids:
            if price >= min_sell_price and to_sell:
                quantity = min(quantity, to_sell)
                product.sell(price, quantity)
                to_sell -= quantity

        price = max(min_sell_price, product.max_volume_ask_price - 1)
        product.sell(price, product.sell_capacity)


class Trader:
    def __init__(self) -> None:
        # all historic data and upcoming Orders will be stored in Product instances
        self.products = {
            "RAINFOREST_RESIN": Product("RAINFOREST_RESIN", 50),
            "KELP": Product("KELP", 50),
            "SQUID_INK": Product("SQUID_INK", 50),

            "CROISSANTS": Product("CROISSANTS", 250),
            "JAMS": Product("JAMS", 350),
            "DJEMBES": Product("DJEMBES", 60),

            "PICNIC_BASKET1": Product("PICNIC_BASKET1", 60),
            "PICNIC_BASKET2": Product("PICNIC_BASKET2", 100),
        }

    def run(self, state: TradingState):
        # load old data from state.traderData + new data from state
        old_trader_data = json.loads(state.traderData) if state.traderData != "" else {}
        for symbol, product in self.products.items():
            # product.load(state, old_trader_data.get(symbol, {}))
            product.load(state, None)

        # store all data in a dict for next iteration
        new_trader_data = dict()
        for symbol, product in self.products.items():
            # new_trader_data[symbol] = product.save()
            pass

        self.trade_rainforest_resin()
        self.trade_kelp()
        self.trade_squid_ink()

        self.trade_croissants()
        self.trade_jams()
        self.trade_djembes() # TO-DO
        self.trade_picnic_basket1()
        self.trade_picnic_basket2()

        result = {
            symbol: product.planned_orders
            for symbol, product in self.products.items()
        }
        conversions = 0

        logger = Logger()
        logger.flush(state, result, conversions, state.traderData)

        return result, 0, json.dumps(new_trader_data)

    def trade_rainforest_resin(self) -> None:
        resin = self.products["RAINFOREST_RESIN"]
        max_buy_price = 9999 + (resin.position_ratio < -0.5)
        min_sell_price = 10_001 - (resin.position_ratio > 0.5)

        Strategy.simple_market_making(
            product=resin,
            max_buy_price=max_buy_price,
            min_sell_price=min_sell_price,
        )

    def trade_kelp(self) -> None:
        kelp = self.products["KELP"]

        f = (kelp.max_volume_ask_price + kelp.max_volume_bid_price + 3) / 2
        f = floor(f) if kelp.position > 0 else ceil(f)

        max_buy_price = (f - 2) - (kelp.position_ratio > 0.45) - (kelp.position_ratio > 0.8)
        min_sell_price = (f - 1) + (kelp.position_ratio < -0.45) + (kelp.position_ratio < -0.8)

        if kelp.position_ratio < -0.9:
            max_buy_price += 1  # desperate, will BUY at bad price

        if kelp.position_ratio > 0.9:
            min_sell_price -= 1  # desperate, will SELL at bad price

        Strategy.simple_market_making(
            product=kelp,
            max_buy_price=max_buy_price,
            min_sell_price=min_sell_price,
        )

    def trade_squid_ink(self) -> None:
        squid_ink = self.products["SQUID_INK"]

        if squid_ink.history_size < 100:
            return

        m, b = squid_ink.linear_regression()

        x = squid_ink.history_size  # iteration number
        fair_value = m * x + b

        max_buy_price = round(fair_value - 7)
        min_sell_price = round(fair_value + 7)

        Strategy.simple_market_making(
            product=squid_ink,
            max_buy_price=max_buy_price,
            min_sell_price=min_sell_price,
        )

    def trade_croissants(self) -> None:
        croissant = self.products["CROISSANTS"]
        m, b = croissant.linear_regression()

        x = croissant.history_size  # iteration number
        fair_value = m * x + b

        max_buy_price = round(fair_value - 5)
        min_sell_price = round(fair_value + 5)

        Strategy.simple_market_making(
            product=croissant,
            max_buy_price=max_buy_price,
            min_sell_price=min_sell_price,
        )

    def trade_jams(self) -> None:
        croissant = self.products["CROISSANTS"]
        m, b = croissant.linear_regression()
        x = croissant.history_size  # iteration number
        croissant_fair_value = m * x + b

        c = self.products["CROISSANTS"].mid_price
        j = self.products["JAMS"].mid_price
        basket2_fair_value = 4 * c + 2 * j

        jams_fair_value = (basket2_fair_value - 4 * croissant_fair_value) / 2

        max_buy_price = round(jams_fair_value - 14)
        min_sell_price = round(jams_fair_value + 14)

        jams = self.products["JAMS"]

        Strategy.simple_market_making(
            product=jams,
            max_buy_price=max_buy_price,
            min_sell_price=min_sell_price,
        )

    def trade_djembes(self) -> None:
        # picnic_basket1 = self.products["PICNIC_BASKET1"]
        # croissant = self.products["CROISSANTS"]
        # jams = self.products["JAMS"]
        # djembes = self.products["DJEMBES"]
        #
        # m, b = croissant.linear_regression()
        # x = croissant.history_size  # iteration number
        # croissant_fair_value = m * x + b
        #
        # c = self.products["CROISSANTS"].mid_price
        # j = self.products["JAMS"].mid_price
        # basket2_fair_value = 4 * c + 2 * j
        #
        # jams_fair_value = (basket2_fair_value - 4 * croissant_fair_value) / 2
        #
        # f = lambda c, j, d: 6 * c + 3 * j + 1 * d
        # sum_of_parts = f(croissant.mid_price, jams.mid_price, djembes.mid_price)
        # basket1_fair_value = 0.55 * sum_of_parts + 0.45 * picnic_basket1.mid_price
        #
        # djembes_fair_value = (basket1_fair_value - 6 * croissant_fair_value - 3 * jams.mid_price)
        # # djembes_fair_value = (0.45 * djembes_fair_value + 0.55 * djembes.mid_price)
        #
        # max_buy_price = round(djembes_fair_value - 15)
        # min_sell_price = round(djembes_fair_value + 15)
        #
        # Strategy.simple_market_making(
        #     product=djembes,
        #     max_buy_price=max_buy_price,
        #     min_sell_price=min_sell_price,
        # )
        pass

    def trade_picnic_basket1(self) -> None:
        picnic_basket1 = self.products["PICNIC_BASKET1"]
        croissants = self.products["CROISSANTS"]
        jams = self.products["JAMS"]
        djembes = self.products["DJEMBES"]

        f = lambda c, j, d: 6 * c + 3 * j + 1 * d

        sum_of_parts = f(croissants.mid_price, jams.mid_price, djembes.mid_price)
        fair_value = 0.55 * sum_of_parts + 0.45 * picnic_basket1.mid_price

        max_buy_price = round(fair_value - 6)
        min_sell_price = round(fair_value + 6)

        cp = croissants.mid_price_history[-100:]
        jp = jams.mid_price_history[-100:]
        dp = djembes.mid_price_history[-100:]
        pp = picnic_basket1.mid_price_history[-100:]

        premiums = [pp[i] - f(cp[i], jp[i], dp[i]) for i in range(len(pp))]

        if len(premiums) >= 2:
            mu, sigma = mean(premiums), stdev(premiums)

            max_buy_price = min(
                max_buy_price,
                round(picnic_basket1.max_volume_ask_price - 0.55 * sigma)
            )

            min_sell_price = max(
                min_sell_price,
                round(picnic_basket1.max_volume_bid_price + 0.55 * sigma),
            )

        Strategy.simple_market_making(
            product=picnic_basket1,
            max_buy_price=max_buy_price,
            min_sell_price=min_sell_price,
        )

    def trade_picnic_basket2(self) -> None:
        picnic_basket2 = self.products["PICNIC_BASKET2"]

        c = self.products["CROISSANTS"].mid_price
        j = self.products["JAMS"].mid_price

        fair_value = 4 * c + 2 * j

        max_buy_price = round(fair_value - 60)
        min_sell_price = round(fair_value - 20)

        Strategy.simple_market_making(
            product=picnic_basket2,
            max_buy_price=max_buy_price,
            min_sell_price=min_sell_price,
        )