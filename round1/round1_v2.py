from datamodel import *
from typing import TypeAlias, Any
import numpy as np
from math import floor, ceil
from statistics import mean
import itertools

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

        position_history (list): Stores our position for the last n iterations.
            position_history[-1] is the current position.

        ask_history (list): Stores outstanding ask orders for the last n iterations.
            [[ask_price_1, ask_volume_1], ...] is appended for each iteration.

        bid_history (list): Similar to ask_history but for bids.
            Volumes are stored as positive here, unlike in order_depth.

        planned_orders (list): List of Orders we plan on placing.
        planned_buy_volume (int): Total volume of all planned BUY orders.
        planned_sell_volume (int): Total volume of all planned SELL orders.
    """

    def __init__(self, symbol: str, limit: int):
        self.symbol: str = symbol
        self.limit: int = limit
        self.position_history: list[int] = []

        self.ask_history: list[list[list[int, int]]] = []
        self.bid_history: list[list[list[int, int]]] = []

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

        if old_data:
            self.position_history = old_data["position_history"]
            self.ask_history = old_data["ask_history"]
            self.bid_history = old_data["bid_history"]

        self.position_history.append(position)
        self.ask_history.append(asks)
        self.bid_history.append(bids)

        for sequence in (self.position_history, self.ask_history, self.bid_history):
            if len(sequence) > 50:
                sequence.pop(0)

    def save(self) -> JSON:
        data = {
            "position_history": self.position_history,
            "ask_history": self.ask_history,
            "bid_history": self.bid_history,
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
    def position(self) -> int:
        return self.position_history[-1]

    @property
    def asks(self) -> list:
        """List of (price, quantity) for asks in increasing order of price.
        Quantities are positive here, unlike in class Order.
        """
        return sorted(self.ask_history[-1])

    @property
    def bids(self) -> list:
        """List of (price, quantity) for bids in decreasing order of price."""
        return sorted(self.bid_history[-1], reverse=True)

    @property
    def best_ask_price(self) -> int:
        return min(price for price, _ in self.asks)

    @property
    def best_bid_price(self) -> int:
        return max(price for price, _ in self.bids)

    @property
    def max_volume_ask_price(self) -> int:
        """Ask price of the order with the highest volume."""
        return max(self.asks, key=lambda t: t[1])[0]

    @property
    def max_volume_bid_price(self) -> int:
        """Bid price of the order with the highest volume."""
        return max(self.bids, key=lambda t: t[1])[0]

    @property
    def buy_capacity(self) -> int:
        """Maximum volume that can be bought without potentially exceeding position limit."""
        return self.limit - self.position - self.planned_buy_volume

    @property
    def sell_capacity(self) -> int:
        """Maximum volume that can be sold without potentially exceeding position limit."""
        return self.limit + self.position - self.planned_sell_volume

    def historic_mid_prices(self, window_size: int) -> list[float]:
        """List of mid-prices.
        More recent ones towards the end.
        mid_price := (max_volume_ask_price + max_volume_bid_price) // 2
        """
        mid_prices = []

        for i in range(-window_size, 0):
            ask = max(self.ask_history[i], key=lambda t: t[1])[0]
            bid = max(self.bid_history[i], key=lambda t: t[1])[0]
            mid_prices.append((ask + bid) / 2)

        return mid_prices

    def linear_regression(self, window_size: int) -> tuple[float, float, float]:
        mid_prices = self.historic_mid_prices(window_size)
        m, b = np.polyfit(range(window_size), mid_prices, 1)

        loss = 0
        for x, y in enumerate(mid_prices):
            loss += (y - m * x - b) ** 2

        return m, b, loss


class Strategy:
    @staticmethod
    def jmerle_style_market_making(
            product: Product,
            max_buy_price: int,
            min_sell_price: int,
    ) -> None:
        """
        Inspired from Jmerle's `MarketMakingStrategy`.
        https://github.com/jmerle/imc-prosperity-2/blob/master/src/submissions/round5.py

        BUY everything @ <= max_buy_price
        Use leftover capacity to BUY @ max_volume_bid_price + 1

        SELL everything @ >= min_sell_price
        Use leftover capacity to SELL @ max_volume_ask_price - 1

        Jmerle also has some "liquidation conditions".
        Excluded because they didn't seem to make any difference.
        """
        for price, quantity in product.asks:
            if price <= max_buy_price:
                quantity = min(quantity, product.buy_capacity)
                product.buy(price, quantity)

        price = min(max_buy_price, product.max_volume_bid_price + 1)
        product.buy(price, product.buy_capacity)

        for price, quantity in product.bids:
            if price >= min_sell_price:
                quantity = min(product.sell_capacity, quantity)
                product.sell(price, quantity)

        price = max(min_sell_price, product.max_volume_ask_price - 1)
        product.sell(price, product.sell_capacity)


class Trader:
    def __init__(self) -> None:
        # all historic data and upcoming Orders will be stored in Product instances
        self.products = {
            "RAINFOREST_RESIN": Product("RAINFOREST_RESIN", 50),
            "KELP": Product("KELP", 50),
            "SQUID_INK": Product("SQUID_INK", 50),
        }
        self.last_squid_buy = 0
        self.last_squid_sell = 0
        self.timestamp = 0

    def run(self, state: TradingState):
        # load old data from state.traderData + new data from state
        old_trader_data = json.loads(state.traderData) if state.traderData != "" else {}
        for symbol, product in self.products.items():
            product.load(state, old_trader_data.get(symbol, {}))

        # store all data in a dict for next iteration
        new_trader_data = dict()
        for symbol, product in self.products.items():
            new_trader_data[symbol] = product.save()

        self.last_squid_buy = new_trader_data.get("last_squid_buy", -10_000)
        self.last_squid_sell = new_trader_data.get("last_squid_buy", -10_000)
        self.timestamp = state.timestamp

        self.trade_rainforest_resin()
        # self.trade_kelp()
        # self.trade_squid_ink()

        result = {
            "RAINFOREST_RESIN": self.products["RAINFOREST_RESIN"].planned_orders,
            "KELP": self.products["KELP"].planned_orders,
            "SQUID_INK": self.products["SQUID_INK"].planned_orders,
        }
        conversions = 0

        new_trader_data["last_squid_buy"] = self.last_squid_buy

        logger = Logger()
        logger.flush(state, result, conversions, state.traderData)

        return result, 0, json.dumps(new_trader_data)

    def trade_rainforest_resin(self) -> None:
        resin = self.products["RAINFOREST_RESIN"]

        max_buy_price = 9998
        min_sell_price = 10_002

        Strategy.jmerle_style_market_making(
            product=resin,
            max_buy_price=max_buy_price,
            min_sell_price=min_sell_price,
        )

    def trade_kelp(self) -> None:
        """
        Use that parallel lines' theory to calculate prices.
        Strategy.jmerle_style_market_making() to place orders.
        """
        kelp = self.products["KELP"]

        f = (kelp.max_volume_ask_price + kelp.max_volume_bid_price + 3) / 2
        f = floor(f) if kelp.position > 0 else ceil(f)

        max_buy_price = (f - 2) - (kelp.position > kelp.limit * 0.25)
        min_sell_price = (f - 1) + (kelp.position < kelp.limit * -0.25)

        return Strategy.jmerle_style_market_making(
            product=kelp,
            max_buy_price=max_buy_price,
            min_sell_price=min_sell_price,
        )

    def trade_squid_ink(self) -> None:
        # squid = self.products["SQUID_INK"]
        #
        # if len(squid.position_history) < 15:
        #     return
        #
        # m, b, _ = squid.linear_regression(15)
        #
        # ideal = squid.historic_mid_prices(1)[0] + m
        #
        # if abs(m) <= 0.05:
        #     mid = (squid.max_volume_ask_price + squid.max_volume_bid_price + 1) / 2
        #     Strategy.jmerle_style_market_making(
        #         squid,
        #         max_buy_price=round(ideal - m),
        #         min_sell_price=round(ideal + m),
        #     )
        pass
