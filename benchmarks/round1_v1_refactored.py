from datamodel import *
from typing import TypeAlias, Any

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
    def __init__(self, symbol: str, limit: int):
        self.symbol: str = symbol
        self.limit: int = limit
        self.position: int = 0

        # a new list of [price, quantity] pairs will be appended each iteration
        # quantity for both ask_history and bid_history are stored as positive values!
        self.ask_history: list[list[list[int, int]]] = []
        self.bid_history: list[list[list[int, int]]] = []

        # to store new Orders we will place
        self.planned_orders: list[Order] = []
        self.planned_buy_volume: int = 0
        self.planned_sell_volume: int = 0

    def load(self, state: TradingState, old_data: JSON) -> None:
        self.position = state.position.get(self.symbol, 0)

        # Don't remove the following lines, this class is not re-instantiated every time
        self.planned_orders: list[Order] = []
        self.planned_buy_volume: int = 0
        self.planned_sell_volume: int = 0

        order_depth = state.order_depths[self.symbol]
        asks = [[price, -quantity] for price, quantity in order_depth.sell_orders.items()]
        bids = [[price, quantity] for price, quantity in order_depth.buy_orders.items()]
        # the quantity for asks in order_depth is always negative, so the sign was flipped here

        if old_data:
            self.ask_history = old_data["ask_history"]
            self.bid_history = old_data["bid_history"]

        self.ask_history.append(asks)
        self.bid_history.append(bids)

        if len(self.ask_history) > 50:
            self.ask_history.pop(0)

        if len(self.bid_history) > 50:
            self.bid_history.pop(0)

    def save(self) -> JSON:
        data = {
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


class Strategy:
    @staticmethod
    def jmerle_style_market_making(
            product: Product,
            max_buy_price: int,
            min_sell_price: int
    ) -> None:
        """
        Inspired from Jmerle's `MarketMakingStrategy`.
        https://github.com/jmerle/imc-prosperity-2/blob/master/src/submissions/round5.py

        BUY everything @ <= max_buy_price
        Use leftover capacity to BUY @ max_volume_bid_price + 1

        SELL everything @ >= min_sell_price
        Use leftover capacity to SELL @ max_volume_ask_price - 1
        """
        to_sell = product.limit + product.position

        for price, quantity in product.asks:
            if price <= max_buy_price:
                quantity = min(quantity, product.buy_capacity)
                product.buy(price, quantity)

        price = min(max_buy_price, product.max_volume_bid_price + 1)
        product.buy(price, product.buy_capacity)

        for price, quantity in product.bids:
            if price >= min_sell_price:
                quantity = min(to_sell, quantity)
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

    def run(self, state: TradingState):
        # load old data from state.traderData + new data from state
        old_trader_data = json.loads(state.traderData) if state.traderData != "" else {}
        for symbol, product in self.products.items():
            product.load(state, old_trader_data.get(symbol, {}))

        # store all data in a dict for next iteration
        new_trader_data = dict()
        for symbol, product in self.products.items():
            new_trader_data[symbol] = product.save()

        self.trade_rainforest_resin()
        self.trade_kelp()

        result = {
            "RAINFOREST_RESIN": self.products["RAINFOREST_RESIN"].planned_orders,
            "KELP": self.products["KELP"].planned_orders,
            "SQUID_INK": self.products["SQUID_INK"].planned_orders,
        }
        conversions = 0

        logger = Logger()
        logger.flush(state, result, conversions, state.traderData)

        return result, 0, json.dumps(new_trader_data)

    def trade_rainforest_resin(self) -> None:
        """
        BUY 2 items @ 9996 and SELL 2 items @ 10,004.
        Use leftover capacity to BUY and SELL @ 9998 and 10,002 respectively.
        """
        resin = self.products["RAINFOREST_RESIN"]

        buy_cheap = min(resin.buy_capacity, 2)
        buy_ok = resin.buy_capacity - buy_cheap

        sell_exp = min(resin.sell_capacity, 2)
        sell_ok = resin.sell_capacity - sell_exp

        resin.buy(9996, buy_cheap)
        resin.buy(9998, buy_ok)

        resin.sell(10_002, sell_ok)
        resin.sell(10_004, sell_exp)

    def trade_kelp(self) -> None:
        """
        Use that parallel lines' theory to calculate prices.
        Strategy.jmerle_style_market_making() to place orders.
        """
        kelp = self.products["KELP"]

        f = (kelp.max_volume_ask_price + kelp.max_volume_bid_price + 3) // 2

        max_buy_price = (f - 2) - (kelp.position > kelp.limit * 0.25)
        min_sell_price = (f - 1) + (kelp.position < kelp.limit * -0.25)

        return Strategy.jmerle_style_market_making(
            product=kelp,
            max_buy_price=max_buy_price,
            min_sell_price=min_sell_price
        )

    def squid_ink(self) -> None:
        pass
