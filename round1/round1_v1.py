from datamodel import *
from typing import TypeAlias

JSON: TypeAlias = dict[str, "JSON"] | list["JSON"] | str | int | float | bool | None


class ProductData:
    def __init__(self, symbol: str, position_limit: int):
        self.symbol = symbol
        self.position_limit = position_limit

        self.position = None
        self.planned_buy_volume = 0  # cumulative volume of BUY orders we're placing next iteration
        self.planned_sell_volume = 0

        # a new list of (price, quantity) pairs will be appended each iteration
        self.ask_history: list[list[tuple[int, int]]] = []
        self.bid_history: list[list[tuple[int, int]]] = []

    def load(self, state: TradingState, old_data: JSON) -> None:
        self.position = state.position[self.symbol]

        order_depth = state.order_depths[self.symbol]
        asks = list(order_depth.sell_orders.items())
        bids = list(order_depth.buy_orders.items())

        if old_data:
            self.ask_history = old_data["ask_history"]
            self.bid_history = old_data["bid_history"]

        self.ask_history.append(asks)
        self.bid_history.append(bids)

    def save(self) -> JSON:
        data = {
            "ask_history": self.ask_history,
            "bid_history": self.bid_history,
        }
        return data

    @property
    def asks(self) -> list:
        """Return list of (price, quantity) for asks."""
        return self.ask_history[-1]

    @property
    def bids(self) -> list:
        """Return list of (price, quantity) for bids."""
        return self.bid_history[-1]

    @property
    def best_ask_price(self) -> int:
        return min(price for price, _ in self.asks)

    @property
    def best_bid_price(self) -> int:
        return max(price for price, _ in self.bids)

    @property
    def max_volume_ask_price(self) -> int:
        """Return the ask price of the order with the highest volume."""
        return max(self.asks, key=lambda t: t[1])[0]

    @property
    def max_volume_bid_price(self) -> int:
        """Return the bid price of the order with the highest volume."""
        return max(self.bids, key=lambda t: t[1])[0]


class Strategy:
    @staticmethod
    def jmerle_style_market_making(product_data: ProductData, fair_price: int) -> list[Order]:
        pass


class Trader:
    def __init__(self) -> None:
        self.products = {
            "RAINFOREST_RESIN": ProductData("RAINFOREST_RESIN", 50),
            "KELP": ProductData("KELP", 50),
            "SQUID_INK": ProductData("SQUID_INK", 50),
        }

    def run(self, state: TradingState):
        # load old data from state.traderData + new data from state
        old_trader_data = json.loads(state.traderData)
        for symbol, product_data in self.products.items():
            product_data.load(state, old_trader_data[symbol])

        # store all data in a dict for next iteration
        new_trader_data = dict()
        for symbol, product_data in self.products.items():
            new_trader_data[symbol] = product_data.save()

        result = {
            "RAINFOREST_RESIN": self.rainforest_resin(),
            "KELP": self.kelp(),
            "SQUID_INK": self.squid_ink()
        }

        return result, 0, json.dumps(new_trader_data)

    def rainforest_resin(self) -> list[Order]:
        pass

    def kelp(self) -> list[Order]:
        pass

    def squid_ink(self) -> list[Order]:
        pass
