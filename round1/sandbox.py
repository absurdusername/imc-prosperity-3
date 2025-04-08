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


class ProductData:
    def __init__(self, symbol: str, limit: int):
        self.symbol: str = symbol
        self.limit: int = limit

        self.position: int = None

        # a new list of [price, quantity] pairs will be appended each iteration
        # quantity for both ask_history and bid_history are stored as positive values!
        self.ask_history: list[list[list[int, int]]] = []
        self.bid_history: list[list[list[int, int]]] = []

    def load(self, state: TradingState, old_data: JSON) -> None:
        self.position = state.position.get(self.symbol, 0)

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

    @property
    def asks(self) -> list:
        """List of (price, quantity) for asks in incresaing order of price.
        Quantities are positive here, unlike in class Order."""
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
        return self.limit - self.position
    
    @property
    def sell_capacity(self) -> int:
        """Maximum volume that can be sold without potentially exceeding position limit."""
        return self.limit + self.position


class Strategy:
    @staticmethod
    def jmerle_style_market_making(
        product_data: ProductData, 
        max_buy_price: int,
        min_sell_price: int
    ) -> list[Order]:
        """ 
        Inspired from Jmerle's `MarketMakingStrategy`.
        https://github.com/jmerle/imc-prosperity-2/blob/master/src/submissions/round5.py

        BUY everything @ <= max_buy_price
        Use leftover capacity to BUY @ max_volume_bid_price

        SELL everything @ >= min_sell_price
        Use leftover capacity to SELL @ max_volume_ask_price
        """
        
        orders = []

        to_buy = product_data.limit - product_data.position
        to_sell = product_data.limit + product_data.position

        for price, quantity in product_data.asks:
            if to_buy and price <= max_buy_price:
                quantity = min(quantity, product_data.buy_capacity)
                orders.append(Order(product_data.symbol, price, quantity))
                to_buy -= quantity

        if to_buy > 0:
            price = min(max_buy_price, product_data.max_volume_bid_price)
            orders.append(Order(product_data.symbol, price, to_buy))

        for price, quantity in product_data.bids:
            if to_sell > 0 and price >= min_sell_price:
                quantity = min(to_sell, quantity)
                orders.append(Order(product_data.symbol, price, -quantity))
                to_sell -= quantity
        
        if to_sell > 0:
            price = max(min_sell_price, product_data.max_volume_ask_price)
            orders.append(Order(product_data.symbol, price, -to_sell))
        
        return orders


class Trader:
    def __init__(self) -> None:
        self.products = {
            "RAINFOREST_RESIN": ProductData("RAINFOREST_RESIN", 50),
            "KELP": ProductData("KELP", 50),
            # "SQUID_INK": ProductData("SQUID_INK", 50),
        }

    def run(self, state: TradingState):
        # load old data from state.traderData + new data from state
        old_trader_data = json.loads(state.traderData) if state.traderData != "" else {}
        for symbol, product_data in self.products.items():
            product_data.load(state, old_trader_data.get(symbol, {}))

        # store all data in a dict for next iteration
        new_trader_data = dict()
        for symbol, product_data in self.products.items():
            new_trader_data[symbol] = product_data.save()

        result = {}
        conversions = 0

        result["RAINFOREST_RESIN"] = self.rainforest_resin()
        result["KELP"] = self.kelp()
        # result["SQUID_INK"] = self.squid_ink()

        logger = Logger()
        logger.flush(state, result, conversions, state.traderData)

        return result, 0, json.dumps(new_trader_data)

    def rainforest_resin(self) -> list[Order]:
        """
        BUY at 9998 and SELL at 10,002.

        Outstanding asks below 10k will always be at 9998.
        And oustanding bids above 10k will always be at 10,002.
        Verified on the 3 days data.

        Only exception are trades. They may happen at +/-2 
        """
        resin_data = self.products["RAINFOREST_RESIN"]

        to_buy = resin_data.limit - resin_data.position
        to_sell = resin_data.limit + resin_data.position

        buy_cheap = min(to_buy, 4)
        buy_ok = to_buy - buy_cheap

        sell_exp = min(to_sell, 4)
        sell_ok = to_sell - sell_exp

        return [
            Order(resin_data.symbol, 9996, buy_cheap),
            Order(resin_data.symbol, 9998, buy_ok),

            Order(resin_data.symbol, 10_002, -sell_ok),
            Order(resin_data.symbol, 10_004, -sell_exp),
        ]

    def kelp(self) -> list[Order]:
        """
        That parallel lines theory.
        """
        kelp_data = self.products["KELP"]

        f = (kelp_data.max_volume_ask_price + kelp_data.max_volume_bid_price + 3) // 2

        max_buy_price = (f - 2) - (kelp_data.position > kelp_data.limit * 0.25)
        min_sell_price = (f - 1) + (kelp_data.position < kelp_data.limit * -0.25)

        return Strategy.jmerle_style_market_making(
            product_data=kelp_data,
            max_buy_price=max_buy_price,
            min_sell_price=min_sell_price
        )

    def squid_ink(self) -> list[Order]:
        return []
