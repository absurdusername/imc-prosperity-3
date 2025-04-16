from datamodel import *
from typing import TypeAlias, Any

from math import floor, ceil, copysign, log, e
from statistics import mean, stdev, mode, median
from collections import deque

import itertools
import numpy as np

import json
from typing import Any, List, Dict, Tuple, Optional
import math
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
        history_size_limit (int): Minimum number of past iterations for which data is preserved.
            The length of position_history, ask_history, and bid_history are affected by this.
            Defaults to 1.

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

    def __init__(self, symbol: str, limit: int, history_size_limit: int = 1):
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
            if len(sequence) > 2 * self.history_size_limit:
                n = self.history_size_limit
                excess = len(sequence) - n

                sequence[:n] = sequence[-n:]
                for _ in range(excess):
                    sequence.pop()

    def save(self) -> JSON:
        # data = {
        #     "position_history": self.position_history,
        #     "ask_history": self.ask_history,
        #     "bid_history": self.bid_history,
        #     "mid_price_history": self.mid_price_history,
        # }

        # TEMPORARY FIX FOR PERFORMANCE
        data = {
            "position_history": [],
            "ask_history": [],
            "bid_history": [],
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
        return len(self.mid_price_history)

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

    @staticmethod
    def estimate_fair_value_LR(product: Product) -> float:
        """
        Fit a line to all available mid-prices.
        Return estimate for what the current mid-price *should have been* according to the line.
        """
        m, b = product.linear_regression()
        x = product.history_size  # iteration number

        fair_value = m * x + b
        return fair_value


class BasketGoodsTrader:
    """Trade all the Round 2 products."""
    def __init__(
            self,
            picnic_basket1: Product,
            picnic_basket2: Product,
            croissants: Product,
            jams: Product,
            djembes: Product
    ):
        self.picnic_basket1 = picnic_basket1
        self.picnic_basket2 = picnic_basket2
        self.croissants = croissants
        self.jams = jams
        self.djembes = djembes

        self.cached_fair_values: dict[Symbol, float] = {
            "CROISSANTS": 0,
            "PICNIC_BASKET1": 0,
            "PICNIC_BASKET2": 0,
        }

    def trade(self) -> None:
        self._trade_croissants()
        self._trade_picnic_basket1()
        self._trade_picnic_basket2()
        self._trade_jams()

    def _trade_croissants(self) -> None:
        fair_value = Strategy.estimate_fair_value_LR(self.croissants)
        self.cached_fair_values["CROISSANTS"] = fair_value

        max_buy_price = round(fair_value - 5)
        min_sell_price = round(fair_value + 5)

        Strategy.simple_market_making(
            product=self.croissants,
            max_buy_price=max_buy_price,
            min_sell_price=min_sell_price,
        )

    def _trade_picnic_basket1(self) -> None:
        f = lambda c, j, d: 6 * c + 3 * j + 1 * d

        sum_of_parts = f(self.croissants.mid_price, self.jams.mid_price, self.djembes.mid_price)
        fair_value = 0.55 * sum_of_parts + 0.45 * self.picnic_basket1.mid_price
        self.cached_fair_values["PICNIC_BASKET1"] = fair_value

        max_buy_price = round(fair_value - 6)
        min_sell_price = round(fair_value + 6)

        cp = self.croissants.mid_price_history[-100:]
        jp = self.jams.mid_price_history[-100:]
        dp = self.djembes.mid_price_history[-100:]
        pp = self.picnic_basket1.mid_price_history[-100:]

        premiums = [pp[i] - f(cp[i], jp[i], dp[i]) for i in range(len(pp))]

        if len(premiums) >= 2:
            mu, sigma = mean(premiums), stdev(premiums)

            max_buy_price = min(
                max_buy_price,
                round(self.picnic_basket1.max_volume_ask_price - 0.55 * sigma)
            )

            min_sell_price = max(
                min_sell_price,
                round(self.picnic_basket1.max_volume_bid_price + 0.55 * sigma),
            )

        Strategy.simple_market_making(
            product=self.picnic_basket1,
            max_buy_price=max_buy_price,
            min_sell_price=min_sell_price,
        )

    def _trade_picnic_basket2(self) -> None:
        f = lambda c, j: 4 * c + 2 * j
        fair_value = f(self.croissants.mid_price, self.jams.mid_price)
        self.cached_fair_values["PICNIC_BASKET2"] = fair_value

        max_buy_price = round(fair_value - 60)
        min_sell_price = round(fair_value - 20)

        Strategy.simple_market_making(
            product=self.picnic_basket2,
            max_buy_price=max_buy_price,
            min_sell_price=min_sell_price,
        )

    def _trade_jams(self) -> None:
        picnic_basket2_fair_value = self.cached_fair_values["PICNIC_BASKET2"]
        croissants_fair_value = self.cached_fair_values["CROISSANTS"]

        fair_value = (picnic_basket2_fair_value - 4 * croissants_fair_value) / 2
        self.cached_fair_values["JAMS"] = fair_value

        max_buy_price = round(fair_value - 14)
        min_sell_price = round(fair_value + 14)

        Strategy.simple_market_making(
            product=self.jams,
            max_buy_price=max_buy_price,
            min_sell_price=min_sell_price,
        )

    def _trade_djembes(self) -> None:
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


class Trader:
    def __init__(self) -> None:
        # all historic data and upcoming Orders will be stored in Product instances
        self.products: dict[Symbol, Product] = {
            "RAINFOREST_RESIN": Product("RAINFOREST_RESIN", 50),
            "KELP": Product("KELP", 50),
            "SQUID_INK": Product("SQUID_INK", 50, history_size_limit=10_000),

            "CROISSANTS": Product("CROISSANTS", 250, history_size_limit=10_000),
            "JAMS": Product("JAMS", 350, history_size_limit=100),
            "DJEMBES": Product("DJEMBES", 60, history_size_limit=100),

            "PICNIC_BASKET1": Product("PICNIC_BASKET1", 60, history_size_limit=100),
            "PICNIC_BASKET2": Product("PICNIC_BASKET2", 100),
        }

        self.basket_goods_trader = BasketGoodsTrader(
            picnic_basket1=self.products["PICNIC_BASKET1"],
            picnic_basket2=self.products["PICNIC_BASKET2"],
            croissants=self.products["CROISSANTS"],
            jams=self.products["JAMS"],
            djembes=self.products["DJEMBES"],
        )

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

    def run(self, state: TradingState):
        ambuj_trader_data, lakshay_trader_data = state.traderData.split("|-|-|") if state.traderData != "" else ("", "")
        # load old data from state.traderData + new data from state

        old_trader_data = json.loads(ambuj_trader_data) if ambuj_trader_data != "" else {}
        for symbol, product in self.products.items():
            product.load(state, old_trader_data.get(symbol, {}))
            # product.load(state, None)

        # store all data in a dict for next iteration
        new_trader_data = dict()
        for symbol, product in self.products.items():
            new_trader_data[symbol] = product.save()

        self.trade_rainforest_resin()
        self.trade_kelp()
        self.trade_squid_ink()

        self.basket_goods_trader.trade()

        result = {
            symbol: product.planned_orders
            for symbol, product in self.products.items()
        }
        ambuj_trader_data_to_save = json.dumps(new_trader_data)

        # self.logger.flush(state, result, conversions, state.traderData)
        #
        # return result, 0, json.dumps(new_trader_data)

        ### VOUCHER STUFF

        all_strategy_states = {}
        # trader_data_from_state = state.traderData if state.traderData else "{}"
        try:
            all_strategy_states = json.loads(lakshay_trader_data)
        except json.JSONDecodeError as e:
            self.logger.print(f"E: Load traderData JSON failed: {e}. Raw: '{lakshay_trader_data[:100]}...'");
            all_strategy_states = {}
        except Exception as e:
            self.logger.print(f"E: Load traderData failed unexpectedly: {e}")
            all_strategy_states = {}

        rock_price = self.calculate_mid_price("VOLCANIC_ROCK", state)
        tte = 0.0
        if rock_price is not None:
            tte = self.calculate_time_to_expiry(state.timestamp)

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

        lakshay_trader_data_to_save = json.dumps(updated_all_strategy_states)
        conversions = 0

        complete_trader_data_to_save = ambuj_trader_data_to_save + "|-|-|" + lakshay_trader_data_to_save

        self.logger.flush(state, result, conversions, complete_trader_data_to_save)
        #
        return result, conversions, complete_trader_data_to_save

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

        fair_value = Strategy.estimate_fair_value_LR(squid_ink)

        max_buy_price = round(fair_value - 7)
        min_sell_price = round(fair_value + 7)

        Strategy.simple_market_making(
            product=squid_ink,
            max_buy_price=max_buy_price,
            min_sell_price=min_sell_price,
        )

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

## VOUCHERS



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