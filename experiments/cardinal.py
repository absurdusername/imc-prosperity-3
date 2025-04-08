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


logger = Logger()
from typing import Dict, List
from round1.datamodel import TradingState, Order
import collections
from collections import defaultdict
import numpy as np

class Trader:
    # Initialize position tracking
    position = {'RAINFOREST_RESIN': 0, 'KELP': 0}
    POSITION_LIMIT = {'RAINFOREST_RESIN': 20, 'KELP': 20}
    volume_traded = {'RAINFOREST_RESIN': 0, 'KELP': 0}
    cpnl = defaultdict(lambda: 0)
    
    # For KELP price tracking
    kelp_cache = []
    kelp_dim = 4
    
    def values_extract(self, order_dict, buy=0):
        tot_vol = 0
        best_val = -1
        mxvol = -1

        for ask, vol in order_dict.items():
            if(buy==0):
                vol *= -1
            tot_vol += vol
            if tot_vol > mxvol:
                mxvol = vol
                best_val = ask
        
        return tot_vol, best_val
    
    def compute_orders_rainforest_resin(self, product, order_depth, acc_bid, acc_ask):
        orders: list[Order] = []

        osell = collections.OrderedDict(sorted(order_depth.sell_orders.items()))
        obuy = collections.OrderedDict(sorted(order_depth.buy_orders.items(), reverse=True))

        sell_vol, best_sell_pr = self.values_extract(osell)
        buy_vol, best_buy_pr = self.values_extract(obuy, 1)

        cpos = self.position[product]

        # Buy orders when price is below true value
        for ask, vol in osell.items():
            if ask < 10000 and cpos < self.POSITION_LIMIT['RAINFOREST_RESIN']:
                order_for = min(-vol, self.POSITION_LIMIT['RAINFOREST_RESIN'] - cpos)
                cpos += order_for
                assert(order_for >= 0)
                orders.append(Order(product, ask, order_for))

        # Place limit buy orders
        if cpos < self.POSITION_LIMIT['RAINFOREST_RESIN']:
            bid_pr = min(best_buy_pr + 1, 9999)  # Buy below true value
            num = min(40, self.POSITION_LIMIT['RAINFOREST_RESIN'] - cpos)
            orders.append(Order(product, bid_pr, num))
            cpos += num
        
        cpos = self.position[product]
        
        # Sell orders when price is above true value
        for bid, vol in obuy.items():
            if bid > 10000 and cpos > -self.POSITION_LIMIT['RAINFOREST_RESIN']:
                order_for = max(-vol, -self.POSITION_LIMIT['RAINFOREST_RESIN']-cpos)
                cpos += order_for
                assert(order_for <= 0)
                orders.append(Order(product, bid, order_for))

        # Place limit sell orders
        if cpos > -self.POSITION_LIMIT['RAINFOREST_RESIN']:
            sell_pr = max(best_sell_pr - 1, 10001)  # Sell above true value
            num = max(-40, -self.POSITION_LIMIT['RAINFOREST_RESIN']-cpos)
            orders.append(Order(product, sell_pr, num))
            cpos += num

        return orders
    
    def compute_orders_kelp(self, product, order_depth):
        orders: list[Order] = []

        osell = collections.OrderedDict(sorted(order_depth.sell_orders.items()))
        obuy = collections.OrderedDict(sorted(order_depth.buy_orders.items(), reverse=True))

        sell_vol, best_sell_pr = self.values_extract(osell)
        buy_vol, best_buy_pr = self.values_extract(obuy, 1)
        
        # Calculate mid price
        mid_price = (best_sell_pr + best_buy_pr) / 2
        
        # Add to price history
        if len(self.kelp_cache) == self.kelp_dim:
            self.kelp_cache.pop(0)
        self.kelp_cache.append(mid_price)
        
        # Implement linear regression strategy
        if len(self.kelp_cache) >= 3:
            # Calculate linear regression
            x = np.array(range(len(self.kelp_cache))).reshape(-1, 1)
            y = np.array(self.kelp_cache).reshape(-1, 1)
            model = np.polyfit(x.flatten(), y.flatten(), 1)
            slope = model[0]
            
            # Calculate volatility for dynamic position sizing
            volatility = np.std(self.kelp_cache)
            
            # Current position
            cpos = self.position[product]
            
            # Calculate regression channel boundaries
            regression_line = np.poly1d(model)
            regression_values = regression_line(x.flatten())
            channel_width = volatility * 2
            upper_channel = regression_values[-1] + channel_width
            lower_channel = regression_values[-1] - channel_width
            
            # Trend strength - stronger slope means more aggressive positions
            trend_strength = min(abs(slope) * 10, 1.0)
            position_size = max(int(self.POSITION_LIMIT[product] * trend_strength), 5)
            
            # Trading strategy based on price position relative to regression channel
            if mid_price < lower_channel and slope > 0:
                # Price below channel in uptrend - strong buy signal
                for ask, vol in osell.items():
                    if cpos < self.POSITION_LIMIT[product]:
                        order_for = min(-vol, self.POSITION_LIMIT[product] - cpos)
                        cpos += order_for
                        orders.append(Order(product, ask, order_for))
                
                # Place aggressive limit buy orders
                if cpos < self.POSITION_LIMIT[product]:
                    bid_pr = best_buy_pr + 1
                    num = min(position_size, self.POSITION_LIMIT[product] - cpos)
                    orders.append(Order(product, bid_pr, num))
            
            elif mid_price > upper_channel and slope < 0:
                # Price above channel in downtrend - strong sell signal
                for bid, vol in obuy.items():
                    if cpos > -self.POSITION_LIMIT[product]:
                        order_for = max(-vol, -self.POSITION_LIMIT[product] - cpos)
                        cpos += order_for
                        orders.append(Order(product, bid, order_for))
                
                # Place aggressive limit sell orders
                if cpos > -self.POSITION_LIMIT[product]:
                    sell_pr = best_sell_pr - 1
                    num = max(-position_size, -self.POSITION_LIMIT[product] - cpos)
                    orders.append(Order(product, sell_pr, num))
            
            # Mean reversion strategy when price is at extremes
            elif mid_price > upper_channel and slope > 0:
                # Price above channel in uptrend - potential reversal
                # Take smaller position size for mean reversion
                mean_reversion_size = max(int(position_size * 0.5), 3)
                if cpos > -self.POSITION_LIMIT[product]:
                    sell_pr = best_sell_pr - 1
                    num = max(-mean_reversion_size, -self.POSITION_LIMIT[product] - cpos)
                    orders.append(Order(product, sell_pr, num))
            
            elif mid_price < lower_channel and slope < 0:
                # Price below channel in downtrend - potential reversal
                # Take smaller position size for mean reversion
                mean_reversion_size = max(int(position_size * 0.5), 3)
                if cpos < self.POSITION_LIMIT[product]:
                    bid_pr = best_buy_pr + 1
                    num = min(mean_reversion_size, self.POSITION_LIMIT[product] - cpos)
                    orders.append(Order(product, bid_pr, num))
            
            # Maximize volume with market making when price is within channel
            else:
                # Calculate optimal spread based on volatility
                spread = max(2, int(volatility))
                
                # Neutralize position if it's too extreme
                if abs(cpos) > self.POSITION_LIMIT[product] * 0.7:
                    # If position is too high, prioritize selling
                    if cpos > 0:
                        sell_pr = best_buy_pr
                        num = max(-min(abs(cpos), 10), -self.POSITION_LIMIT[product] - cpos)
                        orders.append(Order(product, sell_pr, num))
                    # If position is too low, prioritize buying
                    else:
                        bid_pr = best_sell_pr
                        num = min(min(abs(cpos), 10), self.POSITION_LIMIT[product] - cpos)
                        orders.append(Order(product, bid_pr, num))
                else:
                    # Market making with tight spreads to maximize volume
                    if cpos < self.POSITION_LIMIT[product] * 0.8:
                        bid_pr = best_buy_pr + 1
                        num = min(5, self.POSITION_LIMIT[product] - cpos)
                        orders.append(Order(product, bid_pr, num))
                    
                    if cpos > -self.POSITION_LIMIT[product] * 0.8:
                        sell_pr = best_sell_pr - 1
                        num = max(-5, -self.POSITION_LIMIT[product] - cpos)
                        orders.append(Order(product, sell_pr, num))
        
        return orders


        
    def run(self, state: TradingState) -> Dict[str, List[Order]]:
        """
        Only method required. It takes all buy and sell orders for all symbols as an input,
        and outputs a list of orders to be sent
        """
        # Initialize the method output dict as an empty dict
        result = {'RAINFOREST_RESIN': [], 'KELP': []}

        # Update positions
        for key, val in state.position.items():
            self.position[key] = val
        
        print()
        for key, val in self.position.items():
            print(f'{key} position: {val}')

        # Trading RAINFOREST_RESIN with fixed true value of 10,000
        if 'RAINFOREST_RESIN' in state.order_depths:
            order_depth = state.order_depths['RAINFOREST_RESIN']
            orders = self.compute_orders_rainforest_resin('RAINFOREST_RESIN', order_depth, 9999, 10001)
            result['RAINFOREST_RESIN'] += orders

        # Trading KELP with varying value
        if 'KELP' in state.order_depths:
            order_depth = state.order_depths['KELP']
            orders = self.compute_orders_kelp('KELP', order_depth)
            result['KELP'] += orders

        # Update trading statistics
        for product in state.own_trades.keys():
            for trade in state.own_trades[product]:
                if trade.timestamp != state.timestamp-100:
                    continue
                self.volume_traded[product] += abs(trade.quantity)
                if trade.buyer == "SUBMISSION":
                    self.cpnl[product] -= trade.quantity * trade.price
                else:
                    self.cpnl[product] += trade.quantity * trade.price

        # Calculate and display PnL
        totpnl = 0
        for product in state.order_depths.keys():
            settled_pnl = 0
            best_sell = min(state.order_depths[product].sell_orders.keys())
            best_buy = max(state.order_depths[product].buy_orders.keys())

            if self.position[product] < 0:
                settled_pnl += self.position[product] * best_buy
            else:
                settled_pnl += self.position[product] * best_sell
            totpnl += settled_pnl + self.cpnl[product]
            print(f"For product {product}, {settled_pnl + self.cpnl[product]}, {(settled_pnl+self.cpnl[product])/(self.volume_traded[product]+1e-20)}")

        print(f"Timestamp {state.timestamp}, Total PNL ended up being {totpnl}")
        print("End transmission")
        
        logger.flush(state, result, 0, "")
        return result, 0, ""
