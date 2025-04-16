

class Trader:
    def __init__(self):
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

        # Configure Voucher Strategies (DISABLED - Using Directional for 10k now)
        # iv_vouchers = {
        #     "VOLCANIC_ROCK_VOUCHER_10000": 10000, 
        # }
        # default_iv_voucher_params = {"window": 50, "entry_z": 1.5, "exit_z": 0.3, "trade_size": 50, "min_time_premium": 0.1} 
        # iv_voucher_params = {
        #     10000: {"window": 50, "entry_z": 1.5, "exit_z": 0.1, "trade_size": 50, "min_time_premium": 0.1}, # Reverted size to 50
        # }
        # for symbol, strike_price in iv_vouchers.items(): 
        #     params = iv_voucher_params.get(strike_price, default_iv_voucher_params) 
        #     strat_name = f"VoucherIV_MR_{strike_price}"
        #     self.strategies[strat_name] = VoucherStrategy(
        #         logger=self.logger,
        #         target_voucher=symbol,
        #         strike=strike_price,
        #         window=params["window"],
        #         entry_z=params["entry_z"],
        #         exit_z=params["exit_z"],
        #         trade_size=params["trade_size"],
        #         min_time_premium=params["min_time_premium"]
        #     )

        # Configure Rock Strategy
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

    def run(self, state: TradingState) -> Tuple[Dict[Symbol, List[Order]], int, str]:
        all_strategy_states = {}
        trader_data_from_state = state.traderData if state.traderData else "{}"
        try:
            all_strategy_states = json.loads(trader_data_from_state)
        except json.JSONDecodeError as e:
            self.logger.print(f"E: Load traderData JSON failed: {e}. Raw: '{trader_data_from_state[:100]}...'");
            all_strategy_states = {}
        except Exception as e:
            self.logger.print(f"E: Load traderData failed unexpectedly: {e}")
            all_strategy_states = {}

        rock_price = self.calculate_mid_price("VOLCANIC_ROCK", state)
        tte = 0.0
        if rock_price is not None:
            tte = self.calculate_time_to_expiry(state.timestamp)

        result: Dict[Symbol, List[Order]] = {}
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

        trader_data_to_save = json.dumps(updated_all_strategy_states)
        conversions = 0

        self.logger.flush(state, result, conversions, trader_data_to_save)

        return result, conversions, trader_data_to_save