import tkinter as tk
from tkinter import ttk, messagebox, scrolledtext
import threading
from time import sleep
from binance.um_futures import UMFutures
from binance.error import ClientError
import ta
import pandas as pd

from backtesting import Backtest, Strategy
# from backtesting.lib import crossover
# We'll assume GOOG, SMA are for testing the backtesting.py library itself,
# and might not be directly needed for the user's strategy.
# from backtesting.test import SMA, GOOG
# pandas is already imported
# ta is already imported, ensure it's there.


# Placeholder function for klines_extended
def klines_extended(symbol, timeframe, interval_days):
    global client, bot_running, root # UMFutures client, bot_running and root for UI checks
    if not client:
        error_msg = "Error: Binance client not initialized for klines_extended."
        print(error_msg)
        return pd.DataFrame(columns=['Open', 'High', 'Low', 'Close', 'Volume']), error_msg

    limit_per_call = 1000

    end_time = int(pd.Timestamp.now(tz='UTC').timestamp() * 1000)
    start_time_ms = end_time - (interval_days * 24 * 60 * 60 * 1000)

    all_klines_raw = []
    current_start_time = start_time_ms

    print(f"Fetching klines for {symbol} from {pd.to_datetime(start_time_ms, unit='ms')} to {pd.to_datetime(end_time, unit='ms')}")

    while True:
        # Check if the bot is stopped (if running live) or if UI is closed (if running from UI but bot not started)
        # This is a proxy to allow breaking long fetches if the application is shutting down.
        # `root` check is for when backtesting is run before starting the live bot.
        # `bot_running` is for when the live bot is active (though backtesting while live bot runs is not typical).
        is_ui_active = root and root.winfo_exists()
        if not bot_running and not is_ui_active: # If bot not running AND UI is not active
             print("klines_extended: Bot and UI stopped, aborting fetch.")
             break
        try:
            kl = client.klines(symbol=symbol, interval=timeframe, startTime=current_start_time, limit=limit_per_call)
            if not kl:
                break 
            
            all_klines_raw.extend(kl)
            
            last_kline_ts = kl[-1][0]
            current_start_time = last_kline_ts + 1 

            if len(kl) < limit_per_call: 
                break
            if current_start_time >= end_time: 
                break
            
            # sleep(0.1) # Optional delay

        except ClientError as ce:
            print(f"ClientError in klines_extended for {symbol}: {ce}")
            return pd.DataFrame(columns=['Open', 'High', 'Low', 'Close', 'Volume']), f"ClientError: {ce}"
        except Exception as e:
            print(f"Generic error in klines_extended for {symbol}: {e}")
            return pd.DataFrame(columns=['Open', 'High', 'Low', 'Close', 'Volume']), f"Generic error: {e}"
    
    if not all_klines_raw:
        error_msg = f"No kline data fetched for {symbol} with timeframe {timeframe} for {interval_days} days."
        print(error_msg)
        return pd.DataFrame(columns=['Open', 'High', 'Low', 'Close', 'Volume']), error_msg

    df = pd.DataFrame(all_klines_raw)
    df = df.iloc[:, :6]
    df.columns = ['Time', 'Open', 'High', 'Low', 'Close', 'Volume']
    df['Time'] = pd.to_datetime(df['Time'], unit='ms')
    df = df.set_index('Time')
    df = df.astype(float)
    
    df = df[~df.index.duplicated(keep='first')]
    df = df.sort_index()

    print(f"Fetched {len(df)} klines for {symbol} ({timeframe}, {interval_days} days). From {df.index.min()} to {df.index.max()}")
    return df, None

# Indicator helper functions for backtesting
# RSI indicator function. Returns dataframe
def rsi_bt(df_series, period=14): # Renamed to rsi_bt to avoid conflict if app.py has other rsi
    return ta.momentum.RSIIndicator(pd.Series(df_series), window=period).rsi()

def ema_bt(df_series, period=200): # Renamed to ema_bt
    return ta.trend.EMAIndicator(pd.Series(df_series), period).ema_indicator()

def macd_bt(df_series): # Renamed to macd_bt
    return ta.trend.MACD(pd.Series(df_series)).macd()

def signal_h_bt(df_series): # Renamed to signal_h_bt
    return ta.volatility.BollingerBands(pd.Series(df_series)).bollinger_hband()

def signal_l_bt(df_series): # Renamed to signal_l_bt
    return ta.volatility.BollingerBands(pd.Series(df_series)).bollinger_lband()

# Backtest Strategy Wrapper Class
class BacktestStrategyWrapper(Strategy):
    # Default parameters, will be overridden by UI inputs later
    user_tp = 0.03  # Default, will be set from UI
    user_sl = 0.02  # Default, will be set from UI
    
    # Strategy-specific parameters (example for the provided strategy)
    ema_period = 200
    rsi_period = 14

    def init(self):
        # price = self.data.Close  # Unused variable
        self.rsi = self.I(rsi_bt, self.data.Close, self.rsi_period)
        self.macd = self.I(macd_bt, self.data.Close)
        self.ema = self.I(ema_bt, self.data.Close, self.ema_period)
        self.bol_h = self.I(signal_h_bt, self.data.Close)
        self.bol_l = self.I(signal_l_bt, self.data.Close)

    def next(self):
        price = float(self.data.Close[-1])
        if not self.position:
            if self.rsi[-1] < 30: # Corrected from rsi[-2] to rsi[-1] for typical backtesting.py usage
                self.buy(size=0.02, tp=(1 + self.user_tp) * price, sl=(1 - self.user_sl) * price)
        # Corrected sell condition logic based on typical RSI strategy
        # Original had "if not self.position and self.rsi[-2] > 70:" which is for entering short.
        # Assuming it's an exit for a long or entry for a short. Let's stick to the original for now.
        # Re-evaluating the provided strategy: it's for entering new positions, not exiting.
            elif self.rsi[-1] > 70: # Corrected from rsi[-2] to rsi[-1]
                self.sell(size=0.02, tp=(1 - self.user_tp) * price, sl=(1 + self.user_sl) * price)

# Function to execute backtest
def execute_backtest(strategy_id_for_backtest, symbol, timeframe, interval_days, ui_tp, ui_sl):
    print(f"Executing backtest for Strategy ID {strategy_id_for_backtest} on {symbol} ({timeframe}, {interval_days} days)")
    print(f"TP: {ui_tp*100}%, SL: {ui_sl*100}%")

    kl_df, klines_error_msg = klines_extended(symbol, timeframe, interval_days)

    if klines_error_msg:
        return klines_error_msg, None, None # Propagate error message
        
    if kl_df.empty or len(kl_df) < 50: # Basic check for enough data
        print("Error: Not enough kline data for backtest after fetching.")
        return "Not enough kline data for backtest.", None, None

    # Here, you would ideally select or parameterize the Strategy class
    # For now, we use BacktestStrategyWrapper directly.
    # Future: Pass strategy-specific parameters (ema_period, rsi_period) if needed
    
    # Update the class variables for TP/SL before instantiating Backtest
    BacktestStrategyWrapper.user_tp = ui_tp
    BacktestStrategyWrapper.user_sl = ui_sl

    bt = Backtest(kl_df, BacktestStrategyWrapper, cash=10000, margin=1/10, commission=0.0007)
    try:
        stats = bt.run()
        print("Backtest completed.")
        # print(stats) # Stats can be very verbose, printed later or in UI
    except Exception as e:
        print(f"Error during backtest bt.run(): {e}")
        return f"Error during backtest simulation: {e}", None, None

    plot_error_msg = None
    try:
        # Plotting might still fail even if simulation runs, e.g., due to Matplotlib issues or specific data conditions
        bt.plot(open_browser=False) 
    except Exception as e_plot:
        print(f"Error during bt.plot(): {e_plot}")
        plot_error_msg = f"Plotting Error: {e_plot}. Statistics are still available."
        
    return stats, bt, plot_error_msg


# Attempt to import pandas_ta, will be checked in SuperTrend function
try:
    import pandas_ta as pta
    PANDAS_TA_AVAILABLE = True
except ImportError:
    PANDAS_TA_AVAILABLE = False
    print("pandas_ta library not found. Manual SuperTrend will be used if SuperTrend is part of strategy.")

# Import keys
from keys import api_testnet, secret_testnet
try:
    from keys import api_mainnet, secret_mainnet
except ImportError:
    api_mainnet = None
    secret_mainnet = None
    print("Mainnet API keys (api_mainnet, secret_mainnet) not found in keys.py.")

# Global variables for bot state and client
client = None
current_env = "testnet"
bot_running = False
bot_thread = None

# GUI Variables
root = None
status_var = None
current_env_var = None
start_button = None
stop_button = None
testnet_radio = None
mainnet_radio = None
balance_var = None
positions_text_widget = None
history_text_widget = None
# Backtesting UI Variables
backtest_symbol_var = None
backtest_timeframe_var = None
backtest_interval_var = None
backtest_tp_var = None
backtest_sl_var = None
backtest_selected_strategy_var = None
backtest_results_text_widget = None
backtest_run_button = None # Added for enabling/disabling


# Tkinter StringVars for parameters
account_risk_percent_var = None
tp_percent_var = None
sl_percent_var = None
leverage_var = None
qty_concurrent_positions_var = None
local_high_low_lookback_var = None
margin_type_var = None # For Margin Type
target_symbols_var = None # For Target Symbols CSV
activity_status_var = None # For bot activity display
selected_strategy_var = None # For strategy selection

# GUI Variables for Strategy Checkboxes
strategy_checkbox_vars = {}

# Entry widgets (to be made global for enabling/disabling)
account_risk_percent_entry = None
tp_percent_entry = None
sl_percent_entry = None
leverage_entry = None
qty_concurrent_positions_entry = None
local_high_low_lookback_entry = None
target_symbols_entry = None # For Target Symbols CSV Entry
margin_type_isolated_radio = None # For Margin Type Radio
margin_type_cross_radio = None # For Margin Type Radio
strategy_radio_buttons = [] # List to hold strategy radio buttons
params_widgets = [] # List to hold all parameter input widgets (will extend with strategy_radio_buttons)
conditions_text_widget = None # For displaying signal conditions


# --- Backtesting UI Command and Threading Logic ---
def run_backtest_command():
    global backtest_symbol_var, backtest_timeframe_var, backtest_interval_var
    global backtest_tp_var, backtest_sl_var, backtest_selected_strategy_var
    global backtest_results_text_widget, STRATEGIES, root, status_var, backtest_run_button

    if backtest_run_button:
        backtest_run_button.config(state=tk.DISABLED)

    try:
        symbol = backtest_symbol_var.get().strip().upper()
        timeframe = backtest_timeframe_var.get().strip()
        interval_str = backtest_interval_var.get().strip()
        tp_str = backtest_tp_var.get().strip()
        sl_str = backtest_sl_var.get().strip()
        selected_strategy_name = backtest_selected_strategy_var.get()

        if not all([symbol, timeframe, interval_str, tp_str, sl_str, selected_strategy_name]):
            messagebox.showerror("Input Error", "All backtest fields are required.")
            return

        interval_days = int(interval_str)
        # Convert TP/SL from percentage string (e.g., "3.0") to float (e.g., 0.03)
        tp_percentage = float(tp_str) / 100.0
        sl_percentage = float(sl_str) / 100.0

        if not (0 < tp_percentage < 1 and 0 < sl_percentage < 1):
            messagebox.showerror("Input Error", "TP and SL percentages must be between 0 and 100 (exclusive of 0, inclusive of values that result in <1 after division by 100). E.g., 0.01 to 0.99 after conversion.")
            return
        
        strategy_id_for_backtest = None
        for id, name in STRATEGIES.items():
            if name == selected_strategy_name:
                strategy_id_for_backtest = id
                break
        
        if strategy_id_for_backtest is None: # Should not happen with Combobox
            messagebox.showerror("Input Error", "Invalid strategy selected.")
            return

        # Clear previous results
        if backtest_results_text_widget and root and root.winfo_exists():
            backtest_results_text_widget.config(state=tk.NORMAL)
            backtest_results_text_widget.delete('1.0', tk.END)
            backtest_results_text_widget.insert(tk.END, f"Starting backtest for {symbol}, Strategy: {selected_strategy_name}...\n")
            backtest_results_text_widget.config(state=tk.DISABLED)
        if status_var and root and root.winfo_exists():
            status_var.set("Backtest: Initializing...")


        thread = threading.Thread(target=perform_backtest_in_thread, 
                                  args=(strategy_id_for_backtest, symbol, timeframe, interval_days, tp_percentage, sl_percentage),
                                  daemon=True)
        thread.start()

    except ValueError:
        messagebox.showerror("Input Error", "Invalid number format for Interval, TP, or SL.")
        if backtest_run_button: backtest_run_button.config(state=tk.NORMAL) # Re-enable on error
    except Exception as e:
        messagebox.showerror("Error", f"An unexpected error occurred: {e}")
        if backtest_results_text_widget and root and root.winfo_exists():
            backtest_results_text_widget.config(state=tk.NORMAL)
            backtest_results_text_widget.insert(tk.END, f"\nError: {e}")
            backtest_results_text_widget.config(state=tk.DISABLED)
        if backtest_run_button: backtest_run_button.config(state=tk.NORMAL) # Re-enable on error

def perform_backtest_in_thread(strategy_id, symbol, timeframe, interval_days, tp, sl):
    global backtest_results_text_widget, root, status_var, backtest_run_button
    
    if status_var and root and root.winfo_exists():
        root.after(0, lambda: status_var.set("Backtest: Fetching kline data..."))

    stats_output, bt_object, plot_error_message = execute_backtest(strategy_id, symbol, timeframe, interval_days, tp, sl)
    
    if status_var and root and root.winfo_exists():
        if isinstance(stats_output, str) or plot_error_message : # If error string returned or plot error
             root.after(0, lambda: status_var.set("Backtest: Error encountered."))
        else:
             root.after(0, lambda: status_var.set("Backtest: Simulation complete. Preparing results..."))


    def update_ui_with_results():
        final_status_message = "Backtest: Completed."
        print("DEBUG: Entered update_ui_with_results function.") # New

        if backtest_results_text_widget and root and root.winfo_exists():
            backtest_results_text_widget.config(state=tk.NORMAL)
            backtest_results_text_widget.delete('1.0', tk.END) # Clear "Starting..." or previous debug messages
            backtest_results_text_widget.insert(tk.END, "DEBUG: Initial test write to results widget successful.\n") # Test write
            print("DEBUG: Successfully performed initial test write to widget.")
        else:
            print("DEBUG: backtest_results_text_widget is None or not available at the start of update_ui_with_results.")
            # If widget is not available, further UI updates for it are pointless.
            # Consider how to handle this, maybe return or log. For now, printing is fine.

        print(f"DEBUG: Type of stats_output: {type(stats_output)}")
        if stats_output is not None:
            print(f"DEBUG: Content of stats_output (first 500 chars): {str(stats_output)[:500]}")
        else:
            print("DEBUG: stats_output is None.")

        if backtest_results_text_widget and root and root.winfo_exists(): # Check again before complex logic
            # The original logic for displaying stats or errors:
            if isinstance(stats_output, pd.Series):
                stats_str = "Backtest Results:\n"
                stats_str += "--------------------\n"
                for index, value in stats_output.items():
                    if index in ['_equity_curve', '_trades']:
                        continue
                    if isinstance(value, float):
                        stats_str += f"{index}: {value:.2f}\n"
                    else:
                        stats_str += f"{index}: {value}\n"
                print(f"DEBUG: Formatted stats_str (first 500 chars): {stats_str[:500]}") # New print
                backtest_results_text_widget.insert(tk.END, stats_str) # Appending after initial debug message
            elif isinstance(stats_output, str):
                print(f"DEBUG: stats_output is a string (error message): {stats_output}") # New print
                backtest_results_text_widget.insert(tk.END, stats_output) # Appending
            elif stats_output is not None:
                stats_str = str(stats_output)
                print(f"DEBUG: stats_output is other non-None type, converting to str: {stats_str[:500]}") # New print
                backtest_results_text_widget.insert(tk.END, stats_str) # Appending
            
            print("DEBUG: Attempted to insert main content into widget.")

            if plot_error_message:
                print(f"DEBUG: Plot error message: {plot_error_message}") # New print
                backtest_results_text_widget.insert(tk.END, f"\n\n{plot_error_message}")
            
            if bt_object and not plot_error_message:
                # ... (plotting logic, potentially add debug prints around bt.plot() if suspected) ...
                backtest_results_text_widget.insert(tk.END, "\n\nPlot window should have opened (if data was sufficient).")
            
            if stats_output is None and not isinstance(stats_output, str):
                print("DEBUG: stats_output is None and not an error string. Displaying no stats message.") # New
                backtest_results_text_widget.insert(tk.END, "Backtest did not return statistics or an error message.")
            
            backtest_results_text_widget.config(state=tk.DISABLED)
        else:
            print("DEBUG: backtest_results_text_widget became unavailable before main content insertion.")

        if status_var and root and root.winfo_exists():
            root.after(0, lambda: status_var.set(final_status_message)) # final_status_message needs to be set based on outcomes
        if backtest_run_button and root and root.winfo_exists(): 
            backtest_run_button.config(state=tk.NORMAL)
        print("DEBUG: Exiting update_ui_with_results function.") # New

    if root and root.winfo_exists(): 
        root.after(0, update_ui_with_results)
    else: # Fallback if root is not available (e.g., if app is closing)
        print("UI update skipped: Root window not available.")
        if isinstance(stats_output, str): print(stats_output)
        elif stats_output: print(str(stats_output))


# Binance API URLs
BINANCE_MAINNET_URL = "https://fapi.binance.com"
BINANCE_TESTNET_URL = "https://testnet.binancefuture.com"

# Global Configuration Variables
STRATEGIES = {
    0: "Original Scalping",
    1: "EMA Cross + SuperTrend",
    2: "Bollinger Band Mean-Reversion",
    3: "VWAP Breakout Momentum",
    4: "MACD Divergence + Pivot-Point",
    5: "New RSI-Based Strategy" # New strategy added
}
ACTIVE_STRATEGY_ID = 0 # Default to original

# Strategy-specific state variables
strategy1_cooldown_active = False # For EMA Cross + SuperTrend: True if cooling down after a loss
strategy1_last_trade_was_loss = False # Tracks if the last trade for strategy 1 was a loss
# Removed 'entry_candle_klines_count' from S1 info, will rely on dynamic kline fetching for timeout
strategy1_active_trade_info = {'symbol': None, 'entry_time': None, 'entry_price': None, 'position_qty': 0, 'side': None, 'sl_order_id': None, 'tp_order_id': None}

strategy2_active_trades = [] # For Bollinger Bands: List of dicts, e.g., [{'symbol': 'BTCUSDT', 'entry_time': ...}, ...]

strategy3_active_trade_info = {'symbol': None, 'entry_time': None, 'entry_price': None, 'side': None, 'initial_atr_for_profit_target': None, 'vwap_trail_active': False}
strategy4_active_trade_info = {'symbol': None, 'entry_time': None, 'entry_price': None, 'side': None, 'divergence_price_point': None}

# Default reset structures for strategy-specific info
strategy1_active_trade_info_default = {'symbol': None, 'entry_time': None, 'entry_price': None, 'position_qty': 0, 'side': None, 'sl_order_id': None, 'tp_order_id': None}
strategy3_active_trade_info_default = {'symbol': None, 'entry_time': None, 'entry_price': None, 'side': None, 'initial_atr_for_profit_target': None, 'vwap_trail_active': False}
strategy4_active_trade_info_default = {'symbol': None, 'entry_time': None, 'entry_price': None, 'side': None, 'divergence_price_point': None}

# pending_signals = {} # To store {'symbol': {'timestamp': pd.Timestamp, 'side': 'up'/'down', 'last_signal_eval_true': True/False, 'initial_conditions_met_count': 0, 'last_check_time': pd.Timestamp}}

# g_conditional_pending_signals stores signals that have met partial conditions
# and are in a 5-minute monitoring window before full confirmation.
# Key: (symbol, strategy_id) tuple
# Value: dict with fields like 'symbol', 'strategy_id', 'side', 'timestamp', 
#        'current_conditions_met_count', 'conditions_to_start_wait_threshold',
#        'conditions_for_full_signal_threshold', 
#        'all_conditions_status_at_pending_start', 'last_evaluated_all_conditions_status',
#        'entry_price_at_pending_start', 'potential_sl_price', 'potential_tp_price'
g_conditional_pending_signals = {}

TARGET_SYMBOLS = ["BTCUSDT", "ETHUSDT", "USDTUSDT", "XRPUSDT", "BNBUSDT", "SOLUSDT", "USDCUSDT", "DOGEUSDT", "TRXUSDT", "ADAUSDT"]
ACCOUNT_RISK_PERCENT = 0.005
SCALPING_REQUIRED_BUY_CONDITIONS = 1
SCALPING_REQUIRED_SELL_CONDITIONS = 1
TP_PERCENT = 0.01
SL_PERCENT = 0.005
leverage = 5
margin_type_setting = 'ISOLATED' # Margin type
qty_concurrent_positions = 100
LOCAL_HIGH_LOW_LOOKBACK_PERIOD = 20 # New global for breakout logic

# --- GUI Helper Function ---
def update_text_widget_content(widget, content_list):
    if widget and widget.winfo_exists():
        current_scroll_pos = widget.yview()
        widget.config(state=tk.NORMAL)
        widget.delete('1.0', tk.END)
        if content_list:
            for item in content_list:
                widget.insert(tk.END, str(item) + "\n")
        else:
            widget.insert(tk.END, "No data to display or error during fetch.\n")
        widget.config(state=tk.DISABLED)
        widget.yview_moveto(current_scroll_pos[0])

def update_conditions_display_content(symbol, conditions_data, error_message):
    global conditions_text_widget, root
    if not (conditions_text_widget and root and root.winfo_exists() and conditions_text_widget.winfo_exists()): # Check root as well
        return

    conditions_text_widget.config(state=tk.NORMAL)
    conditions_text_widget.delete('1.0', tk.END)
    
    display_text = f"Symbol: {symbol}\n--------------------\n"
    if error_message:
        display_text += f"Error: {error_message}\n"
    elif conditions_data:
        for cond, status in conditions_data.items():
            # Format condition name: replace underscores with spaces and title case
            formatted_cond_name = cond.replace('_', ' ').title()
            display_text += f"{formatted_cond_name}: {status}\n"
    else:
        display_text += "No conditions data available.\n"
        
    conditions_text_widget.insert(tk.END, display_text)
    conditions_text_widget.config(state=tk.DISABLED)

# --- SuperTrend Calculation Functions ---
def calculate_supertrend_manual(kl_df, atr_period=10, multiplier=1.5):
    print("Calculating SuperTrend manually.")
    if not all(col in kl_df.columns for col in ['High', 'Low', 'Close']):
        print("Error: DataFrame for SuperTrend must contain 'High', 'Low', 'Close' columns.")
        return pd.Series(['red'] * len(kl_df), index=kl_df.index)

    # Calculate ATR using ta library if available, else basic EMA of TR
    try:
        atr = ta.volatility.AverageTrueRange(high=kl_df['High'], low=kl_df['Low'], close=kl_df['Close'], window=atr_period).average_true_range()
        if atr is None or atr.empty or atr.isnull().all(): # Check if ATR calculation failed
             raise ValueError("ATR calculation with ta library failed or returned all NaNs.")
        print("ATR calculated using 'ta' library for manual SuperTrend.")
    except Exception as e_ta_atr: # Catch issues with ta.ATR or if it's not suitable
        print(f"Failed to use ta.AverageTrueRange ({e_ta_atr}), falling back to manual TR EMA for ATR.")
        high_low = kl_df['High'] - kl_df['Low']
        high_close_prev = (kl_df['High'] - kl_df['Close'].shift()).abs()
        low_close_prev = (kl_df['Low'] - kl_df['Close'].shift()).abs()
        tr = pd.concat([high_low, high_close_prev, low_close_prev], axis=1).max(axis=1)
        atr = tr.ewm(alpha=1/atr_period, adjust=False).mean()

    kl_df_temp = kl_df.copy() # Work on a copy to avoid modifying original DataFrame if passed by reference

    kl_df_temp['basic_ub'] = (kl_df_temp['High'] + kl_df_temp['Low']) / 2 + multiplier * atr
    kl_df_temp['basic_lb'] = (kl_df_temp['High'] + kl_df_temp['Low']) / 2 - multiplier * atr

    kl_df_temp['final_ub'] = 0.00
    kl_df_temp['final_lb'] = 0.00

    # Fill initial NaNs in atr if any (e.g. if using simple TR EMA)
    first_valid_atr_index = atr.first_valid_index()
    start_index_for_calc = 0 # Default if no valid ATR index
    if first_valid_atr_index is not None: # Ensure there is a valid ATR value
        start_index_for_calc = kl_df_temp.index.get_loc(first_valid_atr_index)
        if start_index_for_calc > 0 : # Initialize first final bands based on first valid ATR
             kl_df_temp.loc[kl_df_temp.index[start_index_for_calc-1], 'final_ub'] = kl_df_temp['basic_ub'].iloc[start_index_for_calc-1]
             kl_df_temp.loc[kl_df_temp.index[start_index_for_calc-1], 'final_lb'] = kl_df_temp['basic_lb'].iloc[start_index_for_calc-1]
    else: # No valid ATR, cannot calculate Supertrend
        print("Error: ATR contains no valid data. Cannot calculate Supertrend.")
        return pd.Series(['red'] * len(kl_df_temp), index=kl_df_temp.index)


    for i in range(start_index_for_calc, len(kl_df_temp)):
        idx = kl_df_temp.index[i]
        prev_idx = kl_df_temp.index[i-1]

        if kl_df_temp.loc[idx, 'basic_ub'] < kl_df_temp.loc[prev_idx, 'final_ub'] or kl_df_temp.loc[prev_idx, 'Close'] > kl_df_temp.loc[prev_idx, 'final_ub']:
            kl_df_temp.loc[idx, 'final_ub'] = kl_df_temp.loc[idx, 'basic_ub']
        else:
            kl_df_temp.loc[idx, 'final_ub'] = kl_df_temp.loc[prev_idx, 'final_ub']

        if kl_df_temp.loc[idx, 'basic_lb'] > kl_df_temp.loc[prev_idx, 'final_lb'] or kl_df_temp.loc[prev_idx, 'Close'] < kl_df_temp.loc[prev_idx, 'final_lb']:
            kl_df_temp.loc[idx, 'final_lb'] = kl_df_temp.loc[idx, 'basic_lb']
        else:
            kl_df_temp.loc[idx, 'final_lb'] = kl_df_temp.loc[prev_idx, 'final_lb']

    kl_df_temp['supertrend_signal'] = 'red'
    # Default to green after initial period for logic below, only if there's enough data
    if len(kl_df_temp) > start_index_for_calc :
         kl_df_temp.loc[kl_df_temp.index[start_index_for_calc]:, 'supertrend_signal'] = 'green'


    for i in range(start_index_for_calc, len(kl_df_temp)):
        idx = kl_df_temp.index[i]
        prev_idx = kl_df_temp.index[i-1]
        if kl_df_temp.loc[prev_idx, 'supertrend_signal'] == 'green': # Currently in uptrend
            if kl_df_temp.loc[idx, 'Close'] < kl_df_temp.loc[idx, 'final_lb']:
                kl_df_temp.loc[idx, 'supertrend_signal'] = 'red' # Price crossed below lower band
            else:
                kl_df_temp.loc[idx, 'supertrend_signal'] = 'green' # Continue uptrend
        elif kl_df_temp.loc[prev_idx, 'supertrend_signal'] == 'red': # Currently in downtrend
            if kl_df_temp.loc[idx, 'Close'] > kl_df_temp.loc[idx, 'final_ub']:
                kl_df_temp.loc[idx, 'supertrend_signal'] = 'green' # Price crossed above upper band
            else:
                kl_df_temp.loc[idx, 'supertrend_signal'] = 'red' # Continue downtrend

    return kl_df_temp['supertrend_signal'].fillna('red')


def calculate_supertrend_pta(kl_df, atr_period=10, multiplier=1.5):
    global PANDAS_TA_AVAILABLE
    if not PANDAS_TA_AVAILABLE:
        print("pandas_ta not available. Falling back to manual SuperTrend.")
        return calculate_supertrend_manual(kl_df, atr_period=atr_period, multiplier=multiplier)
    try:
        print("Attempting SuperTrend calculation using pandas_ta.")
        st = pta.supertrend(high=kl_df['High'], low=kl_df['Low'], close=kl_df['Close'], length=atr_period, multiplier=multiplier)
        if st is None or st.empty:
            print("pandas_ta.supertrend returned None or empty. Falling back to manual.")
            return calculate_supertrend_manual(kl_df, atr_period=atr_period, multiplier=multiplier)

        trend_col = None
        # Common column name patterns: SUPERTd_length_multiplier, SUPERT_length_multiplier
        # pandas_ta might also name it based on the input columns if they are not standard 'high', 'low', 'close'
        # For simplicity, we target the most common ones.
        possible_trend_cols = [f'SUPERTd_{atr_period}_{multiplier}', f'SUPERT_{atr_period}_{multiplier}.0']
        # .0 can be appended by pta if multiple ST are generated or settings are slightly off default names

        for col_name_candidate in possible_trend_cols:
             if col_name_candidate in st.columns:
                 trend_col = col_name_candidate
                 break

        # Broader search if specific names not found
        if trend_col is None:
            for col in st.columns:
                if col.startswith('SUPERTd_') and col.endswith(f'_{atr_period}_{multiplier}'):
                    trend_col = col; break
                if col.startswith('SUPERT_') and col.endswith(f'_{atr_period}_{multiplier}'): # Less specific, might be the line not direction
                    if trend_col is None: trend_col = col # Take if no direction col found yet


        if trend_col and trend_col in st.columns:
            if 'SUPERTd' in trend_col: # Direction column: 1 for uptrend, -1 for downtrend
                 supertrend_signal = pd.Series(['green' if x == 1 else 'red' for x in st[trend_col]], index=kl_df.index)
            else: # Main SuperTrend line column: Close > SuperTrend line means uptrend
                 supertrend_signal = pd.Series(['green' if kl_df['Close'].iloc[i] > st[trend_col].iloc[i] else 'red' for i in range(len(kl_df))], index=kl_df.index)
            print(f"SuperTrend calculated using pandas_ta, column: {trend_col}.")
            return supertrend_signal.fillna('red') # Fill NaNs that might occur from pta result
        else:
            print(f"Could not find expected SuperTrend column in pta output. Columns: {st.columns}. Falling back to manual.")
            return calculate_supertrend_manual(kl_df, atr_period=atr_period, multiplier=multiplier)

    except Exception as e:
        print(f"Error using pandas_ta for SuperTrend: {e}. Falling back to manual.")
        return calculate_supertrend_manual(kl_df, atr_period=atr_period, multiplier=multiplier)

# --- Binance Helper Functions and Strategy Logic --- (Existing functions like get_balance_usdt, klines, etc.)
def get_balance_usdt():
    global client
    if not client: return None
    try:
        # Use client.account() for more detailed futures account information
        account_info = client.account(recvWindow=6000) 
        if account_info and 'assets' in account_info:
            for asset_detail in account_info['assets']:
                if asset_detail['asset'] == 'USDT':
                    # Return the 'availableBalance' for USDT
                    print(f"DEBUG: USDT Asset Detail from client.account(): {asset_detail}") # Add this debug line
                    return float(asset_detail['availableBalance']) 
        print("DEBUG: USDT asset not found or 'assets' key missing in client.account() response.") # Add this debug line
    except ClientError as error:
        if error.error_code == -2015: # Keep existing specific error handling
            msg = (
                "ERROR: Invalid API Key or Permissions for Futures Trading (-2015) in get_balance_usdt (using client.account()).\n"
                "Please check API key validity, 'Enable Futures' permission, and IP restrictions."
            )
            print(msg)
        else:
            print(f"ClientError in get_balance_usdt (using client.account()): {error.error_code} - {error.error_message}")
    except Exception as e:
        print(f"Unexpected error in get_balance_usdt (using client.account()): {e}")
    return None # Return None if balance couldn't be fetched

def get_tickers_usdt():
    global client
    if not client: return []
    tickers = []
    try:
        resp = client.ticker_price()
        for elem in resp:
            if 'USDT' in elem['symbol']: tickers.append(elem['symbol'])
    except ClientError as error: print(f"Error get_tickers_usdt: {error}")
    except Exception as e: print(f"Unexpected error get_tickers_usdt: {e}")
    return tickers

def klines(symbol):
    global client
    if not client: return None
    try:
        resp = pd.DataFrame(client.klines(symbol, '5m'))
        resp = resp.iloc[:,:6]
        resp.columns = ['Time', 'Open', 'High', 'Low', 'Close', 'Volume']
        resp = resp.set_index('Time')
        resp.index = pd.to_datetime(resp.index, unit = 'ms')
        resp = resp.astype(float)
        return resp
    except ClientError: pass # Error already printed by other functions or too noisy
    except Exception: pass
    return None

def scalping_strategy_signal(symbol):
    global LOCAL_HIGH_LOW_LOOKBACK_PERIOD, SCALPING_REQUIRED_BUY_CONDITIONS, SCALPING_REQUIRED_SELL_CONDITIONS
    
    return_data = {
        'signal': 'none',
        'conditions_met_count': 0,
        'conditions_to_start_wait_threshold': 2, # Standard for S0
        'conditions_for_full_signal_threshold': SCALPING_REQUIRED_BUY_CONDITIONS, # Default, adjusted later
        'all_conditions_status': {
            'ema_cross_up': False, 'rsi_long_ok': False, 'supertrend_green': False,
            'volume_spike': False, 'price_breakout_high': False,
            'ema_cross_down': False, 'rsi_short_ok': False, 'supertrend_red': False,
            'price_breakout_low': False,
            'num_buy_conditions_met': 0,
            'num_sell_conditions_met': 0
        },
        'sl_price': None, # Not applicable for S0 at this stage
        'tp_price': None, # Not applicable for S0 at this stage
        'error': None
    }

    kl = klines(symbol)
    if kl is None or len(kl) < max(21, LOCAL_HIGH_LOW_LOOKBACK_PERIOD + 1): # Ensure enough data
        return_data['error'] = 'Insufficient kline data'
        return return_data
    
    try:
        ema9 = ta.trend.EMAIndicator(close=kl['Close'], window=9).ema_indicator()
        ema21 = ta.trend.EMAIndicator(close=kl['Close'], window=21).ema_indicator()
        rsi = ta.momentum.RSIIndicator(close=kl['Close'], window=14).rsi()
        supertrend = calculate_supertrend_pta(kl, atr_period=10, multiplier=1.5)
        volume_ma10 = kl['Volume'].rolling(window=10).mean()

        if any(x is None for x in [ema9, ema21, rsi, supertrend, volume_ma10]) or \
           any(x.empty for x in [ema9, ema21, rsi, supertrend]) or volume_ma10.isnull().all():
            return_data['error'] = 'One or more indicators are None or empty'
            return return_data

        required_length = max(10, LOCAL_HIGH_LOW_LOOKBACK_PERIOD + 1)
        if len(ema9) < 2 or len(ema21) < 2 or len(rsi) < 1 or len(supertrend) < 1 or \
           len(volume_ma10.dropna()) < 1 or len(kl['Volume']) < required_length:
            return_data['error'] = 'Indicator series too short for required length'
            return return_data

        last_ema9, prev_ema9 = ema9.iloc[-1], ema9.iloc[-2]
        last_ema21, prev_ema21 = ema21.iloc[-1], ema21.iloc[-2]
        last_rsi = rsi.iloc[-1]
        last_supertrend = supertrend.iloc[-1]
        last_volume = kl['Volume'].iloc[-1]
        last_volume_ma10 = volume_ma10.iloc[-1] # Might be NaN if window is large and data short
        current_price = kl['Close'].iloc[-1]

        if any(pd.isna(val) for val in [last_ema9, prev_ema9, last_ema21, prev_ema21, last_rsi, last_supertrend, last_volume, current_price]) or pd.isna(last_volume_ma10):
            return_data['error'] = 'NaN value in one of the critical indicators or price'
            return return_data
            
        actual_lookback = min(LOCAL_HIGH_LOW_LOOKBACK_PERIOD, len(kl['High']) - 1)
        recent_high = kl['High'].iloc[-(actual_lookback + 1):-1].max() if actual_lookback > 0 else kl['High'].iloc[-2] if len(kl['High']) > 1 else current_price
        recent_low = kl['Low'].iloc[-(actual_lookback + 1):-1].min() if actual_lookback > 0 else kl['Low'].iloc[-2] if len(kl['Low']) > 1 else current_price

        # Individual conditions
        ema_crossed_up = prev_ema9 < prev_ema21 and last_ema9 > last_ema21
        rsi_valid_long = 50 <= last_rsi <= 70
        supertrend_is_green = last_supertrend == 'green'
        volume_is_strong = last_volume > last_volume_ma10
        price_broke_high = current_price > recent_high

        ema_crossed_down = prev_ema9 > prev_ema21 and last_ema9 < last_ema21
        rsi_valid_short = 30 <= last_rsi <= 50
        supertrend_is_red = last_supertrend == 'red'
        price_broke_low = current_price < recent_low

        buy_conditions_met = [ema_crossed_up, rsi_valid_long, supertrend_is_green, volume_is_strong, price_broke_high]
        sell_conditions_met = [ema_crossed_down, rsi_valid_short, supertrend_is_red, volume_is_strong, price_broke_low]
        
        num_buy_conditions_true = sum(buy_conditions_met)
        num_sell_conditions_true = sum(sell_conditions_met)

        return_data['all_conditions_status'].update({
            'ema_cross_up': ema_crossed_up, 'rsi_long_ok': rsi_valid_long, 
            'supertrend_green': supertrend_is_green, 'volume_spike': volume_is_strong, 
            'price_breakout_high': price_broke_high,
            'ema_cross_down': ema_crossed_down, 'rsi_short_ok': rsi_valid_short, 
            'supertrend_red': supertrend_is_red, 'price_breakout_low': price_broke_low,
            'num_buy_conditions_met': num_buy_conditions_true,
            'num_sell_conditions_met': num_sell_conditions_true
        })

        if num_buy_conditions_true >= SCALPING_REQUIRED_BUY_CONDITIONS:
            return_data['signal'] = 'up'
            return_data['conditions_met_count'] = num_buy_conditions_true
            return_data['conditions_for_full_signal_threshold'] = SCALPING_REQUIRED_BUY_CONDITIONS
        elif num_sell_conditions_true >= SCALPING_REQUIRED_SELL_CONDITIONS:
            return_data['signal'] = 'down'
            return_data['conditions_met_count'] = num_sell_conditions_true
            return_data['conditions_for_full_signal_threshold'] = SCALPING_REQUIRED_SELL_CONDITIONS
        else: # No full signal, set met_count to the higher of the two for partial/wait check
            if num_buy_conditions_true >= num_sell_conditions_true: # Prioritize buy if equal
                return_data['conditions_met_count'] = num_buy_conditions_true
                return_data['conditions_for_full_signal_threshold'] = SCALPING_REQUIRED_BUY_CONDITIONS
            else:
                return_data['conditions_met_count'] = num_sell_conditions_true
                return_data['conditions_for_full_signal_threshold'] = SCALPING_REQUIRED_SELL_CONDITIONS
        
        return return_data

    except Exception as e:
        # print(f"Error in scalping_strategy_signal for {symbol}: {e}") # Can be noisy
        return_data['error'] = str(e)
        return return_data

def strategy_ema_supertrend(symbol):
    base_return = {
        'signal': 'none',
        'conditions_met_count': 0,
        'conditions_to_start_wait_threshold': 2,
        'conditions_for_full_signal_threshold': 3, # Changed from 3 to 2
        'all_conditions_status': {
            'ema_cross_up': False, 'st_green': False, 'rsi_long_ok': False,
            'ema_cross_down': False, 'st_red': False, 'rsi_short_ok': False,
        },
        'sl_price': None,
        'tp_price': None,
        'error': None,
        'account_risk_percent': 0.01 # Strategy 1 specific risk
    }

    global strategy1_cooldown_active
    if strategy1_cooldown_active:
        print(f"Strategy 1 ({symbol}): Cooldown active, skipping signal generation.")
        # The reset of strategy1_cooldown_active will be handled in run_bot_logic or a dedicated cooldown management function.
        base_return['error'] = "Cooldown active after loss"
        return base_return

    kl = klines(symbol) # Default 5m interval
    if kl is None or len(kl) < 21: # Need enough for EMAs and SuperTrend lookback
        base_return['error'] = "Insufficient kline data (need at least 21 candles)"
        return base_return

    try:
        # Calculate Indicators
        ema9 = ta.trend.EMAIndicator(close=kl['Close'], window=9).ema_indicator()
        ema21 = ta.trend.EMAIndicator(close=kl['Close'], window=21).ema_indicator()
        # SuperTrend: ATR 10, Multiplier 3
        supertrend_series = calculate_supertrend_pta(kl, atr_period=10, multiplier=3.0) 
        rsi = ta.momentum.RSIIndicator(close=kl['Close'], window=14).rsi()

        if any(x is None for x in [ema9, ema21, supertrend_series, rsi]) or \
           any(x.empty for x in [ema9, ema21, supertrend_series, rsi]):
            base_return['error'] = "One or more core indicators (EMA, Supertrend, RSI) are None or empty"
            return base_return
        
        # Ensure enough data points after indicator calculation (especially for EMAs)
        if len(ema9) < 2 or len(ema21) < 2 or len(supertrend_series) < 1 or len(rsi) < 1:
            base_return['error'] = "Indicator series too short after calculation"
            return base_return

        # Extract latest values
        last_ema9, prev_ema9 = ema9.iloc[-1], ema9.iloc[-2]
        last_ema21, prev_ema21 = ema21.iloc[-1], ema21.iloc[-2]
        last_supertrend_signal = supertrend_series.iloc[-1]
        last_rsi = rsi.iloc[-1]
        current_price = kl['Close'].iloc[-1]

        if any(pd.isna(v) for v in [last_ema9, prev_ema9, last_ema21, prev_ema21, last_supertrend_signal, last_rsi, current_price]):
            base_return['error'] = "NaN value in critical indicator or price data"
            return base_return
            
        # --- Define Conditions ---
        # Update all_conditions_status in base_return directly
        base_return['all_conditions_status']['ema_cross_up'] = prev_ema9 < prev_ema21 and last_ema9 > last_ema21
        base_return['all_conditions_status']['st_green'] = last_supertrend_signal == 'green'
        base_return['all_conditions_status']['rsi_long_ok'] = 40 <= last_rsi <= 70

        base_return['all_conditions_status']['ema_cross_down'] = prev_ema9 > prev_ema21 and last_ema9 < last_ema21
        base_return['all_conditions_status']['st_red'] = last_supertrend_signal == 'red'
        base_return['all_conditions_status']['rsi_short_ok'] = 30 <= last_rsi <= 60

        # Calculate met conditions count
        met_buy_conditions = sum([base_return['all_conditions_status']['ema_cross_up'], 
                                  base_return['all_conditions_status']['st_green'], 
                                  base_return['all_conditions_status']['rsi_long_ok']])
        met_sell_conditions = sum([base_return['all_conditions_status']['ema_cross_down'], 
                                   base_return['all_conditions_status']['st_red'], 
                                   base_return['all_conditions_status']['rsi_short_ok']])

        final_signal_str = 'none' # Temporary variable for signal before SL/TP validation
        sl_price_calc, tp_price_calc = None, None # Use temporary vars for calculations
        
        if len(kl['Low']) < 3 or len(kl['High']) < 3: 
            base_return['error'] = "Not enough kline data for SL/TP calculation (need T-1, T-2)"
            if met_buy_conditions >= met_sell_conditions: # Prioritize buy if counts are equal or buy is greater
                 base_return['conditions_met_count'] = met_buy_conditions
            else:
                 base_return['conditions_met_count'] = met_sell_conditions
            # all_conditions_status is already populated
            return base_return

        price_precision = get_price_precision(symbol)

        if met_buy_conditions >= 2: # Changed from == base_return['conditions_for_full_signal_threshold']
            final_signal_str = 'up'
            base_return['conditions_met_count'] = met_buy_conditions
            swing_low = min(kl['Low'].iloc[-2], kl['Low'].iloc[-3])
            sl_price_calc = round(swing_low - (kl['Low'].iloc[-1] * 0.001), price_precision) 
            if current_price <= sl_price_calc: 
                base_return['error'] = f"SL price {sl_price_calc} is at or above entry {current_price} for LONG"
                final_signal_str = 'none' 
            else:
                risk_amount = current_price - sl_price_calc
                tp_price_calc = round(current_price + (risk_amount * 1.5), price_precision)
                if tp_price_calc <= current_price: 
                    base_return['error'] = f"TP price {tp_price_calc} is at or below entry {current_price} for LONG"
                    final_signal_str = 'none' 
        
        elif met_sell_conditions >= 2: # Changed from == base_return['conditions_for_full_signal_threshold']
            final_signal_str = 'down'
            base_return['conditions_met_count'] = met_sell_conditions
            swing_high = max(kl['High'].iloc[-2], kl['High'].iloc[-3])
            sl_price_calc = round(swing_high + (kl['High'].iloc[-1] * 0.001), price_precision) 
            if current_price >= sl_price_calc: 
                base_return['error'] = f"SL price {sl_price_calc} is at or below entry {current_price} for SHORT"
                final_signal_str = 'none' 
            else:
                risk_amount = sl_price_calc - current_price
                tp_price_calc = round(current_price - (risk_amount * 1.5), price_precision)
                if tp_price_calc >= current_price: 
                    base_return['error'] = f"TP price {tp_price_calc} is at or above entry {current_price} for SHORT"
                    final_signal_str = 'none' 
        else: # Not a full signal, determine conditions_met_count for partial/wait check
            if met_buy_conditions >= met_sell_conditions:
                 base_return['conditions_met_count'] = met_buy_conditions
            else:
                 base_return['conditions_met_count'] = met_sell_conditions

        base_return['signal'] = final_signal_str 
        if final_signal_str != 'none': # Only set SL/TP if signal is valid
            base_return['sl_price'] = sl_price_calc
            base_return['tp_price'] = tp_price_calc
        else: # Ensure SL/TP are None if signal is 'none'
            base_return['sl_price'] = None
            base_return['tp_price'] = None
            if base_return['error']: # Print error only if it was set (e.g. by SL/TP validation)
                 print(f"Strategy {STRATEGIES[1]} for {symbol}: Signal invalidated due to SL/TP error: {base_return['error']}")
        
        return base_return

    except Exception as e:
        base_return['error'] = f"Exception in strategy_ema_supertrend: {str(e)}"
        # Ensure all_conditions_status is populated even in case of early exception
        # This might be redundant if base_return is initialized with defaults, but safe
        base_return['all_conditions_status'].update({
            'ema_cross_up': False, 'st_green': False, 'rsi_long_ok': False,
            'ema_cross_down': False, 'st_red': False, 'rsi_short_ok': False,
        })
        return base_return

def strategy_bollinger_band_mean_reversion(symbol):
    global strategy2_active_trades # Used to check concurrent trade limit

    base_return = {
        'signal': 'none',
        'conditions_met_count': 0,
        'conditions_to_start_wait_threshold': 2,
        'conditions_for_full_signal_threshold': 2, # Changed from 3 to 2
        'all_conditions_status': {
            'price_below_lower_bb': False, 'rsi_oversold': False, 'volume_confirms_long': False,
            'price_above_upper_bb': False, 'rsi_overbought': False, 'volume_confirms_short': False,
        },
        'sl_price': None,
        'tp_price': None,
        'error': None,
        'account_risk_percent': 0.008 # Strategy 2 specific risk
    }
    
    # Max 2 Concurrent Trades Check for this strategy
    if len(strategy2_active_trades) >= 2:
        base_return['error'] = "Max 2 concurrent S2 trades reached"
        if base_return['error'] == "Max 2 concurrent S2 trades reached": 
             print(f"Strategy 2 ({symbol}): Skipped due to max concurrent trade limit (2).")
             return base_return


    kl = klines(symbol) # Default 5m interval
    min_data_len = 20 # For BB and Volume SMA
    if kl is None or len(kl) < min_data_len:
        base_return['error'] = f"Insufficient kline data (need at least {min_data_len} candles)"
        return base_return

    try:
        # Calculate Indicators
        bb_indicator = ta.volatility.BollingerBands(close=kl['Close'], window=20, window_dev=2)
        rsi_indicator = ta.momentum.RSIIndicator(close=kl['Close'], window=14)
        
        upper_bb = bb_indicator.bollinger_hband()
        lower_bb = bb_indicator.bollinger_lband()
        middle_bb = bb_indicator.bollinger_mavg()
        rsi = rsi_indicator.rsi()
        volume_sma = kl['Volume'].rolling(window=20).mean()

        if any(x is None for x in [upper_bb, lower_bb, middle_bb, rsi, volume_sma]) or \
           any(x.empty for x in [upper_bb, lower_bb, middle_bb, rsi, volume_sma]):
            base_return['error'] = "One or more core indicators (BB, RSI, Vol SMA) are None or empty"
            return base_return
        
        if len(upper_bb) < 1 or len(lower_bb) < 1 or len(middle_bb) < 1 or len(rsi) < 1 or len(volume_sma.dropna()) < 1:
            base_return['error'] = "Indicator series too short after calculation"
            return base_return

        # Extract latest values
        current_price = kl['Close'].iloc[-1]
        last_upper_bb = upper_bb.iloc[-1]
        last_lower_bb = lower_bb.iloc[-1]
        last_middle_bb = middle_bb.iloc[-1]
        last_rsi_val = rsi.iloc[-1]
        last_volume_val = kl['Volume'].iloc[-1]
        last_volume_sma_val = volume_sma.iloc[-1] # This can be NaN if volume_sma window > available data points for it

        if any(pd.isna(v) for v in [current_price, last_upper_bb, last_lower_bb, last_middle_bb, last_rsi_val, last_volume_val]) or pd.isna(last_volume_sma_val):
            base_return['error'] = "NaN value in critical indicator or price data for BB strategy"
            return base_return
            
        # --- Define Conditions ---
        base_return['all_conditions_status']['price_below_lower_bb'] = current_price < last_lower_bb
        base_return['all_conditions_status']['rsi_oversold'] = last_rsi_val < 30
        base_return['all_conditions_status']['volume_confirms_long'] = last_volume_val > last_volume_sma_val
        
        base_return['all_conditions_status']['price_above_upper_bb'] = current_price > last_upper_bb
        base_return['all_conditions_status']['rsi_overbought'] = last_rsi_val > 70
        # For Strategy 2, volume_confirms_short is the same as volume_confirms_long
        base_return['all_conditions_status']['volume_confirms_short'] = last_volume_val > last_volume_sma_val 

        met_buy_conditions = sum([base_return['all_conditions_status']['price_below_lower_bb'],
                                  base_return['all_conditions_status']['rsi_oversold'],
                                  base_return['all_conditions_status']['volume_confirms_long']])
        met_sell_conditions = sum([base_return['all_conditions_status']['price_above_upper_bb'],
                                   base_return['all_conditions_status']['rsi_overbought'],
                                   base_return['all_conditions_status']['volume_confirms_short']])

        final_signal_str = 'none'
        sl_price_calc, tp_price_calc = None, None # Initialize calc variables
        price_precision = get_price_precision(symbol)
        buffer_percentage = 0.001

        if met_buy_conditions >= 2: # Changed from == base_return['conditions_for_full_signal_threshold']
            final_signal_str = 'up'
            base_return['conditions_met_count'] = met_buy_conditions
            sl_price_calc = round(last_lower_bb - (current_price * buffer_percentage), price_precision)
            tp_price_calc = round(last_middle_bb, price_precision)
            if current_price <= sl_price_calc:
                base_return['error'] = f"SL price {sl_price_calc} is at or above entry {current_price} for LONG"
                final_signal_str = 'none'
            elif tp_price_calc <= current_price: # Changed from elif to if for independent check
                base_return['error'] = f"TP price {tp_price_calc} is at or below entry {current_price} for LONG"
                final_signal_str = 'none'
        
        elif met_sell_conditions >= 2: # Changed from == base_return['conditions_for_full_signal_threshold']
            final_signal_str = 'down'
            base_return['conditions_met_count'] = met_sell_conditions
            sl_price_calc = round(last_upper_bb + (current_price * buffer_percentage), price_precision)
            tp_price_calc = round(last_middle_bb, price_precision)
            if current_price >= sl_price_calc:
                base_return['error'] = f"SL price {sl_price_calc} is at or below entry {current_price} for SHORT"
                final_signal_str = 'none'
            elif tp_price_calc >= current_price: # Changed from elif to if for independent check
                base_return['error'] = f"TP price {tp_price_calc} is at or above entry {current_price} for SHORT"
                final_signal_str = 'none'
        else: # Not a full signal
            if met_buy_conditions >= met_sell_conditions: 
                 base_return['conditions_met_count'] = met_buy_conditions
            else:
                 base_return['conditions_met_count'] = met_sell_conditions
        
        base_return['signal'] = final_signal_str
        if final_signal_str != 'none':
            base_return['sl_price'] = sl_price_calc
            base_return['tp_price'] = tp_price_calc
        else:
            base_return['sl_price'] = None
            base_return['tp_price'] = None
            if base_return['error']: 
                 print(f"Strategy {STRATEGIES[2]} for {symbol}: Signal invalidated due to SL/TP error: {base_return['error']}")
        
        return base_return

    except Exception as e:
        base_return['error'] = f"Exception in strategy_bollinger_band_mean_reversion for {symbol}: {str(e)}"
        # Ensure all_conditions_status is populated even in case of early exception
        base_return['all_conditions_status'].update({
            'price_below_lower_bb': False, 'rsi_oversold': False, 'volume_confirms_long': False,
            'price_above_upper_bb': False, 'rsi_overbought': False, 'volume_confirms_short': False,
        })
        return base_return

# --- VWAP Calculation Helper ---
def calculate_daily_vwap(kl_df_current_day):
    if not all(col in kl_df_current_day.columns for col in ['High', 'Low', 'Close', 'Volume']):
        print("Error: VWAP calculation requires 'High', 'Low', 'Close', 'Volume' columns.")
        return None
    if kl_df_current_day.empty or kl_df_current_day['Volume'].sum() == 0:
        # Return a series of NaNs of the same length if volume is zero or df is empty
        return pd.Series([float('nan')] * len(kl_df_current_day), index=kl_df_current_day.index)

    tp = (kl_df_current_day['High'] + kl_df_current_day['Low'] + kl_df_current_day['Close']) / 3
    tpv = tp * kl_df_current_day['Volume']
    
    # Cumulative sum of typical price * volume / cumulative sum of volume
    vwap = tpv.cumsum() / kl_df_current_day['Volume'].cumsum()
    return vwap

def strategy_vwap_breakout_momentum(symbol):
    base_return = {
        'signal': 'none',
        'conditions_met_count': 0,
        'conditions_to_start_wait_threshold': 2,
        'conditions_for_full_signal_threshold': 2, # Changed from 3 to 2
        'all_conditions_status': {
            'price_above_vwap_2bar_long': False, 'macd_positive_rising': False, 'atr_volatility_confirms_long': False,
            'price_below_vwap_2bar_short': False, 'macd_negative_falling': False, 'atr_volatility_confirms_short': False,
        },
        'sl_price': None,
        'tp_price': None,
        'error': None,
        'account_risk_percent': 0.01 # Strategy 3 specific risk
    }

    # Time-based Filter (London/NY Overlap: 13:00-17:00 UTC)
    now_utc = pd.Timestamp.now(tz='UTC')
    if not (13 <= now_utc.hour <= 17):
        base_return['error'] = "Outside London/NY overlap (13-17 UTC)"
        return base_return

    # Fetch Kline Data for the current day
    global client
    if not client:
        base_return['error'] = "Client not initialized for S3"
        return base_return
    
    kl_day_df = None
    try:
        start_of_day_utc = now_utc.normalize() 
        raw_klines = client.klines(symbol, '5m', startTime=int(start_of_day_utc.timestamp() * 1000))
        if not raw_klines:
            base_return['error'] = "No kline data returned for the current day"
            return base_return

        kl_day_df = pd.DataFrame(raw_klines)
        kl_day_df = kl_day_df.iloc[:,:6]
        kl_day_df.columns = ['Time', 'Open', 'High', 'Low', 'Close', 'Volume']
        kl_day_df['Time'] = pd.to_datetime(kl_day_df['Time'], unit='ms')
        kl_day_df = kl_day_df.set_index('Time')
        kl_day_df = kl_day_df.astype(float)

    except Exception as e:
        base_return['error'] = f"Error fetching/processing daily klines for S3 {symbol}: {e}"
        return base_return

    min_data_len_for_indicators = 26 
    if len(kl_day_df) < min_data_len_for_indicators:
        base_return['error'] = f"Insufficient daily kline data for indicators (need {min_data_len_for_indicators}, got {len(kl_day_df)})"
        return base_return

    try:
        # Calculate Indicators
        vwap_series = calculate_daily_vwap(kl_day_df)
        atr_indicator = ta.volatility.AverageTrueRange(high=kl_day_df['High'], low=kl_day_df['Low'], close=kl_day_df['Close'], window=14)
        atr_series = atr_indicator.average_true_range()
        
        macd_obj = ta.trend.MACD(close=kl_day_df['Close'], window_slow=26, window_fast=12, window_sign=9)
        macd_line = macd_obj.macd()
        macd_signal_line = macd_obj.macd_signal() # Not directly used in logic but calculated
        macd_hist = macd_obj.macd_diff()

        if any(s is None for s in [vwap_series, atr_series, macd_line, macd_signal_line, macd_hist]) or \
           any(s.empty for s in [vwap_series, atr_series, macd_line, macd_signal_line, macd_hist]):
            base_return['error'] = "One or more S3 indicators (VWAP, ATR, MACD) are None or empty"
            return base_return
        
        min_lookback = 20 
        if len(vwap_series) < 2 or len(atr_series) < min_lookback or len(macd_hist) < 2 : 
            base_return['error'] = "S3 indicator series too short after calculation"
            return base_return

        # Extract latest values
        current_price = kl_day_df['Close'].iloc[-1]
        last_vwap = vwap_series.iloc[-1]
        prev_vwap = vwap_series.iloc[-2] 
        
        last_macd_hist = macd_hist.iloc[-1]
        prev_macd_hist = macd_hist.iloc[-2]
        
        last_atr = atr_series.iloc[-1]
        avg_atr_20 = atr_series.rolling(window=20).mean().iloc[-1]


        if any(pd.isna(v) for v in [current_price, last_vwap, prev_vwap, last_macd_hist, prev_macd_hist, last_atr, avg_atr_20]):
            base_return['error'] = "NaN value in critical S3 indicator or price data"
            return base_return
            
        # --- Define Conditions ---
        base_return['all_conditions_status']['price_above_vwap_2bar_long'] = kl_day_df['Close'].iloc[-1] > last_vwap and kl_day_df['Close'].iloc[-2] > prev_vwap
        base_return['all_conditions_status']['macd_positive_rising'] = last_macd_hist > 0 and last_macd_hist > prev_macd_hist
        base_return['all_conditions_status']['atr_volatility_confirms_long'] = last_atr > avg_atr_20
        
        base_return['all_conditions_status']['price_below_vwap_2bar_short'] = kl_day_df['Close'].iloc[-1] < last_vwap and kl_day_df['Close'].iloc[-2] < prev_vwap
        base_return['all_conditions_status']['macd_negative_falling'] = last_macd_hist < 0 and last_macd_hist < prev_macd_hist
        # For Strategy 3, atr_volatility_confirms_short is the same as atr_volatility_confirms_long
        base_return['all_conditions_status']['atr_volatility_confirms_short'] = last_atr > avg_atr_20 

        met_buy_conditions = sum([base_return['all_conditions_status']['price_above_vwap_2bar_long'],
                                  base_return['all_conditions_status']['macd_positive_rising'],
                                  base_return['all_conditions_status']['atr_volatility_confirms_long']])
        met_sell_conditions = sum([base_return['all_conditions_status']['price_below_vwap_2bar_short'],
                                   base_return['all_conditions_status']['macd_negative_falling'],
                                   base_return['all_conditions_status']['atr_volatility_confirms_short']])
        
        final_signal_str = 'none'
        sl_price_calc, tp_price_calc = None, None # Initialize calc variables
        price_precision = get_price_precision(symbol)

        if met_buy_conditions >= 2: # Changed from == base_return['conditions_for_full_signal_threshold']
            final_signal_str = 'up'
            base_return['conditions_met_count'] = met_buy_conditions
            sl_price_calc = round(last_vwap, price_precision)
            tp_price_calc = round(current_price + (last_atr * 1.2), price_precision)
            if current_price <= sl_price_calc: 
                base_return['error'] = f"S3 SL price {sl_price_calc} is at or above entry {current_price} for LONG"
                final_signal_str = 'none'
            elif tp_price_calc <= current_price: 
                base_return['error'] = f"S3 TP price {tp_price_calc} is at or below entry {current_price} for LONG"
                final_signal_str = 'none'

        elif met_sell_conditions >= 2: # Changed from == base_return['conditions_for_full_signal_threshold']
            final_signal_str = 'down'
            base_return['conditions_met_count'] = met_sell_conditions
            sl_price_calc = round(last_vwap, price_precision)
            tp_price_calc = round(current_price - (last_atr * 1.2), price_precision)
            if current_price >= sl_price_calc: 
                base_return['error'] = f"S3 SL price {sl_price_calc} is at or above entry {current_price} for SHORT"
                final_signal_str = 'none'
            elif tp_price_calc >= current_price: 
                base_return['error'] = f"S3 TP price {tp_price_calc} is at or above entry {current_price} for SHORT"
                final_signal_str = 'none'
        else: # Not a full signal
            if met_buy_conditions >= met_sell_conditions:
                 base_return['conditions_met_count'] = met_buy_conditions
            else:
                 base_return['conditions_met_count'] = met_sell_conditions

        base_return['signal'] = final_signal_str
        if final_signal_str != 'none':
            base_return['sl_price'] = sl_price_calc
            base_return['tp_price'] = tp_price_calc
        else:
            base_return['sl_price'] = None
            base_return['tp_price'] = None
            if base_return['error']:
                print(f"Strategy {STRATEGIES[3]} for {symbol}: Signal invalidated due to SL/TP error: {base_return['error']}")
        
        return base_return

    except Exception as e:
        base_return['error'] = f"Exception in S3 {symbol}: {str(e)}"
        # Ensure all_conditions_status is populated even in case of early exception
        base_return['all_conditions_status'].update({
            'price_above_vwap_2bar_long': False, 'macd_positive_rising': False, 'atr_volatility_confirms_long': False,
            'price_below_vwap_2bar_short': False, 'macd_negative_falling': False, 'atr_volatility_confirms_short': False,
        })
        return base_return

# --- Pivot Points Calculation Helper ---
def calculate_daily_pivot_points(prev_day_high, prev_day_low, prev_day_close):
    if any(v is None for v in [prev_day_high, prev_day_low, prev_day_close]):
        return None
    try:
        P = (prev_day_high + prev_day_low + prev_day_close) / 3
        S1 = (P * 2) - prev_day_high
        R1 = (P * 2) - prev_day_low
        S2 = P - (prev_day_high - prev_day_low)
        R2 = P + (prev_day_high - prev_day_low)
        return {'P': P, 'S1': S1, 'R1': R1, 'S2': S2, 'R2': R2}
    except Exception as e:
        print(f"Error calculating pivot points: {e}")
        return None

def strategy_macd_divergence_pivot(symbol):
    global client, strategy4_active_trade_info

    base_return = {
        'signal': 'none',
        'conditions': {},
        'sl_price': None,
        'tp_price': None,
        'error': None,
        'divergence_price_point': None, # Specific to S4
        'account_risk_percent': 0.012 # Strategy 4 specific risk
    }

    if strategy4_active_trade_info['symbol'] is not None:
        base_return['error'] = "Strategy 4 already has an active trade."
        # print(f"Strategy 4 ({symbol}): Skipped, active trade exists for {strategy4_active_trade_info['symbol']}") # Can be noisy
        return base_return

    # Fetch Previous Day's Data for Pivots
    pivots = None
    try:
        today_utc = pd.Timestamp.now(tz='UTC').normalize()
        start_prev_day_utc = today_utc - pd.Timedelta(days=1)
        # end_prev_day_utc = today_utc - pd.Timedelta(microseconds=1) # Ensure it's strictly previous day
        # Binance API endTime is exclusive, so to get full previous day, up to 00:00 of current day is fine
        # However, client.klines '1d' interval typically refers to the day starting at startTime.
        # For 1d kline of *previous* day, we can ask for startTime=prev_day_start, limit=1.
        
        prev_day_klines_raw = client.klines(
            symbol=symbol, 
            interval='1d', 
            startTime=int(start_prev_day_utc.timestamp() * 1000),
            limit=1 # We only need the single 1D kline for the previous day
        )

        if prev_day_klines_raw and len(prev_day_klines_raw) > 0:
            # Ensure the kline is indeed for the previous day
            kline_ts = pd.to_datetime(prev_day_klines_raw[0][0], unit='ms').normalize()
            if kline_ts == start_prev_day_utc:
                prev_day_high = float(prev_day_klines_raw[0][2])
                prev_day_low = float(prev_day_klines_raw[0][3])
                prev_day_close = float(prev_day_klines_raw[0][4])
                pivots = calculate_daily_pivot_points(prev_day_high, prev_day_low, prev_day_close)
                if pivots is None:
                    base_return['error'] = "Failed to calculate pivot points."
                    return base_return
            else:
                base_return['error'] = "Could not get valid previous day kline for pivots."
                return base_return
        else:
            base_return['error'] = "No previous day kline data returned for pivots."
            return base_return
    except Exception as e:
        base_return['error'] = f"Error fetching prev day klines for S4 pivots: {e}"
        return base_return

    # Fetch Current 5m Kline Data
    kl = klines(symbol) # Uses the existing klines function, which should fetch e.g. last 500 5m candles
    min_candles_for_indicators = 35 # MACD needs ~26+9, Stoch needs ~14+3. Divergence lookback adds more.
    div_lookback = 15 # Candles for divergence detection
    
    if kl is None or len(kl) < min_candles_for_indicators + div_lookback:
        base_return['error'] = f"Insufficient 5m kline data for S4 (need ~{min_candles_for_indicators + div_lookback})"
        return base_return

    try:
        # Calculate Indicators
        macd_hist = ta.trend.MACD(close=kl['Close'], window_slow=26, window_fast=12, window_sign=9).macd_diff()
        stoch_obj = ta.momentum.StochasticOscillator(high=kl['High'], low=kl['Low'], close=kl['Close'], window=14, smooth_window=3, fillna=False)
        stoch_k = stoch_obj.stoch()
        atr_series = ta.volatility.AverageTrueRange(high=kl['High'], low=kl['Low'], close=kl['Close'], window=14).average_true_range()

        if any(s is None for s in [macd_hist, stoch_k, atr_series]) or \
           any(s.empty for s in [macd_hist, stoch_k, atr_series]):
            base_return['error'] = "One or more S4 indicators (MACD, Stoch, ATR) are None or empty"
            return base_return

        if len(macd_hist) < div_lookback + 1 or len(stoch_k) < 2 or len(atr_series) < 1:
            base_return['error'] = "S4 indicator series too short after calculation"
            return base_return

        # Divergence Detection
        bullish_divergence_found = False
        divergence_low_price = None 
        # Look from 2nd to last candle (-2) back to `div_lookback`+1 candle ago
        for i in range(2, div_lookback + 2): # Ensure index -i is valid
            if len(kl) <= i or len(macd_hist) <=i : continue # Boundary check
            # Standard Bullish: Price makes lower low, MACD makes higher low.
            # Current low is kl['Low'].iloc[-1], prev low is kl['Low'].iloc[-i]
            # Current MACD hist is macd_hist.iloc[-1], prev MACD hist is macd_hist.iloc[-i]
            if kl['Low'].iloc[-1] < kl['Low'].iloc[-i] and macd_hist.iloc[-1] > macd_hist.iloc[-i]:
                bullish_divergence_found = True
                divergence_low_price = kl['Low'].iloc[-1] # The low of the current candle where divergence is confirmed
                base_return['divergence_price_point'] = divergence_low_price 
                break
        
        bearish_divergence_found = False
        divergence_high_price = None
        if not bullish_divergence_found: # Only check for bearish if bullish not found
            for i in range(2, div_lookback + 2):
                if len(kl) <= i or len(macd_hist) <=i : continue
                # Standard Bearish: Price makes higher high, MACD makes lower high.
                if kl['High'].iloc[-1] > kl['High'].iloc[-i] and macd_hist.iloc[-1] < macd_hist.iloc[-i]:
                    bearish_divergence_found = True
                    divergence_high_price = kl['High'].iloc[-1]
                    base_return['divergence_price_point'] = divergence_high_price
                    break
        
        # Extract latest values for conditions
        current_price = kl['Close'].iloc[-1]
        last_stoch_k = stoch_k.iloc[-1]
        prev_stoch_k = stoch_k.iloc[-2] # Requires stoch_k to have at least 2 values
        last_atr = atr_series.iloc[-1]

        if any(pd.isna(v) for v in [current_price, last_stoch_k, prev_stoch_k, last_atr]):
            base_return['error'] = "NaN value in critical S4 indicator/price for entry decision"
            return base_return

        # Entry Conditions
        conditions = base_return['conditions']
        final_signal = 'none'
        
        conditions['bullish_divergence'] = bullish_divergence_found
        conditions['bearish_divergence'] = bearish_divergence_found
        conditions['stoch_k'] = last_stoch_k
        conditions['prev_stoch_k'] = prev_stoch_k
        conditions['pivot_P'] = pivots['P']
        conditions['pivot_S1'] = pivots['S1']
        conditions['pivot_R1'] = pivots['R1']


        if bullish_divergence_found:
            # Price at/above Support (S1 or S2)
            # Check if the low of the signal candle touched or was near S1/S2
            price_at_support = (pivots['S2'] <= kl['Low'].iloc[-1] <= pivots['S1'] * 1.005) or \
                               (pivots['S1'] <= kl['Low'].iloc[-1] <= pivots['P'] * 1.005 and kl['Low'].iloc[-1] < pivots['P']) # Allow if between S1 and P but closer to S1
            stoch_oversold_turning_up = last_stoch_k < 20 and last_stoch_k > prev_stoch_k
            
            conditions['price_at_support_long'] = price_at_support
            conditions['stoch_oversold_turning_up'] = stoch_oversold_turning_up

            if price_at_support: # Removed stoch_oversold_turning_up
                final_signal = 'up'

        elif bearish_divergence_found:
            # Price at/below Resistance (R1 or R2)
            price_at_resistance = (pivots['R1'] * 0.995 <= kl['High'].iloc[-1] <= pivots['R2']) or \
                                  (pivots['P'] * 0.995 <= kl['High'].iloc[-1] <= pivots['R1'] and kl['High'].iloc[-1] > pivots['P'])
            stoch_overbought_turning_down = last_stoch_k > 80 and last_stoch_k < prev_stoch_k

            conditions['price_at_resistance_short'] = price_at_resistance
            conditions['stoch_overbought_turning_down'] = stoch_overbought_turning_down

            if price_at_resistance: # Removed stoch_overbought_turning_down
                final_signal = 'down'
        
        # SL/TP Calculation
        if final_signal != 'none':
            price_precision = get_price_precision(symbol)
            if final_signal == 'up':
                sl_price = round(divergence_low_price - last_atr, price_precision)
                tp_price = round(pivots['P'], price_precision) # Target Pivot P
                if current_price <= sl_price or tp_price <= current_price:
                    base_return['error'] = f"S4 LONG SL/TP invalid: SL={sl_price}, TP={tp_price}, Entry={current_price}"
                    final_signal = 'none'
            elif final_signal == 'down':
                sl_price = round(divergence_high_price + last_atr, price_precision)
                tp_price = round(pivots['P'], price_precision) # Target Pivot P
                if current_price >= sl_price or tp_price >= current_price:
                    base_return['error'] = f"S4 SHORT SL/TP invalid: SL={sl_price}, TP={tp_price}, Entry={current_price}"
                    final_signal = 'none'
            
            if final_signal == 'none' and base_return['error']:
                 print(f"Strategy {STRATEGIES[4]} for {symbol}: Signal invalidated: {base_return['error']}")


        base_return['signal'] = final_signal
        base_return['sl_price'] = sl_price if final_signal != 'none' else None
        base_return['tp_price'] = tp_price if final_signal != 'none' else None
        # divergence_price_point already set when divergence found
        
        return base_return

    except Exception as e:
        # import traceback
        # print(f"Exception in strategy_macd_divergence_pivot for {symbol}: {str(e)}\n{traceback.format_exc()}")
        base_return['error'] = f"Exception in S4 {symbol}: {str(e)}"
        return base_return

def strategy_rsi_enhanced(symbol):
    # print(f"DEBUG: strategy_rsi_enhanced (NEW LOGIC) entered for symbol: {symbol}") # Keep for debugging if necessary
    global SL_PERCENT, TP_PERCENT, STRATEGIES

    base_return = {
        'signal': 'none',
        'conditions_met_count': 0,
        'conditions_to_start_wait_threshold': 2, 
        'conditions_for_full_signal_threshold': 2, 
        'all_conditions_status': {
            'rsi_crossed_above_30': False, 'duration_oversold_met': False, 
            'rsi_slope_up_met': False, 'price_above_sma50_met': False, 'bullish_divergence_found': False,
            'rsi_crossed_below_70': False, 'duration_overbought_met': False, 
            'rsi_slope_down_met': False, 'price_below_sma50_met': False, 'bearish_divergence_found': False,
        },
        'sl_price': None, 'tp_price': None, 'error': None,
        'account_risk_percent': 0.01 
    }

    rsi_period = 14
    sma_period = 50
    duration_lookback = 5 
    slope_lookback_rsi = 3 
    divergence_candles = 15
    min_klines = 75 

    kl = klines(symbol)
    if kl is None or len(kl) < min_klines:
        base_return['error'] = f'Insufficient kline data for S5 {symbol} (need {min_klines}, got {len(kl) if kl is not None else 0})'
        return base_return

    try:
        rsi_series = ta.momentum.RSIIndicator(close=kl['Close'], window=rsi_period).rsi()
        sma50_series = ta.trend.SMAIndicator(close=kl['Close'], window=sma_period).sma_indicator()

        if rsi_series is None or sma50_series is None or rsi_series.empty or sma50_series.empty:
            base_return['error'] = f'S5 Indicator calculation failed for {symbol} (RSI or SMA empty)'
            return base_return
        
        if len(rsi_series) < (duration_lookback + slope_lookback_rsi + 1) or len(sma50_series) < 1 or len(kl['Close']) < (duration_lookback + slope_lookback_rsi + 1) :
             base_return['error'] = f'S5 Indicator series too short for S5 {symbol} after calculation.'
             return base_return
        
        required_rsi_indices = list(range(-1, -(duration_lookback + slope_lookback_rsi + 2), -1)) 
        if any(pd.isna(rsi_series.iloc[i]) for i in required_rsi_indices if abs(i) <= len(rsi_series)) or            pd.isna(sma50_series.iloc[-1]) or pd.isna(kl['Close'].iloc[-1]):
            base_return['error'] = f'S5 NaN value in critical indicator/price data for {symbol}'
            return base_return

        current_price = kl['Close'].iloc[-1]
        
        # --- Buy Signal Conditions ---
        cond_rsi_crossed_above_30 = rsi_series.iloc[-1] >= 30 and rsi_series.iloc[-2] < 30
        base_return['all_conditions_status']['rsi_crossed_above_30'] = cond_rsi_crossed_above_30

        cond_duration_oversold_met = True
        if len(rsi_series) >= (duration_lookback + 2): 
            for i in range(2, duration_lookback + 2): 
                if rsi_series.iloc[-i] >= 30:
                    cond_duration_oversold_met = False
                    break
        else: 
            cond_duration_oversold_met = False
        base_return['all_conditions_status']['duration_oversold_met'] = cond_duration_oversold_met
        
        cond_rsi_slope_up_met = (rsi_series.iloc[-1] - rsi_series.iloc[-3]) >= 10 if len(rsi_series) >= 3 else False
        base_return['all_conditions_status']['rsi_slope_up_met'] = cond_rsi_slope_up_met

        cond_price_above_sma50_met = current_price > sma50_series.iloc[-1]
        base_return['all_conditions_status']['price_above_sma50_met'] = cond_price_above_sma50_met

        cond_bullish_divergence_found = False
        if len(kl['Low']) >= divergence_candles + 1 and len(rsi_series) >= divergence_candles + 1:
            for i in range(1, divergence_candles + 1): 
                past_low_idx = -1 - i
                if kl['Low'].iloc[-1] < kl['Low'].iloc[past_low_idx] and rsi_series.iloc[-1] > rsi_series.iloc[past_low_idx]:
                    cond_bullish_divergence_found = True
                    break
        base_return['all_conditions_status']['bullish_divergence_found'] = cond_bullish_divergence_found
        
        num_core_buy_conditions_met = sum([cond_rsi_crossed_above_30, cond_duration_oversold_met, cond_rsi_slope_up_met, cond_price_above_sma50_met])

        # --- Sell Signal Conditions ---
        cond_rsi_crossed_below_70 = rsi_series.iloc[-1] <= 70 and rsi_series.iloc[-2] > 70
        base_return['all_conditions_status']['rsi_crossed_below_70'] = cond_rsi_crossed_below_70

        cond_duration_overbought_met = True
        if len(rsi_series) >= (duration_lookback + 2):
            for i in range(2, duration_lookback + 2): 
                if rsi_series.iloc[-i] <= 70:
                    cond_duration_overbought_met = False
                    break
        else: 
            cond_duration_overbought_met = False
        base_return['all_conditions_status']['duration_overbought_met'] = cond_duration_overbought_met

        cond_rsi_slope_down_met = (rsi_series.iloc[-3] - rsi_series.iloc[-1]) >= 10 if len(rsi_series) >= 3 else False
        base_return['all_conditions_status']['rsi_slope_down_met'] = cond_rsi_slope_down_met
        
        cond_price_below_sma50_met = current_price < sma50_series.iloc[-1]
        base_return['all_conditions_status']['price_below_sma50_met'] = cond_price_below_sma50_met

        cond_bearish_divergence_found = False
        if len(kl['High']) >= divergence_candles + 1 and len(rsi_series) >= divergence_candles + 1:
            for i in range(1, divergence_candles + 1):
                past_high_idx = -1 - i
                if kl['High'].iloc[-1] > kl['High'].iloc[past_high_idx] and rsi_series.iloc[-1] < rsi_series.iloc[past_high_idx]:
                    cond_bearish_divergence_found = True
                    break
        base_return['all_conditions_status']['bearish_divergence_found'] = cond_bearish_divergence_found

        num_core_sell_conditions_met = sum([cond_rsi_crossed_below_70, cond_duration_overbought_met, cond_rsi_slope_down_met, cond_price_below_sma50_met])

        price_precision = get_price_precision(symbol)
        
        if num_core_buy_conditions_met >= 2: 
            base_return['signal'] = 'up'
            base_return['conditions_met_count'] = num_core_buy_conditions_met
            sl_p = round(current_price * (1 - SL_PERCENT), price_precision)
            tp_p = round(current_price * (1 + TP_PERCENT), price_precision)
            if sl_p >= current_price or tp_p <= current_price: 
                base_return['error'] = f"S5 BUY Invalid SL/TP: SL={sl_p}, TP={tp_p}, Entry={current_price}"
                base_return['signal'] = 'none' 
            else:
                base_return['sl_price'] = sl_p
                base_return['tp_price'] = tp_p
        
        elif num_core_sell_conditions_met >= 2: 
            base_return['signal'] = 'down'
            base_return['conditions_met_count'] = num_core_sell_conditions_met
            sl_p = round(current_price * (1 + SL_PERCENT), price_precision)
            tp_p = round(current_price * (1 - TP_PERCENT), price_precision)
            if sl_p <= current_price or tp_p >= current_price: 
                base_return['error'] = f"S5 SELL Invalid SL/TP: SL={sl_p}, TP={tp_p}, Entry={current_price}"
                base_return['signal'] = 'none' 
            else:
                base_return['sl_price'] = sl_p
                base_return['tp_price'] = tp_p
        else: 
            base_return['conditions_met_count'] = max(num_core_buy_conditions_met, num_core_sell_conditions_met)
            base_return['signal'] = 'none'

        if base_return['error'] and base_return['signal'] == 'none': 
             print(f"Strategy {STRATEGIES.get(5, 'S5')} for {symbol}: Signal invalidated by error: {base_return['error']}")
        
    except IndexError as ie:
        base_return['error'] = f"S5 IndexError for {symbol}: {str(ie)}. RSI len: {len(rsi_series) if 'rsi_series' in locals() else 'N/A'}, KL len: {len(kl) if 'kl' in locals() else 'N/A'}"
        base_return['signal'] = 'none'
        print(f"DEBUG: {base_return['error']}")
    except Exception as e:
        base_return['error'] = f"S5 Exception for {symbol}: {str(e)}"
        base_return['signal'] = 'none'
        print(f"DEBUG: {base_return['error']}")
        for key_cond in base_return['all_conditions_status']:
            if isinstance(base_return['all_conditions_status'][key_cond], bool):
                 base_return['all_conditions_status'][key_cond] = False
    
    return base_return

def set_leverage(symbol, level):
    global client
    if not client: return
    try:
        response = client.change_leverage(symbol=symbol, leverage=level, recvWindow=6000)
        print(response)
    except ClientError as error: print(f"Error setting leverage for {symbol}: {error}")
    except Exception as e: print(f"Unexpected error setting leverage for {symbol}: {e}")

def set_mode(symbol, margin_type):
    print(f"DEBUG: set_mode received symbol='{symbol}', margin_type='{margin_type}' (type: {type(margin_type)})")
    global client
    if not client: return
    try:
        response = client.change_margin_type(symbol=symbol, marginType=margin_type, recvWindow=6000)
        print(response)
    except ClientError as error: print(f"Error setting margin type for {symbol}: {error}")
    except Exception as e: print(f"Unexpected error setting margin type for {symbol}: {e}")

def get_price_precision(symbol):
    global client
    if not client: return 0
    try:
        info = client.exchange_info()
        for item in info['symbols']:
            if item['symbol'] == symbol: return item['pricePrecision']
    except Exception as e: print(f"Error getting price precision for {symbol}: {e}")
    return 0

def get_qty_precision(symbol):
    global client
    if not client: return 0
    try:
        info = client.exchange_info()
        for item in info['symbols']:
            if item['symbol'] == symbol: return item['quantityPrecision']
    except Exception as e: print(f"Error getting quantity precision for {symbol}: {e}")
    return 0

def open_order(symbol, side, strategy_sl=None, strategy_tp=None, strategy_account_risk_percent=None):
    global client, ACCOUNT_RISK_PERCENT, SL_PERCENT, TP_PERCENT # SL_PERCENT, TP_PERCENT used if strategy_sl/tp are None
    if not client: return

    original_side_param = side # Store original for logging
    if side == 'up':
        side = 'buy'
        print(f"INFO: Mapped side parameter from 'up' to 'buy' for symbol {symbol}")
    elif side == 'down':
        side = 'sell'
        print(f"INFO: Mapped side parameter from 'down' to 'sell' for symbol {symbol}")
    elif side not in ['buy', 'sell']: # If it's already 'buy' or 'sell', do nothing, otherwise error
        print(f"ERROR: Invalid side parameter '{original_side_param}' received in open_order for symbol {symbol}. Expected 'up', 'down', 'buy', or 'sell'. Aborting order.")
        return

    try:
        price = float(client.ticker_price(symbol)['price'])
        qty_precision = get_qty_precision(symbol)
        price_precision = get_price_precision(symbol)
        print(f"DEBUG: open_order: Initial price={price}, qty_precision={qty_precision}, price_precision={price_precision}")
        if not isinstance(price_precision, int) or price_precision < 0:
            print(f"Warning: Invalid price_precision '{price_precision}' for {symbol}. Defaulting to 4.")
            price_precision = 4

        account_balance = get_balance_usdt()
        if account_balance is None or account_balance <= 0: return

        current_account_risk = ACCOUNT_RISK_PERCENT # Global default
        if strategy_account_risk_percent is not None and 0 < strategy_account_risk_percent < 1:
            current_account_risk = strategy_account_risk_percent
            print(f"Using strategy-defined account risk: {current_account_risk*100:.2f}% for {symbol}")
        else:
            print(f"Using global account risk: {current_account_risk*100:.2f}% for {symbol}")

        capital_to_risk_usdt = account_balance * current_account_risk
        if capital_to_risk_usdt <= 0:
            print(f"Warning: Capital to risk is {capital_to_risk_usdt:.2f} for {symbol}. Aborting order.")
            return
        
        # Ensure SL_PERCENT is used here for position sizing calculation,
        # as it represents the distance to SL, not the total risk capital for the account.
        # The strategy-specific SL price (strategy_sl) will define this distance.
        # If strategy_sl is None, then global SL_PERCENT is used to calculate sl_actual later.
        # The risk amount (current_account_risk) determines the capital, the SL distance (derived from strategy_sl or SL_PERCENT) determines the quantity.

        sl_for_sizing = SL_PERCENT # Default global SL percent for sizing if strategy SL not provided or invalid
        if strategy_sl is not None:
            if side == 'buy' and strategy_sl < price :
                sl_for_sizing = (price - strategy_sl) / price
            elif side == 'sell' and strategy_sl > price:
                sl_for_sizing = (strategy_sl - price) / price
            else: # Strategy SL is invalid (e.g. on wrong side of price or zero)
                print(f"Warning: Invalid strategy_sl ({strategy_sl}) for sizing on {symbol} {side} at price {price}. Defaulting to global SL_PERCENT for sizing.")
                # sl_for_sizing remains global SL_PERCENT
        
        if sl_for_sizing <= 0:
            print(f"Warning: Stop loss for sizing is {sl_for_sizing*100:.2f}% for {symbol}. Aborting order.")
            return

        position_size_usdt = capital_to_risk_usdt / sl_for_sizing
        # Define a cap for notional position size based on a fraction of available balance
        CAP_FRACTION_OF_BALANCE = 0.50 
        max_permissible_notional_value = account_balance * CAP_FRACTION_OF_BALANCE

        original_calculated_pos_size_usdt = position_size_usdt # For logging

        if position_size_usdt > max_permissible_notional_value:
            position_size_usdt = max_permissible_notional_value
            print(f"INFO: Position size capped for {symbol}. Original calc: {original_calculated_pos_size_usdt:.2f} USDT, Capped to: {position_size_usdt:.2f} USDT (max {CAP_FRACTION_OF_BALANCE*100}% of available balance).")
        calculated_qty_asset = round(position_size_usdt / price, qty_precision)
        if calculated_qty_asset <= 0: # Re-check qty after potential capping
            print(f"Warning: Calculated quantity is {calculated_qty_asset} for {symbol} after potential capping. Aborting order.")
            return

        print(f"Order Details ({symbol}): AvailableBal={account_balance:.2f}, RiskCap={capital_to_risk_usdt:.2f} (using {current_account_risk*100:.2f}% risk), SL_Dist_for_Sizing={sl_for_sizing*100:.2f}%, PosSizeUSD={position_size_usdt:.2f}, Qty={calculated_qty_asset}")

        sl_actual, tp_actual = None, None

        if strategy_sl is not None and strategy_tp is not None:
            sl_actual = strategy_sl
            tp_actual = strategy_tp
            print(f"Using strategy-defined SL: {sl_actual}, TP: {tp_actual} for {symbol} {side}")
        else:
            print(f"DEBUG: Calculating percentage-based SL/TP for {symbol} {side}: price={price}, SL_PERCENT={SL_PERCENT}, TP_PERCENT={TP_PERCENT}, price_precision={price_precision}")
            if side == 'buy':
                sl_actual = round(price - price * SL_PERCENT, price_precision)
                tp_actual = round(price + price * TP_PERCENT, price_precision)
                print(f"DEBUG BUY: sl_actual={sl_actual}, tp_actual={tp_actual}")
                if sl_actual is None:
                    print(f"ERROR: sl_actual became None after round() for {symbol} {side}. Inputs: price={price}, SL_PERCENT={SL_PERCENT}, precision={price_precision}")
                if tp_actual is None:
                    print(f"ERROR: tp_actual became None after round() for {symbol} {side}. Inputs: price={price}, TP_PERCENT={TP_PERCENT}, precision={price_precision}")
            elif side == 'sell':
                sl_actual = round(price + price * SL_PERCENT, price_precision)
                tp_actual = round(price - price * TP_PERCENT, price_precision)
                print(f"DEBUG SELL: sl_actual={sl_actual}, tp_actual={tp_actual}")
                if sl_actual is None:
                    print(f"ERROR: sl_actual became None after round() for {symbol} {side}. Inputs: price={price}, SL_PERCENT={SL_PERCENT}, precision={price_precision}")
                if tp_actual is None:
                    print(f"ERROR: tp_actual became None after round() for {symbol} {side}. Inputs: price={price}, TP_PERCENT={TP_PERCENT}, precision={price_precision}")
            print(f"Using percentage-based SL: {sl_actual} (from {SL_PERCENT*100}%), TP: {tp_actual} (from {TP_PERCENT*100}%) for {symbol} {side}")

        if sl_actual is None or tp_actual is None :
            print(f"Error: SL or TP price could not be determined for {symbol} {side}. Aborting order.")
            return

        # Validate that SL and TP are not impossible (e.g. SL same as entry for a buy)
        if side == 'buy':
            if sl_actual >= price: print(f"Warning: SL price {sl_actual} is at or above entry price {price} for BUY order on {symbol}. Check logic."); return
            if tp_actual <= price: print(f"Warning: TP price {tp_actual} is at or below entry price {price} for BUY order on {symbol}. Check logic."); return
        elif side == 'sell':
            if sl_actual <= price: print(f"Warning: SL price {sl_actual} is at or below entry price {price} for SELL order on {symbol}. Check logic."); return
            if tp_actual >= price: print(f"Warning: TP price {tp_actual} is at or above entry price {price} for SELL order on {symbol}. Check logic."); return


        if side == 'buy':
            resp1 = client.new_order(symbol=symbol, side='BUY', type='LIMIT', quantity=calculated_qty_asset, timeInForce='GTC', price=price, newOrderRespType='FULL')
            print(f"BUY {symbol}: {resp1}")
            sleep(0.2)
            resp2 = client.new_order(symbol=symbol, side='SELL', type='STOP_MARKET', quantity=calculated_qty_asset, timeInForce='GTC', stopPrice=sl_actual, reduceOnly=True, newOrderRespType='FULL')
            print(f"SL for BUY {symbol}: {resp2}")
            sleep(0.2)
            resp3 = client.new_order(symbol=symbol, side='SELL', type='TAKE_PROFIT_MARKET', quantity=calculated_qty_asset, timeInForce='GTC', stopPrice=tp_actual, reduceOnly=True, newOrderRespType='FULL')
            print(f"TP for BUY {symbol}: {resp3}")

            # --- BEGIN MODIFICATION: Update UI after BUY order ---
            sleep(0.5) # Allow time for Binance backend to process
            fresh_positions, fresh_open_orders = get_active_positions_data() # MODIFIED
            formatted_positions = format_positions_for_display(fresh_positions, fresh_open_orders) # MODIFIED
            if root and root.winfo_exists() and positions_text_widget:
                root.after(0, update_text_widget_content, positions_text_widget, formatted_positions)

            fresh_history = get_trade_history(symbol_list=TARGET_SYMBOLS, limit_per_symbol=10)
            if root and root.winfo_exists() and history_text_widget:
                root.after(0, update_text_widget_content, history_text_widget, fresh_history)
            # --- END MODIFICATION ---

        elif side == 'sell':
            resp1 = client.new_order(symbol=symbol, side='SELL', type='LIMIT', quantity=calculated_qty_asset, timeInForce='GTC', price=price, newOrderRespType='FULL')
            print(f"SELL {symbol}: {resp1}")
            sleep(0.2)
            resp2 = client.new_order(symbol=symbol, side='BUY', type='STOP_MARKET', quantity=calculated_qty_asset, timeInForce='GTC', stopPrice=sl_actual, reduceOnly=True, newOrderRespType='FULL')
            print(f"SL for SELL {symbol}: {resp2}")
            sleep(0.2)
            resp3 = client.new_order(symbol=symbol, side='BUY', type='TAKE_PROFIT_MARKET', quantity=calculated_qty_asset, timeInForce='GTC', stopPrice=tp_actual, reduceOnly=True, newOrderRespType='FULL')
            print(f"TP for SELL {symbol}: {resp3}")

            # --- BEGIN MODIFICATION: Update UI after SELL order ---
            sleep(0.5) # Allow time for Binance backend to process
            fresh_positions, fresh_open_orders = get_active_positions_data() # MODIFIED
            formatted_positions = format_positions_for_display(fresh_positions, fresh_open_orders) # MODIFIED
            if root and root.winfo_exists() and positions_text_widget:
                root.after(0, update_text_widget_content, positions_text_widget, formatted_positions)

            fresh_history = get_trade_history(symbol_list=TARGET_SYMBOLS, limit_per_symbol=10)
            if root and root.winfo_exists() and history_text_widget:
                root.after(0, update_text_widget_content, history_text_widget, fresh_history)
            # --- END MODIFICATION ---

    except ClientError as error:
        err_msg_prefix = f"Order Err ({symbol}, {side})"
        if error.error_code == -1111: # Standard Binance code for precision issues
            print(f"{err_msg_prefix}: PRECISION ISSUE. Check quantity/price decimal places. Binance msg: {error.error_message}")
        elif error.error_code == -4014: # Often related to MIN_NOTIONAL for Spot/Margin, UMFutures might use a different one like -4104
            print(f"{err_msg_prefix}: MIN_NOTIONAL or similar filter failure (Code -4014). Order value potentially too small. Binance msg: {error.error_message}")
        elif error.error_code == -4104: # Specific to UMFutures for MIN_NOTIONAL
            print(f"{err_msg_prefix}: MIN_NOTIONAL filter failure (Code -4104). Order value too small. Binance msg: {error.error_message}")
        elif error.error_code == -4003: # Often PRICE_FILTER for Spot/Margin, UMFutures might use -4105
            print(f"{err_msg_prefix}: PRICE_FILTER or similar failure (Code -4003). Price out of bounds or invalid. Binance msg: {error.error_message}")
        elif error.error_code == -4105: # Specific to UMFutures for PRICE_FILTER
            print(f"{err_msg_prefix}: PRICE_FILTER failure (Code -4105). Price out of bounds or invalid. Binance msg: {error.error_message}")
        elif error.error_code == -2010: # Typical for insufficient balance
            print(f"{err_msg_prefix}: INSUFFICIENT BALANCE (Code -2010). Check available margin/funds. Binance msg: {error.error_message}")
        else:
            print(f"{err_msg_prefix}: Code {error.error_code} - {error.error_message if error.error_message else error}")
    except Exception as e: print(f"Order Unexpected Err ({symbol}, {side}): {e}")

def get_active_positions_data(): # Renamed from get_pos for clarity
    global client, status_var, root
    if not client:
        if root and root.winfo_exists() and status_var: root.after(0, lambda: status_var.set("Client not init. Cannot get positions."))
        return None, None # Return None for both positions and orders

    positions_data = []
    open_orders_by_symbol = {}
    active_symbols = set()

    try:
        raw_positions = client.get_position_risk(recvWindow=6000) # Added recvWindow
        if raw_positions:
            for p_data in raw_positions:
                try:
                    pos_amt_float = float(p_data.get('positionAmt', '0'))
                    if pos_amt_float != 0:
                        symbol = p_data['symbol']
                        active_symbols.add(symbol)
                        positions_data.append({
                            'symbol': symbol,
                            'qty': pos_amt_float,
                            'entry_price': float(p_data.get('entryPrice', '0')),
                            'mark_price': float(p_data.get('markPrice', '0')),
                            'pnl': float(p_data.get('unRealizedProfit', '0')),
                            'leverage': p_data.get('leverage', 'N/A')
                        })
                except ValueError as ve:
                    print(f"ValueError parsing position data for {p_data.get('symbol')}: {ve}")
                    continue
        
        # Fetch open orders for active symbols
        for symbol in active_symbols:
            try:
                # print(f"DEBUG: Fetching open orders for active symbol: {symbol}") # Debug print
                orders = client.get_orders(symbol=symbol, recvWindow=6000)
                open_orders_by_symbol[symbol] = [o for o in orders if o.get('status') in ['NEW', 'PARTIALLY_FILLED']]
                # print(f"DEBUG: Found {len(open_orders_by_symbol[symbol])} open orders for {symbol}") # Debug print
            except ClientError as ce:
                print(f"ClientError fetching orders for {symbol}: {ce.error_message if hasattr(ce, 'error_message') else ce}")
                open_orders_by_symbol[symbol] = [] # Store empty list on error
            except Exception as e_orders:
                print(f"Generic error fetching orders for {symbol}: {e_orders}")
                open_orders_by_symbol[symbol] = []


        return positions_data, open_orders_by_symbol

    except ClientError as e:
        msg = f"API Error (Positions/Orders): {e.error_message[:40] if hasattr(e, 'error_message') and e.error_message else str(e)[:40]}"
        print(msg)
        if root and root.winfo_exists() and status_var: root.after(0, lambda s=msg: status_var.set(s))
        return None, None # Return None for both
    except Exception as e_gen:
        msg = f"Error (Positions/Orders): {str(e_gen)[:40]}"
        print(msg)
        if root and root.winfo_exists() and status_var: root.after(0, lambda s=msg: status_var.set(s))
        return None, None # Return None for both

def format_positions_for_display(positions_data_list, open_orders_by_symbol=None):
    if open_orders_by_symbol is None: # Graceful handling if None is passed
        open_orders_by_symbol = {}
        
    # Remove or comment out the placeholder print statement
    # if open_orders_by_symbol: 
    #     print(f"DEBUG: format_positions_for_display received {len(open_orders_by_symbol)} symbols with open orders information.")

    if positions_data_list is None: # This check is for the positions_data_list itself
        return ["Error fetching positions or client not ready."]
    if not positions_data_list:
        return ["No open positions."]

    formatted_strings = []
    for p in positions_data_list:
        symbol = p['symbol']
        tp_price_str = "N/A"
        sl_price_str = "N/A"

        if open_orders_by_symbol and symbol in open_orders_by_symbol:
            orders_for_symbol = open_orders_by_symbol[symbol]
            for order in orders_for_symbol:
                # Binance API typically uses boolean for reduceOnly.
                # Making the check robust for potential stringified booleans as well, though less likely.
                is_reduce_only = False
                reduce_only_val = order.get('reduceOnly')
                if isinstance(reduce_only_val, bool):
                    is_reduce_only = reduce_only_val
                elif isinstance(reduce_only_val, str): # Handle 'true'/'false' if they ever appear
                    is_reduce_only = reduce_only_val.lower() == 'true'

                if is_reduce_only:
                    order_type = order.get('type')
                    stop_price = order.get('stopPrice', "N/A") # Default to N/A if stopPrice key is missing
                    if order_type == 'TAKE_PROFIT_MARKET':
                        tp_price_str = str(stop_price) # Ensure it's a string
                    elif order_type == 'STOP_MARKET':
                        sl_price_str = str(stop_price) # Ensure it's a string
        
        formatted_strings.append(
            f"Sym: {p['symbol']}, Qty: {p['qty']}, Entry: {p['entry_price']:.4f}, " +
            f"SL: {sl_price_str}, TP: {tp_price_str}, MarkP: {p['mark_price']:.4f}, " +
            f"PnL: {p['pnl']:.2f} USDT, Lev: {p['leverage']}"
        )
    return formatted_strings

def get_trade_history(symbol_list=['BTCUSDT', 'ETHUSDT'], limit_per_symbol=15):
    global client, status_var, root
    if not client: return ["Client not initialized. Cannot fetch trade history."]

    all_trades_formatted = []
    if not isinstance(symbol_list, list): symbol_list = [symbol_list]

    for symbol in symbol_list:
        try:
            trades = client.account_trades(symbol=symbol, limit=limit_per_symbol)
            if trades: all_trades_formatted.append(f"--- {symbol} (Last {len(trades)}) ---")
            for trade in trades:
                trade_time = pd.to_datetime(trade['time'], unit='ms')
                all_trades_formatted.append(
                    f"{trade_time.strftime('%y-%m-%d %H:%M:%S')} | {trade['side']} | P: {trade['price']} | Q: {trade['qty']} | Fee: {trade['commission']}{trade['commissionAsset']}"
                )
        except ClientError as e:
            msg = f"History Err ({symbol}): {e.error_message[:35] if hasattr(e, 'error_message') and e.error_message else str(e)[:35]}"
            all_trades_formatted.append(msg)
            if root and root.winfo_exists() and status_var: root.after(0, lambda s=msg: status_var.set(s))
        except Exception as e_gen:
            msg = f"History Err ({symbol}): {str(e_gen)[:35]}"
            all_trades_formatted.append(msg)
            if root and root.winfo_exists() and status_var: root.after(0, lambda s=msg: status_var.set(s))

    if not all_trades_formatted: return ["No trades found or error during fetch."]
    return all_trades_formatted

def check_orders():
    global client
    if not client: return []
    try: return client.get_orders(recvWindow=6000)
    except ClientError as error: print(f"Error checking orders: {error}")
    except Exception as e: print(f"Unexpected error checking orders: {e}")
    return []

def close_open_orders(symbol):
    global client
    if not client: return
    try:
        response = client.cancel_open_orders(symbol=symbol, recvWindow=6000)
        print(f"Closing open orders for {symbol}: {response}")
    except ClientError as error: print(f"Error closing open orders for {symbol}: {error}")
    except Exception as e: print(f"Unexpected error closing open orders for {symbol}: {e}")

# --- Account Summary Helper Functions ---
def get_last_7_days_profit(client_instance):
    global TARGET_SYMBOLS
    if not client_instance:
        print("Error: Client not provided for get_last_7_days_profit.")
        return 0.0

    seven_days_ago_ms = int((pd.Timestamp.now(tz='UTC') - pd.Timedelta(days=7)).timestamp() * 1000)
    total_profit = 0.0

    print(f"Fetching trades for last 7 days for symbols: {TARGET_SYMBOLS}")
    for symbol in TARGET_SYMBOLS:
        try:
            trades = client_instance.account_trades(symbol=symbol, startTime=seven_days_ago_ms, recvWindow=6000)
            symbol_profit = 0.0
            for trade in trades:
                price = float(trade['price'])
                qty = float(trade['qty'])
                commission = float(trade['commission'])
                # commission_asset = trade['commissionAsset'] # Assuming commission is in quote asset (USDT)

                # PNL calculation: (price * qty * (1 if side=='SELL' else -1)) - commission
                if trade['side'].upper() == 'SELL':
                    symbol_profit += (price * qty)
                else: # BUY
                    symbol_profit -= (price * qty)
                symbol_profit -= commission # Subtract commission regardless of side
            
            print(f"Symbol: {symbol}, 7-Day Profit (after commissions): {symbol_profit}")
            total_profit += symbol_profit
        except ClientError as ce:
            print(f"ClientError fetching 7-day trades for {symbol}: {ce}")
        except Exception as e:
            print(f"Error processing 7-day trades for {symbol}: {e}")
    
    print(f"Total 7-Day Profit for all target symbols: {total_profit}")
    return total_profit

def get_overall_profit_loss(client_instance):
    global TARGET_SYMBOLS
    if not client_instance:
        print("Error: Client not provided for get_overall_profit_loss.")
        return 0.0

    total_pnl = 0.0
    print(f"Fetching all trades for symbols: {TARGET_SYMBOLS} (limit 1000 per symbol)")

    for symbol in TARGET_SYMBOLS:
        try:
            # Fetch trades in batches if necessary, for now using limit=1000
            # To get ALL trades, one might need to loop with fromId parameter
            trades = client_instance.account_trades(symbol=symbol, limit=1000, recvWindow=6000) # Max limit is 1000
            symbol_pnl = 0.0
            for trade in trades:
                price = float(trade['price'])
                qty = float(trade['qty'])
                commission = float(trade['commission'])
                # commission_asset = trade['commissionAsset']

                if trade['side'].upper() == 'SELL':
                    symbol_pnl += (price * qty)
                else: # BUY
                    symbol_pnl -= (price * qty)
                symbol_pnl -= commission
            
            print(f"Symbol: {symbol}, Overall PNL (after commissions, based on last 1000 trades): {symbol_pnl}")
            total_pnl += symbol_pnl
        except ClientError as ce:
            print(f"ClientError fetching overall trades for {symbol}: {ce}")
        except Exception as e:
            print(f"Error processing overall trades for {symbol}: {e}")
            
    print(f"Total Overall PNL for all target symbols (based on last 1000 trades per symbol): {total_pnl}")
    return total_pnl

def get_total_unrealized_pnl(client_instance):
    if not client_instance:
        print("Error: Client not provided for get_total_unrealized_pnl.")
        return 0.0

    positions_data, _ = get_active_positions_data() # Uses global client by default, but good practice to pass if refactoring
    
    total_un_pnl = 0.0
    if positions_data:
        for position in positions_data:
            try:
                total_un_pnl += float(position.get('pnl', 0.0))
            except ValueError:
                print(f"Warning: Could not parse PNL for position {position.get('symbol')}")
                continue
        print(f"Total Unrealized PNL: {total_un_pnl}")
        return total_un_pnl
    else:
        # This case includes errors from get_active_positions_data or no open positions
        print("No active positions or error fetching them for unrealized PNL calculation.")
        return 0.0

def update_account_summary_data(): # Renamed from populate_initial_account_summary
    global client, root # Ensure access to global client and root
    global account_summary_balance_var, last_7_days_profit_var, overall_profit_loss_var, total_unrealized_pnl_var

    print("Updating account summary data...") # Changed print statement

    if not client:
        print("Client not initialized. Cannot update account summary data.") # Changed print statement
        if root and root.winfo_exists(): # Check if UI elements exist
            account_summary_balance_var.set("Client N/A")
            last_7_days_profit_var.set("Client N/A")
            overall_profit_loss_var.set("Client N/A")
            total_unrealized_pnl_var.set("Client N/A")
        return

    # Fetch data using existing helper functions
    # Balance
    balance = get_balance_usdt() # Uses global client
    if root and root.winfo_exists():
        root.after(0, lambda bal=balance: account_summary_balance_var.set(f"{bal:.2f} USDT" if bal is not None else "N/A"))

    # Last 7 Days Profit
    seven_day_profit = get_last_7_days_profit(client)
    if root and root.winfo_exists():
        root.after(0, lambda p=seven_day_profit: last_7_days_profit_var.set(f"{p:.2f} USDT" if p is not None else "N/A"))

    # Overall Profit/Loss
    overall_pnl = get_overall_profit_loss(client)
    if root and root.winfo_exists():
        root.after(0, lambda pnl=overall_pnl: overall_profit_loss_var.set(f"{pnl:.2f} USDT" if pnl is not None else "N/A"))

    # Total Unrealized PNL
    unrealized_pnl = get_total_unrealized_pnl(client)
    if root and root.winfo_exists():
        root.after(0, lambda upnl=unrealized_pnl: total_unrealized_pnl_var.set(f"{upnl:.2f} USDT" if upnl is not None else "N/A"))
    
    print("Account summary data update attempt complete.") # Changed print statement

def scheduled_account_summary_update():
    global root, client # bot_running no longer directly controls this function's execution flow

    # Check if the root window still exists before proceeding
    if not (root and root.winfo_exists()):
        print("Root window closed, stopping continuous account summary updates.")
        return

    # Attempt to update data regardless of bot state; update_account_summary_data handles client check
    # Also, client is checked here to avoid unnecessary print/call if known to be None globally already.
    if client:
        print("Continuously updating account summary data (every 5s)...")
        update_account_summary_data()
    else:
        # If client is None, update_account_summary_data will set fields to "Client N/A"
        # So, we can call it to ensure UI reflects the "Client N/A" state correctly.
        print("Client not available. Account summary will show 'Client N/A'.")
        update_account_summary_data() 

    # Always reschedule if root window exists, to keep the update loop alive
    # as long as the application is running.
    root.after(5000, scheduled_account_summary_update) # 5000ms = 5 seconds

# --- GUI Control Functions & Bot Logic --- ( Largely unchanged, but run_bot_logic will use new get_pos )

def reinitialize_client():
    global client, current_env, status_var, current_env_var, api_mainnet, secret_mainnet, api_testnet, secret_testnet, BINANCE_MAINNET_URL, BINANCE_TESTNET_URL
    _status_set = lambda msg: status_var.set(msg) if status_var and root and root.winfo_exists() else None
    print(f"Attempting to reinitialize client for: {current_env}")
    if current_env == "testnet":
        if not api_testnet or not secret_testnet:
            msg = "Error: Testnet keys missing."; _status_set(msg); messagebox.showerror("Error", msg)
            return False
        print("Attempting to use Testnet API keys.")
        try:
            client = UMFutures(key=api_testnet, secret=secret_testnet, base_url=BINANCE_TESTNET_URL)
            print(f"Client initialized with base URL: {client.base_url}")
            msg = "Client: Binance Testnet"; print(msg); _status_set(msg)
        except ClientError as e:
            if e.error_code == -2015:
                msg = (
                    "Error Testnet client: Invalid API Key or Permissions (-2015).\n"
                    "Ensure your Testnet API key is valid, has 'Enable Futures' permission, "
                    "and IP restrictions (if any) are correctly set on Binance Testnet."
                )
            else:
                msg = f"Error Testnet client: {e.error_code} - {e.error_message}"
            print(msg); _status_set(msg); messagebox.showerror("Client Error", msg)
            return False
        except Exception as e:
            msg = f"Error Testnet client: {e}"; print(msg); _status_set(msg); messagebox.showerror("Client Error", msg)
            return False
    elif current_env == "mainnet":
        if not api_mainnet or not secret_mainnet:
            msg = "Error: Mainnet keys missing."; _status_set(msg); messagebox.showerror("Error", msg + " Cannot switch.")
            current_env = "testnet";
            if current_env_var: current_env_var.set("testnet")
            print("Attempting to use Testnet API keys for fallback.")
            try:
                client = UMFutures(key=api_testnet, secret=secret_testnet, base_url=BINANCE_TESTNET_URL)
                print(f"Client initialized with base URL: {client.base_url}")
                _status_set("Fell back to Testnet.")
            except Exception as e_fb: client = None; _status_set(f"Fallback to Testnet failed: {e_fb}")
            return False
        print("Attempting to use Mainnet API keys.")
        try:
            client = UMFutures(key=api_mainnet, secret=secret_mainnet, base_url=BINANCE_MAINNET_URL)
            print(f"Client initialized with base URL: {client.base_url}")
            msg = "Client: Binance Mainnet"; print(msg); _status_set(msg)
        except ClientError as e:
            if e.error_code == -2015:
                msg = (
                    "Error Mainnet client: Invalid API Key or Permissions (-2015).\n"
                    "Ensure your Mainnet API key is valid, has 'Enable Futures' permission, "
                    "and IP restrictions (if any) are correctly set on Binance."
                )
            else:
                msg = f"Error Mainnet client: {e.error_code} - {e.error_message}"
            print(msg); _status_set(msg); messagebox.showerror("Client Error", msg)
            return False
        except Exception as e:
            msg = f"Error Mainnet client: {e}"; print(msg); _status_set(msg); messagebox.showerror("Client Error", msg)
            return False
    else:
        msg = "Error: Invalid environment."; _status_set(msg); messagebox.showerror("Error", msg)
        return False
    return True

def toggle_environment():
    global current_env, bot_running, status_var, current_env_var
    new_env = current_env_var.get()
    _status_set = lambda msg: status_var.set(msg) if status_var and root and root.winfo_exists() else None
    original_env = current_env

    print(f"Attempting to switch environment from {original_env} to {new_env}")

    if bot_running:
        messagebox.showwarning("Warning", "Change environment only when bot is stopped.")
        current_env_var.set(current_env); # Revert GUI selection
        print(f"Switch aborted: Bot is running. Environment remains {current_env}.")
        return

    if current_env != new_env:
        current_env = new_env # Tentatively set current_env for reinitialize_client to use
        _status_set(f"Switching to {current_env}...")
        if reinitialize_client():
            _status_set(f"Client ready for {current_env}.")
            print(f"Environment switch successful to {current_env}")
        else:
            # reinitialize_client might have fallen back to testnet on mainnet key failure
            # so, current_env might have been changed inside reinitialize_client
            # We need to reflect the actual environment if it changed, or revert if it truly failed.
            if current_env != original_env and client is not None : # It means fallback occurred and client is set
                 _status_set(f"Failed switch to {new_env}, but fell back to {current_env} successfully.")
                 print(f"Environment switch: Failed to switch to {new_env}, fell back to {current_env}.")
                 if current_env_var: current_env_var.set(current_env) # Update GUI to reflect fallback
            else: # True failure or no fallback occurred, revert to original
                current_env = original_env # Revert to original_env as the switch failed
                current_env_var.set(current_env) # Revert GUI selection
                _status_set(f"Failed switch to {new_env}. Now: {current_env}.")
                print(f"Environment switch failed, remaining on {current_env}")
    else: # current_env == new_env, so just re-initializing
        _status_set(f"Re-initializing for {current_env}...");
        if reinitialize_client():
            print(f"Client re-initialized successfully for {current_env}.")
        else:
            # If re-init fails, it might clear the client or fallback.
            # The status_var is already set by reinitialize_client on failure.
            print(f"Client re-initialization failed for {current_env}.")
            # Ensure GUI reflects the state if a fallback happened during re-init of same env (e.g. mainnet keys removed)
            if current_env_var and current_env_var.get() != current_env:
                current_env_var.set(current_env)

def run_bot_logic():
    global bot_running, status_var, client, start_button, stop_button, testnet_radio, mainnet_radio
    global qty_concurrent_positions, margin_type_setting, leverage, g_conditional_pending_signals, current_price_var # Changed pending_signals to g_conditional_pending_signals
    global balance_var, positions_text_widget, history_text_widget, activity_status_var
    # Account Summary StringVars for UI update
    global account_summary_balance_var, last_7_days_profit_var, overall_profit_loss_var, total_unrealized_pnl_var
    # Make all strategy-specific trackers global for modification
    global strategy1_active_trade_info, strategy1_cooldown_active, strategy1_last_trade_was_loss
    global strategy2_active_trades
    global strategy3_active_trade_info
    global strategy4_active_trade_info
    # And their default reset structures
    global strategy1_active_trade_info_default, strategy3_active_trade_info_default, strategy4_active_trade_info_default


    _status_set = lambda msg: status_var.set(msg) if status_var and root and root.winfo_exists() else None
    _activity_set = lambda msg: activity_status_var.set(msg) if activity_status_var and root and root.winfo_exists() else None
    
    print("Bot logic thread started."); _status_set("Bot running...")
    if client is None:
        _status_set("Error: Client not initialized. Bot stopping.")
        _activity_set("Bot Idle - Client Error")
        bot_running = False
        # No return here, allow cleanup at the end of function if bot_running is False

    loop_count = 0
    while bot_running:
        try:
            _activity_set("Starting new scan cycle...")
            if client is None:
                _status_set("Client is None. Bot stopping."); _activity_set("Bot Idle - Client Error")
                bot_running = False; break
            
            balance = get_balance_usdt(); sleep(0.1)
            if not bot_running: break

            if root and root.winfo_exists() and balance_var:
                if balance is not None: 
                    root.after(0, lambda bal=balance: balance_var.set(f"{bal:.2f} USDT"))
                    # The line updating account_summary_balance_var has been removed here
                else: 
                    root.after(0, lambda: balance_var.set("Error or N/A"))
                    # The line updating account_summary_balance_var has been removed here
                    # (It was implicitly covered by the `if balance is not None` before,
                    #  so if balance is None, account_summary_balance_var would also be set to N/A by scheduled update)


            if balance is None: # This check should be before trying to use client for other fetches
                msg = 'API/balance error. Retrying...'; print(msg); _status_set(msg); _activity_set("Balance fetch error...")
                for _ in range(60): 
                    if not bot_running: break
                    sleep(1)
                if not bot_running: break
                continue

            current_balance_msg_for_status = f"Bal: {balance:.2f} USDT." if balance is not None else "Bal: Error"

            # --- Update Positions and History UI in main loop ---
            # Note: get_total_unrealized_pnl already calls get_active_positions_data.
            # We can reuse its result or call it again if needed for open_orders_for_ui specifically.
            # For now, let's assume get_active_positions_data is called again for simplicity here,
            # or modify get_total_unrealized_pnl to return both if optimization is critical.
            if bot_running: 
                current_positions_for_ui, open_orders_for_ui = get_active_positions_data() 
                formatted_current_positions_for_ui = format_positions_for_display(current_positions_for_ui, open_orders_for_ui) 
                if root and root.winfo_exists() and positions_text_widget:
                    root.after(0, update_text_widget_content, positions_text_widget, formatted_current_positions_for_ui)
                
                current_history_for_ui = get_trade_history(symbol_list=TARGET_SYMBOLS, limit_per_symbol=12)
                if root and root.winfo_exists() and history_text_widget:
                    root.after(0, update_text_widget_content, history_text_widget, current_history_for_ui)
                
                sleep(0.1) 
            
            # --- Manage Active Conditional Pending Signals (Part 2) ---
            if g_conditional_pending_signals: 
                _activity_set(f"Managing {len(g_conditional_pending_signals)} conditional signals...")
                current_time_utc_for_pending_manage = pd.Timestamp.now(tz='UTC')

                for key, item in list(g_conditional_pending_signals.items()):
                    if not bot_running: break
                    
                    symbol = item['symbol']
                    strategy_id = item['strategy_id'] # This is item['strategy_id'], not current_active_strategy_id
                    side = item['side']
                    timestamp = item['timestamp']
                    conditions_to_start_wait_threshold = item['conditions_to_start_wait_threshold']
                    conditions_for_full_signal_threshold = item['conditions_for_full_signal_threshold'] # Get from item

                    time_elapsed_seconds = (current_time_utc_for_pending_manage - timestamp).total_seconds()

                    current_strategy_output = {}
                    if strategy_id == 0: current_strategy_output = scalping_strategy_signal(symbol)
                    elif strategy_id == 1: current_strategy_output = strategy_ema_supertrend(symbol)
                    elif strategy_id == 2: current_strategy_output = strategy_bollinger_band_mean_reversion(symbol)
                    elif strategy_id == 3: current_strategy_output = strategy_vwap_breakout_momentum(symbol)
                    # Strategy 4 is not part of g_conditional_pending_signals yet
                    
                    current_met_count = 0
                    if current_strategy_output and current_strategy_output.get('all_conditions_status'):
                        all_conds = current_strategy_output['all_conditions_status']
                        if strategy_id == 0:
                            if side == 'up': current_met_count = all_conds.get('num_buy_conditions_met', 0)
                            elif side == 'down': current_met_count = all_conds.get('num_sell_conditions_met', 0)
                        elif strategy_id == 1:
                            if side == 'up': relevant_keys = ['ema_cross_up', 'st_green', 'rsi_long_ok']
                            else: relevant_keys = ['ema_cross_down', 'st_red', 'rsi_short_ok']
                            current_met_count = sum(1 for k, v in all_conds.items() if k in relevant_keys and v)
                        elif strategy_id == 2:
                            if side == 'up': relevant_keys = ['price_below_lower_bb', 'rsi_oversold', 'volume_confirms_long']
                            else: relevant_keys = ['price_above_upper_bb', 'rsi_overbought', 'volume_confirms_short']
                            current_met_count = sum(1 for k, v in all_conds.items() if k in relevant_keys and v)
                        elif strategy_id == 3:
                            if side == 'up': relevant_keys = ['price_above_vwap_2bar_long', 'macd_positive_rising', 'atr_volatility_confirms_long']
                            else: relevant_keys = ['price_below_vwap_2bar_short', 'macd_negative_falling', 'atr_volatility_confirms_short']
                            current_met_count = sum(1 for k, v in all_conds.items() if k in relevant_keys and v)

                    if key in g_conditional_pending_signals: # Check if not deleted by another condition
                        g_conditional_pending_signals[key]['current_conditions_met_count'] = current_met_count
                        if current_strategy_output: # Ensure output exists
                            g_conditional_pending_signals[key]['last_evaluated_all_conditions_status'] = current_strategy_output.get('all_conditions_status', {}).copy()
                            g_conditional_pending_signals[key]['potential_sl_price'] = current_strategy_output.get('sl_price')
                            g_conditional_pending_signals[key]['potential_tp_price'] = current_strategy_output.get('tp_price')
                            if strategy_id == 0: # Update threshold for S0 as it's dynamic
                                g_conditional_pending_signals[key]['conditions_for_full_signal_threshold'] = current_strategy_output.get('conditions_for_full_signal_threshold', conditions_for_full_signal_threshold)
                                conditions_for_full_signal_threshold = g_conditional_pending_signals[key]['conditions_for_full_signal_threshold'] # update local var for current iteration

                    remaining_seconds = max(0, 300 - time_elapsed_seconds)
                    
                    # Enhanced UI Update for pending signals
                    if strategy_id == 0: # Check item's strategy_id here
                        live_signal_is_true = current_strategy_output.get('signal') != 'none' if current_strategy_output else False
                        s0_pending_msg = (f"S0 Pending: Confirming {symbol} ({side}) - {int(remaining_seconds)}s left. "
                                          f"Last eval: {'TRUE' if live_signal_is_true else 'FALSE'}.")
                        _activity_set(s0_pending_msg)
                    else:
                        # Generic message for other strategies if they use this pending system
                        _activity_set(f"S{strategy_id} {symbol} ({side}): {current_met_count}/{conditions_for_full_signal_threshold} cond. {int(remaining_seconds)}s left.")

                    if root and root.winfo_exists() and current_strategy_output:
                         root.after(0, update_conditions_display_content, symbol, current_strategy_output.get('all_conditions_status'), current_strategy_output.get('error'))

                    if current_met_count >= conditions_for_full_signal_threshold and not current_strategy_output.get('error'):
                        print(f"CONFIRMED: {STRATEGIES[strategy_id]} for {symbol} ({side}). Met {current_met_count}/{conditions_for_full_signal_threshold}.")
                        _status_set(f"Ordering {symbol} ({side}) via {STRATEGIES[strategy_id]}...")
                        set_mode(symbol, margin_type_setting); sleep(0.1)
                        set_leverage(symbol, leverage); sleep(0.1)
                        
                        sl_to_use = g_conditional_pending_signals[key]['potential_sl_price'] # Use the latest from re-eval
                        tp_to_use = g_conditional_pending_signals[key]['potential_tp_price'] # Use the latest from re-eval
                        risk_percent = current_strategy_output.get('account_risk_percent')
                        
                        entry_price_for_tracking = None
                        try: entry_price_for_tracking = float(client.ticker_price(symbol)['price'])
                        except Exception as e_track_price: print(f"Error fetching entry price for tracking {symbol}: {e_track_price}")

                        open_order(symbol, side, strategy_sl=sl_to_use, strategy_tp=tp_to_use, strategy_account_risk_percent=risk_percent)
                        
                        trade_entry_ts = pd.Timestamp.now(tz='UTC')
                        if strategy_id == 1: strategy1_active_trade_info.update({'symbol': symbol, 'entry_time': trade_entry_ts, 'entry_price': entry_price_for_tracking, 'side': side})
                        elif strategy_id == 2: strategy2_active_trades.append({'symbol': symbol, 'entry_time': trade_entry_ts, 'entry_price': entry_price_for_tracking, 'side': side})
                        elif strategy_id == 3: strategy3_active_trade_info.update({'symbol': symbol, 'entry_time': trade_entry_ts, 'entry_price': entry_price_for_tracking, 'side': side, 'initial_atr_for_profit_target': current_strategy_output.get('last_atr'), 'vwap_trail_active': False })
                        
                        if key in g_conditional_pending_signals: del g_conditional_pending_signals[key]
                        sleep(1)
                        break    # Trade initiated, break from processing more pending signals this cycle

                    if time_elapsed_seconds >= 300:
                        print(f"TIMEOUT: {STRATEGIES[strategy_id]} for {symbol} ({side}). Did not confirm. Met {current_met_count}/{conditions_for_full_signal_threshold}.")
                        _activity_set(f"TIMEOUT: S{strategy_id} for {symbol} ({side}).")
                        if key in g_conditional_pending_signals: del g_conditional_pending_signals[key]
                        continue

                    if current_met_count < conditions_to_start_wait_threshold:
                        print(f"KILLED (Degraded): {STRATEGIES[strategy_id]} for {symbol} ({side}). Conditions fell to {current_met_count}/{conditions_to_start_wait_threshold}.")
                        _activity_set(f"KILLED (Degraded): S{strategy_id} for {symbol} ({side}).")
                        if key in g_conditional_pending_signals: del g_conditional_pending_signals[key]
                        continue
                    
                    if current_strategy_output.get('error'):
                        print(f"KILLED (Error on re-eval): {STRATEGIES[strategy_id]} for {symbol} ({side}). Error: {current_strategy_output['error']}")
                        _activity_set(f"KILLED (Error): S{strategy_id} for {symbol} ({side}).")
                        if key in g_conditional_pending_signals: del g_conditional_pending_signals[key]
                        continue
                    sleep(0.1) 
            
            _status_set(current_balance_msg_for_status + " Managing active trades & timeouts...")
            # --- Active Trade Management & Timeouts (BEFORE checking for new trades) ---
            now_utc_for_timeout = pd.Timestamp.now(tz='UTC')

            # Strategy 1 Timeout
            if strategy1_active_trade_info['symbol'] is not None and strategy1_active_trade_info['entry_time'] is not None:
                s1_active_symbol = strategy1_active_trade_info['symbol']
                s1_entry_time = strategy1_active_trade_info['entry_time']
                try:
                    kl_s1 = klines(s1_active_symbol) # Fetches 5m klines by default
                    if kl_s1 is not None and not kl_s1.empty and s1_entry_time.tzinfo is not None:
                        # Ensure entry_time is localized if kl_s1.index is localized (it should be UTC)
                        # Get_loc might fail if entry_time is not found or not unique after 'nearest'
                        # This assumes klines are sorted and entry_time is a valid timestamp from a previous kline
                        try:
                            # Ensure s1_entry_time is timezone-aware (UTC) if kl_s1.index is.
                            # kl_s1.index should be UTC from klines()
                            if s1_entry_time.tzinfo is None and kl_s1.index.tzinfo is not None:
                                s1_entry_time = s1_entry_time.tz_localize(kl_s1.index.tzinfo) # Localize to kline's tz
                            elif s1_entry_time.tzinfo is not None and kl_s1.index.tzinfo is None:
                                # This case is less likely if klines() always returns tz-aware
                                print(f"Warning: s1_entry_time is tz-aware but kline index is not for {s1_active_symbol}. Comparing directly.")
                            elif s1_entry_time.tzinfo is not None and kl_s1.index.tzinfo is not None and s1_entry_time.tzinfo != kl_s1.index.tzinfo:
                                s1_entry_time = s1_entry_time.tz_convert(kl_s1.index.tzinfo) # Convert to kline's tz

                            closest_ts = kl_s1.index.asof(s1_entry_time)

                            if pd.isna(closest_ts):
                                print(f"Strategy 1: Entry time {s1_entry_time} for {s1_active_symbol} not found within kline range for timeout check (asof returned NaT).")
                                entry_candle_index = -1 # Sentinel to indicate not found
                            else:
                                entry_candle_index = kl_s1.index.get_loc(closest_ts)

                            if entry_candle_index != -1: # Proceed only if found
                                candles_passed = len(kl_s1) - 1 - entry_candle_index
                                if candles_passed >= 3: # Original timeout logic
                                    print(f"Strategy 1: Closing {s1_active_symbol} due to 3-candle timeout (candles passed: {candles_passed}).")
                                    close_open_orders(s1_active_symbol)
                        except Exception as e_s1_timeout_loc: # Broader exception catch for safety
                            print(f"Strategy 1: Error during entry time lookup for {s1_active_symbol}: {e_s1_timeout_loc}")
                        
                except Exception as e_to_s1: print(f"Error during S1 timeout check for {s1_active_symbol}: {e_to_s1}")
            
            # Strategy 2 Timeout
            for trade_info in list(strategy2_active_trades): # Iterate a copy for safe removal
                s2_active_symbol = trade_info['symbol']
                s2_entry_time = trade_info['entry_time']
                try:
                    kl_s2 = klines(s2_active_symbol)
                    if kl_s2 is not None and not kl_s2.empty and s2_entry_time.tzinfo is not None:
                        try:
                            entry_candle_index = kl_s2.index.get_loc(s2_entry_time, method='nearest')
                            candles_passed = len(kl_s2) - 1 - entry_candle_index
                            if candles_passed >= 2:
                                print(f"Strategy 2: Closing {s2_active_symbol} due to 2-candle timeout.")
                                close_open_orders(s2_active_symbol)
                        except KeyError:
                             print(f"Strategy 2: Entry time for {s2_active_symbol} not found in recent klines for timeout.")
                except Exception as e_to_s2: print(f"Error during S2 timeout check for {s2_active_symbol}: {e_to_s2}")

            # Strategy 4 Timeout (End of Day)
            if strategy4_active_trade_info['symbol'] is not None and strategy4_active_trade_info['entry_time'] is not None:
                s4_active_symbol = strategy4_active_trade_info['symbol']
                s4_entry_time = strategy4_active_trade_info['entry_time'] # This should be a pd.Timestamp
                if now_utc_for_timeout.date() > s4_entry_time.date():
                    print(f"Strategy 4: Closing {s4_active_symbol} due to end-of-day timeout (current: {now_utc_for_timeout.date()}, entry: {s4_entry_time.date()}).")
                    close_open_orders(s4_active_symbol)
            
            if not bot_running: break
            # Refresh positions AND open orders after potential timeout actions
            active_positions_data, active_open_orders_data = get_active_positions_data() # MODIFIED
            if not bot_running: break
            open_position_symbols = [p['symbol'] for p in active_positions_data] if active_positions_data else []
            # active_open_orders_data is now available if needed for further logic here

            # --- Position Closure Detection and Resetting Strategy Trackers ---
            s1_sym = strategy1_active_trade_info['symbol']
            if s1_sym and s1_sym not in open_position_symbols:
                print(f"Strategy 1: Detected closure of trade on {s1_sym}.")
                strategy1_last_trade_was_loss = True 
                strategy1_cooldown_active = True
                _activity_set(f"S1: {s1_sym} closed. Cooldown for 1 cycle.")
                strategy1_active_trade_info = strategy1_active_trade_info_default.copy()

            current_s2_trades = list(strategy2_active_trades) # Iterate over a copy
            for trade_info in current_s2_trades:
                s2_sym = trade_info['symbol']
                if s2_sym not in open_position_symbols:
                    print(f"Strategy 2: Detected closure of trade on {s2_sym}.")
                    strategy2_active_trades.remove(trade_info) # remove from original list
                    _activity_set(f"S2: {s2_sym} closed. Available S2 slots: {2-len(strategy2_active_trades)}.")
            
            s3_sym = strategy3_active_trade_info['symbol']
            if s3_sym and s3_sym not in open_position_symbols:
                print(f"Strategy 3: Detected closure of trade on {s3_sym}.")
                strategy3_active_trade_info = strategy3_active_trade_info_default.copy()
                _activity_set(f"S3: {s3_sym} closed.")

            s4_sym = strategy4_active_trade_info['symbol']
            if s4_sym and s4_sym not in open_position_symbols:
                print(f"Strategy 4: Detected closure of trade on {s4_sym}.")
                strategy4_active_trade_info = strategy4_active_trade_info_default.copy()
                _activity_set(f"S4: {s4_sym} closed.")

            if not bot_running: break
            _status_set(current_balance_msg_for_status + " Determining ability to open new trades...")

            # --- Determine if a new trade can be opened based on current ACTIVE_STRATEGY_ID ---
            can_open_new_trade_overall = False
            current_active_strategy_id = ACTIVE_STRATEGY_ID 
            
            activity_msg_prefix = f"S{current_active_strategy_id}: "
            if current_active_strategy_id == 0:
                # For strategy 0, can_open_new_trade_overall depends on actual open positions,
                # as pending signals don't count towards qty_concurrent_positions yet.
                if len(open_position_symbols) < qty_concurrent_positions: can_open_new_trade_overall = True
                else: can_open_new_trade_overall = False # Explicitly false if max positions are open
                
                # Activity message for S0 should also consider pending signals
                pending_s0_count = len([item for key, item in g_conditional_pending_signals.items() if item['strategy_id'] == 0])
                _activity_set(activity_msg_prefix + 
                              (f"MaxOpen {qty_concurrent_positions}. ActualOpen: {len(open_position_symbols)}. Pending: {pending_s0_count}. " + 
                               ("Seeking/Managing." if can_open_new_trade_overall or pending_s0_count > 0 else "Monitoring.")))
            elif current_active_strategy_id == 1:
                if strategy1_active_trade_info['symbol'] is None:
                    if strategy1_cooldown_active: _activity_set(activity_msg_prefix + "Cooldown active.")
                    else: can_open_new_trade_overall = True; _activity_set(activity_msg_prefix + "Ready. Seeking.")
                else: _activity_set(activity_msg_prefix + f"Trade active on {strategy1_active_trade_info['symbol']}. Monitoring.")
            elif current_active_strategy_id == 2:
                if len(strategy2_active_trades) < 2: can_open_new_trade_overall = True
                _activity_set(activity_msg_prefix + (f"Active: {len(strategy2_active_trades)}/2. " + ("Seeking." if can_open_new_trade_overall else "Max S2 trades.")))
            elif current_active_strategy_id == 3:
                if strategy3_active_trade_info['symbol'] is None: can_open_new_trade_overall = True; _activity_set(activity_msg_prefix + "Ready. Seeking.")
                else: _activity_set(activity_msg_prefix + f"Trade active on {strategy3_active_trade_info['symbol']}. Monitoring.")
            elif current_active_strategy_id == 4:
                if strategy4_active_trade_info['symbol'] is None: can_open_new_trade_overall = True; _activity_set(activity_msg_prefix + "Ready. Seeking.")
                else: _activity_set(activity_msg_prefix + f"Trade active on {strategy4_active_trade_info['symbol']}. Monitoring.")
            elif current_active_strategy_id == 5: # Strategy 5 logic
                # Assuming S5 can open trades as long as general conditions are met (e.g., not exceeding overall max positions if that's a global rule)
                if len(open_position_symbols) < qty_concurrent_positions : # Check against global concurrent positions
                    can_open_new_trade_overall = True
                    _activity_set(activity_msg_prefix + "Ready. Seeking.")
                else:
                    _activity_set(activity_msg_prefix + f"Max open positions ({qty_concurrent_positions}) reached. Monitoring.")
                    can_open_new_trade_overall = False
            
            # --- Manage Pending Signals (Part 2) ---
            # (This loop was inserted in the previous step and is assumed to be correct)
            if g_conditional_pending_signals: 
                _activity_set(f"Managing {len(g_conditional_pending_signals)} conditional signals...")
                current_time_utc_for_pending_manage = pd.Timestamp.now(tz='UTC')
                for key, item in list(g_conditional_pending_signals.items()):
                    if not bot_running: break
                    symbol = item['symbol']
                    strategy_id = item['strategy_id'] 
                    side = item['side']
                    timestamp = item['timestamp']
                    conditions_to_start_wait_threshold = item['conditions_to_start_wait_threshold']
                    conditions_for_full_signal_threshold = item['conditions_for_full_signal_threshold']
                    time_elapsed_seconds = (current_time_utc_for_pending_manage - timestamp).total_seconds()
                    current_strategy_output = {}
                    if strategy_id == 0: current_strategy_output = scalping_strategy_signal(symbol)
                    elif strategy_id == 1: current_strategy_output = strategy_ema_supertrend(symbol)
                    elif strategy_id == 2: current_strategy_output = strategy_bollinger_band_mean_reversion(symbol)
                    elif strategy_id == 3: current_strategy_output = strategy_vwap_breakout_momentum(symbol)
                    current_met_count = 0
                    if current_strategy_output and current_strategy_output.get('all_conditions_status'):
                        all_conds = current_strategy_output['all_conditions_status']
                        if strategy_id == 0:
                            if side == 'up': current_met_count = all_conds.get('num_buy_conditions_met', 0)
                            elif side == 'down': current_met_count = all_conds.get('num_sell_conditions_met', 0)
                        elif strategy_id == 1:
                            if side == 'up': relevant_keys = ['ema_cross_up', 'st_green', 'rsi_long_ok']
                            else: relevant_keys = ['ema_cross_down', 'st_red', 'rsi_short_ok']
                            current_met_count = sum(1 for k, v in all_conds.items() if k in relevant_keys and v)
                        elif strategy_id == 2:
                            if side == 'up': relevant_keys = ['price_below_lower_bb', 'rsi_oversold', 'volume_confirms_long']
                            else: relevant_keys = ['price_above_upper_bb', 'rsi_overbought', 'volume_confirms_short']
                            current_met_count = sum(1 for k, v in all_conds.items() if k in relevant_keys and v)
                        elif strategy_id == 3:
                            if side == 'up': relevant_keys = ['price_above_vwap_2bar_long', 'macd_positive_rising', 'atr_volatility_confirms_long']
                            else: relevant_keys = ['price_below_vwap_2bar_short', 'macd_negative_falling', 'atr_volatility_confirms_short']
                            current_met_count = sum(1 for k, v in all_conds.items() if k in relevant_keys and v)
                    if key in g_conditional_pending_signals:
                        g_conditional_pending_signals[key]['current_conditions_met_count'] = current_met_count
                        if current_strategy_output:
                            g_conditional_pending_signals[key]['last_evaluated_all_conditions_status'] = current_strategy_output.get('all_conditions_status', {}).copy()
                            g_conditional_pending_signals[key]['potential_sl_price'] = current_strategy_output.get('sl_price')
                            g_conditional_pending_signals[key]['potential_tp_price'] = current_strategy_output.get('tp_price')
                            if strategy_id == 0:
                                 g_conditional_pending_signals[key]['conditions_for_full_signal_threshold'] = current_strategy_output.get('conditions_for_full_signal_threshold', conditions_for_full_signal_threshold)
                                 conditions_for_full_signal_threshold = g_conditional_pending_signals[key]['conditions_for_full_signal_threshold']
                    remaining_seconds = max(0, 300 - time_elapsed_seconds)
                    _activity_set(f"S{strategy_id} {symbol} ({side}): {current_met_count}/{conditions_for_full_signal_threshold} cond. {int(remaining_seconds)}s left.")
                    if root and root.winfo_exists() and current_strategy_output:
                         root.after(0, update_conditions_display_content, symbol, current_strategy_output.get('all_conditions_status'), current_strategy_output.get('error'))
                    if current_met_count >= conditions_for_full_signal_threshold and not current_strategy_output.get('error'):
                        print(f"CONFIRMED: {STRATEGIES[strategy_id]} for {symbol} ({side}). Met {current_met_count}/{conditions_for_full_signal_threshold}.")
                        _status_set(f"Ordering {symbol} ({side}) via {STRATEGIES[strategy_id]}...")
                        set_mode(symbol, type); sleep(0.1)
                        set_leverage(symbol, leverage); sleep(0.1)
                        sl_to_use = g_conditional_pending_signals[key]['potential_sl_price']
                        tp_to_use = g_conditional_pending_signals[key]['potential_tp_price']
                        risk_percent = current_strategy_output.get('account_risk_percent')
                        entry_price_for_tracking = None
                        try: entry_price_for_tracking = float(client.ticker_price(symbol)['price'])
                        except Exception as e_track_price: print(f"Error fetching entry price for tracking {symbol}: {e_track_price}")
                        open_order(symbol, side, strategy_sl=sl_to_use, strategy_tp=tp_to_use, strategy_account_risk_percent=risk_percent)
                        trade_entry_ts = pd.Timestamp.now(tz='UTC')
                        if strategy_id == 1: strategy1_active_trade_info.update({'symbol': symbol, 'entry_time': trade_entry_ts, 'entry_price': entry_price_for_tracking, 'side': side})
                        elif strategy_id == 2: strategy2_active_trades.append({'symbol': symbol, 'entry_time': trade_entry_ts, 'entry_price': entry_price_for_tracking, 'side': side})
                        elif strategy_id == 3: strategy3_active_trade_info.update({'symbol': symbol, 'entry_time': trade_entry_ts, 'entry_price': entry_price_for_tracking, 'side': side, 'initial_atr_for_profit_target': current_strategy_output.get('last_atr'), 'vwap_trail_active': False })
                        if key in g_conditional_pending_signals: del g_conditional_pending_signals[key]
                        sleep(1); continue
                    if time_elapsed_seconds >= 300:
                        print(f"TIMEOUT: {STRATEGIES[strategy_id]} for {symbol} ({side}). Did not confirm. Met {current_met_count}/{conditions_for_full_signal_threshold}.")
                        _activity_set(f"TIMEOUT: S{strategy_id} for {symbol} ({side}).")
                        if key in g_conditional_pending_signals: del g_conditional_pending_signals[key]
                        continue
                    if current_met_count < conditions_to_start_wait_threshold:
                        print(f"KILLED (Degraded): {STRATEGIES[strategy_id]} for {symbol} ({side}). Conditions fell to {current_met_count}/{conditions_to_start_wait_threshold}.")
                        _activity_set(f"KILLED (Degraded): S{strategy_id} for {symbol} ({side}).")
                        if key in g_conditional_pending_signals: del g_conditional_pending_signals[key]
                        continue
                    if current_strategy_output.get('error'):
                        print(f"KILLED (Error on re-eval): {STRATEGIES[strategy_id]} for {symbol} ({side}). Error: {current_strategy_output['error']}")
                        _activity_set(f"KILLED (Error): S{strategy_id} for {symbol} ({side}).")
                        if key in g_conditional_pending_signals: del g_conditional_pending_signals[key]
                        continue
                    sleep(0.1) 
            
            _status_set(current_balance_msg_for_status + " Managing active trades & timeouts...") # This line should be after pending mgmt
            # --- Active Trade Management & Timeouts (BEFORE checking for new trades) ---
            now_utc_for_timeout = pd.Timestamp.now(tz='UTC')
            
            # --- Signal Detection and Pending State Initiation (Part 1) ---
            if can_open_new_trade_overall : 
                symbols_to_check = TARGET_SYMBOLS 
                for sym_to_check in symbols_to_check:
                    if not bot_running: break

                    # --- MODIFICATION: Check if position already open for this symbol ---
                    if sym_to_check in open_position_symbols:
                        print(f"Skipping {sym_to_check} as a position is already open for this symbol.")
                        _activity_set(f"Skipping {sym_to_check}, already open.") 
                        sleep(0.1) 
                        continue # Skip to the next symbol
                    # --- END MODIFICATION ---
                    
                    current_active_strategy_id = ACTIVE_STRATEGY_ID 
                    
                    # Price fetching should happen before strategy-specific logic if it's common
                    _activity_set(f"S{current_active_strategy_id}: Scanning {sym_to_check}...") # General scanning message
                    latest_price = None
                    try:
                        ticker = client.ticker_price(sym_to_check)
                        latest_price = float(ticker['price'])
                        if root and root.winfo_exists() and current_price_var:
                            root.after(0, lambda s=sym_to_check, p=latest_price: current_price_var.set(f"Scanning: {s} - Price: {p:.{get_price_precision(s)}f}")) 
                    except Exception as e_price_fetch:
                        print(f"Error fetching price for {sym_to_check} for display: {e_price_fetch}")
                        if root and root.winfo_exists() and current_price_var: 
                            root.after(0, lambda s=sym_to_check: current_price_var.set(f"Scanning: {s} - Price: Error"))
                        sleep(0.1); continue # Skip this symbol if price fetch fails

                    # Strategy-specific handling
                    if current_active_strategy_id == 5: # New Strategy 5 handling
                        print(f"DEBUG: S5: Active for symbol: {sym_to_check}")
                        # S5 does not use g_conditional_pending_signals, it trades directly.
                        # Check if this symbol already has an S5 trade or if S5 has a 1-trade-at-a-time rule (not specified, assuming can trade multiple symbols)

                        _activity_set(f"S5: Scanning {sym_to_check}...")
                        signal_data_s5 = strategy_rsi_enhanced(sym_to_check)
                        print(f"DEBUG: S5: Data for {sym_to_check}: {signal_data_s5}")

                        if root and root.winfo_exists():
                            root.after(0, update_conditions_display_content, sym_to_check, signal_data_s5.get('all_conditions_status'), signal_data_s5.get('error'))
                        
                        if signal_data_s5.get('error'):
                            print(f"S5 Error ({sym_to_check}): {signal_data_s5['error']}")
                            sleep(0.1); continue
                        
                        current_signal_s5 = signal_data_s5.get('signal', 'none')
                        if current_signal_s5 in ['up', 'down']:
                            # Optional: Show a popup for S5 signals if desired (as in the overwritten code)
                            # popup_title = f"Trade Signal: {STRATEGIES.get(5, 'S5')}"
                            # popup_message = f"Symbol: {sym_to_check}\nSignal: {current_signal_s5.upper()}\nStrategy: {STRATEGIES.get(5, 'S5')}"
                            # if root and root.winfo_exists(): root.after(0, lambda title=popup_title, msg=popup_message: messagebox.showinfo(title, msg))
                            
                            print(f"S5: {current_signal_s5.upper()} signal for {sym_to_check}. Ordering...")
                            _status_set(f"S5: Ordering {sym_to_check} ({current_signal_s5.upper()})...")
                            set_mode(sym_to_check, margin_type_setting); sleep(0.1) # Use margin_type_setting
                            set_leverage(sym_to_check, leverage); sleep(0.1)
                            open_order(sym_to_check, current_signal_s5, 
                                       strategy_sl=signal_data_s5.get('sl_price'), 
                                       strategy_tp=signal_data_s5.get('tp_price'), 
                                       strategy_account_risk_percent=signal_data_s5.get('account_risk_percent'))
                            _activity_set(f"S5: Trade initiated for {sym_to_check}.")
                            # S5 might allow multiple trades across symbols, so no 'break' here unless a specific S5 trade management global var is checked.
                            # If S5 was meant to be one-trade-at-a-time like S1/S3/S4, a global tracker for S5's active trade would be needed.
                            sleep(1) # Small delay after placing order
                        else:
                            _activity_set(f"S5: No signal for {sym_to_check}. Next..."); sleep(0.1)


                    elif current_active_strategy_id == 4:
                        if strategy4_active_trade_info['symbol'] is not None:
                            _activity_set(f"S4: Already in an active trade with {strategy4_active_trade_info['symbol']}. Skipping scan for {sym_to_check}")
                            sleep(0.05)
                            continue 

                        _activity_set(f"S4: Scanning {sym_to_check}...") # More specific S4 scan message
                        
                        signal_data_s4 = strategy_macd_divergence_pivot(sym_to_check)
                        current_signal_s4 = signal_data_s4.get('signal', 'none')
                        error_message_s4 = signal_data_s4.get('error')

                        if root and root.winfo_exists():
                            root.after(0, update_conditions_display_content, sym_to_check, signal_data_s4.get('conditions'), error_message_s4)

                        if error_message_s4:
                            print(f"Error from Strategy 4 for {sym_to_check}: {error_message_s4}")
                            sleep(0.1)
                            continue

                        if current_signal_s4 in ['up', 'down']:
                            print(f"Strategy 4: {current_signal_s4.upper()} signal for {sym_to_check}. Ordering immediately...")
                            _status_set(f"S4: {current_signal_s4.upper()} signal for {sym_to_check}. Ordering...")
                            
                            set_mode(sym_to_check, margin_type_setting)
                            sleep(0.1)
                            set_leverage(sym_to_check, leverage)
                            sleep(0.1)
    
                            sl_price_s4 = signal_data_s4.get('sl_price')
                            tp_price_s4 = signal_data_s4.get('tp_price')
                            strategy_risk_s4 = signal_data_s4.get('account_risk_percent')
                            
                            entry_price_for_tracking_s4 = None # Use latest_price if direct pre-order fetch fails
                            try:
                                entry_price_for_tracking_s4 = float(client.ticker_price(sym_to_check)['price'])
                            except Exception as e_price_s4:
                                print(f"S4: Error fetching entry price for tracking {sym_to_check}: {e_price_s4}. Using last known price: {latest_price}")
                                entry_price_for_tracking_s4 = latest_price # Fallback to previously fetched price
                            
                            if entry_price_for_tracking_s4 is None: # Still None, means initial fetch also failed
                                print(f"S4: Critical error - no price available for {sym_to_check} to place order. Skipping.")
                                continue
    
                            open_order(sym_to_check, current_signal_s4, 
                                       strategy_sl=sl_price_s4, 
                                       strategy_tp=tp_price_s4, 
                                       strategy_account_risk_percent=strategy_risk_s4)
                            
                            trade_entry_timestamp_s4 = pd.Timestamp.now(tz='UTC')
                            strategy4_active_trade_info.update({
                                'symbol': sym_to_check, 
                                'entry_time': trade_entry_timestamp_s4, 
                                'entry_price': entry_price_for_tracking_s4,
                                'side': current_signal_s4, 
                                'divergence_price_point': signal_data_s4.get('divergence_price_point')
                            })
                            _activity_set(f"S4: Trade initiated for {sym_to_check}. Monitoring.")
                            sleep(3) 
                            break # Strategy 4 handles one trade at a time
                        else:
                            _activity_set(f"S4: No signal for {sym_to_check}. Next...")
                            sleep(0.1) # After processing S4 for a symbol if no break
                    
                    elif current_active_strategy_id in [0, 1, 2, 3]: # Original strategies that use pending system
                        if (sym_to_check, current_active_strategy_id) in g_conditional_pending_signals:
                            continue # Already pending for this strategy (S0-S3)
                        
                        # Skip if symbol already has an active trade for certain single-trade strategies (S1, S3)
                        # S2 allows multiple trades on different symbols, S0 allows up to overall limit.
                        if current_active_strategy_id == 1 and strategy1_active_trade_info['symbol'] is not None:
                            sleep(0.05); continue 
                        if current_active_strategy_id == 2 and any(t['symbol'] == sym_to_check for t in strategy2_active_trades):
                             sleep(0.05); continue
                        if current_active_strategy_id == 3 and strategy3_active_trade_info['symbol'] is not None:
                             sleep(0.05); continue
                        
                        strategy_output = {}
                        if current_active_strategy_id == 0: strategy_output = scalping_strategy_signal(sym_to_check)
                        elif current_active_strategy_id == 1: strategy_output = strategy_ema_supertrend(sym_to_check)
                        elif current_active_strategy_id == 2: strategy_output = strategy_bollinger_band_mean_reversion(sym_to_check)
                        elif current_active_strategy_id == 3: strategy_output = strategy_vwap_breakout_momentum(sym_to_check)
                        
                        if not strategy_output: 
                            sleep(0.1); continue

                        if root and root.winfo_exists():
                            root.after(0, update_conditions_display_content, sym_to_check, strategy_output.get('all_conditions_status'), strategy_output.get('error'))

                        if strategy_output.get('error'):
                            print(f"Error from {STRATEGIES[current_active_strategy_id]} for {sym_to_check}: {strategy_output['error']}")
                            sleep(0.1); continue

                        side_to_consider = None
                        actual_met_count = 0
                        if current_active_strategy_id == 0:
                            buy_met_s0 = strategy_output['all_conditions_status'].get('num_buy_conditions_met', 0)
                            sell_met_s0 = strategy_output['all_conditions_status'].get('num_sell_conditions_met', 0)
                            wait_threshold_s0 = strategy_output.get('conditions_to_start_wait_threshold', 2)
                            if buy_met_s0 >= wait_threshold_s0 and buy_met_s0 >= sell_met_s0 :
                                side_to_consider = 'up'; actual_met_count = buy_met_s0
                            elif sell_met_s0 >= wait_threshold_s0:
                                side_to_consider = 'down'; actual_met_count = sell_met_s0
                        else: 
                            temp_buy_met, temp_sell_met = 0, 0
                            if current_active_strategy_id == 1:
                                temp_buy_met = sum(1 for k,v in strategy_output['all_conditions_status'].items() if k in ['ema_cross_up', 'st_green', 'rsi_long_ok'] and v)
                                temp_sell_met = sum(1 for k,v in strategy_output['all_conditions_status'].items() if k in ['ema_cross_down', 'st_red', 'rsi_short_ok'] and v)
                            elif current_active_strategy_id == 2:
                                temp_buy_met = sum(1 for k,v in strategy_output['all_conditions_status'].items() if k in ['price_below_lower_bb', 'rsi_oversold', 'volume_confirms_long'] and v)
                                temp_sell_met = sum(1 for k,v in strategy_output['all_conditions_status'].items() if k in ['price_above_upper_bb', 'rsi_overbought', 'volume_confirms_short'] and v)
                            elif current_active_strategy_id == 3:
                                temp_buy_met = sum(1 for k,v in strategy_output['all_conditions_status'].items() if k in ['price_above_vwap_2bar_long', 'macd_positive_rising', 'atr_volatility_confirms_long'] and v)
                                temp_sell_met = sum(1 for k,v in strategy_output['all_conditions_status'].items() if k in ['price_below_vwap_2bar_short', 'macd_negative_falling', 'atr_volatility_confirms_short'] and v)

                            if temp_buy_met >= strategy_output.get('conditions_to_start_wait_threshold', 2) and temp_buy_met >= temp_sell_met:
                                side_to_consider = 'up'; actual_met_count = temp_buy_met
                            elif temp_sell_met >= strategy_output.get('conditions_to_start_wait_threshold', 2):
                                side_to_consider = 'down'; actual_met_count = temp_sell_met
                            if strategy_output.get('signal') in ['up', 'down']: # Full signal overrides partial
                                side_to_consider = strategy_output['signal']
                                actual_met_count = strategy_output.get('conditions_met_count', 0)

                        if side_to_consider and actual_met_count >= strategy_output.get('conditions_to_start_wait_threshold',2):
                            pending_key = (sym_to_check, current_active_strategy_id)
                            full_signal_thresh = strategy_output['conditions_for_full_signal_threshold']
                            g_conditional_pending_signals[pending_key] = {
                                'symbol': sym_to_check, 'strategy_id': current_active_strategy_id,
                                'side': side_to_consider, 'timestamp': pd.Timestamp.now(tz='UTC'),
                                'current_conditions_met_count': actual_met_count,
                                'conditions_to_start_wait_threshold': strategy_output['conditions_to_start_wait_threshold'],
                                'conditions_for_full_signal_threshold': full_signal_thresh,
                                'all_conditions_status_at_pending_start': strategy_output['all_conditions_status'].copy(),
                                'last_evaluated_all_conditions_status': strategy_output['all_conditions_status'].copy(),
                                'entry_price_at_pending_start': latest_price,
                                'potential_sl_price': strategy_output.get('sl_price'), 
                                'potential_tp_price': strategy_output.get('tp_price')  
                            }
                            if current_active_strategy_id == 0:
                                ui_msg = f"S0 New: {sym_to_check} ({side_to_consider}) added. 5min confirmation window started."
                                print(ui_msg); _activity_set(ui_msg)
                            else:
                                log_msg = (f"S{current_active_strategy_id} {sym_to_check} ({side_to_consider}): "
                                           f"{actual_met_count}/{full_signal_thresh} cond. met. Monitoring.")
                                print(log_msg); _activity_set(log_msg)
                        sleep(0.1) 
                    else: # Unknown strategy
                        _activity_set(f"Unknown strategy ID {current_active_strategy_id}. Skipping scan for {sym_to_check}.")
                        sleep(0.1)
                        continue
            
            if not bot_running: break
            
            if strategy1_cooldown_active and strategy1_last_trade_was_loss:
                # print("Strategy 1: Cooldown was active. Resetting cooldown status for next full scan cycle.")
                # strategy1_cooldown_active = False
                # Cooldown is now persistent until manually reset or bot restarts.
                # strategy1_last_trade_was_loss = False # Optional: reset this flag too, or keep it for longer-term tracking
                pass # Explicitly do nothing here, cooldown remains active.
            
            loop_count +=1
            wait_message = f"{current_balance_msg_for_status} Scan cycle {loop_count} done. Waiting..."
            _status_set(wait_message)
            # Update activity status if it was just monitoring due to can_open_new_trade_overall being false
            if not can_open_new_trade_overall: _activity_set(f"S{current_active_strategy_id}: Conditions not met for new trades. Waiting...")
            else: _activity_set("Scan cycle complete. Waiting for next cycle...")

            for _ in range(180): 
                if not bot_running: break
                
        except ClientError as ce:
            err_msg = f"API Error: {ce.error_message if hasattr(ce, 'error_message') and ce.error_message else ce}. Retrying..."
            print(err_msg); _status_set(err_msg); _activity_set("API Error. Retrying...")
            if "signature" in str(ce).lower() or "timestamp" in str(ce).lower(): reinitialize_client()
            for _ in range(60): 
                if not bot_running: break; sleep(1)
        except Exception as e:
            err_msg = f"Bot loop error: {e}. Retrying..."
            print(err_msg); _status_set(err_msg); _activity_set("Loop Error. Retrying...")
            # import traceback # For debugging
            # print(traceback.format_exc()) # For debugging
            for _ in range(60):
                if not bot_running: break; sleep(1)
            if not bot_running: break
            continue

    _activity_set("Bot Idle")
    if root and root.winfo_exists() and current_price_var:
        root.after(0, lambda: current_price_var.set("Scanning: N/A - Price: N/A"))
    if root and root.winfo_exists():
        root.after(0, update_conditions_display_content, "Bot Idle", None, "Bot stopped.")
    print("Bot logic thread stopped.")
    _status_set("Bot stopped.")
    if start_button and root and root.winfo_exists(): start_button.config(state=tk.NORMAL)
    if stop_button and root and root.winfo_exists(): stop_button.config(state=tk.DISABLED)
    if testnet_radio and root and root.winfo_exists(): testnet_radio.config(state=tk.NORMAL)
    if mainnet_radio and root and root.winfo_exists(): mainnet_radio.config(state=tk.NORMAL)
    # Re-enable parameter widgets
    for widget in params_widgets:
        if widget and root and root.winfo_exists() and widget.winfo_exists():
            widget.config(state=tk.NORMAL)

def apply_settings():
    global ACCOUNT_RISK_PERCENT, TP_PERCENT, SL_PERCENT, leverage, qty_concurrent_positions, LOCAL_HIGH_LOW_LOOKBACK_PERIOD, margin_type_setting, TARGET_SYMBOLS, ACTIVE_STRATEGY_ID # 'margin_type_setting' is for margin type

    try:
        # Strategy Selection
        selected_id = selected_strategy_var.get()
        if selected_id in STRATEGIES:
            ACTIVE_STRATEGY_ID = selected_id
            print(f"Active strategy set to ID {ACTIVE_STRATEGY_ID}: {STRATEGIES[ACTIVE_STRATEGY_ID]}")
        else:
            messagebox.showerror("Settings Error", f"Invalid strategy ID selected: {selected_id}")
            return False

        # Target Symbols
        raw_symbols_str = target_symbols_var.get()
        if not raw_symbols_str.strip():
            messagebox.showerror("Settings Error", "Target Symbols list cannot be empty.")
            return False
        
        symbols_list = [s.strip().upper() for s in raw_symbols_str.split(',') if s.strip()]
        if not symbols_list: # Handles case where input was just commas or whitespace
            messagebox.showerror("Settings Error", "Target Symbols list cannot be empty after parsing. Ensure valid symbols separated by commas.")
            return False
        TARGET_SYMBOLS = symbols_list

        # Account Risk Percent
        arp_val = float(account_risk_percent_var.get())
        if not (0 < arp_val <= 100):
            messagebox.showerror("Settings Error", "Account Risk % must be a value between 0 and 100 (e.g., enter 2 for 2%).")
            return False
        ACCOUNT_RISK_PERCENT = arp_val / 100

        # Take Profit Percent
        tp_val = float(tp_percent_var.get())
        if not (0 < tp_val <= 1000):
            messagebox.showerror("Settings Error", "Take Profit % must be a positive value (e.g., enter 1 for 1%, 150 for 150%). Recommended range 0-1000.")
            return False
        TP_PERCENT = tp_val / 100

        # Stop Loss Percent
        sl_val = float(sl_percent_var.get())
        if not (0 < sl_val <= 100):
            messagebox.showerror("Settings Error", "Stop Loss % must be a value between 0 and 100 (e.g., enter 0.5 for 0.5%).")
            return False
        SL_PERCENT = sl_val / 100

        # Leverage
        lev_val = int(leverage_var.get())
        if not (0 < lev_val <= 125): # Binance max leverage
            messagebox.showerror("Settings Error", "Leverage must be an integer between 1 and 125.")
            return False
        leverage = lev_val

        # Max Open Positions
        mop_val = int(qty_concurrent_positions_var.get())
        if not (0 < mop_val <= 1000): # Arbitrary reasonable upper limit
            messagebox.showerror("Settings Error", "Max Open Positions must be an integer > 0.")
            return False
        qty_concurrent_positions = mop_val
        
        # Local High/Low Lookback Period
        lhl_val = int(local_high_low_lookback_var.get())
        if not (0 < lhl_val <= 200): # Arbitrary reasonable upper limit
            messagebox.showerror("Settings Error", "Lookback Period must be an integer > 0 and reasonable (e.g., <= 200).")
            return False
        LOCAL_HIGH_LOW_LOOKBACK_PERIOD = lhl_val

        # Margin Type
        margin_type_setting = margin_type_var.get() # This is already a string 'ISOLATED' or 'CROSS'
        if margin_type_setting not in ["ISOLATED", "CROSS"]: # Should not happen with radio buttons
             messagebox.showerror("Settings Error", "Invalid Margin Type selected.")
             return False

        print(f"Applied settings: Strategy='{STRATEGIES[ACTIVE_STRATEGY_ID]}', Risk={ACCOUNT_RISK_PERCENT}, TP={TP_PERCENT}, SL={SL_PERCENT}, Lev={leverage}, MaxPos={qty_concurrent_positions}, Lookback={LOCAL_HIGH_LOW_LOOKBACK_PERIOD}, Margin={margin_type_setting}, Symbols={TARGET_SYMBOLS}")
        messagebox.showinfo("Settings", "Settings applied successfully!")
        return True

    except ValueError as ve:
        messagebox.showerror("Settings Error", f"Invalid input. Please ensure all parameters are numbers. Error: {ve}")
        return False
    except Exception as e:
        messagebox.showerror("Settings Error", f"An unexpected error occurred while applying settings: {e}")
        return False

def start_bot():
    global bot_running, bot_thread, status_var, client, start_button, stop_button, testnet_radio, mainnet_radio, conditions_text_widget, current_price_var
    _status_set = lambda msg: status_var.set(msg) if status_var and root and root.winfo_exists() else None
    
    if root and root.winfo_exists() and current_price_var:
        current_price_var.set("Starting... Price: N/A")

    if not apply_settings(): # This will show its own error messages
        _status_set("Bot not started. Please correct settings.")
        if root and root.winfo_exists(): root.after(0, update_conditions_display_content, "Settings Error", None, "Correct settings before starting.")
        return

    if bot_running: messagebox.showinfo("Info", "Bot is already running."); return
    if client is None:
        _status_set("Client not set. Initializing...")
        # _activity_set("Client error. Cannot start.") # activity_status_var is set by reinitialize_client or main
        if root and root.winfo_exists(): root.after(0, update_conditions_display_content, "Client Error", None, "Client not initialized.")
        if not reinitialize_client() or client is None: # reinitialize_client shows its own popups
             _status_set("Client init failed."); return # _status_set already done by reinitialize_client
    
    if root and root.winfo_exists(): # Clear conditions display before starting
        root.after(0, update_conditions_display_content, "Bot Starting", None, "Waiting for first scan...")

    bot_running = True; _status_set("Bot starting...") # _activity_set will be updated by run_bot_logic
    if start_button: start_button.config(state=tk.DISABLED)
    if stop_button: stop_button.config(state=tk.NORMAL)
    if testnet_radio: testnet_radio.config(state=tk.DISABLED)
    if mainnet_radio: mainnet_radio.config(state=tk.DISABLED)
    # Disable parameter widgets (already done when starting, this is redundant here)
    # Disable parameter widgets
    for widget in params_widgets:
        if widget and root and root.winfo_exists() and widget.winfo_exists():
            widget.config(state=tk.DISABLED)

    bot_thread = threading.Thread(target=run_bot_logic, daemon=True)
    bot_thread.start()

def stop_bot():
    global bot_running, bot_thread, status_var, start_button, stop_button, testnet_radio, mainnet_radio, activity_status_var, conditions_text_widget, current_price_var
    _status_set = lambda msg: status_var.set(msg) if status_var and root and root.winfo_exists() else None
    _activity_set = lambda msg: activity_status_var.set(msg) if activity_status_var and root and root.winfo_exists() else None

    if root and root.winfo_exists() and current_price_var:
        current_price_var.set("Stopping... Price: N/A")

    if not bot_running and (bot_thread is None or not bot_thread.is_alive()):
        _status_set("Bot is not running.")
        _activity_set("Bot Idle")
        if root and root.winfo_exists() and current_price_var: current_price_var.set("Scanning: N/A - Price: N/A")
        if root and root.winfo_exists(): root.after(0, update_conditions_display_content, "Bot Idle", None, "Bot is not running.")
        if start_button: start_button.config(state=tk.NORMAL)
        if stop_button: stop_button.config(state=tk.DISABLED)
        if testnet_radio: testnet_radio.config(state=tk.NORMAL)
        if mainnet_radio: mainnet_radio.config(state=tk.NORMAL)
        return
    _status_set("Bot stopping..."); _activity_set("Bot stopping...")
    bot_running = False
    if stop_button: stop_button.config(state=tk.DISABLED) # Disable stop, start will be enabled by run_bot_logic end
    print("Stop signal sent to bot thread.")

def handle_strategy_checkbox_select(selected_id):
    global strategy_checkbox_vars, selected_strategy_var, STRATEGIES 

    current_val_for_selected_id = strategy_checkbox_vars[selected_id].get()

    if current_val_for_selected_id: # If the clicked checkbox is now True
        selected_strategy_var.set(selected_id) # Set this as the active strategy ID
        # Uncheck all others
        for s_id, bool_var in strategy_checkbox_vars.items():
            if s_id != selected_id:
                bool_var.set(False)
        print(f"Strategy {STRATEGIES[selected_id]} ({selected_id}) GUI selected via checkbox.")
    else:
        # This logic prevents unchecking the *last* checked box, 
        # effectively making one always selected, similar to radio buttons.
        # If the user tries to uncheck the currently active strategy, re-check it.
        if selected_strategy_var.get() == selected_id: # Check if it's the currently active one
            strategy_checkbox_vars[selected_id].set(True) 
            print(f"Strategy {STRATEGIES[selected_id]} ({selected_id}) remains selected. Cannot uncheck all.")
        # If it's not the active one being unchecked (which shouldn't happen if logic is correct elsewhere,
        # but as a safeguard), this 'else' branch means an already false checkbox was clicked (no change) or
        # a non-active one was unchecked, which is fine if another one is still active.
        # However, the primary goal is to ensure the selected_strategy_var reflects one true checkbox.

# --- Main Application Window ---
if __name__ == "__main__":
    root = tk.Tk(); root.title("Binance Scalping Bot")

    current_price_var = None # Will be initialized after root Tk()

    status_var = tk.StringVar(value="Welcome! Select environment."); current_env_var = tk.StringVar(value=current_env); balance_var = tk.StringVar(value="N/A")
    activity_status_var = tk.StringVar(value="Bot Idle") # Initialize activity status
    selected_strategy_var = tk.IntVar(value=ACTIVE_STRATEGY_ID)

    # Account Summary StringVars
    account_summary_balance_var = tk.StringVar(value="N/A") # Dedicated for account summary section
    last_7_days_profit_var = tk.StringVar(value="N/A")
    overall_profit_loss_var = tk.StringVar(value="N/A")
    total_unrealized_pnl_var = tk.StringVar(value="N/A")

    # Backtesting StringVars
    backtest_symbol_var = tk.StringVar(value="XRPUSDT")
    backtest_timeframe_var = tk.StringVar(value="5m")
    backtest_interval_var = tk.StringVar(value="30")
    backtest_tp_var = tk.StringVar(value="3.0")
    backtest_sl_var = tk.StringVar(value="2.0")
    backtest_selected_strategy_var = tk.StringVar()


    controls_frame = ttk.LabelFrame(root, text="Controls"); controls_frame.pack(padx=10, pady=(5,0), fill="x")
    env_frame = ttk.Frame(controls_frame); env_frame.pack(pady=2, fill="x")
    ttk.Label(env_frame, text="Env:").pack(side=tk.LEFT, padx=(5,2))
    testnet_radio = ttk.Radiobutton(env_frame, text="Testnet", variable=current_env_var, value="testnet", command=toggle_environment); testnet_radio.pack(side=tk.LEFT, padx=2)
    mainnet_radio = ttk.Radiobutton(env_frame, text="Mainnet", variable=current_env_var, value="mainnet", command=toggle_environment); mainnet_radio.pack(side=tk.LEFT, padx=2)

    # Parameter Inputs Frame
    params_input_frame = ttk.LabelFrame(controls_frame, text="Trading Parameters")
    params_input_frame.pack(fill="x", padx=5, pady=5)

    # ACCOUNT_RISK_PERCENT
    ttk.Label(params_input_frame, text="Account Risk %:").grid(row=0, column=0, padx=2, pady=2, sticky='w')
    account_risk_percent_var = tk.StringVar(value=str(ACCOUNT_RISK_PERCENT * 100))
    account_risk_percent_entry = ttk.Entry(params_input_frame, textvariable=account_risk_percent_var, width=10)
    account_risk_percent_entry.grid(row=0, column=1, padx=2, pady=2, sticky='w')

    # TP_PERCENT
    ttk.Label(params_input_frame, text="Take Profit %:").grid(row=0, column=2, padx=2, pady=2, sticky='w') # Next column
    tp_percent_var = tk.StringVar(value=str(TP_PERCENT * 100))
    tp_percent_entry = ttk.Entry(params_input_frame, textvariable=tp_percent_var, width=10)
    tp_percent_entry.grid(row=0, column=3, padx=2, pady=2, sticky='w')

    # SL_PERCENT
    ttk.Label(params_input_frame, text="Stop Loss %:").grid(row=1, column=0, padx=2, pady=2, sticky='w')
    sl_percent_var = tk.StringVar(value=str(SL_PERCENT * 100))
    sl_percent_entry = ttk.Entry(params_input_frame, textvariable=sl_percent_var, width=10)
    sl_percent_entry.grid(row=1, column=1, padx=2, pady=2, sticky='w')

    # leverage
    ttk.Label(params_input_frame, text="Leverage:").grid(row=1, column=2, padx=2, pady=2, sticky='w') # Next column
    leverage_var = tk.StringVar(value=str(leverage))
    leverage_entry = ttk.Entry(params_input_frame, textvariable=leverage_var, width=10)
    leverage_entry.grid(row=1, column=3, padx=2, pady=2, sticky='w')

    # qty_concurrent_positions
    ttk.Label(params_input_frame, text="Max Open Pos:").grid(row=2, column=0, padx=2, pady=2, sticky='w')
    qty_concurrent_positions_var = tk.StringVar(value=str(qty_concurrent_positions))
    qty_concurrent_positions_entry = ttk.Entry(params_input_frame, textvariable=qty_concurrent_positions_var, width=10)
    qty_concurrent_positions_entry.grid(row=2, column=1, padx=2, pady=2, sticky='w')

    # LOCAL_HIGH_LOW_LOOKBACK_PERIOD
    ttk.Label(params_input_frame, text="Lookback (Breakout):").grid(row=2, column=2, padx=2, pady=2, sticky='w') # Next column
    local_high_low_lookback_var = tk.StringVar(value=str(LOCAL_HIGH_LOW_LOOKBACK_PERIOD))
    local_high_low_lookback_entry = ttk.Entry(params_input_frame, textvariable=local_high_low_lookback_var, width=10)
    local_high_low_lookback_entry.grid(row=2, column=3, padx=2, pady=2, sticky='w')

    # Margin Type
    ttk.Label(params_input_frame, text="Margin Type:").grid(row=3, column=0, padx=2, pady=2, sticky='w')
    margin_type_var = tk.StringVar(value=margin_type_setting) # Initialize with global 'margin_type_setting'
    margin_type_isolated_radio = ttk.Radiobutton(params_input_frame, text="ISOLATED", variable=margin_type_var, value="ISOLATED")
    margin_type_isolated_radio.grid(row=3, column=1, padx=2, pady=2, sticky='w')
    margin_type_cross_radio = ttk.Radiobutton(params_input_frame, text="CROSS", variable=margin_type_var, value="CROSS")
    margin_type_cross_radio.grid(row=3, column=2, padx=2, pady=2, sticky='w')

    # Target Symbols Entry
    ttk.Label(params_input_frame, text="Target Symbols (CSV):").grid(row=4, column=0, padx=2, pady=2, sticky='w')
    target_symbols_var = tk.StringVar(value=",".join(TARGET_SYMBOLS))
    target_symbols_entry = ttk.Entry(params_input_frame, textvariable=target_symbols_var, width=40) # Wider entry
    target_symbols_entry.grid(row=4, column=1, columnspan=3, padx=2, pady=2, sticky='we') # Span across 3 columns

    # Populate params_widgets list
    params_widgets = [
        account_risk_percent_entry, tp_percent_entry, sl_percent_entry,
        leverage_entry, qty_concurrent_positions_entry, local_high_low_lookback_entry,
        margin_type_isolated_radio, margin_type_cross_radio, target_symbols_entry
    ]
    # Configure column weights for params_input_frame to make target_symbols_entry expand
    params_input_frame.columnconfigure(1, weight=1)
    params_input_frame.columnconfigure(3, weight=1)

    # Strategy Selection Frame
    strategy_frame = ttk.LabelFrame(controls_frame, text="Strategy Selection")
    strategy_frame.pack(fill="x", padx=5, pady=5)
    
    # Clear strategy_radio_buttons if it was used for old radio buttons,
    # or ensure it doesn't conflict if it's used for other parameter types.
    # For this refactor, we assume strategy_radio_buttons is exclusively for these strategy selectors.
    # We will now add checkboxes to params_widgets directly.
    
    # Remove old radio buttons from params_widgets if they were added by reference
    # This is tricky if params_widgets holds mixed types. Assuming it's safe to rebuild part of it.
    # A safer approach is to filter out the old radio buttons if they had a specific type/name.
    # For now, let's clear strategy_radio_buttons and rebuild the strategy part of params_widgets.
    
    # Filter out previous strategy radio buttons from params_widgets
    # This assumes they were all Radiobuttons and part of strategy_radio_buttons list previously
    # A more robust way would be to tag them or manage them in a dedicated list.
    if strategy_radio_buttons: # If it was populated before
        params_widgets = [widget for widget in params_widgets if widget not in strategy_radio_buttons]
        strategy_radio_buttons.clear() # Clear the old list

    for strategy_id, strategy_name in STRATEGIES.items():
        var = tk.BooleanVar(value=(strategy_id == selected_strategy_var.get()))
        strategy_checkbox_vars[strategy_id] = var
        cb = ttk.Checkbutton(strategy_frame, 
                             text=strategy_name, 
                             variable=var, 
                             command=lambda sid=strategy_id: handle_strategy_checkbox_select(sid))
        cb.pack(anchor='w', padx=5)
        params_widgets.append(cb) # Add the new checkbox to params_widgets
    
    # Note: The global 'strategy_radio_buttons' list is no longer used for these strategy selectors.
    # If it was used for other parameter types, that logic needs to be preserved or adapted.
    # For this subtask, we assume it was primarily for strategy selection widgets.

    api_key_info_label = ttk.Label(controls_frame, text="API Keys from keys.py"); api_key_info_label.pack(pady=2)
    timeframe_label = ttk.Label(controls_frame, text="Timeframe: 5m (fixed)"); timeframe_label.pack(pady=2)

    # Backtesting Engine Frame
    # backtesting_frame = ttk.LabelFrame(controls_frame, text="Backtesting Engine") # MOVED
    # backtesting_frame.pack(fill="x", padx=5, pady=5) # MOVED

    # Backtest Parameters Grid
    # backtest_params_grid = ttk.Frame(backtesting_frame); backtest_params_grid.pack(fill="x", padx=5, pady=5) # MOVED
    # Definitions of child widgets for backtest_params_grid are moved down, after its re-instantiation.
    
    # backtest_run_button = ttk.Button(backtesting_frame, text="Run Backtest", command=run_backtest_command) # MOVED with its parent frame logic
    # backtest_run_button.pack(pady=5) # MOVED


    buttons_frame = ttk.Frame(controls_frame); buttons_frame.pack(pady=2)
    start_button = ttk.Button(buttons_frame, text="Start Bot", command=start_bot); start_button.pack(side=tk.LEFT, padx=5)
    stop_button = ttk.Button(buttons_frame, text="Stop Bot", command=stop_bot, state=tk.DISABLED); stop_button.pack(side=tk.LEFT, padx=5)
    data_frame = ttk.LabelFrame(root, text="Live Data & History"); data_frame.pack(padx=10, pady=5, fill="both", expand=True)

    # New side-by-side frame for Backtesting and Account Summary
    side_by_side_frame = ttk.Frame(data_frame)
    side_by_side_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

    # Backtesting Engine Frame (moved into side_by_side_frame)
    backtesting_frame = ttk.LabelFrame(side_by_side_frame, text="Backtesting Engine")
    backtesting_frame.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 5), expand=False) 

    # Backtest Parameters Grid (inside the moved backtesting_frame)
    backtest_params_grid = ttk.Frame(backtesting_frame)
    backtest_params_grid.pack(fill="x", padx=5, pady=5)

    # Child UI elements of backtest_params_grid are now defined here:
    ttk.Label(backtest_params_grid, text="Symbol:").grid(row=0, column=0, padx=2, pady=2, sticky='w')
    backtest_symbol_entry = ttk.Entry(backtest_params_grid, textvariable=backtest_symbol_var, width=12)
    backtest_symbol_entry.grid(row=0, column=1, padx=2, pady=2, sticky='w')

    ttk.Label(backtest_params_grid, text="Timeframe:").grid(row=0, column=2, padx=2, pady=2, sticky='w')
    backtest_timeframe_entry = ttk.Entry(backtest_params_grid, textvariable=backtest_timeframe_var, width=7)
    backtest_timeframe_entry.grid(row=0, column=3, padx=2, pady=2, sticky='w')

    ttk.Label(backtest_params_grid, text="Interval (days):").grid(row=1, column=0, padx=2, pady=2, sticky='w')
    backtest_interval_entry = ttk.Entry(backtest_params_grid, textvariable=backtest_interval_var, width=7)
    backtest_interval_entry.grid(row=1, column=1, padx=2, pady=2, sticky='w')
    
    ttk.Label(backtest_params_grid, text="Take Profit %:").grid(row=1, column=2, padx=2, pady=2, sticky='w')
    backtest_tp_entry = ttk.Entry(backtest_params_grid, textvariable=backtest_tp_var, width=7)
    backtest_tp_entry.grid(row=1, column=3, padx=2, pady=2, sticky='w')

    ttk.Label(backtest_params_grid, text="Stop Loss %:").grid(row=2, column=0, padx=2, pady=2, sticky='w')
    backtest_sl_entry = ttk.Entry(backtest_params_grid, textvariable=backtest_sl_var, width=7)
    backtest_sl_entry.grid(row=2, column=1, padx=2, pady=2, sticky='w')

    ttk.Label(backtest_params_grid, text="Backtest Strategy:").grid(row=2, column=2, padx=2, pady=2, sticky='w')
    backtest_strategy_combobox = ttk.Combobox(backtest_params_grid, textvariable=backtest_selected_strategy_var, values=list(STRATEGIES.values()), width=25)
    if STRATEGIES: backtest_strategy_combobox.current(0) # Set default selection
    backtest_strategy_combobox.grid(row=2, column=3, padx=2, pady=2, sticky='w')
    
    # backtest_run_button is defined after its parent backtesting_frame is fully configured.
    # It's not a child of backtest_params_grid, so its definition remains separate but after backtesting_frame.
    # The actual definition of backtest_run_button is handled where backtesting_frame's other direct children are defined.
    # For this specific move, we ensure backtest_params_grid children are correct.
    # The backtest_run_button's definition will be placed after backtesting_frame and its main children like backtest_params_grid and backtest_results_frame.

    # Account Summary Frame (New)
    account_summary_frame = ttk.LabelFrame(side_by_side_frame, text="Account Summary")
    account_summary_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
    # ttk.Label(account_summary_frame, text="Account summary data will appear here.").pack(padx=10, pady=10) # Remove placeholder

    # Add new labels to account_summary_frame
    ttk.Label(account_summary_frame, text="Account Balance:").grid(row=0, column=0, sticky="w", padx=5, pady=2)
    ttk.Label(account_summary_frame, textvariable=account_summary_balance_var).grid(row=0, column=1, sticky="w", padx=5, pady=2)

    ttk.Label(account_summary_frame, text="Last 7 Days Profit:").grid(row=1, column=0, sticky="w", padx=5, pady=2)
    ttk.Label(account_summary_frame, textvariable=last_7_days_profit_var).grid(row=1, column=1, sticky="w", padx=5, pady=2)

    ttk.Label(account_summary_frame, text="Overall Profit/Loss:").grid(row=2, column=0, sticky="w", padx=5, pady=2)
    ttk.Label(account_summary_frame, textvariable=overall_profit_loss_var).grid(row=2, column=1, sticky="w", padx=5, pady=2)

    ttk.Label(account_summary_frame, text="Total Unrealized PNL:").grid(row=3, column=0, sticky="w", padx=5, pady=2)
    ttk.Label(account_summary_frame, textvariable=total_unrealized_pnl_var).grid(row=3, column=1, sticky="w", padx=5, pady=2)
    
    activity_display_frame = ttk.Frame(data_frame)
    activity_display_frame.pack(pady=(2,2), padx=5, fill="x")
    ttk.Label(activity_display_frame, text="Current Activity:").pack(side=tk.LEFT, padx=(0,5))
    ttk.Label(activity_display_frame, textvariable=activity_status_var).pack(side=tk.LEFT)

    # Real-time Price Display Frame
    price_display_frame = ttk.Frame(data_frame)
    price_display_frame.pack(pady=(2,2), padx=5, fill="x")
    current_price_var = tk.StringVar(value="Scanning: N/A - Price: N/A") # Initialize here
    price_label = ttk.Label(price_display_frame, textvariable=current_price_var)
    price_label.pack(side=tk.LEFT)

    balance_display_frame = ttk.Frame(data_frame); balance_display_frame.pack(pady=(5,2), padx=5, fill="x")
    ttk.Label(balance_display_frame, text="Account Balance:").pack(side=tk.LEFT, padx=(0,5))
    ttk.Label(balance_display_frame, textvariable=balance_var).pack(side=tk.LEFT)
    
    positions_display_frame = ttk.LabelFrame(data_frame, text="Current Open Positions"); positions_display_frame.pack(pady=2, padx=5, fill="both", expand=True)
    positions_text_widget = scrolledtext.ScrolledText(positions_display_frame, height=6, width=70, state=tk.DISABLED, wrap=tk.WORD); positions_text_widget.pack(pady=5, padx=5, fill="both", expand=True)
    history_display_frame = ttk.LabelFrame(data_frame, text="Recent Trade History (BTC, ETH)"); history_display_frame.pack(pady=(2,5), padx=5, fill="both", expand=True)
    history_text_widget = scrolledtext.ScrolledText(history_display_frame, height=10, width=70, state=tk.DISABLED, wrap=tk.WORD); history_text_widget.pack(pady=5, padx=5, fill="both", expand=True)
    
    conditions_frame = ttk.LabelFrame(data_frame, text="Signal Conditions")
    conditions_frame.pack(pady=5, padx=5, fill="both", expand=True)
    conditions_text_widget = scrolledtext.ScrolledText(conditions_frame, height=8, width=70, state=tk.DISABLED, wrap=tk.WORD)
    conditions_text_widget.pack(pady=5, padx=5, fill="both", expand=True)

    # Backtest Results Display Area (Now inside backtesting_frame)
    backtest_results_frame = ttk.LabelFrame(backtesting_frame, text="Backtest Results") # Parent changed to backtesting_frame
    backtest_results_frame.pack(pady=(2,5), padx=5, fill="both", expand=True) # This should be after backtesting_frame is defined
    backtest_results_text_widget = scrolledtext.ScrolledText(backtest_results_frame, height=10, width=70, state=tk.DISABLED, wrap=tk.WORD)
    backtest_results_text_widget.pack(pady=5, padx=5, fill="both", expand=True)

    # backtest_run_button's definition needs to be here, after backtesting_frame is defined and configured.
    backtest_run_button = ttk.Button(backtesting_frame, text="Run Backtest", command=run_backtest_command) 
    backtest_run_button.pack(pady=5)


    status_label = ttk.Label(root, textvariable=status_var, relief=tk.SUNKEN, anchor=tk.W); status_label.pack(padx=10, pady=(0,5), fill="x", side=tk.BOTTOM)
    root.update_idletasks(); status_label.config(wraplength=root.winfo_width() - 20)
    root.bind("<Configure>", lambda event, widget=status_label: widget.config(wraplength=root.winfo_width() - 20))
    
    print("Attempting initial client initialization on startup...")
    initial_conditions_msg = "Bot Idle - Ready"
    if not reinitialize_client() or client is None:
        start_button.config(state=tk.DISABLED); status_var.set("Client not initialized. Start disabled."); activity_status_var.set("Bot Idle - Client Error")
        initial_conditions_msg = "Client not initialized."
    else:
        status_var.set(f"Client initialized for {current_env}. Bot ready.")
        activity_status_var.set("Bot Idle - Ready")
    
    update_conditions_display_content("System", None, initial_conditions_msg) # Initial display

    # Initial data load for Account Summary
    update_account_summary_data() 

    # Start the continuous 5-second update cycle for Account Summary
    if root and root.winfo_exists(): # Check if root exists before scheduling
        print("Starting continuous 5-second updates for account summary from __main__.")
        root.after(5000, scheduled_account_summary_update) # Start the first scheduled call after 5 seconds

    def on_closing():
        global bot_running, bot_thread, root
        if bot_running:
            if messagebox.askokcancel("Quit", "Bot is running. Quit anyway?"):
                bot_running = False
                if bot_thread and bot_thread.is_alive(): bot_thread.join(timeout=2)
                root.destroy()
        else: root.destroy()
    root.protocol("WM_DELETE_WINDOW", on_closing); root.minsize(450, 500); root.mainloop()
# sleep(0.1) # This line was identified as causing an IndentationError by user - removing it.
# sleep(0.1) # This line was identified as causing an IndentationError by user - removing it.

# Note: Many variables in this script are intentionally unused for clarity, debugging, or future use.
# If you want to further reduce warnings, consider using linters or IDE settings to ignore unused-variable warnings.
