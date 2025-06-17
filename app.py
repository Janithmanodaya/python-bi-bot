import tkinter as tk
from tkinter import ttk, messagebox, scrolledtext
import threading
from time import sleep
from binance.um_futures import UMFutures
from binance.error import ClientError
import ta
import pandas as pd

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


# Binance API URLs
BINANCE_MAINNET_URL = "https://fapi.binance.com"
BINANCE_TESTNET_URL = "https://testnet.binancefuture.com"

# Global Configuration Variables
STRATEGIES = {
    0: "Original Scalping",
    1: "EMA Cross + SuperTrend"
    # Placeholder for more strategies
}
ACTIVE_STRATEGY_ID = 0 # Default to original
strategy1_active_trade_info = {'symbol': None, 'entry_time': None, 'entry_price': None, 'position_qty': 0, 'side': None, 'sl_order_id': None, 'tp_order_id': None}
TARGET_SYMBOLS = ["BTCUSDT", "ETHUSDT", "USDTUSDT", "XRPUSDT", "BNBUSDT", "SOLUSDT", "USDCUSDT", "DOGEUSDT", "TRXUSDT", "ADAUSDT"]
ACCOUNT_RISK_PERCENT = 0.02
TP_PERCENT = 0.01
SL_PERCENT = 0.005
leverage = 5
type = 'ISOLATED' # Margin type
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
        response = client.balance(recvWindow=6000)
        for elem in response:
            if elem['asset'] == 'USDT': return float(elem['balance'])
    except ClientError as error:
        if error.error_code == -2015:
            msg = (
                "ERROR: Invalid API Key or Permissions for Futures Trading (-2015).\n"
                "Please check the following on your Binance account:\n"
                "1. API Key is correctly copied.\n"
                "2. 'Enable Futures' permission is CHECKED for this API key.\n"
                "3. IP access restrictions are correctly configured for this API key if enabled.\n"
                "   Your current IP might not be whitelisted."
            )
            print(f"Error get_balance_usdt: {msg}")
            # Optionally, update GUI status if this function is called from a place that can update UI
            # if status_var and root and root.winfo_exists(): root.after(0, lambda: status_var.set("API Key/Permission Error (-2015)"))
        else:
            print(f"Error get_balance_usdt: {error.error_code} - {error.error_message}")
    except Exception as e:
        print(f"Unexpected error get_balance_usdt: {e}")
    return None

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
    global LOCAL_HIGH_LOW_LOOKBACK_PERIOD
    
    default_conditions = {
        'ema_cross_up': False, 'rsi_long_ok': False, 'supertrend_green': False,
        'volume_spike': False, 'price_breakout_high': False,
        'ema_cross_down': False, 'rsi_short_ok': False, 'supertrend_red': False,
        'price_breakout_low': False
    }
    default_return = {'signal': 'none', 'conditions': default_conditions, 'error': None}

    kl = klines(symbol)
    if kl is None or len(kl) < max(21, LOCAL_HIGH_LOW_LOOKBACK_PERIOD + 1): # Ensure enough data
        default_return['error'] = 'Insufficient kline data'
        return default_return
    
    try:
        ema9 = ta.trend.EMAIndicator(close=kl['Close'], window=9).ema_indicator()
        ema21 = ta.trend.EMAIndicator(close=kl['Close'], window=21).ema_indicator()
        rsi = ta.momentum.RSIIndicator(close=kl['Close'], window=14).rsi()
        supertrend = calculate_supertrend_pta(kl, atr_period=10, multiplier=1.5)
        volume_ma10 = kl['Volume'].rolling(window=10).mean()

        if any(x is None for x in [ema9, ema21, rsi, supertrend, volume_ma10]) or \
           any(x.empty for x in [ema9, ema21, rsi, supertrend]) or volume_ma10.isnull().all():
            default_return['error'] = 'One or more indicators are None or empty'
            return default_return

        required_length = max(10, LOCAL_HIGH_LOW_LOOKBACK_PERIOD + 1)
        if len(ema9) < 2 or len(ema21) < 2 or len(rsi) < 1 or len(supertrend) < 1 or \
           len(volume_ma10.dropna()) < 1 or len(kl['Volume']) < required_length:
            default_return['error'] = 'Indicator series too short for required length'
            return default_return

        last_ema9, prev_ema9 = ema9.iloc[-1], ema9.iloc[-2]
        last_ema21, prev_ema21 = ema21.iloc[-1], ema21.iloc[-2]
        last_rsi = rsi.iloc[-1]
        last_supertrend = supertrend.iloc[-1]
        last_volume = kl['Volume'].iloc[-1]
        last_volume_ma10 = volume_ma10.iloc[-1] # Might be NaN if window is large and data short
        current_price = kl['Close'].iloc[-1]

        if any(pd.isna(val) for val in [last_ema9, prev_ema9, last_ema21, prev_ema21, last_rsi, last_supertrend, last_volume, current_price]) or pd.isna(last_volume_ma10):
            default_return['error'] = 'NaN value in one of the critical indicators or price'
            return default_return
            
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

        conditions_status = {
            'ema_cross_up': ema_crossed_up, 'rsi_long_ok': rsi_valid_long, 
            'supertrend_green': supertrend_is_green, 'volume_spike': volume_is_strong, 
            'price_breakout_high': price_broke_high,
            'ema_cross_down': ema_crossed_down, 'rsi_short_ok': rsi_valid_short, 
            'supertrend_red': supertrend_is_red, 'price_breakout_low': price_broke_low
        }

        final_signal_str = 'none'
        if ema_crossed_up and rsi_valid_long and supertrend_is_green and volume_is_strong and price_broke_high:
            final_signal_str = 'up'
        elif ema_crossed_down and rsi_valid_short and supertrend_is_red and volume_is_strong and price_broke_low:
            final_signal_str = 'down'
        
        return {'signal': final_signal_str, 'conditions': conditions_status, 'error': None}

    except Exception as e:
        # print(f"Error in scalping_strategy_signal for {symbol}: {e}") # Can be noisy
        default_return['error'] = str(e)
        return default_return

def strategy_ema_supertrend(symbol):
    default_conditions = {
        'ema_cross_up': False, 'st_green': False, 'rsi_long_ok': False,
        'ema_cross_down': False, 'st_red': False, 'rsi_short_ok': False,
    }
    base_return = {'signal': 'none', 'conditions': default_conditions, 'sl_price': None, 'tp_price': None, 'error': None}

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
        conditions = {}
        conditions['ema_cross_up'] = prev_ema9 < prev_ema21 and last_ema9 > last_ema21
        conditions['st_green'] = last_supertrend_signal == 'green'
        conditions['rsi_long_ok'] = 40 <= last_rsi <= 70

        conditions['ema_cross_down'] = prev_ema9 > prev_ema21 and last_ema9 < last_ema21
        conditions['st_red'] = last_supertrend_signal == 'red'
        conditions['rsi_short_ok'] = 30 <= last_rsi <= 60 # Adjusted RSI for short as per common practice, original was 30-50

        final_signal = 'none'
        sl_price, tp_price = None, None
        price_precision = get_price_precision(symbol)

        if len(kl['Low']) < 3 or len(kl['High']) < 3: # Need at least T-1 and T-2 for swing
            base_return['error'] = "Not enough kline data for SL/TP calculation (need T-1, T-2)"
            base_return['conditions'] = conditions # Still return conditions evaluated so far
            return base_return

        if conditions['ema_cross_up'] and conditions['st_green'] and conditions['rsi_long_ok']:
            final_signal = 'up'
            swing_low = min(kl['Low'].iloc[-2], kl['Low'].iloc[-3])
            sl_price = round(swing_low - (kl['Low'].iloc[-1] * 0.001), price_precision) # Buffer
            if current_price <= sl_price: # SL would be at or above entry
                base_return['error'] = f"SL price {sl_price} is at or above entry {current_price} for LONG"
                final_signal = 'none' # Invalidate signal
            else:
                risk_amount = current_price - sl_price
                tp_price = round(current_price + (risk_amount * 1.5), price_precision)
                if tp_price <= current_price: # TP would be at or below entry
                    base_return['error'] = f"TP price {tp_price} is at or below entry {current_price} for LONG"
                    final_signal = 'none' # Invalidate signal


        elif conditions['ema_cross_down'] and conditions['st_red'] and conditions['rsi_short_ok']:
            final_signal = 'down'
            swing_high = max(kl['High'].iloc[-2], kl['High'].iloc[-3])
            sl_price = round(swing_high + (kl['High'].iloc[-1] * 0.001), price_precision) # Buffer
            if current_price >= sl_price: # SL would be at or below entry
                base_return['error'] = f"SL price {sl_price} is at or below entry {current_price} for SHORT"
                final_signal = 'none' # Invalidate signal
            else:
                risk_amount = sl_price - current_price
                tp_price = round(current_price - (risk_amount * 1.5), price_precision)
                if tp_price >= current_price: # TP would be at or above entry
                    base_return['error'] = f"TP price {tp_price} is at or above entry {current_price} for SHORT"
                    final_signal = 'none' # Invalidate signal
        
        if final_signal == 'none' and base_return['error']:
             print(f"Strategy {STRATEGIES[1]} for {symbol}: Signal invalidated due to SL/TP error: {base_return['error']}")


        base_return['signal'] = final_signal
        base_return['conditions'] = conditions
        base_return['sl_price'] = sl_price if final_signal != 'none' else None
        base_return['tp_price'] = tp_price if final_signal != 'none' else None
        # Error already set if any specific issue, or remains None
        return base_return

    except Exception as e:
        base_return['error'] = f"Exception in strategy_ema_supertrend: {str(e)}"
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

def open_order(symbol, side, strategy_sl=None, strategy_tp=None):
    global client, ACCOUNT_RISK_PERCENT, SL_PERCENT, TP_PERCENT # SL_PERCENT, TP_PERCENT used if strategy_sl/tp are None
    if not client: return
    try:
        price = float(client.ticker_price(symbol)['price'])
        qty_precision = get_qty_precision(symbol)
        price_precision = get_price_precision(symbol)

        account_balance = get_balance_usdt()
        if account_balance is None or account_balance <= 0: return
        capital_to_risk_usdt = account_balance * ACCOUNT_RISK_PERCENT
        if capital_to_risk_usdt <= 0: return
        if SL_PERCENT <= 0: return

        position_size_usdt = capital_to_risk_usdt / SL_PERCENT
        calculated_qty_asset = round(position_size_usdt / price, qty_precision)
        if calculated_qty_asset <= 0: return

        print(f"Order Details ({symbol}): Bal={account_balance:.2f}, RiskCap={capital_to_risk_usdt:.2f}, PosSize={position_size_usdt:.2f}, Qty={calculated_qty_asset}")

        sl_actual, tp_actual = None, None

        if strategy_sl is not None and strategy_tp is not None:
            sl_actual = strategy_sl
            tp_actual = strategy_tp
            print(f"Using strategy-defined SL: {sl_actual}, TP: {tp_actual} for {symbol} {side}")
        else:
            if side == 'buy':
                sl_actual = round(price - price * SL_PERCENT, price_precision)
                tp_actual = round(price + price * TP_PERCENT, price_precision)
            elif side == 'sell':
                sl_actual = round(price + price * SL_PERCENT, price_precision)
                tp_actual = round(price - price * TP_PERCENT, price_precision)
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
        elif side == 'sell':
            resp1 = client.new_order(symbol=symbol, side='SELL', type='LIMIT', quantity=calculated_qty_asset, timeInForce='GTC', price=price, newOrderRespType='FULL')
            print(f"SELL {symbol}: {resp1}")
            sleep(0.2)
            resp2 = client.new_order(symbol=symbol, side='BUY', type='STOP_MARKET', quantity=calculated_qty_asset, timeInForce='GTC', stopPrice=sl_actual, reduceOnly=True, newOrderRespType='FULL')
            print(f"SL for SELL {symbol}: {resp2}")
            sleep(0.2)
            resp3 = client.new_order(symbol=symbol, side='BUY', type='TAKE_PROFIT_MARKET', quantity=calculated_qty_asset, timeInForce='GTC', stopPrice=tp_actual, reduceOnly=True, newOrderRespType='FULL')
            print(f"TP for SELL {symbol}: {resp3}")
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
        return None

    positions_data = []
    try:
        raw_positions = client.get_position_risk()
        if raw_positions:
            for p_data in raw_positions:
                try:
                    pos_amt_float = float(p_data.get('positionAmt', '0'))
                    if pos_amt_float != 0:
                        positions_data.append({
                            'symbol': p_data['symbol'],
                            'qty': pos_amt_float,
                            'entry_price': float(p_data.get('entryPrice', '0')),
                            'mark_price': float(p_data.get('markPrice', '0')),
                            'pnl': float(p_data.get('unRealizedProfit', '0')),
                            'leverage': p_data.get('leverage', 'N/A')
                        })
                except ValueError as ve:
                    print(f"ValueError parsing position data for {p_data.get('symbol')}: {ve}")
                    continue
        return positions_data
    except ClientError as e:
        msg = f"API Error (Positions): {e.error_message[:40] if hasattr(e, 'error_message') and e.error_message else str(e)[:40]}"
        print(msg)
        if root and root.winfo_exists() and status_var: root.after(0, lambda s=msg: status_var.set(s))
        return None
    except Exception as e_gen:
        msg = f"Error (Positions): {str(e_gen)[:40]}"
        print(msg)
        if root and root.winfo_exists() and status_var: root.after(0, lambda s=msg: status_var.set(s))
        return None

def format_positions_for_display(positions_data_list):
    if positions_data_list is None:
        return ["Error fetching positions or client not ready."]
    if not positions_data_list:
        return ["No open positions."]

    formatted_strings = []
    for p in positions_data_list:
        formatted_strings.append(
            f"Sym: {p['symbol']}, Qty: {p['qty']}, Entry: {p['entry_price']:.4f}, " +
            f"MarkP: {p['mark_price']:.4f}, PnL: {p['pnl']:.2f} USDT, Lev: {p['leverage']}" # leverage is already string "Xx"
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
    global qty_concurrent_positions, type, leverage
    global balance_var, positions_text_widget, history_text_widget, activity_status_var

    _status_set = lambda msg: status_var.set(msg) if status_var and root and root.winfo_exists() else None
    _activity_set = lambda msg: activity_status_var.set(msg) if activity_status_var and root and root.winfo_exists() else None
    
    print("Bot logic thread started."); _status_set("Bot running...")
    if client is None:
        _status_set("Error: Client not initialized. Bot stopping.")
        _activity_set("Bot Idle - Client Error")
        bot_running = False

    loop_count = 0
    while bot_running:
        try:
            _activity_set("Starting new scan cycle...")
            if client is None: # Should ideally not happen if start_bot checks client
                _status_set("Client is None. Bot stopping.")
                _activity_set("Bot Idle - Client Error")
                bot_running = False; break
            
            balance = get_balance_usdt(); sleep(0.1)
            if not bot_running: break # Check immediately after potentially long operation

            if root and root.winfo_exists() and balance_var: # GUI update for balance
                if balance is not None: root.after(0, lambda bal=balance: balance_var.set(f"{bal:.2f} USDT"))
                else: root.after(0, lambda: balance_var.set("Error or N/A"))

            if balance is None: # Handle balance fetch error
                msg = 'API/balance error. Retrying...'; print(msg); _status_set(msg)
                _activity_set("Balance fetch error. Retrying...")
                for _ in range(60): # Wait and retry
                    if not bot_running: break
                    sleep(1)
                if not bot_running: break
                continue # Retry balance fetch

            current_balance_msg_for_status = f"Bal: {balance:.2f} USDT."
            _status_set(current_balance_msg_for_status + " Scanning target symbols...")
            
            active_positions_data = get_active_positions_data() # Structured data or None
            if not bot_running: break # Check after potentially long call

            open_position_symbols = []
            if active_positions_data: # Not None and not empty
                open_position_symbols = [p['symbol'] for p in active_positions_data]
            
            # Strategy 1: Check and clear active trade if position is closed
            if ACTIVE_STRATEGY_ID == 1 and strategy1_active_trade_info['symbol'] is not None:
                is_still_open = False
                if active_positions_data:
                    for pos in active_positions_data:
                        if pos['symbol'] == strategy1_active_trade_info['symbol']:
                            is_still_open = True
                            # Optional: Update qty if it changed and if we were tracking it precisely
                            # strategy1_active_trade_info['position_qty'] = pos['qty'] 
                            break
                if not is_still_open:
                    print(f"Strategy 1: Detected closure of trade on {strategy1_active_trade_info['symbol']}.")
                    # Future: Implement P&L check here for "cooldown after loss"
                    strategy1_active_trade_info = {'symbol': None, 'entry_time': None, 'entry_price': None, 'position_qty': 0, 'side': None, 'sl_order_id': None, 'tp_order_id': None}
                    _activity_set(f"Strategy 1: Trade for {strategy1_active_trade_info['symbol']} closed. Ready for new signal.")


            if not bot_running: break

            # Determine if a new trade can be opened based on strategy and current state
            can_open_new_trade_overall = False
            if ACTIVE_STRATEGY_ID == 1:
                if strategy1_active_trade_info['symbol'] is None:
                    can_open_new_trade_overall = True
                else: # Strategy 1 already has an active trade
                    _activity_set(f"Strategy 1: Trade active on {strategy1_active_trade_info['symbol']}. Monitoring it. Not seeking new trades this cycle.")
            elif len(open_position_symbols) < qty_concurrent_positions: # For Strategy 0 (Original Scalping)
                can_open_new_trade_overall = True
            else: # Max positions reached for Strategy 0
                 _activity_set(f"Strategy 0: At max positions ({len(open_position_symbols)}). Monitoring. Cycle will pause scan for new trades.")

            if can_open_new_trade_overall:
                symbols_to_check = TARGET_SYMBOLS 
                all_open_orders_after_cancel = check_orders() # Refresh orders
                if not bot_running: break
                if all_open_orders_after_cancel is None: all_open_orders_after_cancel = []
                current_symbols_with_any_open_orders = {o.get('symbol') for o in all_open_orders_after_cancel if isinstance(o, dict) and o.get('symbol')}

                for sym_to_check in symbols_to_check:
                    if not bot_running: break
                    _activity_set(f"Scanning: {sym_to_check}...")
                    
                    # General skip if symbol already has a position or open orders (relevant for Strategy 0)
                    if ACTIVE_STRATEGY_ID == 0 and (sym_to_check in open_position_symbols or sym_to_check in current_symbols_with_any_open_orders):
                        _activity_set(f"Strategy 0: Skipping {sym_to_check} (already has position/orders).")
                        sleep(0.05) 
                        continue
                    
                    # --- Signal Generation ---
                    signal_data = {} # Initialize to prevent UnboundLocalError if no strategy matches
                    if ACTIVE_STRATEGY_ID == 0:
                        signal_data = scalping_strategy_signal(sym_to_check)
                    elif ACTIVE_STRATEGY_ID == 1:
                         # This block will only be reached if strategy1_active_trade_info['symbol'] is None (due to can_open_new_trade_overall logic)
                        signal_data = strategy_ema_supertrend(sym_to_check)
                    else: # Unknown strategy
                        signal_data = {'signal': 'none', 'conditions': {"error": f"Unknown strategy ID: {ACTIVE_STRATEGY_ID}"}, 'error': "Unknown strategy ID"}
                        print(f"Unknown strategy ID: {ACTIVE_STRATEGY_ID} for {sym_to_check}")
                    
                    current_signal = signal_data.get('signal', 'none')

                    if root and root.winfo_exists():
                        root.after(0, update_conditions_display_content, sym_to_check, signal_data.get('conditions'), signal_data.get('error'))
                    if not bot_running: break

                    # --- Order Placement ---
                    if current_signal == 'up' or current_signal == 'down':
                        _status_set(f"{current_signal.upper()} signal for {sym_to_check}. Ordering...");
                        set_mode(sym_to_check, type); sleep(0.1)
                        set_leverage(sym_to_check, leverage); sleep(0.1)
                        
                        open_order(sym_to_check, current_signal, 
                                   strategy_sl=signal_data.get('sl_price'), 
                                   strategy_tp=signal_data.get('tp_price'))
                        
                        if ACTIVE_STRATEGY_ID == 1: 
                            strategy1_active_trade_info['symbol'] = sym_to_check
                            strategy1_active_trade_info['entry_time'] = pd.Timestamp.now(tz='UTC')
                            try: 
                                current_price_for_entry = float(client.ticker_price(sym_to_check)['price'])
                                strategy1_active_trade_info['entry_price'] = current_price_for_entry
                            except Exception as e_price:
                                print(f"Could not fetch entry price for {sym_to_check} for Strategy 1 tracking: {e_price}")
                                strategy1_active_trade_info['entry_price'] = None 
                            strategy1_active_trade_info['side'] = current_signal
                            print(f"Strategy 1: Trade initiated for {sym_to_check}. Basic Info: {strategy1_active_trade_info}")
                            _activity_set(f"Strategy 1: Trade initiated for {sym_to_check}. Monitoring.")
                            # can_open_new_trade_overall = False # This was set to false but this is the inner loop, it should break
                            break # For Strategy 1, stop scanning for new symbols after initiating a trade

                        active_positions_data = get_active_positions_data() 
                        if active_positions_data: open_position_symbols = [p['symbol'] for p in active_positions_data]
                        sleep(3) 
                        
                        if ACTIVE_STRATEGY_ID == 0 and len(open_position_symbols) >= qty_concurrent_positions:
                            print(f"Strategy 0: Reached max concurrent positions ({qty_concurrent_positions}). Pausing symbol scan for this cycle.")
                            _activity_set(f"Strategy 0: Max positions reached. Pausing scan.")
                            break 
                    else: # signal == 'none'
                        _activity_set(f"No signal: {sym_to_check}. Next...")
                        sleep(0.1) 
            else: # Not can_open_new_trade_overall
                if ACTIVE_STRATEGY_ID == 1 and strategy1_active_trade_info['symbol'] is not None:
                     _activity_set(f"Strategy 1: Monitoring active trade on {strategy1_active_trade_info['symbol']}.")
                # For strategy 0, message is already set if at max positions.
                # If not at max positions but can_open_new_trade_overall is false for other reasons (future logic), add message here.
                sleep(1) # Small sleep if not actively scanning new symbols this cycle.

            if not bot_running: break # Check before starting long wait
            loop_count +=1
            wait_message = f"{current_balance_msg_for_status} Scan cycle done. Waiting..."
            _status_set(wait_message)
            _activity_set("Scan cycle complete. Waiting for next cycle...")
            for _ in range(180): # 3-minute wait
                if not bot_running: break
                sleep(1)
        except ClientError as ce:
            err_msg = f"API Error: {ce.error_message if hasattr(ce, 'error_message') and ce.error_message else ce}. Retrying..."
            print(err_msg); _status_set(err_msg); _activity_set(f"API Error. Retrying...")
            if "signature" in str(ce).lower() or "timestamp" in str(ce).lower(): reinitialize_client() # Re-init on critical auth errors
            for _ in range(60): # Wait before retrying loop
                if not bot_running: break
                sleep(1)
        except Exception as e:
            err_msg = f"Bot loop error: {e}. Retrying..."
            print(err_msg); _status_set(err_msg); _activity_set(f"Loop Error. Retrying...")
            for _ in range(60): # Wait before retrying loop
                if not bot_running: break
                sleep(1)
            if not bot_running: break # Exit outer loop if stop signal received during wait
            continue # Continue to next iteration of outer loop

    _activity_set("Bot Idle")
    if root and root.winfo_exists(): # Clear conditions on stop
        root.after(0, update_conditions_display_content, "Bot Idle", None, "Bot stopped. No conditions to display.")
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
    global ACCOUNT_RISK_PERCENT, TP_PERCENT, SL_PERCENT, leverage, qty_concurrent_positions, LOCAL_HIGH_LOW_LOOKBACK_PERIOD, type, TARGET_SYMBOLS, ACTIVE_STRATEGY_ID # 'type' is for margin type

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
        if not (0 < arp_val < 1): # Should be a fraction like 0.02
            messagebox.showerror("Settings Error", "Account Risk % must be between 0 and 1 (e.g., 0.02 for 2%).")
            return False
        ACCOUNT_RISK_PERCENT = arp_val

        # Take Profit Percent
        tp_val = float(tp_percent_var.get())
        if not (0 < tp_val < 1): # Should be a fraction like 0.01
            messagebox.showerror("Settings Error", "Take Profit % must be between 0 and 1 (e.g., 0.01 for 1%).")
            return False
        TP_PERCENT = tp_val

        # Stop Loss Percent
        sl_val = float(sl_percent_var.get())
        if not (0 < sl_val < 1): # Should be a fraction like 0.005
            messagebox.showerror("Settings Error", "Stop Loss % must be between 0 and 1 (e.g., 0.005 for 0.5%).")
            return False
        SL_PERCENT = sl_val

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
        type = margin_type_var.get() # This is already a string 'ISOLATED' or 'CROSS'
        if type not in ["ISOLATED", "CROSS"]: # Should not happen with radio buttons
             messagebox.showerror("Settings Error", "Invalid Margin Type selected.")
             return False

        print(f"Applied settings: Strategy='{STRATEGIES[ACTIVE_STRATEGY_ID]}', Risk={ACCOUNT_RISK_PERCENT}, TP={TP_PERCENT}, SL={SL_PERCENT}, Lev={leverage}, MaxPos={qty_concurrent_positions}, Lookback={LOCAL_HIGH_LOW_LOOKBACK_PERIOD}, Margin={type}, Symbols={TARGET_SYMBOLS}")
        messagebox.showinfo("Settings", "Settings applied successfully!")
        return True

    except ValueError as ve:
        messagebox.showerror("Settings Error", f"Invalid input. Please ensure all parameters are numbers. Error: {ve}")
        return False
    except Exception as e:
        messagebox.showerror("Settings Error", f"An unexpected error occurred while applying settings: {e}")
        return False

def start_bot():
    global bot_running, bot_thread, status_var, client, start_button, stop_button, testnet_radio, mainnet_radio, conditions_text_widget
    _status_set = lambda msg: status_var.set(msg) if status_var and root and root.winfo_exists() else None
    
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

    bot_thread = threading.Thread(target=run_bot_logic, daemon=True); bot_thread.start()

def stop_bot():
    global bot_running, bot_thread, status_var, start_button, stop_button, testnet_radio, mainnet_radio, activity_status_var, conditions_text_widget
    _status_set = lambda msg: status_var.set(msg) if status_var and root and root.winfo_exists() else None
    _activity_set = lambda msg: activity_status_var.set(msg) if activity_status_var and root and root.winfo_exists() else None

    if not bot_running and (bot_thread is None or not bot_thread.is_alive()):
        _status_set("Bot is not running.")
        _activity_set("Bot Idle") 
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

# --- Main Application Window ---
if __name__ == "__main__":
    root = tk.Tk(); root.title("Binance Scalping Bot")
    # global status_var, current_env_var, start_button, stop_button, testnet_radio, mainnet_radio, balance_var, positions_text_widget, history_text_widget # This line caused SyntaxError
    global account_risk_percent_var, tp_percent_var, sl_percent_var, leverage_var, qty_concurrent_positions_var, local_high_low_lookback_var, margin_type_var, target_symbols_var, activity_status_var, conditions_text_widget, selected_strategy_var
    global account_risk_percent_entry, tp_percent_entry, sl_percent_entry, leverage_entry, qty_concurrent_positions_entry, local_high_low_lookback_entry, margin_type_isolated_radio, margin_type_cross_radio, target_symbols_entry, strategy_radio_buttons
    global params_widgets

    status_var = tk.StringVar(value="Welcome! Select environment."); current_env_var = tk.StringVar(value=current_env); balance_var = tk.StringVar(value="N/A")
    activity_status_var = tk.StringVar(value="Bot Idle") # Initialize activity status
    selected_strategy_var = tk.IntVar(value=ACTIVE_STRATEGY_ID)

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
    account_risk_percent_var = tk.StringVar(value=str(ACCOUNT_RISK_PERCENT))
    account_risk_percent_entry = ttk.Entry(params_input_frame, textvariable=account_risk_percent_var, width=10)
    account_risk_percent_entry.grid(row=0, column=1, padx=2, pady=2, sticky='w')

    # TP_PERCENT
    ttk.Label(params_input_frame, text="Take Profit %:").grid(row=0, column=2, padx=2, pady=2, sticky='w') # Next column
    tp_percent_var = tk.StringVar(value=str(TP_PERCENT))
    tp_percent_entry = ttk.Entry(params_input_frame, textvariable=tp_percent_var, width=10)
    tp_percent_entry.grid(row=0, column=3, padx=2, pady=2, sticky='w')

    # SL_PERCENT
    ttk.Label(params_input_frame, text="Stop Loss %:").grid(row=1, column=0, padx=2, pady=2, sticky='w')
    sl_percent_var = tk.StringVar(value=str(SL_PERCENT))
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
    margin_type_var = tk.StringVar(value=type) # Initialize with global 'type'
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
    
    strategy_radio_buttons = [] # Re-initialize here before populating
    for strategy_id, strategy_name in STRATEGIES.items():
        rb = ttk.Radiobutton(strategy_frame, 
                             text=strategy_name, 
                             variable=selected_strategy_var, 
                             value=strategy_id,
                             command=lambda sid=strategy_id: print(f"Strategy {STRATEGIES[sid]} ({sid}) GUI selected")) # Updated command
        rb.pack(anchor='w', padx=5)
        strategy_radio_buttons.append(rb)
    
    params_widgets.extend(strategy_radio_buttons) # Add strategy radios to params_widgets for state mgmt

    api_key_info_label = ttk.Label(controls_frame, text="API Keys from keys.py"); api_key_info_label.pack(pady=2)
    timeframe_label = ttk.Label(controls_frame, text="Timeframe: 5m (fixed)"); timeframe_label.pack(pady=2)

    buttons_frame = ttk.Frame(controls_frame); buttons_frame.pack(pady=2)
    start_button = ttk.Button(buttons_frame, text="Start Bot", command=start_bot); start_button.pack(side=tk.LEFT, padx=5)
    stop_button = ttk.Button(buttons_frame, text="Stop Bot", command=stop_bot, state=tk.DISABLED); stop_button.pack(side=tk.LEFT, padx=5)
    data_frame = ttk.LabelFrame(root, text="Live Data & History"); data_frame.pack(padx=10, pady=5, fill="both", expand=True)
    
    activity_display_frame = ttk.Frame(data_frame)
    activity_display_frame.pack(pady=(2,2), padx=5, fill="x")
    ttk.Label(activity_display_frame, text="Current Activity:").pack(side=tk.LEFT, padx=(0,5))
    ttk.Label(activity_display_frame, textvariable=activity_status_var).pack(side=tk.LEFT)

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

    def on_closing():
        global bot_running, bot_thread, root
        if bot_running:
            if messagebox.askokcancel("Quit", "Bot is running. Quit anyway?"):
                bot_running = False
                if bot_thread and bot_thread.is_alive(): bot_thread.join(timeout=2)
                root.destroy()
        else: root.destroy()
    root.protocol("WM_DELETE_WINDOW", on_closing); root.minsize(450, 500); root.mainloop()
