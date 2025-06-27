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

DEFAULT_ATR_MULTIPLIER = 2.0 # Moved here
def calculate_dynamic_sl_tp(symbol: str, entry_price: float, side: str, atr_period: int = 14, rr: float = 1.5, atr_multiplier: float = DEFAULT_ATR_MULTIPLIER): # Changed buffer to atr_multiplier
    # Ensure klines, get_price_precision are available in the scope or passed as arguments if necessary.
    # Assuming they are globally accessible or defined in the same file as per the existing code structure.
    # Also assumes 'ta' and 'pd' (pandas) are imported.

    return_value = {'sl_price': None, 'tp_price': None, 'error': None}
    
    kl_df = klines(symbol) # Fetches 5m klines by default from existing function

    if kl_df is None or kl_df.empty:
        return_value['error'] = f"No kline data returned for {symbol}."
        return return_value

    # Ensure enough data for ATR calculation (atr_period + a few for stability)
    if len(kl_df) < atr_period + 5: # Arbitrary buffer of 5, adjust if needed
        return_value['error'] = f"Insufficient kline data for ATR calculation on {symbol} (need at least {atr_period + 5}, got {len(kl_df)})."
        return return_value

    try:
        # Ensure columns are correct, klines() function seems to provide them
        atr_indicator = ta.volatility.AverageTrueRange(
            high=kl_df['High'],
            low=kl_df['Low'],
            close=kl_df['Close'],
            window=atr_period,
            fillna=False # Ensure NaNs are not filled prematurely by ta library itself
        )
        atr_series = atr_indicator.average_true_range()

        if atr_series is None or atr_series.empty:
            return_value['error'] = f"ATR calculation resulted in an empty series for {symbol}."
            return return_value
        
        # Get the last valid ATR value
        atr_value = atr_series.dropna().iloc[-1] if not atr_series.dropna().empty else None

        if atr_value is None or pd.isna(atr_value) or atr_value == 0: # ATR could be zero in very flat markets
            return_value['error'] = f"Invalid ATR value ({atr_value}) for {symbol}. Cannot calculate SL/TP."
            return return_value

    except Exception as e:
        return_value['error'] = f"Error calculating ATR for {symbol}: {str(e)}"
        return return_value

    price_precision = get_price_precision(symbol)
    if not isinstance(price_precision, int) or price_precision < 0:
        # Fallback precision if get_price_precision fails or returns invalid
        print(f"Warning: Invalid price_precision '{price_precision}' for {symbol} from get_price_precision. Defaulting to 2.")
        price_precision = 2 


    sl_calculated = None
    tp_calculated = None

    if side == 'up':
        sl_calculated = entry_price - (atr_value * atr_multiplier) # Changed calculation
        # Ensure SL is actually below entry for a long trade
        if sl_calculated >= entry_price:
            return_value['error'] = f"Calculated SL ({sl_calculated}) for 'up' side is not below entry price ({entry_price})."
            # Optionally, could adjust SL here, e.g. entry_price - atr_value (without buffer), or just return error.
            # For now, returning error as per strict interpretation.
            return return_value
        tp_calculated = entry_price + (entry_price - sl_calculated) * rr
    elif side == 'down':
        sl_calculated = entry_price + (atr_value * atr_multiplier) # Changed calculation
        # Ensure SL is actually above entry for a short trade
        if sl_calculated <= entry_price:
            return_value['error'] = f"Calculated SL ({sl_calculated}) for 'down' side is not above entry price ({entry_price})."
            return return_value
        tp_calculated = entry_price - (sl_calculated - entry_price) * rr
    else:
        return_value['error'] = f"Invalid side '{side}' provided. Must be 'up' or 'down'."
        return return_value

    # Rounding
    try:
        # Before rounding, check if calculated SL/TP are positive
        if sl_calculated <= 0:
            return_value['error'] = f"Calculated SL price ({sl_calculated}) is not positive for {symbol}."
            return_value['sl_price'] = None
            return_value['tp_price'] = None # Invalidate TP as well if SL is invalid
            return return_value
        
        if tp_calculated <= 0:
            return_value['error'] = f"Calculated TP price ({tp_calculated}) is not positive for {symbol}."
            # SL might still be valid, but TP is not. Depending on strategy, this might be acceptable or invalidate the signal.
            # For now, let's invalidate TP but keep SL if it was positive.
            # However, since TP depends on SL, if SL was the issue, TP would also be affected.
            # If SL was valid but TP calculation (e.g. RR part) made it non-positive, only TP is invalid.
            return_value['tp_price'] = None 
            # If SL was fine, keep it: return_value['sl_price'] = round(sl_calculated, price_precision) 
            # But it's safer to invalidate both if one part of the critical SL/TP pair is bad.
            # Forcing TP to None if it's bad. SL would have been caught by the check above if it was bad.
            # If SL is positive but TP is not, it implies an issue with RR or the difference calculation part.
            # Let's be strict: if TP is bad, and SL was good, we still might want to invalidate the signal.
            # For now, if TP is non-positive, just nullify TP. The calling strategy should decide what to do.
            # Re-evaluating: if tp_calculated is <=0, it's a critical failure. Invalidate both for safety.
            if return_value['sl_price'] is not None: # If SL was previously considered okay
                 return_value['sl_price'] = round(sl_calculated, price_precision) # keep the rounded SL
            # but if tp is bad, the signal might be unusable.
            # The strategy should handle this if tp_price is None.
            # Let's stick to: if sl_calculated <= 0, both are None. If tp_calculated <=0, only tp is None (sl might be valid).
            # This will be handled by the subsequent checks as well.

        return_value['sl_price'] = round(sl_calculated, price_precision)
        return_value['tp_price'] = round(tp_calculated, price_precision)

    except TypeError as te: # Handles if sl_calculated or tp_calculated are None
         return_value['error'] = f"Error rounding SL/TP for {symbol}: {str(te)}. SL Calc: {sl_calculated}, TP Calc: {tp_calculated}, Precision: {price_precision}"
         return_value['sl_price'] = None 
         return_value['tp_price'] = None
         return return_value

    # Final validation for SL/TP prices being positive after rounding
    if return_value['sl_price'] is not None and return_value['sl_price'] <= 0:
        return_value['error'] = f"Rounded SL price ({return_value['sl_price']}) is not positive for {symbol}."
        return_value['sl_price'] = None
        return_value['tp_price'] = None # Invalidate TP as well
        return return_value

    if return_value['tp_price'] is not None and return_value['tp_price'] <= 0:
        return_value['error'] = f"Rounded TP price ({return_value['tp_price']}) is not positive for {symbol}."
        return_value['tp_price'] = None
        # SL might still be valid, but the strategy should check if tp_price is None.
        # No need to return immediately here if SL is valid.

    # Final validation for TP vs Entry (and SL vs Entry was done earlier implicitly)
    if side == 'up':
        if return_value['sl_price'] is not None and return_value['sl_price'] >= entry_price:
             return_value['error'] = f"SL price ({return_value['sl_price']}) for 'up' side not below entry ({entry_price})."
             return_value['sl_price'] = None; return_value['tp_price'] = None; return return_value
        if return_value['tp_price'] is not None and return_value['tp_price'] <= entry_price:
            return_value['error'] = f"TP price ({return_value['tp_price']}) for 'up' side not above entry ({entry_price})."
            return_value['tp_price'] = None 
            # SL might still be valid. Strategy should check.

    elif side == 'down':
        if return_value['sl_price'] is not None and return_value['sl_price'] <= entry_price:
            return_value['error'] = f"SL price ({return_value['sl_price']}) for 'down' side not above entry ({entry_price})."
            return_value['sl_price'] = None; return_value['tp_price'] = None; return return_value
        if return_value['tp_price'] is not None and return_value['tp_price'] >= entry_price:
            return_value['error'] = f"TP price ({return_value['tp_price']}) for 'down' side not below entry ({entry_price})."
            return_value['tp_price'] = None
            # SL might still be valid. Strategy should check.

    return return_value


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
    log_msg_klines_extended = [f"[klines_extended LOG for {symbol}] Fetch range: {pd.to_datetime(start_time_ms, unit='ms')} to {pd.to_datetime(end_time, unit='ms')}"]

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
    
    log_msg_klines_extended.append(f"Raw klines fetched: {len(all_klines_raw)}")
    if not all_klines_raw:
        error_msg = f"No kline data fetched for {symbol} with timeframe {timeframe} for {interval_days} days."
        log_msg_klines_extended.append(error_msg)
        print("\n".join(log_msg_klines_extended))
        return pd.DataFrame(columns=['Open', 'High', 'Low', 'Close', 'Volume']), error_msg

    df = pd.DataFrame(all_klines_raw)
    df = df.iloc[:, :6]
    df.columns = ['Time', 'Open', 'High', 'Low', 'Close', 'Volume']
    df['Time'] = pd.to_datetime(df['Time'], unit='ms')
    df = df.set_index('Time')
    df = df.astype(float)
    
    df = df[~df.index.duplicated(keep='first')]
    df = df.sort_index()
    log_msg_klines_extended.append(f"Processed klines (after deduplication & sort): {len(df)}")
    log_msg_klines_extended.append(f"Fetched {len(df)} klines for {symbol} ({timeframe}, {interval_days} days). From {df.index.min()} to {df.index.max()}")

    # --- BEGIN DEBUG PRINTS ---
    log_msg_klines_extended.append(f"DEBUG klines_extended: DataFrame head:\n{df.head()}")
    log_msg_klines_extended.append(f"DEBUG klines_extended: DataFrame dtypes:\n{df.dtypes}")
    log_msg_klines_extended.append(f"DEBUG klines_extended: DataFrame index:\n{df.index}")
    log_msg_klines_extended.append(f"DEBUG klines_extended: DataFrame shape: {df.shape}")
    log_msg_klines_extended.append(f"DEBUG klines_extended: Any NaNs in DataFrame: {df.isnull().values.any()}")
    log_msg_klines_extended.append(f"DEBUG klines_extended: Sum of NaNs per column:\n{df.isnull().sum()}")
    # --- END DEBUG PRINTS ---
    print("\n".join(log_msg_klines_extended))
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

def atr_bt(high_series, low_series, close_series, window):
    # Ensure inputs are pandas Series
    high = pd.Series(high_series)
    low = pd.Series(low_series)
    close = pd.Series(close_series)
    atr_indicator = ta.volatility.AverageTrueRange(
        high=high,
        low=low,
        close=close,
        window=window,
        fillna=False # Important for backtesting.py to handle NaNs initially
    )
    atr_values = atr_indicator.average_true_range()
    return atr_values # Return the pandas Series

def supertrend_numerical_bt(high_series, low_series, close_series, atr_period, multiplier):
    # Create a DataFrame from the input arrays (which self.I provides)
    df = pd.DataFrame({
        'High': pd.Series(high_series),
        'Low': pd.Series(low_series),
        'Close': pd.Series(close_series)
    })
    
    # calculate_supertrend_pta expects a DataFrame and returns a Series of 'green'/'red'
    st_signal_str = calculate_supertrend_pta(df, atr_period=atr_period, multiplier=multiplier)
    
    # Convert 'green'/'red' to 1/-1
    numerical_signal = st_signal_str.map({'green': 1, 'red': -1}).fillna(0) # Fill NaNs with 0 (neutral)
    return numerical_signal.values # self.I expects a numpy array or list

# --- Candlestick Pattern Helper Functions (Strategy 7) ---
def get_candle_metrics(candle_series: pd.Series) -> dict:
    """Extracts OHLC and calculates body, wicks, and range from a candle series."""
    o = float(candle_series['Open'])
    h = float(candle_series['High'])
    l = float(candle_series['Low'])
    c = float(candle_series['Close'])
    v = float(candle_series['Volume'])
    
    body = abs(c - o)
    upper_wick = h - max(o, c)
    lower_wick = min(o, c) - l
    total_range = h - l
    
    return {
        'o': o, 'h': h, 'l': l, 'c': c, 'v': v,
        'body': body, 'upper_wick': upper_wick, 'lower_wick': lower_wick, 'total_range': total_range,
        'is_bullish': c > o, 'is_bearish': c < o,
        'is_doji_shape': body <= total_range * 0.1 if total_range > 0 else body == 0 
    }

def check_volume_spike(kl_df: pd.DataFrame, lookback_period: int = 20, multiplier: float = 2.0) -> bool:
    if kl_df is None or len(kl_df) < lookback_period + 1:
        return False
    try:
        avg_volume = kl_df['Volume'].iloc[-(lookback_period + 1):-1].mean()
        current_volume = kl_df['Volume'].iloc[-1]
        if pd.isna(avg_volume) or pd.isna(current_volume) or avg_volume == 0:
            return current_volume > 0 if avg_volume == 0 else False
        return current_volume > (avg_volume * multiplier)
    except Exception as e:
        print(f"Error in check_volume_spike: {e}")
        return False

def is_hammer(metrics: dict) -> bool:
    if not metrics: return False
    return metrics['body'] > 0 and metrics['lower_wick'] >= 2 * metrics['body'] and metrics['upper_wick'] < metrics['body'] * 0.8 

def is_inverted_hammer(metrics: dict) -> bool:
    if not metrics: return False
    return metrics['body'] > 0 and metrics['upper_wick'] >= 2 * metrics['body'] and metrics['lower_wick'] < metrics['body'] * 0.8

def is_bullish_engulfing(prev_metrics: dict, curr_metrics: dict) -> bool:
    if not prev_metrics or not curr_metrics: return False
    return prev_metrics['is_bearish'] and curr_metrics['is_bullish'] and \
           curr_metrics['c'] > prev_metrics['o'] and curr_metrics['o'] < prev_metrics['c'] and \
           curr_metrics['body'] > prev_metrics['body']

def is_piercing_line(prev_metrics: dict, curr_metrics: dict) -> bool:
    if not prev_metrics or not curr_metrics: return False
    mid_point_prev_body = prev_metrics['c'] + prev_metrics['body'] / 2 if prev_metrics['is_bearish'] else prev_metrics['o'] + prev_metrics['body'] / 2
    return prev_metrics['is_bearish'] and curr_metrics['is_bullish'] and \
           curr_metrics['o'] < prev_metrics['l'] and \
           curr_metrics['c'] > mid_point_prev_body and \
           curr_metrics['c'] < prev_metrics['o'] 

def is_morning_star(c1m: dict, c2m: dict, c3m: dict) -> bool: 
    if not c1m or not c2m or not c3m: return False
    return c1m['is_bearish'] and (c1m['body'] / (c1m['total_range'] if c1m['total_range'] > 0 else 1)) > 0.6 and \
           (c2m['body'] / (c2m['total_range'] if c2m['total_range'] > 0 else 1)) < 0.3 and \
           max(c2m['o'], c2m['c']) < c1m['c'] and \
           c3m['is_bullish'] and c3m['c'] > (c1m['o'] + c1m['c']) / 2

def is_hanging_man(metrics: dict) -> bool: 
    if not metrics: return False
    return is_hammer(metrics)

def is_shooting_star(metrics: dict) -> bool: 
    if not metrics: return False
    return is_inverted_hammer(metrics)

def is_bearish_engulfing(prev_metrics: dict, curr_metrics: dict) -> bool:
    if not prev_metrics or not curr_metrics: return False
    return prev_metrics['is_bullish'] and curr_metrics['is_bearish'] and \
           curr_metrics['o'] > prev_metrics['c'] and curr_metrics['c'] < prev_metrics['o'] and \
           curr_metrics['body'] > prev_metrics['body']

def is_evening_star(c1m: dict, c2m: dict, c3m: dict) -> bool: 
    if not c1m or not c2m or not c3m: return False
    return c1m['is_bullish'] and (c1m['body'] / (c1m['total_range'] if c1m['total_range'] > 0 else 1)) > 0.6 and \
           (c2m['body'] / (c2m['total_range'] if c2m['total_range'] > 0 else 1)) < 0.3 and \
           min(c2m['o'], c2m['c']) > c1m['c'] and \
           c3m['is_bearish'] and c3m['c'] < (c1m['o'] + c1m['c']) / 2
           
def is_dark_cloud_cover(prev_metrics: dict, curr_metrics: dict) -> bool:
    if not prev_metrics or not curr_metrics: return False
    mid_point_prev_body = prev_metrics['o'] + prev_metrics['body'] / 2 if prev_metrics['is_bullish'] else prev_metrics['c'] + prev_metrics['body'] / 2
    return prev_metrics['is_bullish'] and curr_metrics['is_bearish'] and \
           curr_metrics['o'] > prev_metrics['h'] and \
           curr_metrics['c'] < mid_point_prev_body and \
           curr_metrics['c'] > prev_metrics['o'] 

def is_three_white_soldiers(c1m: dict, c2m: dict, c3m: dict) -> bool:
    if not c1m or not c2m or not c3m: return False
    cond1 = c1m['is_bullish'] and c2m['is_bullish'] and c3m['is_bullish']
    cond2 = all((m['body'] / (m['total_range'] if m['total_range'] > 0 else 1)) > 0.5 for m in [c1m, c2m, c3m]) 
    cond3 = (c1m['o'] < c2m['o'] and c2m['o'] < c1m['c']) and \
            (c2m['o'] < c3m['o'] and c3m['o'] < c2m['c']) 
    cond4 = c3m['c'] > c2m['c'] and c2m['c'] > c1m['c'] 
    cond5 = all(m['upper_wick'] < m['body'] * 0.5 for m in [c1m, c2m, c3m]) 
    return cond1 and cond2 and cond3 and cond4 and cond5

def is_rising_three_methods(kl_df_last5: pd.DataFrame) -> bool:
    if kl_df_last5 is None or len(kl_df_last5) < 5: return False
    m = [get_candle_metrics(kl_df_last5.iloc[i]) for i in range(5)]
    cond1 = m[0]['is_bullish'] and (m[0]['body'] / (m[0]['total_range'] if m[0]['total_range'] > 0 else 1)) > 0.7
    cond2 = m[1]['is_bearish'] and m[2]['is_bearish'] and m[3]['is_bearish']
    cond3 = all((sm_m['body'] / (sm_m['total_range'] if sm_m['total_range'] > 0 else 1)) < 0.4 for sm_m in [m[1],m[2],m[3]]) 
    cond4 = all(sm_m['h'] < m[0]['h'] and sm_m['l'] > m[0]['l'] for sm_m in [m[1],m[2],m[3]]) 
    cond5 = m[4]['is_bullish'] and m[4]['c'] > m[0]['h'] and (m[4]['body'] / (m[4]['total_range'] if m[4]['total_range'] > 0 else 1)) > 0.7
    return cond1 and cond2 and cond3 and cond4 and cond5

def is_three_black_crows(c1m: dict, c2m: dict, c3m: dict) -> bool:
    if not c1m or not c2m or not c3m: return False
    cond1 = c1m['is_bearish'] and c2m['is_bearish'] and c3m['is_bearish']
    cond2 = all((m['body'] / (m['total_range'] if m['total_range'] > 0 else 1)) > 0.5 for m in [c1m, c2m, c3m])
    cond3 = (c1m['c'] < c2m['o'] and c2m['o'] < c1m['o']) and \
            (c2m['c'] < c3m['o'] and c3m['o'] < c2m['o']) 
    cond4 = c3m['c'] < c2m['c'] and c2m['c'] < c1m['c'] 
    cond5 = all(m['lower_wick'] < m['body'] * 0.5 for m in [c1m, c2m, c3m]) 
    return cond1 and cond2 and cond3 and cond4 and cond5

def is_falling_three_methods(kl_df_last5: pd.DataFrame) -> bool:
    if kl_df_last5 is None or len(kl_df_last5) < 5: return False
    m = [get_candle_metrics(kl_df_last5.iloc[i]) for i in range(5)]
    cond1 = m[0]['is_bearish'] and (m[0]['body'] / (m[0]['total_range'] if m[0]['total_range'] > 0 else 1)) > 0.7
    cond2 = m[1]['is_bullish'] and m[2]['is_bullish'] and m[3]['is_bullish']
    cond3 = all((sm_m['body'] / (sm_m['total_range'] if sm_m['total_range'] > 0 else 1)) < 0.4 for sm_m in [m[1],m[2],m[3]])
    cond4 = all(sm_m['h'] < m[0]['h'] and sm_m['l'] > m[0]['l'] for sm_m in [m[1],m[2],m[3]])
    cond5 = m[4]['is_bearish'] and m[4]['c'] < m[0]['l'] and (m[4]['body'] / (m[4]['total_range'] if m[4]['total_range'] > 0 else 1)) > 0.7
    return cond1 and cond2 and cond3 and cond4 and cond5

def is_doji(metrics: dict, body_to_range_ratio_threshold=0.05) -> bool:
    if not metrics: return False
    if metrics['total_range'] == 0: return metrics['body'] == 0 
    return metrics['body'] / metrics['total_range'] <= body_to_range_ratio_threshold

def is_spinning_top(metrics: dict, body_to_range_ratio_threshold=0.3, min_wick_to_body_ratio=1.5) -> bool:
    if not metrics: return False
    small_body = (metrics['body'] / metrics['total_range'] <= body_to_range_ratio_threshold) if metrics['total_range'] > 0 else metrics['body'] == 0
    long_wicks = False
    if metrics['body'] > 0: 
        long_wicks = (metrics['upper_wick'] / metrics['body'] >= min_wick_to_body_ratio) and \
                     (metrics['lower_wick'] / metrics['body'] >= min_wick_to_body_ratio)
    elif metrics['total_range'] > 0 : 
        long_wicks = (metrics['upper_wick'] / metrics['total_range'] > 0.3) and \
                     (metrics['lower_wick'] / metrics['total_range'] > 0.3)
    return small_body and long_wicks

# --- Strategy 7: Candlestick Patterns ---
def strategy_candlestick_patterns_signal(symbol: str) -> dict:
    base_return = {
        'signal': 'none', 'sl_price': None, 'tp_price': None, 'error': None,
        'account_risk_percent': 0.005, 
        'all_conditions_status': {} 
    }
    s7_log_prefix = f"[S7 {symbol}]"

    min_klines_needed = 205 
    kl_df = klines(symbol)
    if kl_df is None or len(kl_df) < min_klines_needed:
        base_return['error'] = f"S7: Insufficient klines (need {min_klines_needed}, got {len(kl_df) if kl_df is not None else 0})"
        return base_return

    now_utc = pd.Timestamp.now(tz='UTC')
    if now_utc.hour == 23 or now_utc.hour == 0:
        base_return['error'] = "S7: Outside trading hours (23:00-01:00 UTC excluded)"
        base_return['all_conditions_status']['liquidity_filter_passed'] = False
        return base_return
    base_return['all_conditions_status']['liquidity_filter_passed'] = True

    try:
        ema200_series = ta.trend.EMAIndicator(close=kl_df['Close'], window=200).ema_indicator()
        if ema200_series is None or ema200_series.empty or ema200_series.isnull().all(): 
            base_return['error'] = "S7: EMA200 calculation resulted in an empty or all-NaN series."
            return base_return
        last_ema200 = ema200_series.iloc[-1]
        if pd.isna(last_ema200):
            base_return['error'] = "S7: EMA200 is NaN for the latest candle."
            return base_return
    except Exception as e:
        base_return['error'] = f"S7: Error calculating EMA200: {str(e)}"
        return base_return
    
    current_price = kl_df['Close'].iloc[-1]
    price_precision = get_price_precision(symbol)

    volume_spike_detected = check_volume_spike(kl_df, lookback_period=20, multiplier=2.0)
    base_return['all_conditions_status']['volume_spike'] = volume_spike_detected

    m_curr = get_candle_metrics(kl_df.iloc[-1])
    m_prev = get_candle_metrics(kl_df.iloc[-2]) if len(kl_df) >= 2 else None
    m_prev2 = get_candle_metrics(kl_df.iloc[-3]) if len(kl_df) >= 3 else None
    df_last5 = kl_df.iloc[-5:] if len(kl_df) >= 5 else None


    detected_pattern_name = "None"
    pattern_side = "none" 
    sl_ref_price = None 

    if m_prev2 and m_prev and is_morning_star(m_prev2, m_prev, m_curr): detected_pattern_name, pattern_side, sl_ref_price = "Morning Star", "up", min(m_prev2['l'], m_prev['l'], m_curr['l'])
    elif m_prev2 and m_prev and is_evening_star(m_prev2, m_prev, m_curr): detected_pattern_name, pattern_side, sl_ref_price = "Evening Star", "down", max(m_prev2['h'], m_prev['h'], m_curr['h'])
    elif m_prev and is_bullish_engulfing(m_prev, m_curr): detected_pattern_name, pattern_side, sl_ref_price = "Bullish Engulfing", "up", m_curr['l'] 
    elif m_prev and is_bearish_engulfing(m_prev, m_curr): detected_pattern_name, pattern_side, sl_ref_price = "Bearish Engulfing", "down", m_curr['h'] 
    elif is_hammer(m_curr): detected_pattern_name, pattern_side, sl_ref_price = "Hammer", "up", m_curr['l']
    elif is_hanging_man(m_curr): detected_pattern_name, pattern_side, sl_ref_price = "Hanging Man", "down", m_curr['h']
    elif is_inverted_hammer(m_curr): detected_pattern_name, pattern_side, sl_ref_price = "Inverted Hammer", "up", m_curr['l']
    elif is_shooting_star(m_curr): detected_pattern_name, pattern_side, sl_ref_price = "Shooting Star", "down", m_curr['h']
    elif m_prev and is_piercing_line(m_prev, m_curr): detected_pattern_name, pattern_side, sl_ref_price = "Piercing Line", "up", m_curr['l']
    elif m_prev and is_dark_cloud_cover(m_prev, m_curr): detected_pattern_name, pattern_side, sl_ref_price = "Dark Cloud Cover", "down", m_curr['h']
    elif m_prev2 and m_prev and is_three_white_soldiers(m_prev2, m_prev, m_curr): detected_pattern_name, pattern_side, sl_ref_price = "Three White Soldiers", "up", m_prev2['l']
    elif m_prev2 and m_prev and is_three_black_crows(m_prev2, m_prev, m_curr): detected_pattern_name, pattern_side, sl_ref_price = "Three Black Crows", "down", m_prev2['h']
    elif df_last5 is not None and is_rising_three_methods(df_last5): detected_pattern_name, pattern_side, sl_ref_price = "Rising Three Methods", "up", df_last5.iloc[0]['Low']
    elif df_last5 is not None and is_falling_three_methods(df_last5): detected_pattern_name, pattern_side, sl_ref_price = "Falling Three Methods", "down", df_last5.iloc[0]['High']

    base_return['all_conditions_status']['detected_pattern'] = detected_pattern_name
    base_return['all_conditions_status']['current_price_for_ema_check'] = f"{current_price:.{price_precision}f}"
    base_return['all_conditions_status']['last_ema200'] = f"{last_ema200:.{price_precision}f}" if not pd.isna(last_ema200) else "N/A"

    if pattern_side != "none":
        ema_filter_passed = (pattern_side == "up" and current_price > last_ema200) or \
                            (pattern_side == "down" and current_price < last_ema200)
        base_return['all_conditions_status']['ema_filter_passed'] = ema_filter_passed
        
        if volume_spike_detected and ema_filter_passed:
            base_return['signal'] = pattern_side
            entry_price = current_price
            sl_calculated, tp_calculated = None, None
            
            atr_val_buffer = 0.0
            try:
                atr_series_buffer = ta.volatility.AverageTrueRange(high=kl_df['High'], low=kl_df['Low'], close=kl_df['Close'], window=14).average_true_range()
                if atr_series_buffer is not None and not atr_series_buffer.empty and not pd.isna(atr_series_buffer.iloc[-1]):
                    atr_val_buffer = atr_series_buffer.iloc[-1] * 0.1 
            except Exception as e_atr_buff: print(f"{s7_log_prefix} ATR buffer calc error: {e_atr_buff}")

            if sl_ref_price is not None:
                if pattern_side == "up":
                    sl_calculated = round(sl_ref_price - atr_val_buffer, price_precision)
                    if sl_calculated >= entry_price: sl_calculated = round(entry_price * (1 - 0.003), price_precision) 
                    if sl_calculated > 0 and entry_price > sl_calculated: 
                         tp_calculated = round(entry_price + (entry_price - sl_calculated) * 1.5, price_precision)
                elif pattern_side == "down":
                    sl_calculated = round(sl_ref_price + atr_val_buffer, price_precision)
                    if sl_calculated <= entry_price: sl_calculated = round(entry_price * (1 + 0.003), price_precision) 
                    if sl_calculated > 0 and entry_price < sl_calculated :
                        tp_calculated = round(entry_price - (sl_calculated - entry_price) * 1.5, price_precision)
            
            if sl_calculated and tp_calculated and sl_calculated > 0 and tp_calculated > 0:
                base_return['sl_price'] = sl_calculated
                base_return['tp_price'] = tp_calculated
            else:
                base_return['signal'] = 'none'
                base_return['error'] = f"S7: SL/TP calc failed for {detected_pattern_name} SL_ref={sl_ref_price}, SL={sl_calculated}, TP={tp_calculated}"
        else: 
            if base_return['signal'] == 'none' and detected_pattern_name != "None": 
                error_details = []
                if not volume_spike_detected: error_details.append("No volume spike")
                if not ema_filter_passed: error_details.append("EMA filter failed")
                if error_details: 
                    base_return['error'] = f"S7: {detected_pattern_name} filters failed: {', '.join(error_details)}"
            
    if base_return['signal'] == 'none' and detected_pattern_name == "None" and not base_return['error'] :
         base_return['error'] = "S7: No specific pattern detected."
    
    if base_return['error'] and base_return['error'] not in ["S7: No specific pattern detected.", f"S7: {detected_pattern_name} filters failed: No volume spike, EMA filter failed", f"S7: {detected_pattern_name} filters failed: No volume spike", f"S7: {detected_pattern_name} filters failed: EMA filter failed"]: 
        print(f"{s7_log_prefix} {base_return['error']}")
    elif base_return['signal'] != 'none':
        print(f"{s7_log_prefix} Signal: {base_return['signal']} for {detected_pattern_name}. SL={base_return['sl_price']}, TP={base_return['tp_price']}")

    return base_return

# Backtest Strategy Wrapper Class
class BacktestStrategyWrapper(Strategy):
    # Default parameters, will be overridden by UI inputs later
    user_tp = 0.03  # Default, will be set from UI
    user_sl = 0.02  # Default, will be set from UI
    
    # Strategy-specific parameters (example for the provided strategy)
    ema_period = 200
    rsi_period = 14
    ATR_PERIOD = 14 # New class variable for ATR period
    ST_ATR_PERIOD_S0 = 10 # Strategy 0 Supertrend ATR Period
    ST_MULTIPLIER_S0 = 1.5  # Strategy 0 Supertrend Multiplier
    
    # Strategy S1 ("EMA Cross + SuperTrend") Parameters
    EMA_SHORT_S1 = 9
    EMA_LONG_S1 = 21
    RSI_PERIOD_S1 = 14
    ST_ATR_PERIOD_S1 = 10 
    ST_MULTIPLIER_S1 = 3.0
    
    current_strategy_id = 5 # Defaulting to 5 for "New RSI-Based Strategy"
    RR = 1.5  # Risk/Reward ratio
    SL_ATR_MULTI = DEFAULT_ATR_MULTIPLIER # Multiplier for ATR to determine SL distance
    PRICE_PRECISION_BT = 4 # Default rounding precision for backtesting

    # Attributes for new SL/TP modes, to be set by execute_backtest
    sl_tp_mode_bt = "ATR/Dynamic"  # Renamed to avoid conflict with global SL_TP_MODE
    sl_pnl_amount_bt = 0.0
    tp_pnl_amount_bt = 0.0
    # user_sl and user_tp (percentages) are already class attributes
    leverage = 1.0 # Default leverage for backtesting


    def init(self):
        # Use self.sl_tp_mode_bt (or whatever name is chosen for the class attribute)
        print(f"DEBUG BacktestStrategyWrapper.init: Initializing for strategy ID {self.current_strategy_id}, SL/TP Mode: {getattr(self, 'sl_tp_mode_bt', 'N/A')}, Leverage: {self.leverage}")
        # --- Jules's Logging ---
        print(f"[BT Strategy LOG for {self.data.symbol if hasattr(self.data, 'symbol') else 'N/A'}] init() called.")
        print(f"[BT Strategy LOG for {self.data.symbol if hasattr(self.data, 'symbol') else 'N/A'}] Data received by strategy: self.data.df shape: {self.data.df.shape}")
        print(f"[BT Strategy LOG for {self.data.symbol if hasattr(self.data, 'symbol') else 'N/A'}] Data head:\n{self.data.df.head()}")
        print(f"[BT Strategy LOG for {self.data.symbol if hasattr(self.data, 'symbol') else 'N/A'}] Data tail:\n{self.data.df.tail()}")
        print(f"[BT Strategy LOG for {self.data.symbol if hasattr(self.data, 'symbol') else 'N/A'}] Data NaN sum:\n{self.data.df.isnull().sum()}")
        # --- End Jules's Logging ---
        if self.current_strategy_id == 5:
            self.rsi_period_val = getattr(self, 'rsi_period', 14) 
            self.rsi = self.I(rsi_bt, self.data.Close, self.rsi_period_val, name='RSI_S5')
            # ATR is always initialized as it's needed for ATR/Dynamic mode by any strategy
            self.atr = self.I(atr_bt, self.data.High, self.data.Low, self.data.Close, self.ATR_PERIOD, name='ATR_dynSLTP_S5')
        elif self.current_strategy_id == 1:
            print(f"DEBUG BacktestStrategyWrapper.init: Initializing indicators for Strategy ID 1 (EMA Cross + SuperTrend)")
            self.ema_short_s1 = self.I(ema_bt, self.data.Close, self.EMA_SHORT_S1, name='EMA_S_S1')
            self.ema_long_s1 = self.I(ema_bt, self.data.Close, self.EMA_LONG_S1, name='EMA_L_S1')
            self.rsi_s1 = self.I(rsi_bt, self.data.Close, self.RSI_PERIOD_S1, name='RSI_S1')
            self.st_s1 = self.I(supertrend_numerical_bt, self.data.High, self.data.Low, self.data.Close, self.ST_ATR_PERIOD_S1, self.ST_MULTIPLIER_S1, name='ST_S1', overlay=True)
            self.atr = self.I(atr_bt, self.data.High, self.data.Low, self.data.Close, self.ATR_PERIOD, name='ATR_dynSLTP_S1')
        elif self.current_strategy_id == 0:
            print(f"DEBUG BacktestStrategyWrapper.init: Initializing indicators for Strategy ID 0 (Original Scalping)")
            self.ema9_s0 = self.I(ema_bt, self.data.Close, 9, name='EMA9_S0')
            self.ema21_s0 = self.I(ema_bt, self.data.Close, 21, name='EMA21_S0')
            self.rsi_s0 = self.I(rsi_bt, self.data.Close, 14, name='RSI_S0')
            self.volume_ma10_s0 = self.I(lambda series, window: pd.Series(series).rolling(window).mean(), self.data.Volume, 10, name='VolumeMA10_S0', overlay=False) 
            self.st_s0 = self.I(supertrend_numerical_bt, self.data.High, self.data.Low, self.data.Close, self.ST_ATR_PERIOD_S0, self.ST_MULTIPLIER_S0, name='Supertrend_S0', overlay=True) 
            self.atr = self.I(atr_bt, self.data.High, self.data.Low, self.data.Close, self.ATR_PERIOD, name='ATR_dynSLTP_S0')
        elif self.current_strategy_id == 6: # Market Structure S/D Strategy
            print(f"DEBUG BacktestStrategyWrapper.init: Initializing for Strategy ID 6 (Market Structure S/D)")
            # For backtesting, pre-calculate swing points and S/D zones on the entire dataset once.
            # Market structure itself will be evaluated dynamically in next() or based on these points.
            try:
                self.bt_swing_highs_bool, self.bt_swing_lows_bool = find_swing_points(self.data.df, order=5)
                # S/D zones also depend on the whole dataset for accurate historical identification
                self.bt_sd_zones = identify_supply_demand_zones(self.data.df, atr_period=14, lookback_candles=10) 
                print(f"DEBUG BT S6: Pre-calculated {len(self.bt_sd_zones)} S/D zones.")
                # ATR is useful for some dynamic calculations or fallbacks, ensure it's available.
                self.atr = self.I(atr_bt, self.data.High, self.data.Low, self.data.Close, self.ATR_PERIOD, name='ATR_S6')
            except Exception as e:
                print(f"ERROR BT S6 init: Failed to pre-calculate swings or S/D zones: {e}")
                # Potentially raise this or handle it to prevent backtest from running with faulty setup
                self.bt_swing_highs_bool = pd.Series([False]*len(self.data.df), index=self.data.df.index)
                self.bt_swing_lows_bool = pd.Series([False]*len(self.data.df), index=self.data.df.index)
                self.bt_sd_zones = []
                self.atr = None # Invalidate ATR if setup fails critically
        elif self.current_strategy_id == 7: # Candlestick Patterns Strategy
            print(f"DEBUG BacktestStrategyWrapper.init: Initializing for Strategy ID 7 (Candlestick Patterns)")
            # Required indicators for S7's filters and logic
            self.ema200_s7 = self.I(ema_bt, self.data.Close, 200, name='EMA200_S7')
            self.atr_s7 = self.I(atr_bt, self.data.High, self.data.Low, self.data.Close, 14, name='ATR_S7') # ATR for general use or SL/TP if ATR/Dynamic mode
            # Candlestick pattern recognition will be done in next() using helper functions
            # on self.data.df slices. No specific self.I needed for the patterns themselves here.
            # Volume data is self.data.Volume
        else: # Default for any other strategy ID, ensure ATR is available
            print(f"DEBUG BacktestStrategyWrapper.init: Strategy ID {self.current_strategy_id} - Defaulting to RSI and ATR.")
            self.rsi_period_val = getattr(self, 'rsi_period', 14)
            self.rsi = self.I(rsi_bt, self.data.Close, self.rsi_period_val, name=f'RSI_default_{self.current_strategy_id}')
            self.atr = self.I(atr_bt, self.data.High, self.data.Low, self.data.Close, self.ATR_PERIOD, name=f'ATR_default_{self.current_strategy_id}')
            print(f"DEBUG BacktestStrategyWrapper.init: Defaulted ATR indicator for strategy ID {self.current_strategy_id}.")

    def next(self):
        # print(f"DEBUG BacktestStrategyWrapper.next: Executing for strategy ID {self.current_strategy_id}, Mode: {self.sl_tp_mode_bt}") # DEBUG
        price = float(self.data.Close[-1]) 
        trade_symbol = self.data.symbol if hasattr(self.data, 'symbol') else 'N/A'
        log_prefix_bt_next = f"[BT Strategy LOG {trade_symbol} - Bar {len(self.data.Close)-1} - Price {price:.4f}]" # Adjusted precision
        
        # --- SL/TP Price Calculation Block (Common for all strategies) ---
        sl_final_price = None
        tp_final_price = None
        sl_long, tp_long, sl_short, tp_short = None, None, None, None # Initialize here
        entry_price_for_calc = price # Current price for SL/TP calculation
        
        # For PnL calculation, estimate asset quantity based on equity fraction
        # This assumes self.buy/sell size parameter (e.g., 0.02) is a fraction of equity.
        # backtesting.py's `size` param in `self.buy` can be absolute units or fraction of equity.
        # If it's fraction of equity (e.g. size=0.1 for 10% of equity), this calc is okay.
        # If `size` is absolute units, then asset_qty_for_pnl_calc should just be that `size`.
        # Assuming fractional equity for now as per typical backtesting.py examples.
        trade_size_fraction_bt = 0.02 # Default/Example size, should match what strategies use
        asset_qty_for_pnl_calc = (self.equity * trade_size_fraction_bt) / entry_price_for_calc if entry_price_for_calc > 0 else 0

        if self.sl_tp_mode_bt == "Percentage":
            # Validate user_sl and user_tp to be between 0 (exclusive) and 1 (exclusive for SL, less critical for TP but good practice)
            if 0 < self.user_sl < 1 and 0 < self.user_tp: 
                # For a potential long trade
                _sl_long_raw = entry_price_for_calc * (1 - self.user_sl)
                _tp_long_raw = entry_price_for_calc * (1 + self.user_tp)
                # For a potential short trade
                _sl_short_raw = entry_price_for_calc * (1 + self.user_sl)
                _tp_short_raw = entry_price_for_calc * (1 - self.user_tp)

                # Ensure SL/TP are positive before rounding
                if _sl_long_raw > 0: sl_long = round(_sl_long_raw, self.PRICE_PRECISION_BT)
                if _tp_long_raw > 0: tp_long = round(_tp_long_raw, self.PRICE_PRECISION_BT)
                if _sl_short_raw > 0: sl_short = round(_sl_short_raw, self.PRICE_PRECISION_BT)
                if _tp_short_raw > 0: tp_short = round(_tp_short_raw, self.PRICE_PRECISION_BT)
            else:
                print(f"DEBUG BT Percentage: Invalid user_sl ({self.user_sl}) or user_tp ({self.user_tp}). Must be > 0 (and SL < 1).")
                # sl_long, tp_long, etc., remain None

        elif self.sl_tp_mode_bt == "Fixed PnL":
            if self.sl_pnl_amount_bt > 0 and self.tp_pnl_amount_bt > 0 and asset_qty_for_pnl_calc > 0:
                # For a potential long trade
                _sl_long_raw = entry_price_for_calc - (self.sl_pnl_amount_bt / asset_qty_for_pnl_calc)
                _tp_long_raw = entry_price_for_calc + (self.tp_pnl_amount_bt / asset_qty_for_pnl_calc)
                # For a potential short trade
                _sl_short_raw = entry_price_for_calc + (self.sl_pnl_amount_bt / asset_qty_for_pnl_calc)
                _tp_short_raw = entry_price_for_calc - (self.tp_pnl_amount_bt / asset_qty_for_pnl_calc)

                if _sl_long_raw > 0: sl_long = round(_sl_long_raw, self.PRICE_PRECISION_BT)
                if _tp_long_raw > 0: tp_long = round(_tp_long_raw, self.PRICE_PRECISION_BT)
                if _sl_short_raw > 0: sl_short = round(_sl_short_raw, self.PRICE_PRECISION_BT)
                if _tp_short_raw > 0: tp_short = round(_tp_short_raw, self.PRICE_PRECISION_BT)
            else:
                print(f"DEBUG BT Fixed PnL: Cannot calculate. Amounts positive? ({self.sl_pnl_amount_bt > 0}, {self.tp_pnl_amount_bt > 0}), AssetQty positive? ({asset_qty_for_pnl_calc > 0})")
                # sl_long, tp_long, etc., remain None
        
        # If ATR/Dynamic, sl_final_price and tp_final_price remain None; strategy logic will calculate them.
        # --- End of Common SL/TP Price Calculation Block ---

        # Actual trade size to be used by self.buy/sell
        # This should be consistent with how strategies determine trade size.
        # For now, using the example fractional size.
        trade_size_to_use_in_order = trade_size_fraction_bt

        # --- Strategy Specific Logic ---
        if self.current_strategy_id == 0: # Original Scalping Strategy
            if self.position:
                # print(f"{log_prefix_bt_next} S0: Already in position. Skipping.") # Basic log, can be expanded
                return

            # Indicator calculations
            # Ensure enough data for all indicators and lookbacks
            min_len_s0 = max(21, LOCAL_HIGH_LOW_LOOKBACK_PERIOD + 2, 10) # Max of EMA21, lookback+prev, VolumeMA10
            if len(self.data.Close) < min_len_s0:
                # print(f"{log_prefix_bt_next} S0: Insufficient data length ({len(self.data.Close)} < {min_len_s0}). Skipping.")
                return

            last_ema9 = self.ema9_s0[-1]; prev_ema9 = self.ema9_s0[-2]
            last_ema21 = self.ema21_s0[-1]; prev_ema21 = self.ema21_s0[-2]
            last_rsi = self.rsi_s0[-1]; last_st = self.st_s0[-1]
            last_vol = self.data.Volume[-1]; last_vol_ma10 = self.volume_ma10_s0[-1]
            
            lookback = LOCAL_HIGH_LOW_LOOKBACK_PERIOD
            recent_high = self.data.High[-lookback-1:-1].max(); recent_low = self.data.Low[-lookback-1:-1].min()

            # Log indicator values
            # print(f"{log_prefix_bt_next} S0 Indicators: EMA9={last_ema9:.2f}, EMA21={last_ema21:.2f}, RSI={last_rsi:.2f}, ST={last_st}, VOL={last_vol}, VOL_MA10={last_vol_ma10:.2f}, RecentHigh={recent_high:.2f}, RecentLow={recent_low:.2f}")

            # Conditions
            ema_up = prev_ema9 < prev_ema21 and last_ema9 > last_ema21
            rsi_long = 50 <= last_rsi <= 70; st_green = (last_st == 1)
            vol_strong = last_vol > last_vol_ma10 if not pd.isna(last_vol_ma10) and last_vol_ma10 != 0 else False
            price_break_high = price > recent_high
            buy_conds_met_list = [ema_up, rsi_long, st_green, vol_strong, price_break_high]
            num_buy = sum(buy_conds_met_list)
            # print(f"{log_prefix_bt_next} S0 Buy Conditions: EMA_UP={ema_up}, RSI_LONG={rsi_long}, ST_GREEN={st_green}, VOL_STRONG={vol_strong}, PRICE_BREAK_H={price_break_high} -> Met: {num_buy}")


            ema_down = prev_ema9 > prev_ema21 and last_ema9 < last_ema21
            rsi_short = 30 <= last_rsi <= 50; st_red = (last_st == -1) # ST is -1 for red
            price_break_low = price < recent_low
            sell_conds_met_list = [ema_down, rsi_short, st_red, vol_strong, price_break_low]
            num_sell = sum(sell_conds_met_list)
            # print(f"{log_prefix_bt_next} S0 Sell Conditions: EMA_DOWN={ema_down}, RSI_SHORT={rsi_short}, ST_RED={st_red}, VOL_STRONG={vol_strong}, PRICE_BREAK_L={price_break_low} -> Met: {num_sell}")

            trade_side_s0 = None
            if num_buy >= SCALPING_REQUIRED_BUY_CONDITIONS: trade_side_s0 = 'buy'
            elif num_sell >= SCALPING_REQUIRED_SELL_CONDITIONS: trade_side_s0 = 'sell'

            if trade_side_s0:
                entry_price_s0 = price
                sl_price, tp_price = None, None # Initialize
                # print(f"{log_prefix_bt_next} S0 Signal: {trade_side_s0.upper()}. Entry: {entry_price_s0:.2f}, SL/TP Mode: {self.sl_tp_mode_bt}")
                if self.sl_tp_mode_bt == "ATR/Dynamic":
                    if self.atr is None or len(self.atr) < 1 or pd.isna(self.atr[-1]) or self.atr[-1] == 0:
                        # print(f"{log_prefix_bt_next} S0: ATR invalid for dynamic SL/TP (ATR: {self.atr[-1] if self.atr and len(self.atr)>0 else 'N/A'}). Skipping trade.")
                        return 
                    current_atr_s0 = self.atr[-1]
                    if trade_side_s0 == 'buy':
                        sl_price = round(entry_price_s0 - (current_atr_s0 * self.SL_ATR_MULTI), self.PRICE_PRECISION_BT)
                        tp_price = round(entry_price_s0 + ((entry_price_s0 - sl_price) * self.RR), self.PRICE_PRECISION_BT)
                        if sl_price >= entry_price_s0 or tp_price <= entry_price_s0: 
                            # print(f"{log_prefix_bt_next} S0 ATR Buy: Invalid SL/TP. SL={sl_price}, TP={tp_price}. Skipping."); 
                            return
                    else: # sell
                        sl_price = round(entry_price_s0 + (current_atr_s0 * self.SL_ATR_MULTI), self.PRICE_PRECISION_BT)
                        tp_price = round(entry_price_s0 - ((sl_price - entry_price_s0) * self.RR), self.PRICE_PRECISION_BT)
                        if sl_price <= entry_price_s0 or tp_price >= entry_price_s0: 
                            # print(f"{log_prefix_bt_next} S0 ATR Sell: Invalid SL/TP. SL={sl_price}, TP={tp_price}. Skipping."); 
                            return
                elif self.sl_tp_mode_bt == "Percentage":
                    sl_price = sl_long if trade_side_s0 == 'buy' else sl_short
                    tp_price = tp_long if trade_side_s0 == 'buy' else tp_short
                    if sl_price is None or tp_price is None or sl_price <=0 or tp_price <=0: 
                        # print(f"{log_prefix_bt_next} S0 Percentage: SL/TP calculation failed or invalid. SL={sl_price}, TP={tp_price}. Skipping."); 
                        return
                    if trade_side_s0 == 'buy' and (sl_price >= entry_price_s0 or tp_price <= entry_price_s0): 
                        # print(f"{log_prefix_bt_next} S0 Percentage Buy: Invalid SL/TP. SL={sl_price}, TP={tp_price}. Skipping."); 
                        return
                    if trade_side_s0 == 'sell' and (sl_price <= entry_price_s0 or tp_price >= entry_price_s0): 
                        # print(f"{log_prefix_bt_next} S0 Percentage Sell: Invalid SL/TP. SL={sl_price}, TP={tp_price}. Skipping."); 
                        return
                elif self.sl_tp_mode_bt == "Fixed PnL":
                    sl_price = sl_long if trade_side_s0 == 'buy' else sl_short
                    tp_price = tp_long if trade_side_s0 == 'buy' else tp_short
                    if sl_price is None or tp_price is None or sl_price <=0 or tp_price <=0: 
                        # print(f"{log_prefix_bt_next} S0 Fixed PnL: SL/TP calculation failed or invalid. SL={sl_price}, TP={tp_price}. Skipping."); 
                        return
                    if trade_side_s0 == 'buy' and (sl_price >= entry_price_s0 or tp_price <= entry_price_s0): 
                        # print(f"{log_prefix_bt_next} S0 Fixed PnL Buy: Invalid SL/TP. SL={sl_price}, TP={tp_price}. Skipping."); 
                        return
                    if trade_side_s0 == 'sell' and (sl_price <= entry_price_s0 or tp_price >= entry_price_s0): 
                        # print(f"{log_prefix_bt_next} S0 Fixed PnL Sell: Invalid SL/TP. SL={sl_price}, TP={tp_price}. Skipping."); 
                        return
                else: 
                    # print(f"{log_prefix_bt_next} S0: Unknown SL/TP mode '{self.sl_tp_mode_bt}'. Skipping."); 
                    return

                # print(f"{log_prefix_bt_next} S0 Placing Trade: Side={trade_side_s0}, Entry={entry_price_s0:.2f}, SL={sl_price:.2f}, TP={tp_price:.2f}, Size={trade_size_to_use_in_order}")
                if trade_side_s0 == 'buy': self.buy(sl=sl_price, tp=tp_price, size=trade_size_to_use_in_order)
                else: self.sell(sl=sl_price, tp=tp_price, size=trade_size_to_use_in_order)
            # else:
                # print(f"{log_prefix_bt_next} S0: No trade signal this bar.")


        elif self.current_strategy_id == 1: # EMA Cross + SuperTrend
            if self.position: 
                # print(f"{log_prefix_bt_next} S1: Already in position. Skipping."); 
                return
            if len(self.ema_short_s1) < 2 or len(self.ema_long_s1) < 2 or len(self.rsi_s1) < 1 or len(self.st_s1) < 1: 
                # print(f"{log_prefix_bt_next} S1: Insufficient indicator length. EMA_S={len(self.ema_short_s1)}, EMA_L={len(self.ema_long_s1)}, RSI={len(self.rsi_s1)}, ST={len(self.st_s1)}. Skipping."); 
                return

            last_ema_short = self.ema_short_s1[-1]; prev_ema_short = self.ema_short_s1[-2]
            last_ema_long = self.ema_long_s1[-1]; prev_ema_long = self.ema_long_s1[-2]
            last_rsi_s1 = self.rsi_s1[-1]; last_st_s1 = self.st_s1[-1] # ST is numerical: 1 green, -1 red

            # print(f"{log_prefix_bt_next} S1 Indicators: EMA_S={last_ema_short:.2f} (Prev:{prev_ema_short:.2f}), EMA_L={last_ema_long:.2f} (Prev:{prev_ema_long:.2f}), RSI={last_rsi_s1:.2f}, ST={last_st_s1}")

            ema_crossed_up_s1 = prev_ema_short < prev_ema_long and last_ema_short > last_ema_long
            st_green_s1 = (last_st_s1 == 1); rsi_long_ok_s1 = 40 <= last_rsi_s1 <= 70
            buy_conds_s1_met_list = [ema_crossed_up_s1, st_green_s1, rsi_long_ok_s1]
            num_buy_s1 = sum(buy_conds_s1_met_list)
            # print(f"{log_prefix_bt_next} S1 Buy Conditions: EMA_X_UP={ema_crossed_up_s1}, ST_GREEN={st_green_s1}, RSI_OK={rsi_long_ok_s1} -> Met: {num_buy_s1}")


            ema_crossed_down_s1 = prev_ema_short > prev_ema_long and last_ema_short < last_ema_long
            st_red_s1 = (last_st_s1 == -1); rsi_short_ok_s1 = 30 <= last_rsi_s1 <= 60
            sell_conds_s1_met_list = [ema_crossed_down_s1, st_red_s1, rsi_short_ok_s1]
            num_sell_s1 = sum(sell_conds_s1_met_list)
            # print(f"{log_prefix_bt_next} S1 Sell Conditions: EMA_X_DOWN={ema_crossed_down_s1}, ST_RED={st_red_s1}, RSI_OK={rsi_short_ok_s1} -> Met: {num_sell_s1}")
            
            REQUIRED_S1 = 2; trade_side_s1 = None
            if num_buy_s1 >= REQUIRED_S1: trade_side_s1 = 'buy'
            elif num_sell_s1 >= REQUIRED_S1: trade_side_s1 = 'sell'

            if trade_side_s1:
                entry_price_s1 = price
                sl_to_use_s1, tp_to_use_s1 = None, None
                # print(f"{log_prefix_bt_next} S1 Signal: {trade_side_s1.upper()}. Entry: {entry_price_s1:.2f}, SL/TP Mode: {self.sl_tp_mode_bt}")
                if self.sl_tp_mode_bt == "ATR/Dynamic":
                    if self.atr is None or len(self.atr) < 1 or pd.isna(self.atr[-1]) or self.atr[-1] == 0:
                        # print(f"{log_prefix_bt_next} S1: ATR invalid for dynamic SL/TP (ATR: {self.atr[-1] if self.atr and len(self.atr)>0 else 'N/A'}). Skipping trade.")
                        return
                    current_atr_s1 = self.atr[-1]
                    if trade_side_s1 == 'buy':
                        sl_to_use_s1 = round(entry_price_s1 - (current_atr_s1 * self.SL_ATR_MULTI), self.PRICE_PRECISION_BT)
                        tp_to_use_s1 = round(entry_price_s1 + ((entry_price_s1 - sl_to_use_s1) * self.RR), self.PRICE_PRECISION_BT)
                        if sl_to_use_s1 >= entry_price_s1 or tp_to_use_s1 <= entry_price_s1: 
                            # print(f"{log_prefix_bt_next} S1 ATR Buy: Invalid SL/TP. SL={sl_to_use_s1}, TP={tp_to_use_s1}. Skipping."); 
                            return
                    else: # sell
                        sl_to_use_s1 = round(entry_price_s1 + (current_atr_s1 * self.SL_ATR_MULTI), self.PRICE_PRECISION_BT)
                        tp_to_use_s1 = round(entry_price_s1 - ((sl_to_use_s1 - entry_price_s1) * self.RR), self.PRICE_PRECISION_BT)
                        if sl_to_use_s1 <= entry_price_s1 or tp_to_use_s1 >= entry_price_s1: 
                            # print(f"{log_prefix_bt_next} S1 ATR Sell: Invalid SL/TP. SL={sl_to_use_s1}, TP={tp_to_use_s1}. Skipping."); 
                            return
                elif self.sl_tp_mode_bt == "Percentage":
                    sl_to_use_s1 = sl_long if trade_side_s1 == 'buy' else sl_short
                    tp_to_use_s1 = tp_long if trade_side_s1 == 'buy' else tp_short
                    if sl_to_use_s1 is None or tp_to_use_s1 is None or sl_to_use_s1 <= 0 or tp_to_use_s1 <= 0: 
                        # print(f"{log_prefix_bt_next} S1 Percentage: SL/TP calc failed. SL={sl_to_use_s1}, TP={tp_to_use_s1}. Skipping."); 
                        return
                    if trade_side_s1 == 'buy' and (sl_to_use_s1 >= entry_price_s1 or tp_to_use_s1 <= entry_price_s1): 
                        # print(f"{log_prefix_bt_next} S1 Percentage Buy: Invalid SL/TP. SL={sl_to_use_s1}, TP={tp_to_use_s1}. Skipping."); 
                        return
                    if trade_side_s1 == 'sell' and (sl_to_use_s1 <= entry_price_s1 or tp_to_use_s1 >= entry_price_s1): 
                        # print(f"{log_prefix_bt_next} S1 Percentage Sell: Invalid SL/TP. SL={sl_to_use_s1}, TP={tp_to_use_s1}. Skipping."); 
                        return
                elif self.sl_tp_mode_bt == "Fixed PnL":
                    sl_to_use_s1 = sl_long if trade_side_s1 == 'buy' else sl_short
                    tp_to_use_s1 = tp_long if trade_side_s1 == 'buy' else tp_short
                    if sl_to_use_s1 is None or tp_to_use_s1 is None or sl_to_use_s1 <= 0 or tp_to_use_s1 <= 0: 
                        # print(f"{log_prefix_bt_next} S1 Fixed PnL: SL/TP calc failed. SL={sl_to_use_s1}, TP={tp_to_use_s1}. Skipping."); 
                        return
                    if trade_side_s1 == 'buy' and (sl_to_use_s1 >= entry_price_s1 or tp_to_use_s1 <= entry_price_s1): 
                        # print(f"{log_prefix_bt_next} S1 Fixed PnL Buy: Invalid SL/TP. SL={sl_to_use_s1}, TP={tp_to_use_s1}. Skipping."); 
                        return
                    if trade_side_s1 == 'sell' and (sl_to_use_s1 <= entry_price_s1 or tp_to_use_s1 >= entry_price_s1): 
                        # print(f"{log_prefix_bt_next} S1 Fixed PnL Sell: Invalid SL/TP. SL={sl_to_use_s1}, TP={tp_to_use_s1}. Skipping."); 
                        return
                
                if sl_to_use_s1 is None or tp_to_use_s1 is None: 
                    # print(f"{log_prefix_bt_next} S1: Final SL/TP check failed. SL={sl_to_use_s1}, TP={tp_to_use_s1}. Skipping."); 
                    return

                # print(f"{log_prefix_bt_next} S1 Placing Trade: Side={trade_side_s1}, Entry={entry_price_s1:.2f}, SL={sl_to_use_s1:.2f}, TP={tp_to_use_s1:.2f}, Size={trade_size_to_use_in_order}")
                if trade_side_s1 == 'buy': self.buy(sl=sl_to_use_s1, tp=tp_to_use_s1, size=trade_size_to_use_in_order)
                else: self.sell(sl=sl_to_use_s1, tp=tp_to_use_s1, size=trade_size_to_use_in_order)
            # else:
                # print(f"{log_prefix_bt_next} S1: No trade signal this bar.")

        elif self.current_strategy_id == 5: # New RSI-Based Strategy
            if self.position: 
                # print(f"{log_prefix_bt_next} S5: Already in position. Skipping."); 
                return

            # S5 uses self.rsi and self.atr (ATR for dynamic SL/TP)
            if len(self.rsi) < 1: 
                # print(f"{log_prefix_bt_next} S5: Insufficient RSI length ({len(self.rsi)}). Skipping."); 
                return
            if self.sl_tp_mode_bt == "ATR/Dynamic" and (self.atr is None or len(self.atr) < 1):
                # print(f"{log_prefix_bt_next} S5: ATR/Dynamic mode but ATR not available or too short ({len(self.atr) if self.atr else 'None'}). Skipping."); 
                return

            # print(f"{log_prefix_bt_next} S5 Indicators: RSI={self.rsi[-1]:.2f}" + (f", ATR={self.atr[-1]:.4f}" if self.sl_tp_mode_bt == "ATR/Dynamic" and self.atr and len(self.atr)>0 else ""))

            take_long_s5 = self.rsi[-1] < 30
            take_short_s5 = self.rsi[-1] > 70
            # print(f"{log_prefix_bt_next} S5 Conditions: RSI<30={take_long_s5}, RSI>70={take_short_s5}")

            trade_side_s5 = None
            if take_long_s5: trade_side_s5 = 'buy'
            elif take_short_s5: trade_side_s5 = 'sell'

            if trade_side_s5:
                entry_price_s5 = price
                sl_to_use_s5, tp_to_use_s5 = None, None
                # print(f"{log_prefix_bt_next} S5 Signal: {trade_side_s5.upper()}. Entry: {entry_price_s5:.2f}, SL/TP Mode: {self.sl_tp_mode_bt}")

                if self.sl_tp_mode_bt == "ATR/Dynamic":
                    if pd.isna(self.atr[-1]) or self.atr[-1] == 0: 
                        # print(f"{log_prefix_bt_next} S5: ATR invalid for dynamic SL/TP (ATR: {self.atr[-1]}). Skipping."); 
                        return
                    current_atr_s5 = self.atr[-1]
                    if trade_side_s5 == 'buy':
                        sl_to_use_s5 = round(entry_price_s5 - (current_atr_s5 * self.SL_ATR_MULTI), self.PRICE_PRECISION_BT)
                        tp_to_use_s5 = round(entry_price_s5 + ((entry_price_s5 - sl_to_use_s5) * self.RR), self.PRICE_PRECISION_BT)
                        if sl_to_use_s5 >= entry_price_s5 or tp_to_use_s5 <= entry_price_s5: 
                            # print(f"{log_prefix_bt_next} S5 ATR Buy: Invalid SL/TP. SL={sl_to_use_s5}, TP={tp_to_use_s5}. Skipping."); 
                            return
                    else: # sell
                        sl_to_use_s5 = round(entry_price_s5 + (current_atr_s5 * self.SL_ATR_MULTI), self.PRICE_PRECISION_BT)
                        tp_to_use_s5 = round(entry_price_s5 - ((sl_to_use_s5 - entry_price_s5) * self.RR), self.PRICE_PRECISION_BT)
                        if sl_to_use_s5 <= entry_price_s5 or tp_to_use_s5 >= entry_price_s5: 
                            # print(f"{log_prefix_bt_next} S5 ATR Sell: Invalid SL/TP. SL={sl_to_use_s5}, TP={tp_to_use_s5}. Skipping."); 
                            return
                elif self.sl_tp_mode_bt == "Percentage":
                    sl_to_use_s5 = sl_long if trade_side_s5 == 'buy' else sl_short
                    tp_to_use_s5 = tp_long if trade_side_s5 == 'buy' else tp_short
                    if sl_to_use_s5 is None or tp_to_use_s5 is None or sl_to_use_s5 <= 0 or tp_to_use_s5 <= 0: 
                        # print(f"{log_prefix_bt_next} S5 Percentage: SL/TP calc failed. SL={sl_to_use_s5}, TP={tp_to_use_s5}. Skipping."); 
                        return
                    if trade_side_s5 == 'buy' and (sl_to_use_s5 >= entry_price_s5 or tp_to_use_s5 <= entry_price_s5): 
                        # print(f"{log_prefix_bt_next} S5 Percentage Buy: Invalid SL/TP. SL={sl_to_use_s5}, TP={tp_to_use_s5}. Skipping."); 
                        return
                    if trade_side_s5 == 'sell' and (sl_to_use_s5 <= entry_price_s5 or tp_to_use_s5 >= entry_price_s5): 
                        # print(f"{log_prefix_bt_next} S5 Percentage Sell: Invalid SL/TP. SL={sl_to_use_s5}, TP={tp_to_use_s5}. Skipping."); 
                        return
                elif self.sl_tp_mode_bt == "Fixed PnL":
                    sl_to_use_s5 = sl_long if trade_side_s5 == 'buy' else sl_short
                    tp_to_use_s5 = tp_long if trade_side_s5 == 'buy' else tp_short
                    if sl_to_use_s5 is None or tp_to_use_s5 is None or sl_to_use_s5 <= 0 or tp_to_use_s5 <= 0: 
                        # print(f"{log_prefix_bt_next} S5 Fixed PnL: SL/TP calc failed. SL={sl_to_use_s5}, TP={tp_to_use_s5}. Skipping."); 
                        return
                    if trade_side_s5 == 'buy' and (sl_to_use_s5 >= entry_price_s5 or tp_to_use_s5 <= entry_price_s5): 
                        # print(f"{log_prefix_bt_next} S5 Fixed PnL Buy: Invalid SL/TP. SL={sl_to_use_s5}, TP={tp_to_use_s5}. Skipping."); 
                        return
                    if trade_side_s5 == 'sell' and (sl_to_use_s5 <= entry_price_s5 or tp_to_use_s5 >= entry_price_s5): 
                        # print(f"{log_prefix_bt_next} S5 Fixed PnL Sell: Invalid SL/TP. SL={sl_to_use_s5}, TP={tp_to_use_s5}. Skipping."); 
                        return

                if sl_to_use_s5 is None or tp_to_use_s5 is None: 
                    # print(f"{log_prefix_bt_next} S5: Final SL/TP check failed. SL={sl_to_use_s5}, TP={tp_to_use_s5}. Skipping."); 
                    return # Redundant but safe
                
                # print(f"{log_prefix_bt_next} S5 Placing Trade: Side={trade_side_s5}, Entry={entry_price_s5:.2f}, SL={sl_to_use_s5:.2f}, TP={tp_to_use_s5:.2f}, Size={trade_size_to_use_in_order}")
                if trade_side_s5 == 'buy': self.buy(sl=sl_to_use_s5, tp=tp_to_use_s5, size=trade_size_to_use_in_order)
                else: self.sell(sl=sl_to_use_s5, tp=tp_to_use_s5, size=trade_size_to_use_in_order)
            # else:
                # print(f"{log_prefix_bt_next} S5: No trade signal this bar.")
        
        elif self.current_strategy_id == 6: # Market Structure S/D Strategy
            # Ensure log_prefix_bt_next uses a dynamic precision based on self.PRICE_PRECISION_BT
            # However, price itself is already a float here. Formatting is for the print string.
            # For consistency, let's define precision for logging here.
            log_price_precision = getattr(self, 'PRICE_PRECISION_BT', 4) # Default to 4 if not set

            if self.position:
                print(f"{log_prefix_bt_next} S6: Already in position. Skipping.")
                return

            if not hasattr(self, 'bt_swing_highs_bool') or not hasattr(self, 'bt_sd_zones') or self.atr is None:
                print(f"{log_prefix_bt_next} S6: Required attributes (swings, zones, atr) not initialized. Skipping.")
                return
            
            current_data_slice_df = self.data.df.iloc[:len(self.data.Close)] 
            if len(current_data_slice_df) < 20: 
                print(f"{log_prefix_bt_next} S6: Insufficient data for S6 logic ({len(current_data_slice_df)} bars).")
                return

            current_bar_timestamp = self.data.index[-1]
            entry_price_s6 = price # Current closing price as potential entry

            # --- DEBUG PRINT: Initial Info ---
            print(f"{log_prefix_bt_next} S6 DEBUG: Bar Timestamp: {current_bar_timestamp}, Current Price: {entry_price_s6:.{log_price_precision}f}")

            current_swing_highs = self.bt_swing_highs_bool[self.bt_swing_highs_bool.index.isin(current_data_slice_df.index)]
            current_swing_lows = self.bt_swing_lows_bool[self.bt_swing_lows_bool.index.isin(current_data_slice_df.index)]
            
            try:
                market_structure = identify_market_structure(current_data_slice_df, current_swing_highs, current_swing_lows)
                # --- DEBUG PRINT: Market Structure ---
                print(f"{log_prefix_bt_next} S6 DEBUG: Market Bias: {market_structure['trend_bias']}, Valid High: {market_structure.get('last_valid_high_price', 'N/A')}, Valid Low: {market_structure.get('last_valid_low_price', 'N/A')}")
            except Exception as e_ms:
                print(f"{log_prefix_bt_next} S6: Error in identify_market_structure: {e_ms}")
                return

            historical_sd_zones = [
                zone for zone in self.bt_sd_zones 
                if zone['timestamp_end'] < current_bar_timestamp 
            ]
            
            if not historical_sd_zones:
                print(f"{log_prefix_bt_next} S6: No historical S/D zones found relative to current bar.")
                return

            potential_trade_s6 = None
            
            # Check trend bias: if 'ranging', skip (as per problem description "skips only truly ranging markets")
            # The identify_market_structure function should ideally handle this.
            # If it returns 'ranging', we should not proceed to look for entries.
            if market_structure['trend_bias'] == 'ranging':
                print(f"{log_prefix_bt_next} S6: Market trend_bias is 'ranging'. Skipping S/D zone checks.")
                return # Do not proceed if market is ranging

            if market_structure['trend_bias'] == 'bullish':
                demand_zones = [z for z in historical_sd_zones if z['type'] == 'demand']
                demand_zones.sort(key=lambda z: z['timestamp_end'], reverse=True) 
                if demand_zones:
                    zone = demand_zones[0] # Most recent historical demand zone
                    
                    # Add 0.5% buffer to zone edges for entry condition
                    buffered_zone_start = zone['price_start'] * (1 - 0.005)
                    buffered_zone_end = zone['price_end'] * (1 + 0.005)

                    # --- DEBUG PRINT: Zone Info (Bullish) ---
                    print(f"{log_prefix_bt_next} S6 DEBUG Bullish: Checking Demand Zone: Original Start={zone['price_start']:.{log_price_precision}f}, Original End={zone['price_end']:.{log_price_precision}f}. Buffered Entry Range: {buffered_zone_start:.{log_price_precision}f} - {buffered_zone_end:.{log_price_precision}f}")

                    if buffered_zone_start <= entry_price_s6 <= buffered_zone_end: # Price in buffered demand zone
                        # SL calculation uses original zone boundary + small percentage of zone height
                        sl_s6_raw = zone['price_start'] - ((zone['price_end'] - zone['price_start']) * 0.10) # Example: 10% of zone height as SL distance from bottom
                        sl_s6 = round(sl_s6_raw, self.PRICE_PRECISION_BT)
                        
                        tp_s6_raw = market_structure.get('last_valid_high_price') or market_structure.get('current_swing_high_price')
                        if tp_s6_raw is None: 
                            print(f"{log_prefix_bt_next} S6 BT Bullish: No valid TP target (swing high). Skipping."); return
                        tp_s6 = round(tp_s6_raw, self.PRICE_PRECISION_BT)

                        # --- DEBUG PRINT: SL/TP Info (Bullish) ---
                        print(f"{log_prefix_bt_next} S6 DEBUG Bullish: Potential Entry: {entry_price_s6:.{log_price_precision}f}, SL: {sl_s6:.{log_price_precision}f} (RawSL based on original zone: {sl_s6_raw:.{log_price_precision}f}), TP: {tp_s6:.{log_price_precision}f} (RawTP: {tp_s6_raw:.{log_price_precision}f})")

                        if tp_s6 > entry_price_s6 and sl_s6 < entry_price_s6: # Basic SL/TP validity
                            reward = tp_s6 - entry_price_s6
                            risk = entry_price_s6 - sl_s6
                            if risk > 0:
                                rr_ratio_s6 = reward / risk
                                # --- DEBUG PRINT: R:R (Bullish) ---
                                print(f"{log_prefix_bt_next} S6 DEBUG Bullish: Reward={reward:.{log_price_precision}f}, Risk={risk:.{log_price_precision}f}, R:R={rr_ratio_s6:.2f}")
                                
                                if rr_ratio_s6 >= 1.5: # Lowered R:R threshold
                                    potential_trade_s6 = {'signal': 'buy', 'sl': sl_s6, 'tp': tp_s6}
                                else: print(f"{log_prefix_bt_next} S6 BT Bullish: R:R too low ({rr_ratio_s6:.2f} < 1.5). Skipping.")
                            else: print(f"{log_prefix_bt_next} S6 BT Bullish: Risk is not positive ({risk:.{log_price_precision}f}). Skipping.")
                        else: print(f"{log_prefix_bt_next} S6 BT Bullish: SL/TP ({sl_s6}, {tp_s6}) invalid relative to entry ({entry_price_s6}). Skipping.")
                    else: print(f"{log_prefix_bt_next} S6 BT Bullish: Price {entry_price_s6:.{log_price_precision}f} not in buffered demand zone [{buffered_zone_start:.{log_price_precision}f} - {buffered_zone_end:.{log_price_precision}f}]. Orig Zone: [{zone['price_start']:.{log_price_precision}f} - {zone['price_end']:.{log_price_precision}f}]")
                else: print(f"{log_prefix_bt_next} S6 BT Bullish: No relevant historical demand zones found.")
            
            elif market_structure['trend_bias'] == 'bearish':
                supply_zones = [z for z in historical_sd_zones if z['type'] == 'supply']
                supply_zones.sort(key=lambda z: z['timestamp_end'], reverse=True)
                if supply_zones:
                    zone = supply_zones[0] # Most recent historical supply zone

                    buffered_zone_start = zone['price_start'] * (1 - 0.005)
                    buffered_zone_end = zone['price_end'] * (1 + 0.005)

                    # --- DEBUG PRINT: Zone Info (Bearish) ---
                    print(f"{log_prefix_bt_next} S6 DEBUG Bearish: Checking Supply Zone: Original Start={zone['price_start']:.{log_price_precision}f}, Original End={zone['price_end']:.{log_price_precision}f}. Buffered Entry Range: {buffered_zone_start:.{log_price_precision}f} - {buffered_zone_end:.{log_price_precision}f}")

                    if buffered_zone_start <= entry_price_s6 <= buffered_zone_end: # Price in buffered supply zone
                        sl_s6_raw = zone['price_end'] + ((zone['price_end'] - zone['price_start']) * 0.10) # Example: 10% of zone height as SL distance from top
                        sl_s6 = round(sl_s6_raw, self.PRICE_PRECISION_BT)
                        
                        tp_s6_raw = market_structure.get('last_valid_low_price') or market_structure.get('current_swing_low_price')
                        if tp_s6_raw is None: 
                            print(f"{log_prefix_bt_next} S6 BT Bearish: No valid TP target (swing low). Skipping."); return
                        tp_s6 = round(tp_s6_raw, self.PRICE_PRECISION_BT)

                        # --- DEBUG PRINT: SL/TP Info (Bearish) ---
                        print(f"{log_prefix_bt_next} S6 DEBUG Bearish: Potential Entry: {entry_price_s6:.{log_price_precision}f}, SL: {sl_s6:.{log_price_precision}f} (RawSL based on original zone: {sl_s6_raw:.{log_price_precision}f}), TP: {tp_s6:.{log_price_precision}f} (RawTP: {tp_s6_raw:.{log_price_precision}f})")

                        if tp_s6 < entry_price_s6 and sl_s6 > entry_price_s6: # Basic SL/TP validity
                            reward = entry_price_s6 - tp_s6
                            risk = sl_s6 - entry_price_s6
                            if risk > 0:
                                rr_ratio_s6 = reward / risk
                                # --- DEBUG PRINT: R:R (Bearish) ---
                                print(f"{log_prefix_bt_next} S6 DEBUG Bearish: Reward={reward:.{log_price_precision}f}, Risk={risk:.{log_price_precision}f}, R:R={rr_ratio_s6:.2f}")

                                if rr_ratio_s6 >= 1.5: # Lowered R:R threshold
                                    potential_trade_s6 = {'signal': 'sell', 'sl': sl_s6, 'tp': tp_s6}
                                else: print(f"{log_prefix_bt_next} S6 BT Bearish: R:R too low ({rr_ratio_s6:.2f} < 1.5). Skipping.")
                            else: print(f"{log_prefix_bt_next} S6 BT Bearish: Risk is not positive ({risk:.{log_price_precision}f}). Skipping.")
                        else: print(f"{log_prefix_bt_next} S6 BT Bearish: SL/TP ({sl_s6}, {tp_s6}) invalid relative to entry ({entry_price_s6}). Skipping.")
                    else: print(f"{log_prefix_bt_next} S6 BT Bearish: Price {entry_price_s6:.{log_price_precision}f} not in buffered supply zone [{buffered_zone_start:.{log_price_precision}f} - {buffered_zone_end:.{log_price_precision}f}]. Orig Zone: [{zone['price_start']:.{log_price_precision}f} - {zone['price_end']:.{log_price_precision}f}]")
                else: print(f"{log_prefix_bt_next} S6 BT Bearish: No relevant historical supply zones found.")
            # Implicitly, if trend_bias is not 'bullish' or 'bearish' (e.g. 'ranging' or other), no action is taken here.
            # The earlier check `if market_structure['trend_bias'] == 'ranging': return` handles explicit ranging.

            if potential_trade_s6:
                print(f"{log_prefix_bt_next} S6 Placing Trade: Side={potential_trade_s6['signal']}, Entry={entry_price_s6:.{log_price_precision}f}, SL={potential_trade_s6['sl']:.{log_price_precision}f}, TP={potential_trade_s6['tp']:.{log_price_precision}f}, Size={trade_size_to_use_in_order}")
                if potential_trade_s6['signal'] == 'buy':
                    self.buy(sl=potential_trade_s6['sl'], tp=potential_trade_s6['tp'], size=trade_size_to_use_in_order)
                else: # sell
                    self.sell(sl=potential_trade_s6['sl'], tp=potential_trade_s6['tp'], size=trade_size_to_use_in_order)
            else:
                 # This 'else' means no 'potential_trade_s6' was formed.
                 # Debug prints within the logic above should indicate why (e.g., R:R too low, price not in zone, etc.)
                 print(f"{log_prefix_bt_next} S6: No valid trade signal this bar based on refined conditions.")
        
        elif self.current_strategy_id == 7: # Candlestick Patterns Strategy (Backtesting Logic)
            if self.position:
                # print(f"{log_prefix_bt_next} S7 BT: Already in position. Skipping.")
                return

            min_bars_s7 = 205 # Matches live strategy's min_klines_needed for EMA200 and volume lookback
            if len(self.data.Close) < min_bars_s7:
                # print(f"{log_prefix_bt_next} S7 BT: Insufficient data length ({len(self.data.Close)} < {min_bars_s7}). Skipping.")
                return
            
            if self.ema200_s7 is None or len(self.ema200_s7) < 1 or pd.isna(self.ema200_s7[-1]):
                # print(f"{log_prefix_bt_next} S7 BT: EMA200 not available or NaN. Skipping.")
                return
            
            # Liquidity filter (simplified for backtesting - assuming data is for active hours)
            # Live strategy checks current UTC hour. For backtesting, this is harder without explicit hour data per bar.
            # We'll assume data provided for backtesting is within valid trading hours.

            # Prepare data for pattern functions (last 5 candles for some patterns)
            # self.data.df provides Open, High, Low, Close, Volume
            current_df_slice = self.data.df.iloc[len(self.data.Close)-5 : len(self.data.Close)] if len(self.data.Close) >=5 else self.data.df.iloc[:len(self.data.Close)]
            
            # Mimic `strategy_candlestick_patterns_signal` structure
            m_curr = get_candle_metrics(current_df_slice.iloc[-1]) if len(current_df_slice) >= 1 else None
            m_prev = get_candle_metrics(current_df_slice.iloc[-2]) if len(current_df_slice) >= 2 else None
            m_prev2 = get_candle_metrics(current_df_slice.iloc[-3]) if len(current_df_slice) >= 3 else None
            # df_last5 for Rising/Falling Three Methods will be current_df_slice itself if len >=5

            if not m_curr: # Must have at least current candle metrics
                # print(f"{log_prefix_bt_next} S7 BT: Not enough candle metrics. Skipping.")
                return

            # Volume Spike Check for Backtesting
            # Use self.data.Volume which is available in backtesting.py
            # The check_volume_spike function expects a DataFrame with a 'Volume' column.
            # We can pass a slice of self.data.df for this.
            # The lookback for volume average is 20, plus current candle.
            volume_lookback_period_s7_bt = 20
            if len(self.data.Volume) < volume_lookback_period_s7_bt + 1:
                # print(f"{log_prefix_bt_next} S7 BT: Not enough volume data for spike check. Skipping.")
                volume_spike_detected_s7_bt = False
            else:
                # Pass the relevant part of the historical data DataFrame to check_volume_spike
                # Ensure the slice ends at the current bar being processed.
                # self.data.df contains all historical data up to the current point in `next()`
                volume_data_slice_for_spike_check = self.data.df.iloc[len(self.data.Close) - (volume_lookback_period_s7_bt + 1) : len(self.data.Close)]
                volume_spike_detected_s7_bt = check_volume_spike(volume_data_slice_for_spike_check, lookback_period=volume_lookback_period_s7_bt, multiplier=2.0)

            # EMA Filter
            last_ema200_s7_bt = self.ema200_s7[-1]
            current_price_s7_bt = price # price is self.data.Close[-1]

            detected_pattern_name_s7_bt = "None"
            pattern_side_s7_bt = "none"
            sl_ref_price_s7_bt = None # Reference price from pattern for SL (e.g., wick low/high)

            # Pattern Detection Logic (copied and adapted from live strategy)
            if m_prev2 and m_prev and is_morning_star(m_prev2, m_prev, m_curr): detected_pattern_name_s7_bt, pattern_side_s7_bt, sl_ref_price_s7_bt = "Morning Star", "up", min(m_prev2['l'], m_prev['l'], m_curr['l'])
            elif m_prev2 and m_prev and is_evening_star(m_prev2, m_prev, m_curr): detected_pattern_name_s7_bt, pattern_side_s7_bt, sl_ref_price_s7_bt = "Evening Star", "down", max(m_prev2['h'], m_prev['h'], m_curr['h'])
            elif m_prev and is_bullish_engulfing(m_prev, m_curr): detected_pattern_name_s7_bt, pattern_side_s7_bt, sl_ref_price_s7_bt = "Bullish Engulfing", "up", m_curr['l']
            elif m_prev and is_bearish_engulfing(m_prev, m_curr): detected_pattern_name_s7_bt, pattern_side_s7_bt, sl_ref_price_s7_bt = "Bearish Engulfing", "down", m_curr['h']
            elif is_hammer(m_curr): detected_pattern_name_s7_bt, pattern_side_s7_bt, sl_ref_price_s7_bt = "Hammer", "up", m_curr['l']
            elif is_hanging_man(m_curr): detected_pattern_name_s7_bt, pattern_side_s7_bt, sl_ref_price_s7_bt = "Hanging Man", "down", m_curr['h']
            elif is_inverted_hammer(m_curr): detected_pattern_name_s7_bt, pattern_side_s7_bt, sl_ref_price_s7_bt = "Inverted Hammer", "up", m_curr['l']
            elif is_shooting_star(m_curr): detected_pattern_name_s7_bt, pattern_side_s7_bt, sl_ref_price_s7_bt = "Shooting Star", "down", m_curr['h']
            elif m_prev and is_piercing_line(m_prev, m_curr): detected_pattern_name_s7_bt, pattern_side_s7_bt, sl_ref_price_s7_bt = "Piercing Line", "up", m_curr['l']
            elif m_prev and is_dark_cloud_cover(m_prev, m_curr): detected_pattern_name_s7_bt, pattern_side_s7_bt, sl_ref_price_s7_bt = "Dark Cloud Cover", "down", m_curr['h']
            elif m_prev2 and m_prev and is_three_white_soldiers(m_prev2, m_prev, m_curr): detected_pattern_name_s7_bt, pattern_side_s7_bt, sl_ref_price_s7_bt = "Three White Soldiers", "up", m_prev2['l']
            elif m_prev2 and m_prev and is_three_black_crows(m_prev2, m_prev, m_curr): detected_pattern_name_s7_bt, pattern_side_s7_bt, sl_ref_price_s7_bt = "Three Black Crows", "down", m_prev2['h']
            elif len(current_df_slice) >= 5 and is_rising_three_methods(current_df_slice): detected_pattern_name_s7_bt, pattern_side_s7_bt, sl_ref_price_s7_bt = "Rising Three Methods", "up", current_df_slice.iloc[0]['Low']
            elif len(current_df_slice) >= 5 and is_falling_three_methods(current_df_slice): detected_pattern_name_s7_bt, pattern_side_s7_bt, sl_ref_price_s7_bt = "Falling Three Methods", "down", current_df_slice.iloc[0]['High']

            if pattern_side_s7_bt != "none":
                # print(f"{log_prefix_bt_next} S7 BT: Pattern detected: {detected_pattern_name_s7_bt} ({pattern_side_s7_bt})")
                ema_filter_passed_s7_bt = (pattern_side_s7_bt == "up" and current_price_s7_bt > last_ema200_s7_bt) or \
                                          (pattern_side_s7_bt == "down" and current_price_s7_bt < last_ema200_s7_bt)
                
                if volume_spike_detected_s7_bt and ema_filter_passed_s7_bt:
                    # print(f"{log_prefix_bt_next} S7 BT: Filters passed (Volume Spike: {volume_spike_detected_s7_bt}, EMA Filter: {ema_filter_passed_s7_bt})")
                    # Signal is confirmed, now use the SL/TP mode from backtest settings
                    entry_price_s7 = current_price_s7_bt
                    sl_to_use_s7, tp_to_use_s7 = None, None

                    if self.sl_tp_mode_bt == "ATR/Dynamic":
                        if self.atr_s7 is None or len(self.atr_s7) < 1 or pd.isna(self.atr_s7[-1]) or self.atr_s7[-1] == 0:
                            # print(f"{log_prefix_bt_next} S7 BT: ATR invalid for ATR/Dynamic SL/TP. Skipping.")
                            return
                        current_atr_s7_bt = self.atr_s7[-1]
                        
                        # For S7, the live strategy uses pattern wick + ATR buffer.
                        # For backtesting in ATR/Dynamic mode, we can try to mimic this or use the standard ATR SL.
                        # Let's try to use the sl_ref_price_s7_bt if available, otherwise fallback to standard ATR.
                        if sl_ref_price_s7_bt is not None:
                            atr_buffer_s7_bt = current_atr_s7_bt * 0.1 # 10% of ATR as buffer
                            if pattern_side_s7_bt == "up":
                                sl_candidate = round(sl_ref_price_s7_bt - atr_buffer_s7_bt, self.PRICE_PRECISION_BT)
                                if sl_candidate >= entry_price_s7: # If SL is too tight or above entry
                                    sl_candidate = round(entry_price_s7 - (current_atr_s7_bt * self.SL_ATR_MULTI), self.PRICE_PRECISION_BT) # Fallback to standard ATR SL
                                sl_to_use_s7 = sl_candidate
                                if sl_to_use_s7 < entry_price_s7 : tp_to_use_s7 = round(entry_price_s7 + (entry_price_s7 - sl_to_use_s7) * self.RR, self.PRICE_PRECISION_BT)
                            else: # down
                                sl_candidate = round(sl_ref_price_s7_bt + atr_buffer_s7_bt, self.PRICE_PRECISION_BT)
                                if sl_candidate <= entry_price_s7:
                                    sl_candidate = round(entry_price_s7 + (current_atr_s7_bt * self.SL_ATR_MULTI), self.PRICE_PRECISION_BT)
                                sl_to_use_s7 = sl_candidate
                                if sl_to_use_s7 > entry_price_s7 : tp_to_use_s7 = round(entry_price_s7 - (sl_to_use_s7 - entry_price_s7) * self.RR, self.PRICE_PRECISION_BT)
                        else: # No specific sl_ref_price from pattern, use standard ATR SL/TP
                            if pattern_side_s7_bt == "up":
                                sl_to_use_s7 = round(entry_price_s7 - (current_atr_s7_bt * self.SL_ATR_MULTI), self.PRICE_PRECISION_BT)
                                if sl_to_use_s7 < entry_price_s7 : tp_to_use_s7 = round(entry_price_s7 + (entry_price_s7 - sl_to_use_s7) * self.RR, self.PRICE_PRECISION_BT)
                            else: # down
                                sl_to_use_s7 = round(entry_price_s7 + (current_atr_s7_bt * self.SL_ATR_MULTI), self.PRICE_PRECISION_BT)
                                if sl_to_use_s7 > entry_price_s7 : tp_to_use_s7 = round(entry_price_s7 - (sl_to_use_s7 - entry_price_s7) * self.RR, self.PRICE_PRECISION_BT)
                        
                    elif self.sl_tp_mode_bt == "Percentage":
                        sl_to_use_s7 = sl_long if pattern_side_s7_bt == "up" else sl_short
                        tp_to_use_s7 = tp_long if pattern_side_s7_bt == "up" else tp_short
                    elif self.sl_tp_mode_bt == "Fixed PnL":
                        sl_to_use_s7 = sl_long if pattern_side_s7_bt == "up" else sl_short
                        tp_to_use_s7 = tp_long if pattern_side_s7_bt == "up" else tp_short
                    
                    # Final validation of calculated SL/TP for S7
                    if sl_to_use_s7 is None or tp_to_use_s7 is None or sl_to_use_s7 <= 0 or tp_to_use_s7 <= 0:
                        # print(f"{log_prefix_bt_next} S7 BT: SL/TP calculation failed or invalid. SL={sl_to_use_s7}, TP={tp_to_use_s7}. Skipping."); 
                        return
                    if pattern_side_s7_bt == "up" and (sl_to_use_s7 >= entry_price_s7 or tp_to_use_s7 <= entry_price_s7):
                        # print(f"{log_prefix_bt_next} S7 BT UP: Invalid SL/TP relation. SL={sl_to_use_s7}, TP={tp_to_use_s7}, Entry={entry_price_s7}. Skipping."); 
                        return
                    if pattern_side_s7_bt == "down" and (sl_to_use_s7 <= entry_price_s7 or tp_to_use_s7 >= entry_price_s7):
                        # print(f"{log_prefix_bt_next} S7 BT DOWN: Invalid SL/TP relation. SL={sl_to_use_s7}, TP={tp_to_use_s7}, Entry={entry_price_s7}. Skipping."); 
                        return

                    # print(f"{log_prefix_bt_next} S7 BT Placing Trade: Side={pattern_side_s7_bt}, Entry={entry_price_s7:.{self.PRICE_PRECISION_BT}f}, SL={sl_to_use_s7:.{self.PRICE_PRECISION_BT}f}, TP={tp_to_use_s7:.{self.PRICE_PRECISION_BT}f}, Size={trade_size_to_use_in_order}")
                    if pattern_side_s7_bt == "up": self.buy(sl=sl_to_use_s7, tp=tp_to_use_s7, size=trade_size_to_use_in_order)
                    else: self.sell(sl=sl_to_use_s7, tp=tp_to_use_s7, size=trade_size_to_use_in_order)
                # else:
                    # print(f"{log_prefix_bt_next} S7 BT: Filters failed for {detected_pattern_name_s7_bt}. Volume Spike: {volume_spike_detected_s7_bt}, EMA Filter: {ema_filter_passed_s7_bt}")
            # else:
                # print(f"{log_prefix_bt_next} S7 BT: No candlestick pattern detected this bar.")


        # else: # Other strategies not yet updated for this new SL/TP structure
            # print(f"{log_prefix_bt_next} Strategy ID {self.current_strategy_id} not fully updated for new SL/TP modes in next().")
            # Basic RSI logic as a placeholder if other strategies are selected without full logic
            # This part should be removed or adapted as each strategy is fully integrated
            # if not self.position and hasattr(self, 'rsi') and len(self.rsi) > 0:
            #     if self.rsi[-1] < 30 and self.sl_tp_mode_bt != "ATR/Dynamic": # Example buy with non-ATR SL/TP
            #         if sl_long is not None and tp_long is not None and sl_long < price and tp_long > price:
            #              self.buy(size=trade_size_to_use_in_order, sl=sl_long, tp=tp_long)
            #     elif self.rsi[-1] > 70 and self.sl_tp_mode_bt != "ATR/Dynamic": # Example sell
            #         if sl_short is not None and tp_short is not None and sl_short > price and tp_short < price:
            #              self.sell(size=trade_size_to_use_in_order, sl=sl_short, tp=tp_short)


# Function to execute backtest
def execute_backtest(strategy_id_for_backtest, symbol, timeframe, interval_days, 
                     ui_tp_percentage, ui_sl_percentage, # Percentage based
                     sl_tp_mode, sl_pnl_amount_val, tp_pnl_amount_val, # PnL and Mode based
                     starting_capital, # New parameter for starting capital
                     leverage_bt # New parameter for leverage
                     ):
    print(f"Executing backtest for Strategy ID {strategy_id_for_backtest} on {symbol} ({timeframe}, {interval_days} days), Start Capital: ${starting_capital:.2f}, Leverage: {leverage_bt}x, Mode: {sl_tp_mode}") # Added leverage_bt to print
    if sl_tp_mode == "Percentage":
        print(f"  TP %: {ui_tp_percentage*100:.2f}%, SL %: {ui_sl_percentage*100:.2f}%")
    elif sl_tp_mode == "Fixed PnL":
        print(f"  SL PnL: ${sl_pnl_amount_val:.2f}, TP PnL: ${tp_pnl_amount_val:.2f}")
    elif sl_tp_mode == "ATR/Dynamic":
        print(f"  Using ATR/Dynamic SL/TP defined in strategy.")


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
    print(f"DEBUG execute_backtest: Received ui_tp_percentage: {ui_tp_percentage*100:.2f}%, ui_sl_percentage: {ui_sl_percentage*100:.2f}%") # Corrected var names
    BacktestStrategyWrapper.current_strategy_id = strategy_id_for_backtest
    BacktestStrategyWrapper.sl_tp_mode_bt = sl_tp_mode # Use the renamed class attribute
    
    print(f"DEBUG execute_backtest: Set BacktestStrategyWrapper.current_strategy_id to: {strategy_id_for_backtest}")
    print(f"DEBUG execute_backtest: Set BacktestStrategyWrapper.sl_tp_mode_bt to: {sl_tp_mode}")

    if sl_tp_mode == "Percentage":
        BacktestStrategyWrapper.user_tp = ui_tp_percentage 
        BacktestStrategyWrapper.user_sl = ui_sl_percentage 
        # Reset PnL amounts for clarity if this mode is chosen
        BacktestStrategyWrapper.sl_pnl_amount_bt = 0.0
        BacktestStrategyWrapper.tp_pnl_amount_bt = 0.0
        print(f"DEBUG execute_backtest: Using Percentage SL/TP - TP: {BacktestStrategyWrapper.user_tp*100:.2f}%, SL: {BacktestStrategyWrapper.user_sl*100:.2f}%")
    elif sl_tp_mode == "Fixed PnL":
        BacktestStrategyWrapper.sl_pnl_amount_bt = sl_pnl_amount_val 
        BacktestStrategyWrapper.tp_pnl_amount_bt = tp_pnl_amount_val 
        # Reset percentage SL/TP for clarity
        BacktestStrategyWrapper.user_tp = 0 
        BacktestStrategyWrapper.user_sl = 0
        print(f"DEBUG execute_backtest: Using Fixed PnL SL/TP - SL PnL: ${BacktestStrategyWrapper.sl_pnl_amount_bt:.2f}, TP PnL: ${BacktestStrategyWrapper.tp_pnl_amount_bt:.2f}")
    elif sl_tp_mode == "ATR/Dynamic" or sl_tp_mode == "StrategyDefined_SD": # Added StrategyDefined_SD
        # Reset other modes' params for clarity
        BacktestStrategyWrapper.user_tp = 0 
        BacktestStrategyWrapper.user_sl = 0
        BacktestStrategyWrapper.sl_pnl_amount_bt = 0.0
        BacktestStrategyWrapper.tp_pnl_amount_bt = 0.0
        # For "StrategyDefined_SD", RR and SL_ATR_MULTI might not be directly used from wrapper defaults if strategy_market_structure_sd fully defines SL/TP.
        # However, logging them might still be informative if the backtesting strategy falls back or uses parts of ATR logic.
        print(f"DEBUG execute_backtest: Using {sl_tp_mode} SL/TP. Strategy will define SL/TP. Wrapper RR: {BacktestStrategyWrapper.RR}, SL_ATR_MULTI: {BacktestStrategyWrapper.SL_ATR_MULTI}")
    
    print(f"DEBUG execute_backtest: Strategy RSI period: {getattr(BacktestStrategyWrapper, 'rsi_period', 'N/A')}") # Also log rsi_period

    # Set leverage for the strategy wrapper instance (though it's mainly for logging within the wrapper)
    BacktestStrategyWrapper.leverage = leverage_bt

    # Calculate margin for backtesting.py: margin = 1 / leverage
    # Ensure leverage_bt >= 1. If leverage_bt is 0 or less, or less than 1, default to 1 (no leverage, margin=1).
    if leverage_bt < 1:
        print(f"Warning: Invalid leverage {leverage_bt}x provided for backtest. Defaulting to 1x (no leverage).")
        margin_val = 1.0
        BacktestStrategyWrapper.leverage = 1.0 # Also update the class attribute for consistency in logs
    else:
        margin_val = 1 / leverage_bt
    
    print(f"DEBUG execute_backtest: Calculated margin for Backtest: {margin_val} (from leverage: {leverage_bt}x)")

    bt = Backtest(kl_df, BacktestStrategyWrapper, cash=starting_capital, margin=margin_val, commission=0.0007)
    try:
        stats = bt.run()
        print("Backtest completed.")
        # print(stats) # Stats can be very verbose, printed later or in UI
        print(f"DEBUG execute_backtest: Stats object type: {type(stats)}")
        if isinstance(stats, pd.Series):
            print(f"DEBUG execute_backtest: Stats content (first few entries):\n{stats.head()}")
        elif isinstance(stats, dict): # backtesting.py can sometimes return dicts
             print(f"DEBUG execute_backtest: Stats content (dict):\n{stats}")
        else:
            print(f"DEBUG execute_backtest: Stats content:\n{stats}")

        # Add a print for the _trades attribute if it exists
        if hasattr(stats, '_trades') and not stats['_trades'].empty:
            print(f"DEBUG execute_backtest: Trades DataFrame head:\n{stats['_trades'].head()}")
        elif hasattr(stats, '_trades'):
            print("DEBUG execute_backtest: Trades DataFrame is empty.")
        else:
            print("DEBUG execute_backtest: _trades attribute not found in stats.")

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
# New PnL based SL/TP StringVars for Trading Parameters
sl_pnl_amount_var = None
tp_pnl_amount_var = None
sl_tp_mode_var = None

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
# New PnL based SL/TP Entry/Combobox widgets for Trading Parameters
sl_pnl_amount_entry = None
tp_pnl_amount_entry = None
sl_tp_mode_combobox = None

leverage_entry = None
qty_concurrent_positions_entry = None
local_high_low_lookback_entry = None
target_symbols_entry = None # For Target Symbols CSV Entry
margin_type_isolated_radio = None # For Margin Type Radio
margin_type_cross_radio = None # For Margin Type Radio
strategy_radio_buttons = [] # List to hold strategy radio buttons
params_widgets = [] # List to hold all parameter input widgets (will extend with strategy_radio_buttons)
conditions_text_widget = None # For displaying signal conditions

# Backtesting UI Variables - New PnL and Mode vars
backtest_sl_pnl_amount_var = None
backtest_tp_pnl_amount_var = None
backtest_sl_tp_mode_var = None
backtest_starting_capital_var = None # New var for starting capital


# --- Backtesting UI Command and Threading Logic ---
def run_backtest_command():
    global backtest_symbol_var, backtest_timeframe_var, backtest_interval_var
    global backtest_tp_var, backtest_sl_var, backtest_selected_strategy_var
    # Add new global vars for backtesting PnL and Mode
    global backtest_sl_pnl_amount_var, backtest_tp_pnl_amount_var, backtest_sl_tp_mode_var, backtest_starting_capital_var, backtest_leverage_var # Added backtest_leverage_var
    global backtest_results_text_widget, STRATEGIES, root, status_var, backtest_run_button

    if backtest_run_button:
        backtest_run_button.config(state=tk.DISABLED)

    try:
        symbols_str = backtest_symbol_var.get().strip().upper() # Changed variable name
        if not symbols_str:
            messagebox.showerror("Input Error", "Symbol(s) field cannot be empty.")
            if backtest_run_button: backtest_run_button.config(state=tk.NORMAL)
            return
        
        symbols_list = [s.strip() for s in symbols_str.split(',') if s.strip()]
        if not symbols_list:
            messagebox.showerror("Input Error", "No valid symbols provided. Please enter symbols separated by commas.")
            if backtest_run_button: backtest_run_button.config(state=tk.NORMAL)
            return

        timeframe = backtest_timeframe_var.get().strip()
        interval_str = backtest_interval_var.get().strip()
        selected_strategy_name = backtest_selected_strategy_var.get()
        starting_capital_str = backtest_starting_capital_var.get().strip()
        leverage_str = backtest_leverage_var.get().strip() # Get leverage string
        
        # Get SL/TP mode for backtesting
        current_backtest_sl_tp_mode = backtest_sl_tp_mode_var.get()

        # Default values, will be overwritten if not 'Percentage'
        tp_percentage = 0.0
        sl_percentage = 0.0
        sl_pnl = 0.0
        tp_pnl = 0.0
        starting_capital = 10000 # Default starting capital
        leverage_val = 1.0 # Default leverage

        # Validate Starting Capital
        if not starting_capital_str:
            messagebox.showerror("Input Error", "Starting Capital is required.")
            if backtest_run_button: backtest_run_button.config(state=tk.NORMAL)
            return
        try:
            starting_capital = float(starting_capital_str)
            if starting_capital <= 0:
                messagebox.showerror("Input Error", "Starting Capital must be a positive value.")
                if backtest_run_button: backtest_run_button.config(state=tk.NORMAL)
                return
        except ValueError:
            messagebox.showerror("Input Error", "Invalid number format for Starting Capital.")
            if backtest_run_button: backtest_run_button.config(state=tk.NORMAL)
            return
        
        # Validate Leverage
        if not leverage_str:
            messagebox.showerror("Input Error", "Leverage is required.")
            if backtest_run_button: backtest_run_button.config(state=tk.NORMAL)
            return
        try:
            leverage_val = float(leverage_str)
            if leverage_val < 1: # Leverage must be >= 1
                messagebox.showerror("Input Error", "Leverage must be 1 or greater.")
                if backtest_run_button: backtest_run_button.config(state=tk.NORMAL)
                return
        except ValueError:
            messagebox.showerror("Input Error", "Invalid number format for Leverage.")
            if backtest_run_button: backtest_run_button.config(state=tk.NORMAL)
            return

        if current_backtest_sl_tp_mode == "Percentage":
            tp_str = backtest_tp_var.get().strip() # TP %
            sl_str = backtest_sl_var.get().strip() # SL %
            # Check if symbols_list is empty, timeframe, interval_str, etc. are present
            if not all([symbols_list, timeframe, interval_str, tp_str, sl_str, selected_strategy_name]): # Changed 'symbol' to 'symbols_list'
                messagebox.showerror("Input Error", "All backtest fields (including Symbol(s) and TP/SL Percentages) are required for 'Percentage' mode.")
                if backtest_run_button: backtest_run_button.config(state=tk.NORMAL)
                return
            tp_percentage = float(tp_str) / 100.0
            sl_percentage = float(sl_str) / 100.0
            if not (0 < tp_percentage < 1 and 0 < sl_percentage < 1):
                messagebox.showerror("Input Error", "TP and SL percentages must be between 0 and 100 (exclusive of 0).")
                if backtest_run_button: backtest_run_button.config(state=tk.NORMAL)
                return
        elif current_backtest_sl_tp_mode == "Fixed PnL":
            sl_pnl_str = backtest_sl_pnl_amount_var.get().strip()
            tp_pnl_str = backtest_tp_pnl_amount_var.get().strip()
            if not all([symbols_list, timeframe, interval_str, sl_pnl_str, tp_pnl_str, selected_strategy_name]): # Changed 'symbol' to 'symbols_list'
                messagebox.showerror("Input Error", "All backtest fields (including Symbol(s) and SL/TP PnL Amounts) are required for 'Fixed PnL' mode.")
                if backtest_run_button: backtest_run_button.config(state=tk.NORMAL)
                return
            sl_pnl = float(sl_pnl_str)
            tp_pnl = float(tp_pnl_str)
            if not (sl_pnl > 0 and tp_pnl > 0): # PnL amounts must be positive
                messagebox.showerror("Input Error", "SL and TP PnL amounts must be positive values.")
                if backtest_run_button: backtest_run_button.config(state=tk.NORMAL)
                return
        elif current_backtest_sl_tp_mode == "ATR/Dynamic":
            # For ATR/Dynamic, specific TP/SL inputs might not be needed from main UI
            # as strategy itself calculates them. Or, UI could provide ATR multiplier and RR.
            # For now, assume strategy handles it, no direct numeric inputs for ATR SL/TP from here.
            if not all([symbols_list, timeframe, interval_str, selected_strategy_name]): # Changed 'symbol' to 'symbols_list'
                 messagebox.showerror("Input Error", "Symbol(s), Timeframe, Interval, and Strategy are required for 'ATR/Dynamic' mode.")
                 if backtest_run_button: backtest_run_button.config(state=tk.NORMAL)
                 return
            # sl_percentage and tp_percentage (and sl_pnl, tp_pnl) remain 0.0 as they are not used.
            # The BacktestStrategyWrapper will use its internal ATR logic (or strategy-defined for StrategyDefined_SD).
        elif current_backtest_sl_tp_mode == "StrategyDefined_SD":
            # Similar to ATR/Dynamic, no specific numeric inputs from UI here for SL/TP values themselves,
            # as the strategy (S6) will calculate them.
            if not all([symbols_list, timeframe, interval_str, selected_strategy_name]):
                 messagebox.showerror("Input Error", "Symbol(s), Timeframe, Interval, and Strategy are required for 'StrategyDefined_SD' mode.")
                 if backtest_run_button: backtest_run_button.config(state=tk.NORMAL)
                 return
            # Ensure the selected strategy is compatible (i.e., Strategy 6)
            # This check could be more robust by linking modes to strategies.
            strategy_id_temp = None
            for id_val, name_val in STRATEGIES.items():
                if name_val == selected_strategy_name:
                    strategy_id_temp = id_val
                    break
            if strategy_id_temp != 6: # Assuming ID 6 is Market Structure S/D
                messagebox.showerror("Input Error", "'StrategyDefined_SD' mode is intended for the Market Structure S/D strategy.")
                if backtest_run_button: backtest_run_button.config(state=tk.NORMAL)
                return
        else: # Should not happen with Combobox
            messagebox.showerror("Input Error", "Invalid SL/TP Mode selected for backtest.")
            if backtest_run_button: backtest_run_button.config(state=tk.NORMAL)
            return

        interval_days = int(interval_str)
        # if not (0 < tp_percentage < 1 and 0 < sl_percentage < 1): # Validation moved up
        #     messagebox.showerror("Input Error", "TP and SL percentages must be between 0 and 100 (exclusive of 0, inclusive of values that result in <1 after division by 100). E.g., 0.01 to 0.99 after conversion.")
        #     return # This return was problematic, removed as validation is mode-specific now.
            
        strategy_id_for_backtest = None
        for id, name in STRATEGIES.items():
            if name == selected_strategy_name:
                strategy_id_for_backtest = id
                break
        
        if strategy_id_for_backtest is None: # Should not happen with Combobox
            messagebox.showerror("Input Error", "Invalid strategy selected.")
            if backtest_run_button: backtest_run_button.config(state=tk.NORMAL)
            return

        # Clear previous results
        if backtest_results_text_widget and root and root.winfo_exists():
            backtest_results_text_widget.config(state=tk.NORMAL)
            backtest_results_text_widget.delete('1.0', tk.END) # Clear previous results before starting new batch
            backtest_results_text_widget.insert(tk.END, f"Starting backtests for {len(symbols_list)} symbol(s): {', '.join(symbols_list)}\nStrategy: {selected_strategy_name}, SL/TP Mode: {current_backtest_sl_tp_mode}...\n\n")
            backtest_results_text_widget.config(state=tk.DISABLED)
        
        if status_var and root and root.winfo_exists():
            status_var.set(f"Backtest: Initializing for {len(symbols_list)} symbols...")

        # Threading for the entire loop of symbols
        thread = threading.Thread(target=perform_backtest_for_multiple_symbols, 
                                  args=(symbols_list, strategy_id_for_backtest, timeframe, interval_days, 
                                        tp_percentage, sl_percentage,
                                        current_backtest_sl_tp_mode, sl_pnl, tp_pnl,
                                        starting_capital, leverage_val),
                                  daemon=True)
        thread.start()

    except ValueError: # Catches float/int conversion errors for Interval, TP/SL %, PnL Amounts, Starting Capital, or Leverage
        messagebox.showerror("Input Error", "Invalid number format for Interval, TP/SL %, PnL Amounts, Starting Capital, or Leverage.")
        if backtest_run_button: backtest_run_button.config(state=tk.NORMAL) 
    except Exception as e:
        messagebox.showerror("Error", f"An unexpected error occurred in run_backtest_command: {e}")
        if backtest_results_text_widget and root and root.winfo_exists():
            backtest_results_text_widget.config(state=tk.NORMAL)
            backtest_results_text_widget.insert(tk.END, f"\nError setting up backtest batch: {e}")
            backtest_results_text_widget.config(state=tk.DISABLED)
        if backtest_run_button: backtest_run_button.config(state=tk.NORMAL)


# New function to handle the loop and call perform_backtest_in_thread for each symbol
def perform_backtest_for_multiple_symbols(symbols_list, strategy_id, timeframe, interval_days,
                                          tp_percentage_val, sl_percentage_val,
                                          passed_sl_tp_mode, sl_pnl_val, tp_pnl_val,
                                          starting_capital_val, leverage_val):
    global backtest_results_text_widget, root, status_var, backtest_run_button
    all_symbols_completed = True

    for i, symbol_item in enumerate(symbols_list):
        if not (root and root.winfo_exists()): # Check if UI is still alive
            print("UI closed, aborting multi-symbol backtest.")
            break 
        
        print(f"\nProcessing symbol {i+1}/{len(symbols_list)}: {symbol_item}")
        if status_var:
             root.after(0, lambda s=symbol_item, num=i+1, total=len(symbols_list): status_var.set(f"Backtest: Processing {s} ({num}/{total})"))
        
        # This will call the original perform_backtest_in_thread logic for a single symbol
        # But we need to adapt how results are displayed.
        # For simplicity, let's make perform_backtest_in_thread directly update UI or return results to here
        
        # Re-evaluating: It's better to call execute_backtest directly here and manage UI updates
        # to avoid complex threading handoffs for each symbol's detailed results.
        
        # Update UI to show which symbol is being processed
        if backtest_results_text_widget:
            root.after(0, lambda sym=symbol_item: (
                backtest_results_text_widget.config(state=tk.NORMAL),
                backtest_results_text_widget.insert(tk.END, f"--- Backtesting for {sym} ---\nFetching data...\n"),
                backtest_results_text_widget.see(tk.END), # Scroll to end
                backtest_results_text_widget.config(state=tk.DISABLED)
            ))

        stats_output, bt_object, plot_error_message = execute_backtest(
            strategy_id, symbol_item, timeframe, interval_days,
            tp_percentage_val, sl_percentage_val,
            passed_sl_tp_mode, sl_pnl_val, tp_pnl_val,
            starting_capital_val, leverage_val
        )

        # Update UI with results for this specific symbol
        if root and root.winfo_exists() and backtest_results_text_widget:
            def update_ui_for_symbol_result(sym, stats, plot_err):
                backtest_results_text_widget.config(state=tk.NORMAL)
                if isinstance(stats, pd.Series):
                    stats_str = f"Results for {sym}:\n"
                    stats_str += "--------------------\n"
                    for index, value in stats.items():
                        if index in ['_equity_curve', '_trades']: continue
                        stats_str += f"{index}: {value:.2f}\n" if isinstance(value, float) else f"{index}: {value}\n"
                    backtest_results_text_widget.insert(tk.END, stats_str)
                elif isinstance(stats, str): # Error message from execute_backtest
                    backtest_results_text_widget.insert(tk.END, f"Error for {sym}: {stats}\n")
                elif stats is not None:
                    backtest_results_text_widget.insert(tk.END, f"Results for {sym}:\n{str(stats)}\n")
                else:
                    backtest_results_text_widget.insert(tk.END, f"No statistics returned for {sym}.\n")

                if plot_err:
                    backtest_results_text_widget.insert(tk.END, f"Plotting Error for {sym}: {plot_err}\n")
                # Plot window for each symbol would have opened via execute_backtest if successful

                backtest_results_text_widget.insert(tk.END, "\n---\n\n") # Separator
                backtest_results_text_widget.see(tk.END)
                backtest_results_text_widget.config(state=tk.DISABLED)

            root.after(0, update_ui_for_symbol_result, symbol_item, stats_output, plot_error_message)
        
        if isinstance(stats_output, str): # If execute_backtest returned an error string
            all_symbols_completed = False # Mark that at least one symbol failed
            print(f"Error during backtest for {symbol_item}: {stats_output}")
        
        sleep(1) # Small delay between symbols if needed, e.g., for API rate limits or UI responsiveness

    # Final status update after all symbols are processed
    if root and root.winfo_exists():
        final_msg = "All symbol backtests completed." if all_symbols_completed else "Backtests completed with some errors."
        root.after(0, lambda: status_var.set(f"Backtest: {final_msg}"))
        if backtest_run_button:
            root.after(0, lambda: backtest_run_button.config(state=tk.NORMAL))

# Original perform_backtest_in_thread is no longer directly called by run_backtest_command's thread.
# It is effectively replaced by perform_backtest_for_multiple_symbols which calls execute_backtest in a loop.
# We can remove or comment out the old perform_backtest_in_thread if it's no longer used.
# For now, let's comment it out to avoid confusion.

# def perform_backtest_in_thread(strategy_id, symbol, timeframe, interval_days, 
#                                tp_percentage_val, sl_percentage_val, # Explicitly named percentage params
#                                passed_sl_tp_mode, sl_pnl_val, tp_pnl_val, # Explicitly named mode and PnL params
#                                starting_capital_val, leverage_val): # Added leverage_val
#     global backtest_results_text_widget, root, status_var, backtest_run_button
    
#     print(f"DEBUG perform_backtest_in_thread: Strategy ID: {strategy_id}, Symbol: {symbol}, Timeframe: {timeframe}, Interval: {interval_days} days, Start Capital: ${starting_capital_val:.2f}, Leverage: {leverage_val}x") 
#     print(f"DEBUG perform_backtest_in_thread: SL/TP Mode: {passed_sl_tp_mode}")
#     if passed_sl_tp_mode == "Percentage":
#         print(f"DEBUG perform_backtest_in_thread: TP %: {tp_percentage_val*100:.2f}%, SL %: {sl_percentage_val*100:.2f}%")
#     elif passed_sl_tp_mode == "Fixed PnL":
#         print(f"DEBUG perform_backtest_in_thread: SL PnL: ${sl_pnl_val:.2f}, TP PnL: ${tp_pnl_val:.2f}")
    
#     if status_var and root and root.winfo_exists():
#         root.after(0, lambda: status_var.set(f"Backtest: Fetching kline data for {symbol}...")) # Modified for symbol

#     stats_output, bt_object, plot_error_message = execute_backtest(
#         strategy_id, symbol, timeframe, interval_days, 
#         tp_percentage_val, sl_percentage_val, 
#         passed_sl_tp_mode, sl_pnl_val, tp_pnl_val, 
#         starting_capital_val, leverage_val 
#     )
    
#     print(f"DEBUG perform_backtest_in_thread ({symbol}): execute_backtest returned stats_output type: {type(stats_output)}")
#     if isinstance(stats_output, str): 
#         print(f"DEBUG perform_backtest_in_thread ({symbol}): execute_backtest returned error string: {stats_output}")
#     print(f"DEBUG perform_backtest_in_thread ({symbol}): execute_backtest returned plot_error_message: {plot_error_message}")
    
#     if status_var and root and root.winfo_exists():
#         if isinstance(stats_output, str) or plot_error_message : 
#              root.after(0, lambda: status_var.set(f"Backtest: Error for {symbol}.")) # Modified for symbol
#         else:
#              root.after(0, lambda: status_var.set(f"Backtest: Sim complete for {symbol}. Preparing results...")) # Modified for symbol


#     def update_ui_with_results_per_symbol(current_symbol, stats_data, plot_err_msg, bt_obj): # Added current_symbol
#         final_status_message_symbol = f"Backtest: Completed for {current_symbol}."
        
#         if backtest_results_text_widget and root and root.winfo_exists():
#             backtest_results_text_widget.config(state=tk.NORMAL)
#             # Append results, do not delete previous ones if running multiple symbols
#             backtest_results_text_widget.insert(tk.END, f"\n*** Backtest Results for {current_symbol} ***\n") # Header for symbol
            
#             if isinstance(stats_data, pd.Series):
#                 stats_str = "--------------------\n"
#                 for index, value in stats_data.items():
#                     if index in ['_equity_curve', '_trades']:
#                         continue
#                     if isinstance(value, float):
#                         stats_str += f"{index}: {value:.2f}\n"
#                     else:
#                         stats_str += f"{index}: {value}\n"
#                 backtest_results_text_widget.insert(tk.END, stats_str)
#             elif isinstance(stats_data, str): # Error message
#                 backtest_results_text_widget.insert(tk.END, f"Error: {stats_data}\n")
#             elif stats_data is not None:
#                 backtest_results_text_widget.insert(tk.END, str(stats_data) + "\n")
#             else:
#                 backtest_results_text_widget.insert(tk.END, "Backtest did not return statistics or an error message.\n")

#             if plot_err_msg:
#                 backtest_results_text_widget.insert(tk.END, f"\n{plot_err_msg}\n")
            
#             if bt_obj and not plot_err_msg:
#                 backtest_results_text_widget.insert(tk.END, f"Plot window for {current_symbol} should have opened (if data was sufficient).\n")
            
#             backtest_results_text_widget.insert(tk.END, "---\n") # Separator
#             backtest_results_text_widget.config(state=tk.DISABLED)
#             backtest_results_text_widget.see(tk.END) # Scroll to the end
#         else:
#             print(f"UI update skipped for {current_symbol}: Results widget not available.")

#         if status_var and root and root.winfo_exists():
#             # Status var will be updated by the calling loop (perform_backtest_for_multiple_symbols)
#             pass 
#         # backtest_run_button re-enabling will be handled by the outer loop function

#     if root and root.winfo_exists(): 
#         root.after(0, update_ui_with_results_per_symbol, symbol, stats_output, plot_error_message, bt_object)
#     else: 
#         print(f"UI update skipped for {symbol}: Root window not available.")
#         if isinstance(stats_output, str): print(f"Error for {symbol}: {stats_output}")
#         elif stats_output: print(f"Stats for {symbol}:\n{str(stats_output)}")


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
    5: "New RSI-Based Strategy", # New strategy added
    6: "Market Structure S/D", # New strategy for market structure and supply/demand
    7: "Candlestick Patterns" # Strategy 7
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

TARGET_SYMBOLS = ["BTCUSDT", "ETHUSDT", "XRPUSDT", "BNBUSDT", "SOLUSDT", "TRXUSDT", "DOGEUSDT", "ADAUSDT", "HYPEUSDT", "BCHUSDT", "SUIUSDT", "LINKUSDT", "LEOUSDT", "AVAXUSDT", "XLMUSDT", "TONUSDT", "SHIBUSDT", "LTCUSDT", "HBARUSDT", "XMRUSDT", "BGBUSDT", "DOTUSDT", "UNIUSDT", "PIUSDT", "AAVEUSDT", "PEPEUSDT", "APTUSDT", "OKBUSDT", "TAOUSDT", "NEARUSDT", "ICPUSDT", "CROUSDT", "ETCUSDT", "ONDOUSDT", "MNTUSDT", "KASUSDT", "GTUSDT", "POLUSDT", "TRUMPUSDT", "VETUSDT", "SKYUSDT", "SEIUSDT", "FETUSDT", "RENDERUSDT", "ENAUSDT", "ATOMUSDT", "ARBUSDT", "ALGOUSDT", "FILUSDT", "WLDUSDT", "KCSUSDT", "QNTUSDT", "JUPUSDT", "FLRUSDT"
]
ACCOUNT_RISK_PERCENT = 0.005
SCALPING_REQUIRED_BUY_CONDITIONS = 1
SCALPING_REQUIRED_SELL_CONDITIONS = 1
TP_PERCENT = 0.01
SL_PERCENT = 0.005
leverage = 5
margin_type_setting = 'ISOLATED' # Margin type
qty_concurrent_positions = 100
LOCAL_HIGH_LOW_LOOKBACK_PERIOD = 20 # New global for breakout logic
# DEFAULT_ATR_MULTIPLIER = 2.0 # No longer here

# New Global variables for PnL based SL/TP settings
SL_TP_MODE = "ATR/Dynamic" # Default SL/TP mode
SL_PNL_AMOUNT = 10.0       # Default SL PnL amount in $
TP_PNL_AMOUNT = 20.0       # Default TP PnL amount in $


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


# --- Market Structure and S/D Zone Helper Functions ---
def find_swing_points(data: pd.DataFrame, order: int = 5) -> tuple[pd.Series, pd.Series]:
    """
    Identifies swing highs and lows in the price data.
    A swing high is a price peak higher than 'order' bars on each side.
    A swing low is a price trough lower than 'order' bars on each side.
    
    Args:
        data: Pandas DataFrame with 'High' and 'Low' columns.
        order: Number of bars on each side to check for a peak/trough.
        
    Returns:
        A tuple of two boolean Pandas Series: (is_swing_high, is_swing_low)
    """
    if 'High' not in data.columns or 'Low' not in data.columns:
        raise ValueError("Dataframe must contain 'High' and 'Low' columns.")
    if not isinstance(order, int) or order <= 0:
        raise ValueError("'order' must be a positive integer.")
    if len(data) < (2 * order + 1):
        # Not enough data to find swings with the given order, return empty/false series
        false_series = pd.Series([False] * len(data), index=data.index)
        return false_series, false_series

    # Using scipy.signal.find_peaks
    try:
        from scipy.signal import find_peaks
    except ImportError:
        raise ImportError("scipy library is required for find_swing_points. Please install it.")

    high_peaks_indices, _ = find_peaks(data['High'], distance=order, width=order) # distance & width can help refine
    low_peaks_indices, _ = find_peaks(-data['Low'], distance=order, width=order) # Find peaks in negative Lows for troughs

    is_swing_high = pd.Series(False, index=data.index)
    is_swing_low = pd.Series(False, index=data.index)

    is_swing_high.iloc[high_peaks_indices] = True
    is_swing_low.iloc[low_peaks_indices] = True
    
    # Alternative/Refinement: Manual check if find_peaks is too aggressive or not quite right.
    # For now, relying on find_peaks with appropriate parameters.
    # A common manual approach:
    # is_swing_high = pd.Series(False, index=data.index)
    # is_swing_low = pd.Series(False, index=data.index)
    # for i in range(order, len(data) - order):
    #     is_high = True
    #     for j in range(1, order + 1):
    #         if data['High'].iloc[i] <= data['High'].iloc[i-j] or \
    #            data['High'].iloc[i] <= data['High'].iloc[i+j]:
    #             is_high = False
    #             break
    #     if is_high:
    #         is_swing_high.iloc[i] = True

    #     is_low = True
    #     for j in range(1, order + 1):
    #         if data['Low'].iloc[i] >= data['Low'].iloc[i-j] or \
    #            data['Low'].iloc[i] >= data['Low'].iloc[i+j]:
    #             is_low = False
    #             break
    #     if is_low:
    #         is_swing_low.iloc[i] = True
            
    return is_swing_high, is_swing_low


def identify_market_structure(data: pd.DataFrame, swing_highs_bool: pd.Series, swing_lows_bool: pd.Series) -> dict:
    """
    Identifies market structure: trend bias, valid swing highs/lows.
    
    Args:
        data: Pandas DataFrame with 'High', 'Low', 'Close' columns.
        swing_highs_bool: Boolean series indicating swing high points.
        swing_lows_bool: Boolean series indicating swing low points.
        
    Returns:
        A dictionary containing market structure information:
        {
            'trend_bias': 'bullish'/'bearish'/'ranging',
            'last_valid_high_price': float or None,
            'last_valid_low_price': float or None,
            'current_swing_high_price': float or None,
            'current_swing_low_price': float or None,
            'break_of_structure_event': 'bos_high'/'bos_low'/'none' # Indicates if the most recent event was a BoS
        }
    """
    if not all(col in data.columns for col in ['High', 'Low', 'Close']):
        raise ValueError("Dataframe must contain 'High', 'Low', 'Close' columns.")

    swing_high_prices = data['High'][swing_highs_bool]
    swing_low_prices = data['Low'][swing_lows_bool]

    # Combine and sort all swing points by time
    all_swings = []
    for idx, price in swing_high_prices.items():
        all_swings.append({'time': idx, 'type': 'high', 'price': price})
    for idx, price in swing_low_prices.items():
        all_swings.append({'time': idx, 'type': 'low', 'price': price})
    
    all_swings.sort(key=lambda x: x['time'])

    if not all_swings:
        return {
            'trend_bias': 'ranging', 'last_valid_high_price': None, 'last_valid_low_price': None,
            'current_swing_high_price': None, 'current_swing_low_price': None, 'break_of_structure_event': 'none'
        }

    trend_bias = 'ranging' # Initial assumption
    last_confirmed_high = None
    last_confirmed_low = None
    
    # These will store the price of the latest confirmed valid swing points
    last_valid_high_price = None
    last_valid_low_price = None

    # These track the most recent swing points, regardless of validity for trend change
    current_swing_high_price = swing_high_prices.iloc[-1] if not swing_high_prices.empty else None
    current_swing_low_price = swing_low_prices.iloc[-1] if not swing_low_prices.empty else None
    
    break_of_structure_event = 'none' # Tracks the most recent BoS event

    # Simplified logic: Iterate through swings to determine current state.
    # A more robust implementation might need to track sequences of HH/HL or LL/LH.
    # This version focuses on the "valid swing break" rule.

    # Initialize with the first two swings to establish an initial context if possible
    if len(all_swings) >= 2:
        if all_swings[0]['type'] == 'low' and all_swings[1]['type'] == 'high':
            last_confirmed_low = all_swings[0]
            last_confirmed_high = all_swings[1]
            if last_confirmed_high['price'] > last_confirmed_low['price']: # Basic check
                # Potentially start bullish if high is higher than low.
                # For trend bias, we need a break. Let's assume ranging until a valid break.
                 pass # last_valid_low_price = last_confirmed_low['price']
        elif all_swings[0]['type'] == 'high' and all_swings[1]['type'] == 'low':
            last_confirmed_high = all_swings[0]
            last_confirmed_low = all_swings[1]
            if last_confirmed_low['price'] < last_confirmed_high['price']:
                # Potentially start bearish.
                pass # last_valid_high_price = last_confirmed_high['price']
    elif len(all_swings) == 1: # Only one swing point
        if all_swings[0]['type'] == 'high': last_confirmed_high = all_swings[0]
        else: last_confirmed_low = all_swings[0]


    # Iterate through swings to establish current state based on breaks
    # This loop sets the *initial* valid high/low and trend based on historical breaks.
    temp_last_valid_high = None
    temp_last_valid_low = None
    
    # Pass 1: Establish initial valid swings and trend bias from history
    # This part needs careful state management.
    # Let's simplify: the "last_valid_high/low" are the PIVOTAL swings.
    # An UPTREND is HH and HL. A DOWNTREND is LL and LH.
    # A swing low becomes "valid" (part of uptrend confirmation) when price breaks the PREVIOUS swing high.
    # A swing high becomes "valid" (part of downtrend confirmation) when price breaks the PREVIOUS swing low.
    
    # The user's rule: "Only switch trend bias when a valid swing is broken."
    # This means we need to identify the "valid swing" that, if broken, changes the trend.
    # In an uptrend, the "last valid swing" to watch is the last Higher Low (HL). If broken -> trend change.
    # In a downtrend, the "last valid swing" to watch is the last Lower High (LH). If broken -> trend change.

    # Simplified approach for current state:
    # Iterate backwards from the most recent swings to find the last confirmed structure.
    
    idx = len(all_swings) - 1
    # Initialize last_valid_high/low_price from the latest swings if available
    if current_swing_high_price: temp_last_valid_high = {'price': current_swing_high_price, 'time': swing_high_prices.index[-1]}
    if current_swing_low_price: temp_last_valid_low = {'price': current_swing_low_price, 'time': swing_low_prices.index[-1]}

    # Backwards iteration to find the most recent structure
    # These will store the price AND TIME of the latest confirmed valid swing points
    # last_valid_high/low_price are just the prices from these.
    confirmed_valid_high_point = None # {'price': float, 'time': pd.Timestamp}
    confirmed_valid_low_point = None  # {'price': float, 'time': pd.Timestamp}

    # Store most recent swing high/low encountered, irrespective of confirmed validity for trend
    latest_swing_high_point = None # {'price': float, 'time': pd.Timestamp}
    latest_swing_low_point = None  # {'price': float, 'time': pd.Timestamp}

    # For state transitions and pending confirmations
    # A potential HL that needs confirmation (break of prior high) to become a valid_low_point in an uptrend
    pending_hl_confirmation = None # {'price': float, 'time': pd.Timestamp}
    # A potential LH that needs confirmation (break of prior low) to become a valid_high_point in a downtrend
    pending_lh_confirmation = None # {'price': float, 'time': pd.Timestamp}

    # Iterate through all_swings to determine structure
    for i, current_swing in enumerate(all_swings):
        current_price_at_swing_time = data['Close'].asof(current_swing['time'])

        if current_swing['type'] == 'high':
            latest_swing_high_point = current_swing
            
            if trend_bias == 'bullish':
                if confirmed_valid_high_point and current_swing['price'] > confirmed_valid_high_point['price']: # Higher High (HH)
                    if pending_hl_confirmation: # This HH confirms the pending_hl_confirmation as a valid low
                        confirmed_valid_low_point = pending_hl_confirmation
                        last_valid_low_price = confirmed_valid_low_point['price']
                        pending_hl_confirmation = None # Clear pending HL
                    confirmed_valid_high_point = current_swing # Update to new HH
                    last_valid_high_price = confirmed_valid_high_point['price']
                    break_of_structure_event = 'bos_high' # Continued bullish BoS
                else: # Failed to make a HH or no prior valid high
                    # This current_swing high could be a lower high (LH) if the trend is to change.
                    # We don't change trend yet, but mark this as a potential LH.
                    pending_lh_confirmation = current_swing
                    # If confirmed_valid_low_point exists and current price < confirmed_valid_low_point['price']
                    # This check is complex because current_swing is a high. A break of low happens with price action, not just a swing.
                    # The rule is "switch trend bias when a VALID swing is broken".
                    # For bullish, the valid swing is confirmed_valid_low_point (the last HL).
                    # This check should happen when processing price action against confirmed_valid_low_point.
                    # Let's refine: if price (e.g. data['Close']) between this swing and next swing breaks confirmed_valid_low_point.
                    # Simplified: if we are at a high, and later a low breaks the confirmed_valid_low_point.
                    pass # Keep trend_bias, this high is just a swing high for now.

            elif trend_bias == 'bearish':
                # This is a potential Lower High (LH)
                if confirmed_valid_high_point is None or current_swing['price'] < confirmed_valid_high_point['price']:
                    pending_lh_confirmation = current_swing 
                # If current_swing['price'] > confirmed_valid_high_point['price'] (Break of valid LH)
                # This is a change of character (CHoCH) / potential BoS upwards.
                # Check if price *after* this high breaks this current_swing high.
                # This high (current_swing) could become the first HH of a new uptrend if followed by a HL and then a break of this high.
                # For now, if it breaks the *confirmed_valid_high_point* (last LH of downtrend):
                elif confirmed_valid_high_point and current_swing['price'] > confirmed_valid_high_point['price']:
                    trend_bias = 'bullish'
                    break_of_structure_event = 'bos_high' # Trend changed to bullish
                    confirmed_valid_high_point = current_swing # This is the first HH
                    last_valid_high_price = confirmed_valid_high_point['price']
                    if pending_hl_confirmation: # The low before this break becomes the first HL
                        confirmed_valid_low_point = pending_hl_confirmation
                        last_valid_low_price = confirmed_valid_low_point['price']
                    else: # If no clear pending_hl, use the latest_swing_low_point before this break
                        # Find the latest_swing_low_point that occurred before current_swing
                        prev_lows = [s for s in all_swings[:i] if s['type'] == 'low']
                        if prev_lows:
                           confirmed_valid_low_point = prev_lows[-1]
                           last_valid_low_price = confirmed_valid_low_point['price']
                        else: # Should ideally not happen if swings alternate
                           confirmed_valid_low_point = None; last_valid_low_price = None
                    pending_lh_confirmation = None
                    pending_hl_confirmation = None


            elif trend_bias == 'ranging':
                if latest_swing_low_point and current_swing['price'] > latest_swing_low_point['price']:
                    # Initial move up, potential start of uptrend if this high is broken later by price
                    confirmed_valid_high_point = current_swing # Tentative first high
                    last_valid_high_price = current_swing['price']
                    # The latest_swing_low_point becomes a candidate for the first HL
                    pending_hl_confirmation = latest_swing_low_point
                    # Don't change trend_bias to 'bullish' yet, wait for HH and HL confirmation (break of this current_swing['price'])
                    trend_bias = 'bullish' # Tentatively bullish after first H and L sequence
                    break_of_structure_event = 'bos_high'


        elif current_swing['type'] == 'low':
            latest_swing_low_point = current_swing

            if trend_bias == 'bearish':
                if confirmed_valid_low_point and current_swing['price'] < confirmed_valid_low_point['price']: # Lower Low (LL)
                    if pending_lh_confirmation: # This LL confirms the pending_lh_confirmation as a valid high
                        confirmed_valid_high_point = pending_lh_confirmation
                        last_valid_high_price = confirmed_valid_high_point['price']
                        pending_lh_confirmation = None
                    confirmed_valid_low_point = current_swing # Update to new LL
                    last_valid_low_price = confirmed_valid_low_point['price']
                    break_of_structure_event = 'bos_low' # Continued bearish BoS
                else: # Failed to make LL
                    pending_hl_confirmation = current_swing
                    pass # Keep trend_bias

            elif trend_bias == 'bullish':
                # This is a potential Higher Low (HL)
                if confirmed_valid_low_point is None or current_swing['price'] > confirmed_valid_low_point['price']:
                     pending_hl_confirmation = current_swing
                # If current_swing['price'] < confirmed_valid_low_point['price'] (Break of valid HL)
                # This is a CHoCH / BoS downwards
                elif confirmed_valid_low_point and current_swing['price'] < confirmed_valid_low_point['price']:
                    trend_bias = 'bearish'
                    break_of_structure_event = 'bos_low' # Trend changed to bearish
                    confirmed_valid_low_point = current_swing # This is the first LL
                    last_valid_low_price = confirmed_valid_low_point['price']
                    if pending_lh_confirmation: # The high before this break becomes the first LH
                        confirmed_valid_high_point = pending_lh_confirmation
                        last_valid_high_price = confirmed_valid_high_point['price']
                    else: # Find latest_swing_high_point before this break
                        prev_highs = [s for s in all_swings[:i] if s['type'] == 'high']
                        if prev_highs:
                            confirmed_valid_high_point = prev_highs[-1]
                            last_valid_high_price = confirmed_valid_high_point['price']
                        else:
                            confirmed_valid_high_point = None; last_valid_high_price = None
                    pending_hl_confirmation = None
                    pending_lh_confirmation = None
            
            elif trend_bias == 'ranging':
                if latest_swing_high_point and current_swing['price'] < latest_swing_high_point['price']:
                    # Initial move down, potential start of downtrend
                    confirmed_valid_low_point = current_swing # Tentative first low
                    last_valid_low_price = current_swing['price']
                    pending_lh_confirmation = latest_swing_high_point
                    trend_bias = 'bearish' # Tentatively bearish
                    break_of_structure_event = 'bos_low'

    # Update current_swing_high/low_price based on the absolute latest swings found
    if not swing_high_prices.empty: current_swing_high_price = swing_high_prices.iloc[-1]
    else: current_swing_high_price = None
    if not swing_low_prices.empty: current_swing_low_price = swing_low_prices.iloc[-1]
    else: current_swing_low_price = None
    
    # If still ranging at the end, use latest swings as "valid" for S/D context
    if trend_bias == 'ranging':
        last_valid_high_price = current_swing_high_price
        last_valid_low_price = current_swing_low_price

    return {
        'trend_bias': trend_bias,
        'last_valid_high_price': last_valid_high_price,
        'last_valid_low_price': last_valid_low_price,
        'current_swing_high_price': current_swing_high_price, 
        'current_swing_low_price': current_swing_low_price,   
        'break_of_structure_event': break_of_structure_event
    }

def identify_supply_demand_zones(data: pd.DataFrame, atr_period: int = 14, lookback_candles: int = 10, consolidation_atr_factor: float = 0.7, sharp_move_atr_factor: float = 1.5) -> list[dict]:
    """
    Identifies potential Supply and Demand zones.
    Looks for consolidation (low volatility) followed by a sharp price move.

    Args:
        data: Pandas DataFrame with 'High', 'Low', 'Close', 'Open' columns.
        atr_period: Period for ATR calculation.
        lookback_candles: Number of candles to check for consolidation before a sharp move.
        consolidation_atr_factor: A factor of ATR. If avg range of 'lookback_candles' < this*ATR, it's consolidation.
        sharp_move_atr_factor: A factor of ATR. If a candle's range > this*ATR, it's a sharp move.

    Returns:
        A list of dictionaries, each representing a zone:
        {'type': 'demand'/'supply', 'price_start': float, 'price_end': float, 
         'timestamp_start': pd.Timestamp, 'timestamp_end': pd.Timestamp, 'trigger_candle_timestamp': pd.Timestamp}
    """
    if not all(col in data.columns for col in ['High', 'Low', 'Close', 'Open']):
        raise ValueError("Dataframe must contain 'High', 'Low', 'Close', 'Open' columns.")
    if len(data) < atr_period + lookback_candles + 1:
        return [] # Not enough data

    try:
        from talib import ATR as talib_ATR # Using talib for ATR if available
        atr = talib_ATR(data['High'], data['Low'], data['Close'], timeperiod=atr_period)
    except ImportError:
        # Fallback to pandas_ta or manual calculation if TA-Lib not available
        try:
            if 'pta' not in globals(): import pandas_ta as pta # Ensure pta is imported
            atr_series_pta = pta.atr(high=data['High'], low=data['Low'], close=data['Close'], length=atr_period)
            if atr_series_pta is None: raise ValueError("pandas_ta.atr returned None")
            atr = atr_series_pta
        except Exception as e_pta:
            print(f"TA-Lib not found, pandas_ta ATR failed ({e_pta}). S/D zones might be less accurate. Manual ATR fallback not implemented here for brevity.")
            return []


    zones = []
    # Iterate from (atr_period + lookback_candles) up to the second to last candle (to have a candle after the zone)
    for i in range(atr_period + lookback_candles, len(data) - 1):
        current_atr = atr.iloc[i-1] # ATR of the candle just before the potential sharp move
        if pd.isna(current_atr) or current_atr == 0:
            continue

        # 1. Check for consolidation in the 'lookback_candles' period BEFORE the current candle 'i'
        consolidation_data = data.iloc[i - lookback_candles : i]
        avg_candle_range_consolidation = (consolidation_data['High'] - consolidation_data['Low']).mean()

        if avg_candle_range_consolidation < (consolidation_atr_factor * current_atr):
            # Consolidation detected. Now check for a sharp move at candle 'i'
            sharp_move_candle = data.iloc[i]
            sharp_move_range = sharp_move_candle['High'] - sharp_move_candle['Low']
            sharp_move_body = abs(sharp_move_candle['Close'] - sharp_move_candle['Open'])

            if sharp_move_range > (sharp_move_atr_factor * current_atr) or \
               sharp_move_body > (sharp_move_atr_factor * current_atr * 0.7): # Consider body too

                zone_start_price = consolidation_data['Low'].min()
                zone_end_price = consolidation_data['High'].max()
                zone_type = None

                if sharp_move_candle['Close'] > sharp_move_candle['Open']: # Bullish sharp move
                    zone_type = 'demand'
                    # For demand, the zone is the consolidation area prices *before* the up-move.
                    # Prices: low of consolidation to high of consolidation.
                elif sharp_move_candle['Close'] < sharp_move_candle['Open']: # Bearish sharp move
                    zone_type = 'supply'
                    # For supply, similar concept.
                
                if zone_type:
                    # Ensure zone has some height
                    if zone_end_price > zone_start_price:
                        zones.append({
                            'type': zone_type,
                            'price_start': zone_start_price, # Bottom of the zone
                            'price_end': zone_end_price,     # Top of the zone
                            'timestamp_start': consolidation_data.index[0],
                            'timestamp_end': consolidation_data.index[-1],
                            'trigger_candle_timestamp': sharp_move_candle.name 
                        })
    
    # Optional: Merge overlapping zones of the same type
    # This is a simplified version and does not merge.
    return zones


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
        
        if return_data['signal'] != 'none':
            if SL_TP_MODE == "ATR/Dynamic": # Only calculate SL/TP here if mode is ATR/Dynamic
                try:
                    entry_price = kl['Close'].iloc[-1]
                    # Pass the global DEFAULT_ATR_MULTIPLIER and strategy's RR if applicable, or use defaults
                    # For S0, RR is not explicitly defined, so calculate_dynamic_sl_tp will use its default RR.
                    # We could add an RR to S0's base_return if we want strategy-specific RR for ATR mode.
                    # For now, using default RR from calculate_dynamic_sl_tp.
                    # S0 also doesn't have a specific ATR multiplier in its definition, so default is used.
                    dynamic_sl_tp_result = calculate_dynamic_sl_tp(
                        symbol=symbol,
                        entry_price=entry_price,
                        side=return_data['signal'] 
                        # atr_multiplier and rr will use defaults in calculate_dynamic_sl_tp
                    )

                    if dynamic_sl_tp_result['error']:
                        error_msg = f"S0 ATR SL/TP calc error ({symbol}, {return_data['signal']}): {dynamic_sl_tp_result['error']}"
                        print(error_msg)
                        # Invalidate signal if SL/TP calculation fails in ATR mode
                        return_data['signal'] = 'none' 
                        if return_data['error'] is None: return_data['error'] = error_msg
                        else: return_data['error'] += f"; {error_msg}"
                    else:
                        return_data['sl_price'] = dynamic_sl_tp_result['sl_price']
                        return_data['tp_price'] = dynamic_sl_tp_result['tp_price']
                        if return_data['sl_price'] is None or return_data['tp_price'] is None:
                            error_msg = f"S0 ATR SL/TP ({symbol}, {return_data['signal']}) resulted in None. Orig. err: {dynamic_sl_tp_result.get('error', 'N/A')}"
                            print(error_msg)
                            return_data['signal'] = 'none' # Invalidate signal
                            if return_data['error'] is None: return_data['error'] = error_msg
                            else: return_data['error'] += f"; {error_msg}"
                except Exception as e:
                    error_msg = f"S0 Exception during ATR SL/TP integration ({symbol}): {str(e)}"
                    print(error_msg)
                    return_data['signal'] = 'none' # Invalidate signal
                    if return_data['error'] is None: return_data['error'] = error_msg
                    else: return_data['error'] += f"; {error_msg}"
                    return_data['sl_price'] = None # Ensure these are None on such error
                    return_data['tp_price'] = None
            else: # For "Percentage" or "Fixed PnL" modes
                return_data['sl_price'] = None # SL/TP calculation deferred to open_order
                return_data['tp_price'] = None
                print(f"Strategy {STRATEGIES.get(0, 'S0')} for {symbol}: SL/TP calculation deferred to open_order (Mode: {SL_TP_MODE}).")
        
        # If signal was invalidated due to SL/TP issues in ATR mode, ensure error is propagated if not already set
        if return_data['signal'] == 'none' and return_data['error'] is None and SL_TP_MODE == "ATR/Dynamic":
            # This case might occur if internal validation in calculate_dynamic_sl_tp (e.g. SL > entry) nullifies SL/TP
            # but doesn't set an explicit error string in its return, and our logic above sets signal to 'none'.
            # To be safe, we add a generic error if signal is none after ATR processing without a specific error.
            # However, calculate_dynamic_sl_tp is designed to return an error string in such cases.
            # This is more of a defensive check.
            pass


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

        final_signal_str = 'none' 
        calculated_sl, calculated_tp = None, None # For SL/TP values from dynamic function

        # Determine potential signal based on conditions first
        if met_buy_conditions >= 2: # Changed from == base_return['conditions_for_full_signal_threshold']
            final_signal_str = 'up'
            base_return['conditions_met_count'] = met_buy_conditions
        elif met_sell_conditions >= 2: # Changed from == base_return['conditions_for_full_signal_threshold']
            final_signal_str = 'down'
            base_return['conditions_met_count'] = met_sell_conditions
        else: # Not a full signal, determine conditions_met_count for partial/wait check
            if met_buy_conditions >= met_sell_conditions:
                 base_return['conditions_met_count'] = met_buy_conditions
            else:
                 base_return['conditions_met_count'] = met_sell_conditions
            # No signal, so SL/TP remains None, error remains None unless set by pre-checks
            base_return['signal'] = 'none'
            return base_return # Exit early if no initial signal based on met conditions

        # If a signal ('up' or 'down') is determined, proceed to calculate SL/TP based on mode
        if final_signal_str != 'none':
            if SL_TP_MODE == "ATR/Dynamic":
                entry_price = current_price
                # S1 has RR defined in its BacktestStrategyWrapper, but not directly in its signal function.
                # It uses default RR from calculate_dynamic_sl_tp. ATR multiplier is also default.
                dynamic_sl_tp_result = calculate_dynamic_sl_tp(symbol, entry_price, final_signal_str)

                if dynamic_sl_tp_result['error']:
                    error_msg = f"S1 ATR SL/TP Error ({symbol}, {final_signal_str}): {dynamic_sl_tp_result['error']}"
                    print(error_msg)
                    final_signal_str = 'none' 
                    if base_return['error'] is None: base_return['error'] = error_msg
                    else: base_return['error'] += f"; {error_msg}"
                else:
                    calculated_sl = dynamic_sl_tp_result['sl_price']
                    calculated_tp = dynamic_sl_tp_result['tp_price']

                    if calculated_sl is None or calculated_tp is None:
                        error_msg = f"S1 ATR SL/TP calc ({symbol}, {final_signal_str}) returned None. Orig. err: {dynamic_sl_tp_result.get('error', 'N/A')}"
                        print(error_msg)
                        final_signal_str = 'none'
                        if base_return['error'] is None: base_return['error'] = error_msg
                        else: base_return['error'] += f"; {error_msg}"
                    # Validation of SL/TP vs entry price is handled by calculate_dynamic_sl_tp,
                    # which will set its own error and return None for sl/tp if invalid.
                    # So, we primarily check if they are None here.
                
                if final_signal_str != 'none':
                    base_return['sl_price'] = calculated_sl
                    base_return['tp_price'] = calculated_tp
                else: 
                    base_return['sl_price'] = None
                    base_return['tp_price'] = None
                    if not base_return['error']: base_return['error'] = f"S1 signal for {symbol} invalidated during ATR SL/TP processing."
                    print(f"Strategy {STRATEGIES.get(1, 'S1')} for {symbol} (ATR Mode): Signal invalidated. Error: {base_return['error']}")
            
            else: # "Percentage" or "Fixed PnL" mode
                base_return['sl_price'] = None
                base_return['tp_price'] = None
                print(f"Strategy {STRATEGIES.get(1, 'S1')} for {symbol}: SL/TP calculation deferred to open_order (Mode: {SL_TP_MODE}).")

        base_return['signal'] = final_signal_str
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
        calculated_sl, calculated_tp = None, None

        if met_buy_conditions >= 2: # Changed from == base_return['conditions_for_full_signal_threshold']
            final_signal_str = 'up'
            base_return['conditions_met_count'] = met_buy_conditions
        elif met_sell_conditions >= 2: # Changed from == base_return['conditions_for_full_signal_threshold']
            final_signal_str = 'down'
            base_return['conditions_met_count'] = met_sell_conditions
        else: # Not a full signal
            if met_buy_conditions >= met_sell_conditions: 
                 base_return['conditions_met_count'] = met_buy_conditions
            else:
                 base_return['conditions_met_count'] = met_sell_conditions
            base_return['signal'] = 'none' # Ensure signal is none
            return base_return # Exit early if no initial signal

        # If a signal ('up' or 'down') is determined, proceed to calculate SL/TP based on mode
        if final_signal_str != 'none':
            if SL_TP_MODE == "ATR/Dynamic":
                entry_price = current_price
                # S2 specifics for calculate_dynamic_sl_tp (if any, like RR or ATR Multiplier)
                # Using defaults from calculate_dynamic_sl_tp for now.
                dynamic_sl_tp_result = calculate_dynamic_sl_tp(symbol, entry_price, final_signal_str)

                if dynamic_sl_tp_result['error']:
                    error_msg = f"S2 ATR SL/TP Error ({symbol}, {final_signal_str}): {dynamic_sl_tp_result['error']}"
                    print(error_msg)
                    final_signal_str = 'none'
                    if base_return['error'] is None: base_return['error'] = error_msg
                    else: base_return['error'] += f"; {error_msg}"
                else:
                    calculated_sl = dynamic_sl_tp_result['sl_price']
                    calculated_tp = dynamic_sl_tp_result['tp_price']
                    if calculated_sl is None or calculated_tp is None:
                        error_msg = f"S2 ATR SL/TP calc ({symbol}, {final_signal_str}) returned None. Orig. err: {dynamic_sl_tp_result.get('error', 'N/A')}"
                        print(error_msg)
                        final_signal_str = 'none'
                        if base_return['error'] is None: base_return['error'] = error_msg
                        else: base_return['error'] += f"; {error_msg}"

                if final_signal_str != 'none':
                    base_return['sl_price'] = calculated_sl
                    base_return['tp_price'] = calculated_tp
                else: 
                    base_return['sl_price'] = None
                    base_return['tp_price'] = None
                    if not base_return['error']: base_return['error'] = f"S2 signal for {symbol} invalidated during ATR SL/TP processing."
                    print(f"Strategy {STRATEGIES.get(2, 'S2')} for {symbol} (ATR Mode): Signal invalidated. Error: {base_return['error']}")
            else: # "Percentage" or "Fixed PnL"
                base_return['sl_price'] = None
                base_return['tp_price'] = None
                print(f"Strategy {STRATEGIES.get(2, 'S2')} for {symbol}: SL/TP calculation deferred to open_order (Mode: {SL_TP_MODE}).")
        
        base_return['signal'] = final_signal_str
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
        calculated_sl, calculated_tp = None, None

        if met_buy_conditions >= 2: # Changed from == base_return['conditions_for_full_signal_threshold']
            final_signal_str = 'up'
            base_return['conditions_met_count'] = met_buy_conditions
        elif met_sell_conditions >= 2: # Changed from == base_return['conditions_for_full_signal_threshold']
            final_signal_str = 'down'
            base_return['conditions_met_count'] = met_sell_conditions
        else: # Not a full signal
            if met_buy_conditions >= met_sell_conditions:
                 base_return['conditions_met_count'] = met_buy_conditions
            else:
                 base_return['conditions_met_count'] = met_sell_conditions
            base_return['signal'] = 'none' # Ensure signal is none
            return base_return # Exit early if no initial signal

        # If a signal ('up' or 'down') is determined, proceed to calculate SL/TP based on mode
        if final_signal_str != 'none':
            if SL_TP_MODE == "ATR/Dynamic":
                entry_price = current_price
                # S3 specifics for calculate_dynamic_sl_tp (if any)
                # Using defaults from calculate_dynamic_sl_tp for now.
                dynamic_sl_tp_result = calculate_dynamic_sl_tp(symbol, entry_price, final_signal_str)

                if dynamic_sl_tp_result['error']:
                    error_msg = f"S3 ATR SL/TP Error ({symbol}, {final_signal_str}): {dynamic_sl_tp_result['error']}"
                    print(error_msg)
                    final_signal_str = 'none'
                    if base_return['error'] is None: base_return['error'] = error_msg
                    else: base_return['error'] += f"; {error_msg}"
                else:
                    calculated_sl = dynamic_sl_tp_result['sl_price']
                    calculated_tp = dynamic_sl_tp_result['tp_price']
                    if calculated_sl is None or calculated_tp is None:
                        error_msg = f"S3 ATR SL/TP calc ({symbol}, {final_signal_str}) returned None. Orig. err: {dynamic_sl_tp_result.get('error', 'N/A')}"
                        print(error_msg)
                        final_signal_str = 'none'
                        if base_return['error'] is None: base_return['error'] = error_msg
                        else: base_return['error'] += f"; {error_msg}"
                
                if final_signal_str != 'none':
                    base_return['sl_price'] = calculated_sl
                    base_return['tp_price'] = calculated_tp
                else: 
                    base_return['sl_price'] = None
                    base_return['tp_price'] = None
                    if not base_return['error']: base_return['error'] = f"S3 signal for {symbol} invalidated during ATR SL/TP processing."
                    print(f"Strategy {STRATEGIES.get(3, 'S3')} for {symbol} (ATR Mode): Signal invalidated. Error: {base_return['error']}")
            else: # "Percentage" or "Fixed PnL"
                base_return['sl_price'] = None
                base_return['tp_price'] = None
                print(f"Strategy {STRATEGIES.get(3, 'S3')} for {symbol}: SL/TP calculation deferred to open_order (Mode: {SL_TP_MODE}).")
        
        base_return['signal'] = final_signal_str
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
        
        # SL/TP Calculation based on SL_TP_MODE
        if final_signal != 'none':
            if SL_TP_MODE == "ATR/Dynamic":
                entry_price = current_price
                # S4 specifics for calculate_dynamic_sl_tp (if any)
                # Using defaults from calculate_dynamic_sl_tp for now.
                dynamic_sl_tp_result = calculate_dynamic_sl_tp(symbol, entry_price, final_signal)
                
                if dynamic_sl_tp_result['error']:
                    error_msg = f"S4 ATR SL/TP Error ({symbol}, {final_signal}): {dynamic_sl_tp_result['error']}"
                    print(error_msg)
                    final_signal = 'none'
                    if base_return['error'] is None: base_return['error'] = error_msg
                    else: base_return['error'] += f"; {error_msg}"
                else:
                    calculated_sl = dynamic_sl_tp_result['sl_price']
                    calculated_tp = dynamic_sl_tp_result['tp_price']
                    if calculated_sl is None or calculated_tp is None:
                        error_msg = f"S4 ATR SL/TP calc ({symbol}, {final_signal}) returned None. Orig. err: {dynamic_sl_tp_result.get('error', 'N/A')}"
                        print(error_msg)
                        final_signal = 'none'
                        if base_return['error'] is None: base_return['error'] = error_msg
                        else: base_return['error'] += f"; {error_msg}"

                if final_signal != 'none':
                    base_return['sl_price'] = calculated_sl
                    base_return['tp_price'] = calculated_tp
                else: 
                    base_return['sl_price'] = None
                    base_return['tp_price'] = None
                    if not base_return['error']: base_return['error'] = f"S4 signal for {symbol} invalidated during ATR SL/TP processing."
                    print(f"Strategy {STRATEGIES.get(4, 'S4')} for {symbol} (ATR Mode): Signal invalidated. Error: {base_return['error']}")
            else: # "Percentage" or "Fixed PnL"
                base_return['sl_price'] = None
                base_return['tp_price'] = None
                print(f"Strategy {STRATEGIES.get(4, 'S4')} for {symbol}: SL/TP calculation deferred to open_order (Mode: {SL_TP_MODE}).")

        base_return['signal'] = final_signal
        # divergence_price_point already set when divergence found
        
        return base_return

    except Exception as e:
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

        # Initialize SL/TP in base_return to None before any calculation attempt
        base_return['sl_price'] = None
        base_return['tp_price'] = None
        calculated_sl = None # To store results from dynamic_sl_tp
        calculated_tp = None # To store results from dynamic_sl_tp
        
        # Determine initial signal based on conditions
        if num_core_buy_conditions_met >= 2: 
            base_return['signal'] = 'up'
            base_return['conditions_met_count'] = num_core_buy_conditions_met
        elif num_core_sell_conditions_met >= 2: 
            base_return['signal'] = 'down'
            base_return['conditions_met_count'] = num_core_sell_conditions_met
        else: 
            base_return['conditions_met_count'] = max(num_core_buy_conditions_met, num_core_sell_conditions_met)
            base_return['signal'] = 'none'

        # If a signal is determined, calculate SL/TP based on mode
        if base_return['signal'] != 'none':
            if SL_TP_MODE == "ATR/Dynamic":
                entry_price = current_price
                # S5 specifics for calculate_dynamic_sl_tp (if any)
                # Using defaults from calculate_dynamic_sl_tp for now.
                dynamic_sl_tp_result = calculate_dynamic_sl_tp(symbol, entry_price, base_return['signal'])

                if dynamic_sl_tp_result['error']:
                    error_msg = f"S5 ATR SL/TP Error ({symbol}, {base_return['signal']}): {dynamic_sl_tp_result['error']}"
                    print(error_msg)
                    base_return['signal'] = 'none'
                    if base_return['error'] is None: base_return['error'] = error_msg
                    else: base_return['error'] += f"; {error_msg}"
                else:
                    calculated_sl = dynamic_sl_tp_result['sl_price']
                    calculated_tp = dynamic_sl_tp_result['tp_price']
                    if calculated_sl is None or calculated_tp is None:
                        error_msg = f"S5 ATR SL/TP calc ({symbol}, {base_return['signal']}) returned None. Orig. err: {dynamic_sl_tp_result.get('error', 'N/A')}"
                        print(error_msg)
                        base_return['signal'] = 'none'
                        if base_return['error'] is None: base_return['error'] = error_msg
                        else: base_return['error'] += f"; {error_msg}"
                
                if base_return['signal'] != 'none':
                    base_return['sl_price'] = calculated_sl
                    base_return['tp_price'] = calculated_tp
                else: 
                    base_return['sl_price'] = None
                    base_return['tp_price'] = None
                    if not base_return['error']: base_return['error'] = f"S5 signal for {symbol} invalidated during ATR SL/TP processing."
            else: # "Percentage" or "Fixed PnL"
                base_return['sl_price'] = None
                base_return['tp_price'] = None
                print(f"Strategy {STRATEGIES.get(5, 'S5')} for {symbol}: SL/TP calculation deferred to open_order (Mode: {SL_TP_MODE}).")

        if base_return['error'] and base_return['signal'] == 'none': 
             print(f"Strategy {STRATEGIES.get(5, 'S5')} for {symbol}: Signal invalidated or error occurred: {base_return['error']}") # Ensure error is logged if signal becomes none
        
    except IndexError as ie:
        base_return['error'] = f"S5 IndexError for {symbol}: {str(ie)}. RSI len: {len(rsi_series) if 'rsi_series' in locals() and rsi_series is not None else 'N/A'}, KL len: {len(kl) if 'kl' in locals() and kl is not None else 'N/A'}"
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

# --- Strategy 6: Market Structure S/D ---
def strategy_market_structure_sd(symbol: str) -> dict:
    """
    Strategy based on Market Structure (HH, HL, LL, LH), Supply/Demand zones,
    and a minimum Risk-Reward ratio of 2.5:1.
    """
    base_return = {
        'signal': 'none', # 'up', 'down', or 'none'
        'conditions_met_count': 0, 
        'conditions_to_start_wait_threshold': 1, 
        'conditions_for_full_signal_threshold': 1, 
        'all_conditions_status': { 
            'trend_bias': 'N/A',
            'zone_type_found': 'N/A', 
            'price_in_zone': False,
            'rr_ok': False,
            'last_valid_low': None,
            'last_valid_high': None,
            'current_sd_zone_start': None,
            'current_sd_zone_end': None,
        },
        'sl_price': None,
        'tp_price': None,
        'entry_price_estimate': None, 
        'error': None,
        'account_risk_percent': 0.01 
    }

    min_klines_needed = 60
    s6_log_prefix = f"[S6 LOG {symbol}]"
    print(f"{s6_log_prefix} Initiating strategy_market_structure_sd.")
    kl_df = klines(symbol)

    # Data Sufficiency Check (Initial)
    if kl_df is None:
        error_msg = f"Insufficient kline data for {symbol} (kl_df is None)"
        print(f"{s6_log_prefix} Error: {error_msg}")
        base_return['error'] = error_msg
        return base_return

    # Assert and Log kline length
    try:
        assert len(kl_df) >= 60, f"S6 Error: Insufficient klines for {symbol} (need >=60, got {len(kl_df)})"
        print(f"{s6_log_prefix} Kline data length check passed: {len(kl_df)} candles.")
    except AssertionError as ae:
        print(f"{s6_log_prefix} Error: {str(ae)}")
        base_return['error'] = str(ae)
        return base_return

    # Fallback check if min_klines_needed is different from 60 for other reasons (though problem implies 60 is the target)
    if len(kl_df) < min_klines_needed: # min_klines_needed is currently 60, so this is redundant if assert passes
        error_msg = f"Insufficient kline data for {symbol} (need {min_klines_needed}, got {len(kl_df)})"
        print(f"{s6_log_prefix} Error: {error_msg}") # Should ideally not be reached if assert for 60 is active
        base_return['error'] = error_msg
        return base_return
    # This print is now covered by the one after assertion. Can be removed or kept for verbosity.
    # print(f"{s6_log_prefix} Kline data length: {len(kl_df)}") 

    try:
        # 1. Market Structure Analysis
        print(f"{s6_log_prefix} Finding swing points...")
        swing_highs_bool, swing_lows_bool = find_swing_points(kl_df, order=5) 
        # Swing Count Logging
        print(f"{s6_log_prefix} Swings: {swing_highs_bool.sum()} highs – {swing_lows_bool.sum()} lows")
        
        print(f"{s6_log_prefix} Identifying market structure...")
        market_structure = identify_market_structure(kl_df, swing_highs_bool, swing_lows_bool)
        print(f"{s6_log_prefix} Market Structure: {market_structure}")
        base_return['all_conditions_status']['trend_bias'] = market_structure['trend_bias']
        base_return['all_conditions_status']['last_valid_low'] = market_structure['last_valid_low_price']
        base_return['all_conditions_status']['last_valid_high'] = market_structure['last_valid_high_price']

        # 2. Identify Supply & Demand Zones
        print(f"{s6_log_prefix} Identifying S/D zones...")
        sd_zones = identify_supply_demand_zones(kl_df, atr_period=14, lookback_candles=10, consolidation_atr_factor=0.7, sharp_move_atr_factor=1.5)
        # Zone Count Logging
        print(f"{s6_log_prefix} Zones detected: {len(sd_zones)}")

        if not sd_zones:
            print(f"{s6_log_prefix} No S/D zones identified. No trade signal.")
            base_return['all_conditions_status']['zone_type_found'] = 'None'
            return base_return

        current_price = kl_df['Close'].iloc[-1]
        print(f"{s6_log_prefix} Current price: {current_price}")
        potential_trade = None
        
        # 3. Strategy Logic
        print(f"{s6_log_prefix} Current trend bias: {market_structure['trend_bias']}")
        if market_structure['trend_bias'] == 'bullish':
            relevant_zones = [z for z in sd_zones if z['type'] == 'demand' and z['timestamp_end'] < kl_df.index[-1]]
            relevant_zones.sort(key=lambda z: z['timestamp_end'], reverse=True)
            print(f"{s6_log_prefix} Found {len(relevant_zones)} relevant demand zones for bullish trend.")

            if relevant_zones:
                zone = relevant_zones[0]
                print(f"{s6_log_prefix} Considering demand zone: Start={zone['price_start']}, End={zone['price_end']}, TimestampEnd={zone['timestamp_end']}")
                base_return['all_conditions_status']['zone_type_found'] = 'demand'
                base_return['all_conditions_status']['current_sd_zone_start'] = zone['price_start']
                base_return['all_conditions_status']['current_sd_zone_end'] = zone['price_end']
                
                # Price-in-Zone Logging (Bullish)
                print(f"{s6_log_prefix} Bullish Zone Check: Zone bounds: {zone['price_start']}–{zone['price_end']}, Price: {current_price}")
                if zone['price_start'] <= current_price <= zone['price_end']:
                    print(f"{s6_log_prefix} Price {current_price} is within demand zone.")
                    base_return['all_conditions_status']['price_in_zone'] = True
                    entry_price = current_price 
                    
                    sl_price_raw = zone['price_start']
                    sl_buffer = (zone['price_end'] - zone['price_start']) * 0.10 # 10% of zone height as buffer
                    sl_price = round(sl_price_raw - sl_buffer, get_price_precision(symbol))
                    # Ensure SL is actually below the zone start after buffer and rounding
                    if sl_price >= zone['price_start']: sl_price = round(zone['price_start'] - (kl_df['Close'].iloc[-1] * 0.001), get_price_precision(symbol)) # Minimal fallback SL if buffer logic fails

                    print(f"{s6_log_prefix} Calculated SL for long: {sl_price} (Zone bottom: {sl_price_raw}, Buffer: {sl_buffer})")

                    tp_price = market_structure.get('last_valid_high_price') or market_structure.get('current_swing_high_price')
                    if tp_price: tp_price = round(tp_price, get_price_precision(symbol))
                    print(f"{s6_log_prefix} Calculated TP for long: {tp_price} (LastValidHigh: {market_structure.get('last_valid_high_price')}, CurrentSwingHigh: {market_structure.get('current_swing_high_price')})")

                    if tp_price is None or tp_price <= entry_price:
                        base_return['error'] = f"Invalid TP for bullish (TP {tp_price} <= entry {entry_price} or None)."
                        print(f"{s6_log_prefix} {base_return['error']}")
                    elif sl_price >= entry_price:
                        base_return['error'] = f"Invalid SL for bullish (SL {sl_price} >= entry {entry_price})."
                        print(f"{s6_log_prefix} {base_return['error']}")
                    else:
                        reward_potential = tp_price - entry_price
                        risk_potential = entry_price - sl_price
                        if risk_potential <= 0: 
                            base_return['error'] = "Risk potential <= 0 for bullish."
                            print(f"{s6_log_prefix} {base_return['error']}")
                        else:
                            rr_ratio = reward_potential / risk_potential
                            # Risk-Reward Logging (Bullish)
                            print(f"{s6_log_prefix} Bullish Risk/Reward: Risk: {risk_potential:.4f}, Reward: {reward_potential:.4f}, R:R = {rr_ratio:.2f}")
                            # Adjust R:R Threshold
                            if rr_ratio >= 1.5: # Temporarily lowered from 2.5
                                base_return['all_conditions_status']['rr_ok'] = True
                                potential_trade = {'signal': 'up', 'sl': sl_price, 'tp': tp_price, 'entry': entry_price}
                                print(f"{s6_log_prefix} Bullish trade meets R:R. Signal: up, SL: {sl_price}, TP: {tp_price}")
                            else:
                                base_return['error'] = f"R:R too low ({rr_ratio:.2f}) for demand."
                                print(f"{s6_log_prefix} {base_return['error']}")
                else:
                    print(f"{s6_log_prefix} Price {current_price} not in most recent demand zone.")
            else:
                print(f"{s6_log_prefix} No relevant historical demand zones found for bullish trend.")
        
        elif market_structure['trend_bias'] == 'bearish':
            relevant_zones = [z for z in sd_zones if z['type'] == 'supply' and z['timestamp_end'] < kl_df.index[-1]]
            relevant_zones.sort(key=lambda z: z['timestamp_end'], reverse=True)
            print(f"{s6_log_prefix} Found {len(relevant_zones)} relevant supply zones for bearish trend.")

            if relevant_zones:
                zone = relevant_zones[0]
                print(f"{s6_log_prefix} Considering supply zone: Start={zone['price_start']}, End={zone['price_end']}, TimestampEnd={zone['timestamp_end']}")
                base_return['all_conditions_status']['zone_type_found'] = 'supply'
                base_return['all_conditions_status']['current_sd_zone_start'] = zone['price_start']
                base_return['all_conditions_status']['current_sd_zone_end'] = zone['price_end']

                # Price-in-Zone Logging (Bearish)
                print(f"{s6_log_prefix} Bearish Zone Check: Zone bounds: {zone['price_start']}–{zone['price_end']}, Price: {current_price}")
                if zone['price_start'] <= current_price <= zone['price_end']:
                    print(f"{s6_log_prefix} Price {current_price} is within supply zone.")
                    base_return['all_conditions_status']['price_in_zone'] = True
                    entry_price = current_price
                    
                    sl_price_raw = zone['price_end']
                    sl_buffer = (zone['price_end'] - zone['price_start']) * 0.10 # 10% of zone height as buffer
                    sl_price = round(sl_price_raw + sl_buffer, get_price_precision(symbol))
                    # Ensure SL is actually above the zone end after buffer and rounding
                    if sl_price <= zone['price_end']: sl_price = round(zone['price_end'] + (kl_df['Close'].iloc[-1] * 0.001), get_price_precision(symbol)) # Minimal fallback SL

                    print(f"{s6_log_prefix} Calculated SL for short: {sl_price} (Zone top: {sl_price_raw}, Buffer: {sl_buffer})")

                    tp_price = market_structure.get('last_valid_low_price') or market_structure.get('current_swing_low_price')
                    if tp_price: tp_price = round(tp_price, get_price_precision(symbol))
                    print(f"{s6_log_prefix} Calculated TP for short: {tp_price} (LastValidLow: {market_structure.get('last_valid_low_price')}, CurrentSwingLow: {market_structure.get('current_swing_low_price')})")

                    if tp_price is None or tp_price >= entry_price:
                        base_return['error'] = f"Invalid TP for bearish (TP {tp_price} >= entry {entry_price} or None)."
                        print(f"{s6_log_prefix} {base_return['error']}")
                    elif sl_price <= entry_price:
                         base_return['error'] = f"Invalid SL for bearish (SL {sl_price} <= entry {entry_price})."
                         print(f"{s6_log_prefix} {base_return['error']}")
                    else:
                        reward_potential = entry_price - tp_price
                        risk_potential = sl_price - entry_price
                        if risk_potential <= 0: 
                            base_return['error'] = "Risk potential <= 0 for bearish."
                            print(f"{s6_log_prefix} {base_return['error']}")
                        else:
                            rr_ratio = reward_potential / risk_potential
                            # Risk-Reward Logging (Bearish)
                            print(f"{s6_log_prefix} Bearish Risk/Reward: Risk: {risk_potential:.4f}, Reward: {reward_potential:.4f}, R:R = {rr_ratio:.2f}")
                            # Adjust R:R Threshold
                            if rr_ratio >= 1.5: # Temporarily lowered from 2.5
                                base_return['all_conditions_status']['rr_ok'] = True
                                potential_trade = {'signal': 'down', 'sl': sl_price, 'tp': tp_price, 'entry': entry_price}
                                print(f"{s6_log_prefix} Bearish trade meets R:R. Signal: down, SL: {sl_price}, TP: {tp_price}")
                            else:
                                base_return['error'] = f"R:R too low ({rr_ratio:.2f}) for supply."
                                print(f"{s6_log_prefix} {base_return['error']}")
                else:
                    print(f"{s6_log_prefix} Price {current_price} not in most recent supply zone.")
            else:
                print(f"{s6_log_prefix} No relevant historical supply zones found for bearish trend.")
        else: # Ranging trend
            print(f"{s6_log_prefix} Trend is ranging. No S/D trades taken in ranging market.")
            base_return['error'] = "Trend is ranging, no trade."


        if potential_trade:
            base_return['signal'] = potential_trade['signal']
            base_return['sl_price'] = potential_trade['sl'] 
            base_return['tp_price'] = potential_trade['tp'] # TP already rounded
            base_return['entry_price_estimate'] = round(potential_trade['entry'], get_price_precision(symbol))
            base_return['conditions_met_count'] = 1 
            if base_return['error'] and base_return['all_conditions_status']['rr_ok']:
                print(f"{s6_log_prefix} Clearing previous error as R:R is OK for trade: {base_return['error']}")
                base_return['error'] = None 
        else:
            # This 'else' means no 'potential_trade' was formed.
            # The error might have been set above (e.g. R:R too low, invalid SL/TP)
            # Or, no conditions to form a trade were met (e.g., price not in zone, no relevant zones)
            base_return['signal'] = 'none'
            base_return['sl_price'] = None
            base_return['tp_price'] = None
            if not base_return['error']: # If no specific error was set by R:R or SL/TP validation
                if base_return['all_conditions_status']['zone_type_found'] not in ['N/A', 'None']:
                    if not base_return['all_conditions_status']['price_in_zone']:
                        base_return['error'] = "Price not in S/D zone."
                elif market_structure['trend_bias'] != 'ranging': # Only set this if not ranging and no other error
                     base_return['error'] = "No valid trade setup found (e.g. no zone entry, or other rule failed)."
            if base_return['error']: print(f"{s6_log_prefix} No trade signal. Reason: {base_return['error']}")
            else: print(f"{s6_log_prefix} No trade signal. Conditions not met.")


    except ImportError as e_imp: 
        base_return['error'] = f"ImportError S6 {symbol}: {e_imp}. Check scipy/talib/pandas_ta."
        print(f"{s6_log_prefix} {base_return['error']}")
    except ValueError as ve: 
        base_return['error'] = f"ValueError S6 {symbol}: {ve}"
        print(f"{s6_log_prefix} {base_return['error']}")
    except Exception as e:
        import traceback
        base_return['error'] = f"Unexpected error S6 {symbol}: {str(e)}"
        print(f"{s6_log_prefix} Full Traceback: {traceback.format_exc()}")

    # Final validation and logging
    if base_return['signal'] != 'none':
        if base_return['sl_price'] is None or base_return['tp_price'] is None:
            final_error = (base_return['error'] or "") + " SL/TP is None post-processing."
            print(f"{s6_log_prefix} Invalidating signal due to None SL/TP. Error: {final_error}")
            base_return['error'] = final_error
            base_return['signal'] = 'none'
        elif base_return['signal'] == 'up' and (base_return['sl_price'] >= base_return['entry_price_estimate'] or base_return['tp_price'] <= base_return['entry_price_estimate']):
            final_error = (base_return['error'] or "") + " Invalid SL/TP relation to entry for UP signal."
            print(f"{s6_log_prefix} Invalidating UP signal due to SL/TP relation. Error: {final_error}")
            base_return['error'] = final_error
            base_return['signal'] = 'none'
        elif base_return['signal'] == 'down' and (base_return['sl_price'] <= base_return['entry_price_estimate'] or base_return['tp_price'] >= base_return['entry_price_estimate']):
            final_error = (base_return['error'] or "") + " Invalid SL/TP relation to entry for DOWN signal."
            print(f"{s6_log_prefix} Invalidating DOWN signal due to SL/TP relation. Error: {final_error}")
            base_return['error'] = final_error
            base_return['signal'] = 'none'
            
    print(f"{s6_log_prefix} Final decision: Signal='{base_return['signal']}', SL={base_return['sl_price']}, TP={base_return['tp_price']}, Error='{base_return['error']}'")
    # Surface Errors/Status (Final base_return log)
    print(f"{s6_log_prefix} Final base_return: {base_return}")
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
    global client, ACCOUNT_RISK_PERCENT, SL_PERCENT, TP_PERCENT
    # Add new globals for PnL based SL/TP and mode
    global SL_TP_MODE, SL_PNL_AMOUNT, TP_PNL_AMOUNT
    
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

        capital_to_risk_usdt = 0.0
        sl_for_sizing_percentage = 0.0

        print(f"INFO: Current SL/TP Mode: {SL_TP_MODE} for {symbol}")

        if SL_TP_MODE == "Fixed PnL":
            if SL_PNL_AMOUNT <= 0:
                print(f"Warning: SL PnL Amount ($ {SL_PNL_AMOUNT}) must be positive for 'Fixed PnL' mode. Aborting order for {symbol}.")
                return
            capital_to_risk_usdt = SL_PNL_AMOUNT  # The fixed $ amount to risk
            # For Fixed PnL, quantity is determined by this risk amount and the price difference to SL.
            # The SL price itself will be entry_price - (SL_PNL_AMOUNT / quantity).
            # So, sl_for_sizing_percentage is not directly used to set the SL distance here,
            # but rather to estimate a reasonable notional position size.
            # We can use a reference SL_PERCENT for this initial notional sizing.
            sl_for_sizing_percentage = SL_PERCENT 
            print(f"Using Fixed PnL mode: Capital to Risk = ${capital_to_risk_usdt:.2f}, SL_PERCENT for initial sizing ref = {sl_for_sizing_percentage*100:.2f}%")
        else: # Percentage or ATR/Dynamic mode
            current_account_risk = ACCOUNT_RISK_PERCENT # Global default
            if strategy_account_risk_percent is not None and 0 < strategy_account_risk_percent < 1:
                current_account_risk = strategy_account_risk_percent
                print(f"Using strategy-defined account risk: {current_account_risk*100:.2f}% for {symbol}")
            else:
                print(f"Using global account risk: {current_account_risk*100:.2f}% for {symbol}")
            capital_to_risk_usdt = account_balance * current_account_risk

            if SL_TP_MODE == "ATR/Dynamic" and strategy_sl is not None:
                # Ensure strategy_sl is valid for the side before calculating percentage
                if (side == 'buy' and strategy_sl < price) or \
                   (side == 'sell' and strategy_sl > price):
                    sl_for_sizing_percentage = abs(price - strategy_sl) / price
                else:
                    print(f"Warning: Invalid strategy_sl ({strategy_sl}) for ATR/Dynamic sizing on {symbol} {side} at price {price}. Defaulting to global SL_PERCENT.")
                    sl_for_sizing_percentage = SL_PERCENT
                print(f"Using ATR/Dynamic mode: SL for sizing derived from strategy_sl ({strategy_sl}), effective SL % for sizing: {sl_for_sizing_percentage*100:.2f}%")
            else: # Percentage mode (or ATR/Dynamic without strategy_sl, fallback to Percentage)
                sl_for_sizing_percentage = SL_PERCENT
                if SL_TP_MODE == "ATR/Dynamic": # Log if ATR/Dynamic is falling back
                     print(f"INFO: SL_TP_MODE is 'ATR/Dynamic' but strategy_sl not provided. Falling back to SL_PERCENT for sizing reference.")
                print(f"Using Percentage mode (or fallback): SL_PERCENT for sizing = {sl_for_sizing_percentage*100:.2f}%")

        if capital_to_risk_usdt <= 0:
            print(f"Warning: Capital to risk is {capital_to_risk_usdt:.2f} for {symbol}. Aborting order.")
            return
        if sl_for_sizing_percentage <= 0: # This check is crucial
            print(f"Warning: SL for sizing percentage is {sl_for_sizing_percentage*100:.2f}%. Cannot calculate position size. Aborting order for {symbol}.")
            return

        position_size_usdt_notional = capital_to_risk_usdt / sl_for_sizing_percentage
        
        CAP_FRACTION_OF_BALANCE = 0.50 
        max_permissible_notional_value = account_balance * CAP_FRACTION_OF_BALANCE
        if position_size_usdt_notional > max_permissible_notional_value:
            original_calculated_pos_size_usdt = position_size_usdt_notional
            position_size_usdt_notional = max_permissible_notional_value
            print(f"INFO: Position size capped for {symbol}. Original calc: {original_calculated_pos_size_usdt:.2f} USDT, Capped to: {position_size_usdt_notional:.2f} USDT.")
        
        calculated_qty_asset = round(position_size_usdt_notional / price, qty_precision)
        if calculated_qty_asset <= 0:
            print(f"Warning: Calculated quantity is {calculated_qty_asset} for {symbol}. Aborting order.")
            return

        print(f"Order Details ({symbol}): SL/TP Mode='{SL_TP_MODE}', Bal={account_balance:.2f}, RiskCap=${capital_to_risk_usdt:.2f}, SL_Sizing%={sl_for_sizing_percentage*100:.2f}%, NotionalPosUSD={position_size_usdt_notional:.2f}, Qty={calculated_qty_asset}")

        sl_actual, tp_actual = None, None

        if SL_TP_MODE == "Fixed PnL":
            if TP_PNL_AMOUNT <= 0: 
                print(f"Warning: TP PnL Amount ($ {TP_PNL_AMOUNT}) must be positive for 'Fixed PnL' mode. Aborting order for {symbol}.")
                return
            if calculated_qty_asset == 0: 
                print(f"Error: Calculated quantity is zero for Fixed PnL mode, cannot determine PnL-based SL/TP for {symbol}. Aborting.")
                return

            if side == 'buy':
                sl_actual = round(price - (SL_PNL_AMOUNT / calculated_qty_asset), price_precision)
                tp_actual = round(price + (TP_PNL_AMOUNT / calculated_qty_asset), price_precision)
            elif side == 'sell':
                sl_actual = round(price + (SL_PNL_AMOUNT / calculated_qty_asset), price_precision)
                tp_actual = round(price - (TP_PNL_AMOUNT / calculated_qty_asset), price_precision)
            print(f"Using Fixed PnL: SL: {sl_actual}, TP: {tp_actual} for {symbol} {side} (Qty: {calculated_qty_asset})")

        elif SL_TP_MODE == "ATR/Dynamic" or SL_TP_MODE == "StrategyDefined_SD": # Modified to include new mode
            if strategy_sl is not None and strategy_tp is not None:
                sl_actual = strategy_sl
                tp_actual = strategy_tp
                print(f"Using {SL_TP_MODE} (strategy-defined): SL: {sl_actual}, TP: {tp_actual} for {symbol} {side}")
            else:
                # Fallback to percentage if strategy doesn't provide SL/TP for these modes
                print(f"INFO: {SL_TP_MODE} mode selected but no strategy_sl/tp provided for {symbol}. Falling back to Percentage SL/TP.")
                if side == 'buy':
                    sl_actual = round(price - price * SL_PERCENT, price_precision)
                    tp_actual = round(price + price * TP_PERCENT, price_precision)
                elif side == 'sell':
                    sl_actual = round(price + price * SL_PERCENT, price_precision)
                    tp_actual = round(price - price * TP_PERCENT, price_precision)
                print(f"Using Fallback Percentage for ATR/Dynamic: SL: {sl_actual}, TP: {tp_actual} for {symbol} {side}")
        
        elif SL_TP_MODE == "Percentage": # Explicitly Percentage Mode
            if side == 'buy':
                sl_actual = round(price - price * SL_PERCENT, price_precision)
                tp_actual = round(price + price * TP_PERCENT, price_precision)
            elif side == 'sell':
                sl_actual = round(price + price * SL_PERCENT, price_precision)
                tp_actual = round(price - price * TP_PERCENT, price_precision)
            print(f"Using Percentage-based: SL: {sl_actual} (from {SL_PERCENT*100}%), TP: {tp_actual} (from {TP_PERCENT*100}%) for {symbol} {side}")
        
        else: # Should not be reached if SL_TP_MODE is validated earlier
            print(f"Error: Unknown SL_TP_MODE '{SL_TP_MODE}' in open_order. Aborting.")
            return

        if sl_actual is None or tp_actual is None:
            print(f"Error: SL or TP price could not be determined for {symbol} {side} with mode {SL_TP_MODE}. Aborting order.")
            return
        
        # Validate that SL and TP are not impossible 
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
        try: # Outer try for the main loop
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
            elif current_active_strategy_id == 6: # Strategy 6 logic for can_open_new_trade_overall
                if len(open_position_symbols) < qty_concurrent_positions :
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
                    
                    elif current_active_strategy_id == 6: # Market Structure S/D Strategy
                        print(f"DEBUG: S6: Active for symbol: {sym_to_check}")
                        _activity_set(f"S6: Scanning {sym_to_check}...")
                        signal_data_s6 = strategy_market_structure_sd(sym_to_check) # Call the new strategy function
                        print(f"DEBUG: S6: Data for {sym_to_check}: {signal_data_s6}")

                        if root and root.winfo_exists(): # Update UI with conditions
                            root.after(0, update_conditions_display_content, sym_to_check, signal_data_s6.get('all_conditions_status'), signal_data_s6.get('error'))
                        
                        if signal_data_s6.get('error'):
                            # Error message already printed by strategy_market_structure_sd if it's significant
                            # print(f"S6 Error ({sym_to_check}): {signal_data_s6['error']}") 
                            sleep(0.1); continue # Move to next symbol
                        
                        current_signal_s6 = signal_data_s6.get('signal', 'none')
                        if current_signal_s6 in ['up', 'down']:
                            print(f"S6: {current_signal_s6.upper()} signal for {sym_to_check}. Ordering...")
                            _status_set(f"S6: Ordering {sym_to_check} ({current_signal_s6.upper()})...")
                            set_mode(sym_to_check, margin_type_setting); sleep(0.1)
                            set_leverage(sym_to_check, leverage); sleep(0.1)
                            
                            # Pass SL, TP, and risk from strategy directly
                            open_order(sym_to_check, current_signal_s6, 
                                       strategy_sl=signal_data_s6.get('sl_price'), 
                                       strategy_tp=signal_data_s6.get('tp_price'), 
                                       strategy_account_risk_percent=signal_data_s6.get('account_risk_percent'))
                            
                            _activity_set(f"S6: Trade initiated for {sym_to_check}.")
                            # S6 might also allow multiple trades across symbols unless a specific tracker is added.
                            sleep(1) 
                        else:
                            _activity_set(f"S6: No signal for {sym_to_check}. Next..."); sleep(0.1)

                    elif current_active_strategy_id == 7: # Candlestick Patterns Strategy
                        print(f"DEBUG: S7: Active for symbol: {sym_to_check}")
                        _activity_set(f"S7: Scanning {sym_to_check}...")
                        signal_data_s7 = strategy_candlestick_patterns_signal(sym_to_check)
                        print(f"DEBUG: S7: Data for {sym_to_check}: {signal_data_s7}")

                        if root and root.winfo_exists(): # Update UI with conditions
                            root.after(0, update_conditions_display_content, sym_to_check, signal_data_s7.get('all_conditions_status'), signal_data_s7.get('error'))
                        
                        # Process S7 error: only continue to order if signal is present, even if there's a non-critical error string
                        # Critical errors (like insufficient klines) should already return signal='none' from the strategy.
                        # Non-critical errors (like "No specific pattern detected") might still accompany signal='none'.
                        if signal_data_s7.get('error') and signal_data_s7.get('signal', 'none') == 'none':
                            # Log the error if it's not one of the common "no signal" messages
                            if signal_data_s7['error'] not in ["S7: No specific pattern detected.", f"S7: {signal_data_s7.get('all_conditions_status',{}).get('detected_pattern','N/A')} filters failed: No volume spike, EMA filter failed", f"S7: {signal_data_s7.get('all_conditions_status',{}).get('detected_pattern','N/A')} filters failed: No volume spike", f"S7: {signal_data_s7.get('all_conditions_status',{}).get('detected_pattern','N/A')} filters failed: EMA filter failed"]:
                                print(f"S7 Error ({sym_to_check}): {signal_data_s7['error']}")
                            sleep(0.1); continue # Move to next symbol
                        
                        current_signal_s7 = signal_data_s7.get('signal', 'none')
                        if current_signal_s7 in ['up', 'down']:
                            print(f"S7: {current_signal_s7.upper()} signal for {sym_to_check}. Pattern: {signal_data_s7.get('all_conditions_status', {}).get('detected_pattern', 'N/A')}. Ordering...")
                            _status_set(f"S7: Ordering {sym_to_check} ({current_signal_s7.upper()})...")
                            set_mode(sym_to_check, margin_type_setting); sleep(0.1)
                            set_leverage(sym_to_check, leverage); sleep(0.1)
                            
                            open_order(sym_to_check, current_signal_s7, 
                                       strategy_sl=signal_data_s7.get('sl_price'), 
                                       strategy_tp=signal_data_s7.get('tp_price'), 
                                       strategy_account_risk_percent=signal_data_s7.get('account_risk_percent'))
                            
                            _activity_set(f"S7: Trade initiated for {sym_to_check}.")
                            # S7, like S5 and S6, might allow multiple trades across symbols.
                            # No specific single-trade tracker for S7 implemented yet.
                            sleep(1) 
                        else: # No signal from S7 after evaluation
                            _activity_set(f"S7: No signal for {sym_to_check}. Next..."); sleep(0.1)

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
                # Removed the specific try-except for ClientError and Exception from here,
                # as they will be caught by the main loop's new try-except block.
                
        except ClientError as ce: # This will now be part of the outer try-except
            err_msg = f"API Error: {ce.error_message if hasattr(ce, 'error_message') and ce.error_message else ce}. Recovering and continuing..."
            print(err_msg); _status_set(err_msg); _activity_set("API Error. Recovering and continuing...")
            if "signature" in str(ce).lower() or "timestamp" in str(ce).lower():
                print("Reinitializing client due to signature/timestamp error in API call.")
                reinitialize_client()
            # The existing cool-down loop for retrying
            for _ in range(60): 
                if not bot_running: break
                sleep(1)
            if not bot_running: break
            continue # Ensure the loop continues after this specific error handling

        except Exception as e: # This is the new main loop exception handler
            err_msg = f"Bot loop error: {e}. Recovering and continuing..."
            print(err_msg)
            if root and root.winfo_exists(): # Check if UI elements exist before updating
                _status_set(err_msg)
                _activity_set("Error caught. Recovering...")
            
            # import traceback # For debugging, uncomment if needed
            # print(traceback.format_exc()) # For debugging

            # The existing cool-down loop for retrying provides a pause before the next cycle
            for _ in range(60):
                if not bot_running: break
                sleep(1)
            if not bot_running: break # Exit if bot was stopped during sleep
            continue # Explicitly continue to the next iteration of the while loop

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
    global ACCOUNT_RISK_PERCENT, TP_PERCENT, SL_PERCENT, leverage, qty_concurrent_positions, LOCAL_HIGH_LOW_LOOKBACK_PERIOD, margin_type_setting, TARGET_SYMBOLS, ACTIVE_STRATEGY_ID
    # Add new globals for PnL SL/TP settings
    global SL_TP_MODE, SL_PNL_AMOUNT, TP_PNL_AMOUNT 
    # And their corresponding tk StringVars
    global sl_tp_mode_var, sl_pnl_amount_var, tp_pnl_amount_var

    try:
        # SL/TP Mode
        selected_mode = sl_tp_mode_var.get()
        # Added "StrategyDefined_SD" to the list of valid modes
        if selected_mode not in ["Percentage", "ATR/Dynamic", "Fixed PnL", "StrategyDefined_SD"]:
            messagebox.showerror("Settings Error", "Invalid SL/TP Mode selected.")
            return False
        SL_TP_MODE = selected_mode

        # PnL Amounts (only validate if mode is Fixed PnL, but read them anyway)
        try:
            sl_pnl = float(sl_pnl_amount_var.get())
            tp_pnl = float(tp_pnl_amount_var.get())
            if SL_TP_MODE == "Fixed PnL":
                if not (sl_pnl > 0 and tp_pnl > 0):
                    messagebox.showerror("Settings Error", "For 'Fixed PnL' mode, SL and TP PnL amounts must be positive values.")
                    return False
            SL_PNL_AMOUNT = sl_pnl
            TP_PNL_AMOUNT = tp_pnl
        except ValueError:
            if SL_TP_MODE == "Fixed PnL": # Only critical if this mode is selected
                messagebox.showerror("Settings Error", "Invalid number format for SL/TP PnL Amounts.")
                return False
            # If not Fixed PnL mode, we can ignore non-float values for PnL amounts for now, or default them
            # However, it's better to ensure they are always valid numbers if the entries exist.
            # For now, we'll keep the logic that they are only strictly validated for Fixed PnL mode,
            # and assume they might hold non-numeric defaults if another mode is active.
            # A more robust solution might involve disabling/clearing these when not in Fixed PnL mode.
            if not sl_pnl_amount_var.get().strip(): SL_PNL_AMOUNT = 0.0
            else: SL_PNL_AMOUNT = float(sl_pnl_amount_var.get()) # Attempt conversion, might fail if truly not Fixed PnL
            
            if not tp_pnl_amount_var.get().strip(): TP_PNL_AMOUNT = 0.0
            else: TP_PNL_AMOUNT = float(tp_pnl_amount_var.get())


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
        if not (0 < tp_val <= 1000): # Allow up to 1000% for TP flexibility
            messagebox.showerror("Settings Error", "Take Profit % must be a positive value (e.g., enter 1 for 1%). Recommended 0-1000.")
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
        margin_type_setting = margin_type_var.get() 
        if margin_type_setting not in ["ISOLATED", "CROSS"]: 
             messagebox.showerror("Settings Error", "Invalid Margin Type selected.")
             return False

        print(f"Applied settings: Strategy='{STRATEGIES[ACTIVE_STRATEGY_ID]}', Risk={ACCOUNT_RISK_PERCENT*100:.2f}%, SL/TP Mode='{SL_TP_MODE}'")
        if SL_TP_MODE == "Percentage":
            print(f"  SL={SL_PERCENT*100:.2f}%, TP={TP_PERCENT*100:.2f}%")
        elif SL_TP_MODE == "Fixed PnL":
            print(f"  SL PnL=${SL_PNL_AMOUNT:.2f}, TP PnL=${TP_PNL_AMOUNT:.2f}")
        # ATR/Dynamic mode will use strategy's internal RR and ATR Multiplier, not printed here directly from these global settings
        print(f"  Lev={leverage}, MaxPos={qty_concurrent_positions}, Lookback={LOCAL_HIGH_LOW_LOOKBACK_PERIOD}, Margin={margin_type_setting}, Symbols={TARGET_SYMBOLS}")
        messagebox.showinfo("Settings", "Settings applied successfully!")
        return True

    except ValueError as ve: 
        messagebox.showerror("Settings Error", f"Invalid input for one or more parameters. Please ensure they are valid numbers. Error: {ve}")
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

    # --- Scrollable Main Container Setup ---
    # Create a main frame that will hold the canvas and scrollbars
    scrollable_canvas_frame = ttk.Frame(root)
    scrollable_canvas_frame.pack(fill=tk.BOTH, expand=True)

    canvas = tk.Canvas(scrollable_canvas_frame)
    canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

    v_scrollbar = ttk.Scrollbar(scrollable_canvas_frame, orient=tk.VERTICAL, command=canvas.yview)
    v_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
    h_scrollbar = ttk.Scrollbar(root, orient=tk.HORIZONTAL, command=canvas.xview) # Place h_scrollbar in root to be below canvas
    h_scrollbar.pack(fill=tk.X, side=tk.BOTTOM)


    canvas.configure(yscrollcommand=v_scrollbar.set, xscrollcommand=h_scrollbar.set)
    
    # This frame will contain all the application's content and be placed inside the canvas
    main_content_frame = ttk.Frame(canvas)
    canvas.create_window((0, 0), window=main_content_frame, anchor="nw")

    def on_main_content_configure(event):
        canvas.configure(scrollregion=canvas.bbox("all"))

    def on_canvas_configure(event):
        # Adjust the width of the main_content_frame to the canvas width for horizontal scrolling
        canvas.itemconfig(main_content_window_id, width=event.width)


    main_content_frame.bind("<Configure>", on_main_content_configure)
    # Store the window ID for later use in on_canvas_configure
    main_content_window_id = canvas.create_window((0,0), window=main_content_frame, anchor='nw')
    canvas.bind('<Configure>', on_canvas_configure)


    # --- End Scrollable Main Container Setup ---


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
    # SL/TP Percentage Vars for Backtesting (still needed for "Percentage" mode)
    backtest_tp_var = tk.StringVar(value="3.0") # Represents percentage if mode is Percentage
    backtest_sl_var = tk.StringVar(value="2.0") # Represents percentage if mode is Percentage
    # New PnL and Mode Vars for Backtesting
    backtest_sl_pnl_amount_var = tk.StringVar(value="10") # Default $10 SL PnL
    backtest_tp_pnl_amount_var = tk.StringVar(value="20") # Default $20 TP PnL
    backtest_sl_tp_mode_var = tk.StringVar(value="ATR/Dynamic") # Default mode
    backtest_starting_capital_var = tk.StringVar(value="10000") # Default starting capital
    backtest_leverage_var = tk.StringVar(value="1") # New StringVar for backtest leverage, default 1x

    backtest_selected_strategy_var = tk.StringVar()

    # --- All main UI elements will be children of main_content_frame ---
    controls_frame = ttk.LabelFrame(main_content_frame, text="Controls"); controls_frame.pack(padx=10, pady=(5,0), fill="x") # Changed parent
    env_frame = ttk.Frame(controls_frame); env_frame.pack(pady=2, fill="x")
    ttk.Label(env_frame, text="Env:").pack(side=tk.LEFT, padx=(5,2))
    testnet_radio = ttk.Radiobutton(env_frame, text="Testnet", variable=current_env_var, value="testnet", command=toggle_environment); testnet_radio.pack(side=tk.LEFT, padx=2)
    mainnet_radio = ttk.Radiobutton(env_frame, text="Mainnet", variable=current_env_var, value="mainnet", command=toggle_environment); mainnet_radio.pack(side=tk.LEFT, padx=2)

    # Parameter Inputs Frame
    params_input_frame = ttk.LabelFrame(controls_frame, text="Trading Parameters")
    params_input_frame.pack(fill="x", padx=5, pady=5)

    # Row 0: Account Risk %, SL/TP Mode
    ttk.Label(params_input_frame, text="Account Risk %:").grid(row=0, column=0, padx=2, pady=2, sticky='w')
    account_risk_percent_var = tk.StringVar(value=str(ACCOUNT_RISK_PERCENT * 100))
    account_risk_percent_entry = ttk.Entry(params_input_frame, textvariable=account_risk_percent_var, width=10)
    account_risk_percent_entry.grid(row=0, column=1, padx=2, pady=2, sticky='w')

    ttk.Label(params_input_frame, text="SL/TP Mode:").grid(row=0, column=2, padx=2, pady=2, sticky='w')
    sl_tp_modes = ["Percentage", "ATR/Dynamic", "Fixed PnL", "StrategyDefined_SD"] # Added new mode
    sl_tp_mode_var = tk.StringVar(value=SL_TP_MODE) 
    sl_tp_mode_combobox = ttk.Combobox(params_input_frame, textvariable=sl_tp_mode_var, values=sl_tp_modes, width=17, state="readonly") # Increased width
    sl_tp_mode_combobox.grid(row=0, column=3, padx=2, pady=2, sticky='w')

    # Row 1: SL Percent, TP Percent (for Percentage mode)
    ttk.Label(params_input_frame, text="Stop Loss %:").grid(row=1, column=0, padx=2, pady=2, sticky='w')
    sl_percent_var = tk.StringVar(value=str(SL_PERCENT * 100))
    sl_percent_entry = ttk.Entry(params_input_frame, textvariable=sl_percent_var, width=10)
    sl_percent_entry.grid(row=1, column=1, padx=2, pady=2, sticky='w')

    ttk.Label(params_input_frame, text="Take Profit %:").grid(row=1, column=2, padx=2, pady=2, sticky='w')
    tp_percent_var = tk.StringVar(value=str(TP_PERCENT * 100))
    tp_percent_entry = ttk.Entry(params_input_frame, textvariable=tp_percent_var, width=10)
    tp_percent_entry.grid(row=1, column=3, padx=2, pady=2, sticky='w')

    # Row 2: SL PnL Amount ($), TP PnL Amount ($) (for Fixed PnL mode)
    ttk.Label(params_input_frame, text="SL PnL Amount ($):").grid(row=2, column=0, padx=2, pady=2, sticky='w')
    sl_pnl_amount_var = tk.StringVar(value=str(SL_PNL_AMOUNT)) 
    sl_pnl_amount_entry = ttk.Entry(params_input_frame, textvariable=sl_pnl_amount_var, width=10)
    sl_pnl_amount_entry.grid(row=2, column=1, padx=2, pady=2, sticky='w')

    ttk.Label(params_input_frame, text="TP PnL Amount ($):").grid(row=2, column=2, padx=2, pady=2, sticky='w')
    tp_pnl_amount_var = tk.StringVar(value=str(TP_PNL_AMOUNT)) 
    tp_pnl_amount_entry = ttk.Entry(params_input_frame, textvariable=tp_pnl_amount_var, width=10)
    tp_pnl_amount_entry.grid(row=2, column=3, padx=2, pady=2, sticky='w')
    
    # Row 3: Leverage, Max Open Pos
    ttk.Label(params_input_frame, text="Leverage:").grid(row=3, column=0, padx=2, pady=2, sticky='w')
    leverage_var = tk.StringVar(value=str(leverage))
    leverage_entry = ttk.Entry(params_input_frame, textvariable=leverage_var, width=10)
    leverage_entry.grid(row=3, column=1, padx=2, pady=2, sticky='w')

    ttk.Label(params_input_frame, text="Max Open Pos:").grid(row=3, column=2, padx=2, pady=2, sticky='w')
    qty_concurrent_positions_var = tk.StringVar(value=str(qty_concurrent_positions))
    qty_concurrent_positions_entry = ttk.Entry(params_input_frame, textvariable=qty_concurrent_positions_var, width=10)
    qty_concurrent_positions_entry.grid(row=3, column=3, padx=2, pady=2, sticky='w')

    # Row 4: Lookback, Margin Type
    ttk.Label(params_input_frame, text="Lookback (Breakout):").grid(row=4, column=0, padx=2, pady=2, sticky='w')
    local_high_low_lookback_var = tk.StringVar(value=str(LOCAL_HIGH_LOW_LOOKBACK_PERIOD))
    local_high_low_lookback_entry = ttk.Entry(params_input_frame, textvariable=local_high_low_lookback_var, width=10)
    local_high_low_lookback_entry.grid(row=4, column=1, padx=2, pady=2, sticky='w')

    ttk.Label(params_input_frame, text="Margin Type:").grid(row=4, column=2, padx=2, pady=2, sticky='w')
    margin_type_var = tk.StringVar(value=margin_type_setting)
    margin_type_frame = ttk.Frame(params_input_frame) 
    margin_type_frame.grid(row=4, column=3, padx=2, pady=2, sticky='w')
    margin_type_isolated_radio = ttk.Radiobutton(margin_type_frame, text="ISOLATED", variable=margin_type_var, value="ISOLATED")
    margin_type_isolated_radio.pack(side=tk.LEFT)
    margin_type_cross_radio = ttk.Radiobutton(margin_type_frame, text="CROSS", variable=margin_type_var, value="CROSS")
    margin_type_cross_radio.pack(side=tk.LEFT, padx=(5,0))


    # Row 5: Target Symbols Entry
    ttk.Label(params_input_frame, text="Target Symbols (CSV):").grid(row=5, column=0, padx=2, pady=2, sticky='w')
    target_symbols_var = tk.StringVar(value=",".join(TARGET_SYMBOLS))
    target_symbols_entry = ttk.Entry(params_input_frame, textvariable=target_symbols_var, width=40) 
    target_symbols_entry.grid(row=5, column=1, columnspan=3, padx=2, pady=2, sticky='we')

    params_widgets = [
        account_risk_percent_entry, sl_tp_mode_combobox,
        sl_percent_entry, tp_percent_entry,
        sl_pnl_amount_entry, tp_pnl_amount_entry,
        leverage_entry, qty_concurrent_positions_entry,
        local_high_low_lookback_entry, margin_type_isolated_radio, margin_type_cross_radio,
        target_symbols_entry
    ]
    params_input_frame.columnconfigure(1, weight=1) 
    params_input_frame.columnconfigure(3, weight=1) 

    strategy_frame = ttk.LabelFrame(controls_frame, text="Strategy Selection")
    strategy_frame.pack(fill="x", padx=5, pady=5)
    
    if strategy_radio_buttons: 
        params_widgets = [widget for widget in params_widgets if widget not in strategy_radio_buttons]
        strategy_radio_buttons.clear() 

    for strategy_id, strategy_name in STRATEGIES.items():
        var = tk.BooleanVar(value=(strategy_id == selected_strategy_var.get()))
        strategy_checkbox_vars[strategy_id] = var
        cb = ttk.Checkbutton(strategy_frame, 
                             text=strategy_name, 
                             variable=var, 
                             command=lambda sid=strategy_id: handle_strategy_checkbox_select(sid))
        cb.pack(anchor='w', padx=5)
        params_widgets.append(cb) 
    
    api_key_info_label = ttk.Label(controls_frame, text="API Keys from keys.py"); api_key_info_label.pack(pady=2)
    timeframe_label = ttk.Label(controls_frame, text="Timeframe: 5m (fixed)"); timeframe_label.pack(pady=2)

    buttons_frame = ttk.Frame(controls_frame); buttons_frame.pack(pady=2)
    start_button = ttk.Button(buttons_frame, text="Start Bot", command=start_bot); start_button.pack(side=tk.LEFT, padx=5)
    stop_button = ttk.Button(buttons_frame, text="Stop Bot", command=stop_bot, state=tk.DISABLED); stop_button.pack(side=tk.LEFT, padx=5)
    
    data_frame = ttk.LabelFrame(main_content_frame, text="Live Data & History"); data_frame.pack(padx=10, pady=5, fill="both", expand=True) # Changed parent

    side_by_side_frame = ttk.Frame(data_frame)
    side_by_side_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

    backtesting_frame = ttk.LabelFrame(side_by_side_frame, text="Backtesting Engine")
    backtesting_frame.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 5), expand=False) 

    backtest_params_grid = ttk.Frame(backtesting_frame)
    backtest_params_grid.pack(fill="x", padx=5, pady=5)

    ttk.Label(backtest_params_grid, text="Symbol:").grid(row=0, column=0, padx=2, pady=2, sticky='w')
    backtest_symbol_entry = ttk.Entry(backtest_params_grid, textvariable=backtest_symbol_var, width=12)
    backtest_symbol_entry.grid(row=0, column=1, padx=2, pady=2, sticky='w')

    ttk.Label(backtest_params_grid, text="Timeframe:").grid(row=0, column=2, padx=2, pady=2, sticky='w')
    backtest_timeframe_entry = ttk.Entry(backtest_params_grid, textvariable=backtest_timeframe_var, width=7)
    backtest_timeframe_entry.grid(row=0, column=3, padx=2, pady=2, sticky='w')

    ttk.Label(backtest_params_grid, text="Interval (days):").grid(row=1, column=0, padx=2, pady=2, sticky='w')
    backtest_interval_entry = ttk.Entry(backtest_params_grid, textvariable=backtest_interval_var, width=7)
    backtest_interval_entry.grid(row=1, column=1, padx=2, pady=2, sticky='w')
    
    ttk.Label(backtest_params_grid, text="SL/TP Mode:").grid(row=1, column=2, padx=2, pady=2, sticky='w')
    backtest_sl_tp_mode_combobox = ttk.Combobox(backtest_params_grid, textvariable=backtest_sl_tp_mode_var, values=sl_tp_modes, width=17, state="readonly") # Increased width
    backtest_sl_tp_mode_combobox.grid(row=1, column=3, padx=2, pady=2, sticky='w')

    ttk.Label(backtest_params_grid, text="Stop Loss %:").grid(row=2, column=0, padx=2, pady=2, sticky='w')
    backtest_sl_entry = ttk.Entry(backtest_params_grid, textvariable=backtest_sl_var, width=7) 
    backtest_sl_entry.grid(row=2, column=1, padx=2, pady=2, sticky='w')

    ttk.Label(backtest_params_grid, text="Take Profit %:").grid(row=2, column=2, padx=2, pady=2, sticky='w')
    backtest_tp_entry = ttk.Entry(backtest_params_grid, textvariable=backtest_tp_var, width=7) 
    backtest_tp_entry.grid(row=2, column=3, padx=2, pady=2, sticky='w')

    ttk.Label(backtest_params_grid, text="SL PnL Amt ($):").grid(row=3, column=0, padx=2, pady=2, sticky='w')
    backtest_sl_pnl_amount_entry = ttk.Entry(backtest_params_grid, textvariable=backtest_sl_pnl_amount_var, width=7)
    backtest_sl_pnl_amount_entry.grid(row=3, column=1, padx=2, pady=2, sticky='w')

    ttk.Label(backtest_params_grid, text="TP PnL Amt ($):").grid(row=3, column=2, padx=2, pady=2, sticky='w')
    backtest_tp_pnl_amount_entry = ttk.Entry(backtest_params_grid, textvariable=backtest_tp_pnl_amount_var, width=7)
    backtest_tp_pnl_amount_entry.grid(row=3, column=3, padx=2, pady=2, sticky='w')

    ttk.Label(backtest_params_grid, text="Start Capital ($):").grid(row=4, column=0, padx=2, pady=2, sticky='w')
    backtest_starting_capital_entry = ttk.Entry(backtest_params_grid, textvariable=backtest_starting_capital_var, width=12)
    backtest_starting_capital_entry.grid(row=4, column=1, padx=2, pady=2, sticky='w')

    ttk.Label(backtest_params_grid, text="Leverage (e.g., 10):").grid(row=5, column=0, padx=2, pady=2, sticky='w')
    backtest_leverage_entry = ttk.Entry(backtest_params_grid, textvariable=backtest_leverage_var, width=7)
    backtest_leverage_entry.grid(row=5, column=1, padx=2, pady=2, sticky='w')
    
    ttk.Label(backtest_params_grid, text="Backtest Strategy:").grid(row=6, column=0, padx=2, pady=2, sticky='w')
    backtest_strategy_combobox = ttk.Combobox(backtest_params_grid, textvariable=backtest_selected_strategy_var, values=list(STRATEGIES.values()), width=25, state="readonly")
    if STRATEGIES: backtest_strategy_combobox.current(ACTIVE_STRATEGY_ID if ACTIVE_STRATEGY_ID in STRATEGIES else 0) 
    backtest_strategy_combobox.grid(row=6, column=1, columnspan=3, padx=2, pady=2, sticky='w') 
    
    account_summary_frame = ttk.LabelFrame(side_by_side_frame, text="Account Summary")
    account_summary_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

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

    price_display_frame = ttk.Frame(data_frame)
    price_display_frame.pack(pady=(2,2), padx=5, fill="x")
    current_price_var = tk.StringVar(value="Scanning: N/A - Price: N/A") 
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

    backtest_results_frame = ttk.LabelFrame(backtesting_frame, text="Backtest Results") 
    backtest_results_frame.pack(pady=(2,5), padx=5, fill="both", expand=True) 
    backtest_results_text_widget = scrolledtext.ScrolledText(backtest_results_frame, height=10, width=70, state=tk.DISABLED, wrap=tk.WORD)
    backtest_results_text_widget.pack(pady=5, padx=5, fill="both", expand=True)

    backtest_run_button = ttk.Button(backtesting_frame, text="Run Backtest", command=run_backtest_command) 
    backtest_run_button.pack(pady=5)

    # Status label is now a child of root, packed after scrollable_canvas_frame and h_scrollbar
    status_label = ttk.Label(root, textvariable=status_var, relief=tk.SUNKEN, anchor=tk.W)
    status_label.pack(padx=10, pady=(0,5), fill="x", side=tk.BOTTOM) # Keep it at the very bottom
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
