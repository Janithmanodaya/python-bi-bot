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

# Binance API URLs
BINANCE_MAINNET_URL = "https://fapi.binance.com"
BINANCE_TESTNET_URL = "https://testnet.binancefuture.com"

# Global Configuration Variables
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
    except ClientError as error: print(f"Error get_balance_usdt: {error}")
    except Exception as e: print(f"Unexpected error get_balance_usdt: {e}")
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
    kl = klines(symbol)
    if kl is None or len(kl) < max(21, LOCAL_HIGH_LOW_LOOKBACK_PERIOD + 1): # Ensure enough data for all indicators
        return 'none'
    try:
        ema9 = ta.trend.EMAIndicator(close=kl['Close'], window=9).ema_indicator()
        ema21 = ta.trend.EMAIndicator(close=kl['Close'], window=21).ema_indicator()
        rsi = ta.momentum.RSIIndicator(close=kl['Close'], window=14).rsi()

        # Use the new SuperTrend calculation
        supertrend = calculate_supertrend_pta(kl, atr_period=10, multiplier=1.5) # kl is the DataFrame

        volume_ma10 = kl['Volume'].rolling(window=10).mean()

        if any(x is None for x in [ema9, ema21, rsi, supertrend, volume_ma10]) or \
           any(x.empty for x in [ema9, ema21, rsi, supertrend]) or volume_ma10.isnull().all():
            return 'none'

        # Ensure last values are not NaN before iloc[-1]
        required_length = max(10, LOCAL_HIGH_LOW_LOOKBACK_PERIOD +1) # Min length for volume_ma10 and lookback
        if len(ema9) < 1 or len(ema21) < 1 or len(rsi) < 1 or len(supertrend) < 1 or len(volume_ma10.dropna()) < 1 or len(kl['Volume']) < required_length:
            return 'none'

        last_ema9 = ema9.iloc[-1]
        prev_ema9 = ema9.iloc[-2] if len(ema9) > 1 else last_ema9
        last_ema21 = ema21.iloc[-1]
        prev_ema21 = ema21.iloc[-2] if len(ema21) > 1 else last_ema21
        last_rsi = rsi.iloc[-1]
        last_supertrend = supertrend.iloc[-1]
        last_volume = kl['Volume'].iloc[-1]
        last_volume_ma10 = volume_ma10.iloc[-1]
        current_price = kl['Close'].iloc[-1]

        if any(pd.isna(val) for val in [last_ema9, prev_ema9, last_ema21, prev_ema21, last_rsi, last_supertrend, last_volume, last_volume_ma10, current_price]):
            return 'none'

        # Refined Price Breakout Logic
        actual_lookback = min(LOCAL_HIGH_LOW_LOOKBACK_PERIOD, len(kl['High']) - 1)

        if actual_lookback > 0:
            recent_high = kl['High'].iloc[-(actual_lookback + 1):-1].max()
            recent_low = kl['Low'].iloc[-(actual_lookback + 1):-1].min()
        elif len(kl['High']) > 1 :
            recent_high = kl['High'].iloc[-2]
            recent_low = kl['Low'].iloc[-2]
        else:
            recent_high = current_price
            recent_low = current_price

        long_condition = (prev_ema9 < prev_ema21 and last_ema9 > last_ema21 and
                          50 <= last_rsi <= 70 and
                          last_supertrend == 'green' and
                          last_volume > last_volume_ma10 and
                          current_price > recent_high)

        short_condition = (prev_ema9 > prev_ema21 and last_ema9 < last_ema21 and
                           30 <= last_rsi <= 50 and
                           last_supertrend == 'red' and
                           last_volume > last_volume_ma10 and
                           current_price < recent_low)
        if long_condition: return 'up'
        elif short_condition: return 'down'
        else: return 'none'
    except Exception as e:
        # print(f"Error in scalping_strategy_signal for {symbol}: {e}") # Can be noisy
        return 'none'

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

def open_order(symbol, side):
    global client, ACCOUNT_RISK_PERCENT, SL_PERCENT, TP_PERCENT
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

        if side == 'buy':
            resp1 = client.new_order(symbol=symbol, side='BUY', type='LIMIT', quantity=calculated_qty_asset, timeInForce='GTC', price=price)
            print(f"BUY {symbol}: {resp1}")
            sleep(0.2); sl_price = round(price - price * SL_PERCENT, price_precision)
            resp2 = client.new_order(symbol=symbol, side='SELL', type='STOP_MARKET', quantity=calculated_qty_asset, timeInForce='GTC', stopPrice=sl_price, reduceOnly=True)
            print(f"SL for BUY {symbol}: {resp2}")
            sleep(0.2); tp_price = round(price + price * TP_PERCENT, price_precision)
            resp3 = client.new_order(symbol=symbol, side='SELL', type='TAKE_PROFIT_MARKET', quantity=calculated_qty_asset, timeInForce='GTC', stopPrice=tp_price, reduceOnly=True)
            print(f"TP for BUY {symbol}: {resp3}")
        elif side == 'sell':
            resp1 = client.new_order(symbol=symbol, side='SELL', type='LIMIT', quantity=calculated_qty_asset, timeInForce='GTC', price=price)
            print(f"SELL {symbol}: {resp1}")
            sleep(0.2); sl_price = round(price + price * SL_PERCENT, price_precision)
            resp2 = client.new_order(symbol=symbol, side='BUY', type='STOP_MARKET', quantity=calculated_qty_asset, timeInForce='GTC', stopPrice=sl_price, reduceOnly=True)
            print(f"SL for SELL {symbol}: {resp2}")
            sleep(0.2); tp_price = round(price - price * TP_PERCENT, price_precision)
            resp3 = client.new_order(symbol=symbol, side='BUY', type='TAKE_PROFIT_MARKET', quantity=calculated_qty_asset, timeInForce='GTC', stopPrice=tp_price, reduceOnly=True)
            print(f"TP for SELL {symbol}: {resp3}")
    except ClientError as error: print(f"Order Err ({symbol}, {side}): {error.error_message if error.error_message else error}")
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
            trades = client.my_trades(symbol=symbol, limit=limit_per_symbol)
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
        try:
            client = UMFutures(key=api_testnet, secret=secret_testnet, base_url=BINANCE_TESTNET_URL)
            msg = "Client: Binance Testnet"; print(msg); _status_set(msg)
        except Exception as e:
            msg = f"Error Testnet client: {e}"; print(msg); _status_set(msg); messagebox.showerror("Client Error", msg)
            return False
    elif current_env == "mainnet":
        if not api_mainnet or not secret_mainnet:
            msg = "Error: Mainnet keys missing."; _status_set(msg); messagebox.showerror("Error", msg + " Cannot switch.")
            current_env = "testnet";
            if current_env_var: current_env_var.set("testnet")
            try: client = UMFutures(key=api_testnet, secret=secret_testnet, base_url=BINANCE_TESTNET_URL); _status_set("Fell back to Testnet.")
            except Exception as e_fb: client = None; _status_set(f"Fallback to Testnet failed: {e_fb}")
            return False
        try:
            client = UMFutures(key=api_mainnet, secret=secret_mainnet, base_url=BINANCE_MAINNET_URL)
            msg = "Client: Binance Mainnet"; print(msg); _status_set(msg)
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
    if bot_running:
        messagebox.showwarning("Warning", "Change environment only when bot is stopped.")
        current_env_var.set(current_env); return
    if current_env != new_env:
        current_env = new_env; _status_set(f"Switching to {current_env}...")
        if reinitialize_client(): _status_set(f"Client ready for {current_env}.")
        else: current_env_var.set(current_env); _status_set(f"Failed switch to {new_env}. Now: {current_env}.")
    else:
        _status_set(f"Re-initializing for {current_env}...");
        if not reinitialize_client(): _status_set(f"Client re-init failed for {current_env}.")

def run_bot_logic():
    global bot_running, status_var, client, start_button, stop_button, testnet_radio, mainnet_radio
    global qty_concurrent_positions, type, leverage
    global balance_var, positions_text_widget, history_text_widget

    _status_set = lambda msg: status_var.set(msg) if status_var and root and root.winfo_exists() else None
    print("Bot logic thread started."); _status_set("Bot running...")
    if client is None: _status_set("Error: Client not initialized. Bot stopping."); bot_running = False

    loop_count = 0
    while bot_running:
        try:
            if client is None: _status_set("Client is None. Bot stopping."); bot_running = False; break
            balance = get_balance_usdt(); sleep(0.1)
            if not bot_running: break

            if root and root.winfo_exists() and balance_var:
                if balance is not None: root.after(0, lambda bal=balance: balance_var.set(f"{bal:.2f} USDT"))
                else: root.after(0, lambda: balance_var.set("Error or N/A"))

            if balance is None:
                msg = 'API/balance error. Retrying...'; print(msg); _status_set(msg)
                for _ in range(60):
                    if not bot_running: break
                    sleep(1)
                if not bot_running: break; continue

            current_balance_msg_for_status = f"Bal: {balance:.2f} USDT."
            _status_set(current_balance_msg_for_status + " Scanning...")

            active_positions_data = get_active_positions_data() # Structured data or None
            open_position_symbols = []
            if active_positions_data: # Not None and not empty
                open_position_symbols = [p['symbol'] for p in active_positions_data]

            current_positions_list_for_gui = format_positions_for_display(active_positions_data)
            if root and root.winfo_exists() and positions_text_widget:
                root.after(0, update_text_widget_content, positions_text_widget, current_positions_list_for_gui)
            if not bot_running: break

            if loop_count % 1 == 0:
                history_symbols_to_fetch = ['BTCUSDT', 'ETHUSDT']
                trade_history_list = get_trade_history(symbol_list=history_symbols_to_fetch, limit_per_symbol=10)
                if root and root.winfo_exists() and history_text_widget:
                    root.after(0, update_text_widget_content, history_text_widget, trade_history_list)
            if not bot_running: break

            all_open_orders_raw = check_orders()
            if all_open_orders_raw is None: all_open_orders_raw = []
            symbols_with_any_open_orders = {o.get('symbol') for o in all_open_orders_raw if isinstance(o, dict) and o.get('symbol')}

            for order_symbol in symbols_with_any_open_orders:
                 if order_symbol not in open_position_symbols:
                    close_open_orders(order_symbol)
            if not bot_running: break

            if len(open_position_symbols) < qty_concurrent_positions:
                symbols_to_check = get_tickers_usdt()
                if symbols_to_check is None: symbols_to_check = []

                all_open_orders_after_cancel = check_orders() # Refresh orders
                if all_open_orders_after_cancel is None: all_open_orders_after_cancel = []
                symbols_with_any_open_orders = {o.get('symbol') for o in all_open_orders_after_cancel if isinstance(o, dict) and o.get('symbol')}

                for sym_to_check in symbols_to_check:
                    if not bot_running: break
                    if sym_to_check == 'USDCUSDT': continue
                    if sym_to_check in open_position_symbols or sym_to_check in symbols_with_any_open_orders: continue

                    signal = scalping_strategy_signal(sym_to_check)
                    if signal == 'up' or signal == 'down':
                        _status_set(f"{signal.upper()} signal for {sym_to_check}. Ordering...");
                        set_mode(sym_to_check, type); sleep(0.1)
                        set_leverage(sym_to_check, leverage); sleep(0.1)
                        open_order(sym_to_check, signal)
                        active_positions_data = get_active_positions_data() # Refresh positions after order
                        if active_positions_data: open_position_symbols = [p['symbol'] for p in active_positions_data]
                        sleep(3)
                        if len(open_position_symbols) >= qty_concurrent_positions: break

            if not bot_running: break
            loop_count +=1
            wait_message = f"{current_balance_msg_for_status} Cycle done. Waiting 3 min..."
            _status_set(wait_message)
            for _ in range(180):
                if not bot_running: break; sleep(1)
        except ClientError as ce:
            err_msg = f"API Error: {ce.error_message if hasattr(ce, 'error_message') and ce.error_message else ce}. Waiting..."
            print(err_msg); _status_set(err_msg)
            if "signature" in str(ce).lower() or "timestamp" in str(ce).lower(): reinitialize_client()
            for _ in range(60):
                if not bot_running: break; sleep(1)
        except Exception as e:
            err_msg = f"Bot loop error: {e}. Waiting..."
            print(err_msg); _status_set(err_msg)
            for _ in range(60):
                if not bot_running: break; sleep(1)
            if not bot_running: break; continue

    print("Bot logic thread stopped.")
    _status_set("Bot stopped.")
    if start_button and root and root.winfo_exists(): start_button.config(state=tk.NORMAL)
    if stop_button and root and root.winfo_exists(): stop_button.config(state=tk.DISABLED)
    if testnet_radio and root and root.winfo_exists(): testnet_radio.config(state=tk.NORMAL)
    if mainnet_radio and root and root.winfo_exists(): mainnet_radio.config(state=tk.NORMAL)

def start_bot():
    global bot_running, bot_thread, status_var, client, start_button, stop_button, testnet_radio, mainnet_radio
    _status_set = lambda msg: status_var.set(msg) if status_var and root and root.winfo_exists() else None
    if bot_running: messagebox.showinfo("Info", "Bot is already running."); return
    if client is None:
        _status_set("Client not set. Initializing...")
        if not reinitialize_client() or client is None:
             messagebox.showerror("Error", "Client init failed."); _status_set("Client init failed."); return
    bot_running = True; _status_set("Bot starting...")
    if start_button: start_button.config(state=tk.DISABLED)
    if stop_button: stop_button.config(state=tk.NORMAL)
    if testnet_radio: testnet_radio.config(state=tk.DISABLED)
    if mainnet_radio: mainnet_radio.config(state=tk.DISABLED)
    bot_thread = threading.Thread(target=run_bot_logic, daemon=True); bot_thread.start()

def stop_bot():
    global bot_running, bot_thread, status_var, start_button, stop_button, testnet_radio, mainnet_radio
    _status_set = lambda msg: status_var.set(msg) if status_var and root and root.winfo_exists() else None
    if not bot_running and (bot_thread is None or not bot_thread.is_alive()):
        _status_set("Bot is not running.")
        if start_button: start_button.config(state=tk.NORMAL)
        if stop_button: stop_button.config(state=tk.DISABLED)
        if testnet_radio: testnet_radio.config(state=tk.NORMAL)
        if mainnet_radio: mainnet_radio.config(state=tk.NORMAL)
        return
    _status_set("Bot stopping..."); bot_running = False
    if stop_button: stop_button.config(state=tk.DISABLED)
    print("Stop signal sent to bot thread.")

# --- Main Application Window ---
if __name__ == "__main__":
    root = tk.Tk(); root.title("Binance Scalping Bot")
    # global status_var, current_env_var, start_button, stop_button, testnet_radio, mainnet_radio, balance_var, positions_text_widget, history_text_widget # This line caused SyntaxError
    status_var = tk.StringVar(value="Welcome! Select environment."); current_env_var = tk.StringVar(value=current_env); balance_var = tk.StringVar(value="N/A")
    controls_frame = ttk.LabelFrame(root, text="Controls"); controls_frame.pack(padx=10, pady=(5,0), fill="x")
    env_frame = ttk.Frame(controls_frame); env_frame.pack(pady=2, fill="x")
    ttk.Label(env_frame, text="Env:").pack(side=tk.LEFT, padx=(5,2))
    testnet_radio = ttk.Radiobutton(env_frame, text="Testnet", variable=current_env_var, value="testnet", command=toggle_environment); testnet_radio.pack(side=tk.LEFT, padx=2)
    mainnet_radio = ttk.Radiobutton(env_frame, text="Mainnet", variable=current_env_var, value="mainnet", command=toggle_environment); mainnet_radio.pack(side=tk.LEFT, padx=2)
    api_key_info_label = ttk.Label(controls_frame, text="API Keys from keys.py"); api_key_info_label.pack(pady=2)
    buttons_frame = ttk.Frame(controls_frame); buttons_frame.pack(pady=2)
    start_button = ttk.Button(buttons_frame, text="Start Bot", command=start_bot); start_button.pack(side=tk.LEFT, padx=5)
    stop_button = ttk.Button(buttons_frame, text="Stop Bot", command=stop_bot, state=tk.DISABLED); stop_button.pack(side=tk.LEFT, padx=5)
    data_frame = ttk.LabelFrame(root, text="Live Data & History"); data_frame.pack(padx=10, pady=5, fill="both", expand=True)
    balance_display_frame = ttk.Frame(data_frame); balance_display_frame.pack(pady=(5,2), padx=5, fill="x")
    ttk.Label(balance_display_frame, text="Account Balance:").pack(side=tk.LEFT, padx=(0,5))
    ttk.Label(balance_display_frame, textvariable=balance_var).pack(side=tk.LEFT)
    positions_display_frame = ttk.LabelFrame(data_frame, text="Current Open Positions"); positions_display_frame.pack(pady=2, padx=5, fill="both", expand=True)
    positions_text_widget = scrolledtext.ScrolledText(positions_display_frame, height=6, width=70, state=tk.DISABLED, wrap=tk.WORD); positions_text_widget.pack(pady=5, padx=5, fill="both", expand=True)
    history_display_frame = ttk.LabelFrame(data_frame, text="Recent Trade History (BTC, ETH)"); history_display_frame.pack(pady=(2,5), padx=5, fill="both", expand=True)
    history_text_widget = scrolledtext.ScrolledText(history_display_frame, height=10, width=70, state=tk.DISABLED, wrap=tk.WORD); history_text_widget.pack(pady=5, padx=5, fill="both", expand=True)
    status_label = ttk.Label(root, textvariable=status_var, relief=tk.SUNKEN, anchor=tk.W); status_label.pack(padx=10, pady=(0,5), fill="x", side=tk.BOTTOM)
    root.update_idletasks(); status_label.config(wraplength=root.winfo_width() - 20)
    root.bind("<Configure>", lambda event, widget=status_label: widget.config(wraplength=root.winfo_width() - 20))
    print("Attempting initial client initialization on startup...")
    if not reinitialize_client() or client is None:
        start_button.config(state=tk.DISABLED); status_var.set("Client not initialized. Start disabled.")
    else: status_var.set(f"Client initialized for {current_env}. Bot ready.")
    def on_closing():
        global bot_running, bot_thread, root
        if bot_running:
            if messagebox.askokcancel("Quit", "Bot is running. Quit anyway?"):
                bot_running = False
                if bot_thread and bot_thread.is_alive(): bot_thread.join(timeout=2)
                root.destroy()
        else: root.destroy()
    root.protocol("WM_DELETE_WINDOW", on_closing); root.minsize(450, 500); root.mainloop()
