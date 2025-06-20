import pandas as pd
import abc
import pandas_ta as pta

class BaseIndicator(abc.ABC):
    def __init__(self, name: str, **params):
        self.name = name
        self.params = params

    @abc.abstractmethod
    def validate_params(self):
        pass

    @abc.abstractmethod
    def calculate(self, data: pd.DataFrame):
        pass

    @abc.abstractmethod
    def required_data_length(self):
        pass

class IndicatorRegistry:
    def __init__(self):
        self._indicators = {}

    def register_indicator(self, name: str, indicator_class):
        if not issubclass(indicator_class, BaseIndicator):
            raise TypeError(f"Indicator class {indicator_class.__name__} must inherit from BaseIndicator")
        self._indicators[name] = indicator_class

    def get_indicator(self, name: str, **params):
        if name not in self._indicators:
            raise ValueError(f"Indicator '{name}' not registered.")
        
        indicator_class = self._indicators[name]
        # Params for the indicator instance should be passed as a dictionary
        instance = indicator_class(name=name, **params)
        
        try:
            instance.validate_params()
        except ValueError as e:
            raise ValueError(f"Invalid parameters for indicator '{name}': {e}")
            
        return instance

class EMAIndicator(BaseIndicator):
    def __init__(self, name: str, **params):
        super().__init__(name, **params)
        # 'period' is expected to be in params, validated by validate_params
        if 'period' in self.params:
             self.period = self.params['period']
        else:
            # This case should ideally be caught by validate_params before this,
            # but as a safeguard:
            raise ValueError("Missing 'period' parameter for EMAIndicator.")


    def validate_params(self):
        if 'period' not in self.params:
            raise ValueError("Parameter 'period' is required for EMA.")
        
        period = self.params['period']
        if not isinstance(period, int) or period <= 0:
            raise ValueError(f"Parameter 'period' must be a positive integer, got {period}.")

    def required_data_length(self):
        return self.params['period']

    def calculate(self, data: pd.DataFrame):
        if 'close' not in data.columns:
            raise ValueError("Input DataFrame must contain a 'close' column.")
        
        if len(data) < self.required_data_length():
            raise ValueError(f"Insufficient data length. Required: {self.required_data_length()}, Available: {len(data)}")
        
        # Calculate EMA using pandas_ta
        ema_series = pta.ema(data['close'], length=self.params['period'])
        if ema_series is None:
            raise RuntimeError(f"pandas_ta.ema returned None for period {self.params['period']}. This might indicate an issue with the input data or pandas_ta itself.")
        return ema_series

class BollingerBandsIndicator(BaseIndicator):
    def __init__(self, name: str, **params):
        super().__init__(name, **params)
        if 'window' in self.params and 'std_dev' in self.params:
            self.window = self.params['window']
            self.std_dev = self.params['std_dev']
        else:
            # This case should ideally be caught by validate_params
            raise ValueError("Missing 'window' or 'std_dev' parameter for BollingerBandsIndicator.")

    def validate_params(self):
        if 'window' not in self.params or 'std_dev' not in self.params:
            raise ValueError("Parameters 'window' and 'std_dev' are required for Bollinger Bands.")
        
        window = self.params['window']
        std_dev = self.params['std_dev']

        if not isinstance(window, int) or window <= 0:
            raise ValueError(f"Parameter 'window' must be a positive integer, got {window}.")
        
        if not isinstance(std_dev, (int, float)) or std_dev <= 0:
            raise ValueError(f"Parameter 'std_dev' must be a positive number, got {std_dev}.")

    def required_data_length(self):
        return self.params['window']

    def calculate(self, data: pd.DataFrame):
        if 'close' not in data.columns:
            raise ValueError("Input DataFrame must contain a 'close' column.")
        
        if len(data) < self.required_data_length():
            raise ValueError(f"Insufficient data length. Required: {self.required_data_length()}, Available: {len(data)}")
        
        bbands_df = pta.bbands(data['close'], length=self.params['window'], std=self.params['std_dev'])
        if bbands_df is None:
            raise RuntimeError(f"pandas_ta.bbands returned None for window {self.params['window']} and std_dev {self.params['std_dev']}. This might indicate an issue with the input data or pandas_ta itself.")
        return bbands_df

class RSIIndicator(BaseIndicator):
    def __init__(self, name: str, **params):
        super().__init__(name, **params)
        if 'period' in self.params:
            self.period = self.params['period']
        else:
            raise ValueError("Missing 'period' parameter for RSIIndicator.")

    def validate_params(self):
        if 'period' not in self.params:
            raise ValueError("Parameter 'period' is required for RSI.")
        
        period = self.params['period']
        if not isinstance(period, int) or period <= 0:
            raise ValueError(f"Parameter 'period' must be a positive integer, got {period}.")

    def required_data_length(self):
        # pandas-ta's RSI might produce NaNs for the first `period` entries, 
        # but technically calculation can start with `period` data points.
        return self.params['period'] 

    def calculate(self, data: pd.DataFrame):
        if 'close' not in data.columns:
            raise ValueError("Input DataFrame must contain a 'close' column.")
        
        if len(data) < self.required_data_length():
            # It's often period + lookback for RSI to stabilize, but pandas-ta handles this.
            # We ensure at least 'period' points for the first possible non-NaN value.
            raise ValueError(f"Insufficient data length. Required: {self.required_data_length()}, Available: {len(data)}")
        
        rsi_series = pta.rsi(data['close'], length=self.params['period'])
        if rsi_series is None:
            raise RuntimeError(f"pandas_ta.rsi returned None for period {self.params['period']}. This might indicate an issue with the input data or pandas_ta itself.")
        return rsi_series

class MACDIndicator(BaseIndicator):
    def __init__(self, name: str, **params):
        super().__init__(name, **params)
        if 'fast_period' in self.params and 'slow_period' in self.params and 'signal_period' in self.params:
            self.fast_period = self.params['fast_period']
            self.slow_period = self.params['slow_period']
            self.signal_period = self.params['signal_period']
        else:
            raise ValueError("Missing 'fast_period', 'slow_period', or 'signal_period' for MACDIndicator.")

    def validate_params(self):
        required_params = ['fast_period', 'slow_period', 'signal_period']
        for p in required_params:
            if p not in self.params:
                raise ValueError(f"Parameter '{p}' is required for MACD.")

        fast = self.params['fast_period']
        slow = self.params['slow_period']
        signal = self.params['signal_period']

        if not all(isinstance(p_val, int) and p_val > 0 for p_val in [fast, slow, signal]):
            raise ValueError("Parameters 'fast_period', 'slow_period', and 'signal_period' must all be positive integers.")
        
        if fast >= slow:
            raise ValueError(f"Parameter 'fast_period' ({fast}) must be less than 'slow_period' ({slow}).")

    def required_data_length(self):
        # A common rule of thumb: slow_period for initial EMA + signal_period for signal line EMA.
        # pandas-ta might have specific internal needs, but this is a good baseline.
        return self.params['slow_period'] + self.params['signal_period'] 

    def calculate(self, data: pd.DataFrame):
        if 'close' not in data.columns:
            raise ValueError("Input DataFrame must contain a 'close' column.")
        
        if len(data) < self.required_data_length():
            raise ValueError(f"Insufficient data length. Required: {self.required_data_length()}, Available: {len(data)}")
            
        macd_df = pta.macd(data['close'], 
                           fast=self.params['fast_period'], 
                           slow=self.params['slow_period'], 
                           signal=self.params['signal_period'])
        if macd_df is None:
            raise RuntimeError(f"pandas_ta.macd returned None for fast={self.params['fast_period']}, slow={self.params['slow_period']}, signal={self.params['signal_period']}. This might indicate an issue with the input data or pandas_ta itself.")
        return macd_df
