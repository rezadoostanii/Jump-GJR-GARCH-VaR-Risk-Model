"""
Jump + GJR-GARCH Forecast for Financial Assets
Full version with Backtest, Heatmap, and SQLite integration
WITH VAR BACKTESTING (Kupiec & Christoffersen Tests) - FIXED
Author: [Reza Doostani]
Date: 2026-05-10
"""

import os
import sqlite3
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from arch import arch_model
from typing import Dict, Optional, List
from pathlib import Path
from scipy.stats import chi2, norm
import copy
import warnings
warnings.filterwarnings('ignore')


class VaRBacktest:
    """
    VaR Backtesting Methods:
    - Kupiec Test (Unconditional Coverage)
    - Christoffersen Test (Conditional Coverage)
    """
    
    @staticmethod
    def kupiec_test(exceptions: int, n_observations: int, confidence_level: float = 0.95) -> Dict:
        """
        Kupiec Test (Proportion of Failures Test)
        """
        expected_rate = 1 - confidence_level
        expected_exceptions = n_observations * expected_rate
        
        if expected_exceptions <= 0:
            return {'error': 'Invalid parameters'}
        
        # LR statistic: -2 * ln(L(null) / L(alternative))
        if exceptions == 0:
            lr_ratio = -2 * (n_observations * np.log(1 - expected_rate))
        elif exceptions == n_observations:
            lr_ratio = -2 * (n_observations * np.log(expected_rate))
        else:
            p_hat = exceptions / n_observations
            lr_ratio = -2 * (np.log((1 - expected_rate)**(n_observations - exceptions) * 
                                     expected_rate**exceptions) -
                            np.log((1 - p_hat)**(n_observations - exceptions) * 
                                   p_hat**exceptions))
        
        p_value = 1 - chi2.cdf(lr_ratio, df=1)
        reject_h0 = p_value < 0.05
        
        result = {
            'expected_exceptions': expected_exceptions,
            'actual_exceptions': exceptions,
            'failure_rate': exceptions / n_observations,
            'expected_rate': expected_rate,
            'lr_ratio': lr_ratio,
            'p_value': p_value,
            'reject_h0': reject_h0,
            'interpretation': 'Model is ACCURATE' if not reject_h0 else 'Model is INACCURATE (too many/few exceptions)'
        }
        
        # Additional interpretation
        if exceptions > expected_exceptions * 1.5:
            result['detailed'] = 'Model UNDERESTIMATES risk (too many exceptions)'
        elif exceptions < expected_exceptions * 0.5:
            result['detailed'] = 'Model OVERESTIMATES risk (too few exceptions)'
        else:
            result['detailed'] = 'Failure rate within acceptable range'
        
        return result
    
    @staticmethod
    def christoffersen_test(exceptions_series: np.ndarray, confidence_level: float = 0.95) -> Dict:
        """
        Christoffersen Test (Conditional Coverage)
        
        Parameters:
        -----------
        exceptions_series : np.ndarray
            Binary array where 1 = exception (VaR violation), 0 = no violation
        """
        n = len(exceptions_series)
        exceptions = int(exceptions_series.sum())  # Convert to int
        expected_rate = 1 - confidence_level
        
        if exceptions == 0 or exceptions == n:
            return {
                'error': 'Cannot test independence with 0 or all exceptions',
                'interpretation': 'Need more exceptions for independence test'
            }
        
        # Transition counts
        n00 = n01 = n10 = n11 = 0
        
        for i in range(1, n):
            if exceptions_series[i-1] == 0 and exceptions_series[i] == 0:
                n00 += 1
            elif exceptions_series[i-1] == 0 and exceptions_series[i] == 1:
                n01 += 1
            elif exceptions_series[i-1] == 1 and exceptions_series[i] == 0:
                n10 += 1
            elif exceptions_series[i-1] == 1 and exceptions_series[i] == 1:
                n11 += 1
        
        # Transition probabilities
        pi01 = n01 / (n00 + n01) if (n00 + n01) > 0 else 0
        pi11 = n11 / (n10 + n11) if (n10 + n11) > 0 else 0
        pi2 = (n01 + n11) / (n00 + n01 + n10 + n11)
        
        # Likelihood ratios
        if pi01 > 0 and pi11 > 0 and pi2 > 0 and pi2 < 1:
            lr_ind = -2 * (np.log((1 - pi2)**(n00 + n10) * pi2**(n01 + n11)) -
                          np.log((1 - pi01)**n00 * pi01**n01 *
                                 (1 - pi11)**n10 * pi11**n11))
        else:
            lr_ind = 0
        
        p_value_ind = 1 - chi2.cdf(lr_ind, df=1) if lr_ind > 0 else 0.5
        independence_ok = p_value_ind > 0.05
        
        # Overall Conditional Coverage test (combines Kupiec + Independence)
        test_kupiec = VaRBacktest.kupiec_test(exceptions, n, confidence_level)
        lr_cc = test_kupiec['lr_ratio'] + lr_ind
        p_value_cc = 1 - chi2.cdf(lr_cc, df=2)
        conditional_coverage_ok = p_value_cc > 0.05
        
        result = {
            'n_observations': n,
            'exceptions': exceptions,
            'failure_rate': exceptions / n,
            'expected_rate': expected_rate,
            'transition_counts': {'n00': n00, 'n01': n01, 'n10': n10, 'n11': n11},
            'transition_probabilities': {'pi01': pi01, 'pi11': pi11, 'pi2': pi2},
            'lr_ind': lr_ind,
            'p_value_ind': p_value_ind,
            'independence_ok': independence_ok,
            'lr_cc': lr_cc,
            'p_value_cc': p_value_cc,
            'conditional_coverage_ok': conditional_coverage_ok,
            'interpretation': VaRBacktest._get_interpretation(exceptions, n, expected_rate, 
                                                       independence_ok, conditional_coverage_ok)
        }
        
        return result
    
    @staticmethod
    def _get_interpretation(exceptions, n, expected_rate, independence_ok, conditional_ok):
        """Generate interpretation text."""
        actual_rate = exceptions / n
        
        if conditional_ok:
            if independence_ok:
                if abs(actual_rate - expected_rate) < 0.01:
                    return "✅ EXCELLENT: Model passes both tests perfectly"
                else:
                    return "✅ ACCEPTABLE: Conditional coverage is good"
            else:
                return "⚠️ WARNING: Exceptions are clustered (model misses crisis periods)"
        else:
            if exceptions > n * expected_rate * 1.5:
                return "❌ FAIL: Model UNDERESTIMATES risk significantly"
            elif exceptions < n * expected_rate * 0.5:
                return "⚠️ CAUTION: Model OVERESTIMATES risk (too conservative)"
            else:
                return "❌ FAIL: Model does not pass conditional coverage test"
    
    @staticmethod
    def comprehensive_backtest(returns: pd.Series, var_forecasts: pd.Series, 
                                confidence_level: float = 0.95) -> Dict:
        """
        Comprehensive VaR backtest combining multiple methods.
        """
        # Align data
        common_idx = returns.index.intersection(var_forecasts.index)
        returns_aligned = returns[common_idx]
        var_aligned = var_forecasts[common_idx]
        
        # Exceptions: actual loss > VaR
        losses = -returns_aligned.values
        var_values = var_aligned.values / 100  # Convert percentage to decimal
        
        exceptions = (losses > var_values).astype(int)
        n = len(exceptions)
        n_exceptions = int(exceptions.sum())
        
        # Run tests - FIXED: pass exceptions directly (it's already a numpy array)
        kupiec = VaRBacktest.kupiec_test(n_exceptions, n, confidence_level)
        christoffersen = VaRBacktest.christoffersen_test(exceptions, confidence_level)
        
        # Also calculate hit ratio
        hit_ratio = 1 - (n_exceptions / n)
        expected_hit_ratio = confidence_level
        
        # Calculate Mean Relative Scaled Error (MRSE) for VaR
        exception_losses = losses[exceptions == 1]
        exception_var = var_values[exceptions == 1]
        
        if len(exception_losses) > 0:
            var_error = (exception_losses - exception_var) / (exception_var + 1e-10)
            mse_var = np.mean(var_error ** 2)
            rmse_var = np.sqrt(mse_var)
        else:
            mse_var = np.nan
            rmse_var = np.nan
        
        result = {
            'n_observations': n,
            'exceptions': n_exceptions,
            'exception_rate': n_exceptions / n,
            'hit_rate': hit_ratio,
            'expected_hit_rate': expected_hit_ratio,
            'kupiec': kupiec,
            'christoffersen': christoffersen,
            'rmse_var': rmse_var,
            'final_verdict': VaRBacktest._final_verdict(kupiec, christoffersen)
        }
        
        return result
    
    @staticmethod
    def _final_verdict(kupiec, christoffersen):
        """Generate final verdict combining all tests."""
        if 'error' in christoffersen:
            if not kupiec['reject_h0']:
                return "PASS (Basic) - Independence test inconclusive"
            else:
                return "FAIL - Kupiec test indicates coverage issues"
        
        kupiec_pass = not kupiec['reject_h0']
        independence_pass = christoffersen['independence_ok']
        conditional_pass = christoffersen['conditional_coverage_ok']
        
        score = (kupiec_pass + independence_pass + conditional_pass) / 3
        
        if score == 1:
            return "⭐⭐⭐ EXCELLENT - Passes all tests"
        elif score >= 0.67:
            return "⭐⭐ GOOD - Passes most tests"
        elif score >= 0.33:
            return "⭐ ACCEPTABLE - Passes some tests"
        else:
            return "❌ POOR - Fails all tests"


class JumpGARCHModel:
    """
    GJR-GARCH(1,1) with Jump Diffusion and t-distribution innovations.
    """
    
    def __init__(self, 
                 trading_days_per_year: int = 252,
                 confidence_level: float = 0.05,
                 random_seed: int = 12345):
        
        self.trading_days = trading_days_per_year
        self.confidence_level = confidence_level
        self.random_seed = random_seed
        np.random.seed(random_seed)
        
        self.model = None
        self.params = None
        self.mu_pct = None
        self.nu = None
        self.lambda_jump = None
        self.mu_J = None
        self.sigma_J = None
        self.data = None
        self.returns = None
        self.ticker = None
        
    def load_from_sqlite(self, 
                         db_path: str, 
                         ticker: str, 
                         table_name: str = None,
                         start_date: str = None, 
                         end_date: str = None) -> pd.DataFrame:
        """Load price data from SQLite database."""
        if not os.path.exists(db_path):
            raise FileNotFoundError(f"Database not found: {db_path}")
        
        self.ticker = ticker
        
        if table_name is None:
            table_name = f"{ticker.lower()}_prices"
        
        print(f"📥 Loading data for {ticker} from SQLite database...")
        print(f"   Database: {db_path}")
        print(f"   Table: {table_name}")
        
        conn = sqlite3.connect(db_path)
        
        cursor = conn.cursor()
        cursor.execute(f"PRAGMA table_info({table_name})")
        columns = [col[1] for col in cursor.fetchall()]
        
        date_col = None
        for col in ['Date', 'date', 'datetime', 'Timestamp']:
            if col in columns:
                date_col = col
                break
        
        if date_col is None:
            conn.close()
            raise ValueError(f"No date column found in table {table_name}. Available columns: {columns}")
        
        price_col = None
        for col in ['Close', 'close', 'CLOSE', 'Price', 'price', 'Adj Close']:
            if col in columns:
                price_col = col
                break
        
        if price_col is None:
            conn.close()
            raise ValueError(f"No price column found in table {table_name}. Available columns: {columns}")
        
        query = f'SELECT "{date_col}" as Date, "{price_col}" as Close FROM {table_name}'
        
        if start_date:
            query += f" WHERE Date >= '{start_date}'"
        if end_date:
            if start_date:
                query += f" AND Date <= '{end_date}'"
            else:
                query += f" WHERE Date <= '{end_date}'"
        
        query += " ORDER BY Date ASC"
        
        df = pd.read_sql(query, conn)
        conn.close()
        
        if len(df) == 0:
            raise ValueError(f"No data found for {ticker} in table {table_name}.")
        
        df['Date'] = pd.to_datetime(df['Date'])
        df = df.set_index('Date').sort_index()
        df['Close'] = pd.to_numeric(df['Close'], errors='coerce')
        df = df.dropna()
        df['Return'] = np.log(df['Close'] / df['Close'].shift(1))
        df = df.dropna()
        
        self.data = df
        self.returns = df['Return']
        
        print(f"✅ Loaded {len(df)} observations from {df.index[0].date()} to {df.index[-1].date()}")
        print(f"   Last close price for {ticker}: ${df['Close'].iloc[-1]:,.2f}")
        
        return df
    
    def load_csv_data(self, file_path: str, date_col: str = None, price_col: str = None) -> pd.DataFrame:
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")
        
        df = pd.read_csv(file_path)
        
        if date_col is None:
            if '<DTYYYYMMDD>' in df.columns:
                date_col = '<DTYYYYMMDD>'
                df['Date'] = pd.to_datetime(df[date_col].astype(str), format='%Y%m%d')
            elif 'Date' in df.columns:
                df['Date'] = pd.to_datetime(df['Date'])
            else:
                raise ValueError("Could not detect date column.")
        
        if price_col is None:
            if '<Close>' in df.columns:
                price_col = '<Close>'
            elif 'Close' in df.columns:
                price_col = 'Close'
            else:
                raise ValueError("Could not detect price column.")
        
        df = df.set_index('Date').sort_index()
        df = df[[price_col]].copy()
        df.columns = ['Close']
        df['Return'] = np.log(df['Close'] / df['Close'].shift(1))
        df = df.dropna()
        
        self.data = df
        self.returns = df['Return']
        self.ticker = "CSV_DATA"
        
        print(f"✅ Loaded {len(df)} observations from {df.index[0].date()} to {df.index[-1].date()}")
        print(f"   Last close price: {df['Close'].iloc[-1]:,.2f}")
        
        return df
    
    def fit(self, returns_pct: Optional[pd.Series] = None, verbose: bool = True):
        """Fit GJR-GARCH(1,1) model with t-distribution."""
        if returns_pct is None:
            if self.returns is None:
                raise ValueError("No returns data.")
            returns_pct = self.returns * 100
        
        # Remove NaN/Inf values
        returns_pct = returns_pct.replace([np.inf, -np.inf], np.nan).dropna()
        
        self.model = arch_model(
            returns_pct, 
            vol='GARCH', 
            p=1, 
            o=1,
            q=1, 
            dist='t', 
            mean='Constant'
        )
        
        self.params = self.model.fit(disp='off')
        self.mu_pct = float(self.params.params['mu'])
        self.nu = float(self.params.params['nu'])
        
        if verbose:
            print("\n📊 GJR-GARCH(1,1) Model Summary:")
            print(self.params.summary())
            
            gamma = float(self.params.params['gamma[1]'])
            if gamma > 0:
                print(f"\n💡 Leverage Effect: γ = {gamma:.4f} > 0")
                print("   → Bad news increase volatility more than good news")
            else:
                print(f"\n📊 Leverage Effect: γ = {gamma:.4f}")
        
        return self.params
    
    def estimate_jump_params(self, jump_sigma_multiplier: float = 3.0, verbose: bool = True):
        """Estimate jump intensity and size."""
        if self.returns is None:
            raise ValueError("No returns data.")
        
        mu_daily = self.returns.mean()
        sigma_daily = self.returns.std()
        
        jump_threshold = jump_sigma_multiplier * sigma_daily
        is_jump = self.returns.abs() > jump_threshold
        self.lambda_jump = is_jump.sum() / len(self.returns) * self.trading_days
        
        jump_sizes = self.returns[is_jump] - mu_daily
        if len(jump_sizes) == 0:
            self.mu_J = 0.0
            self.sigma_J = 0.0
        else:
            self.mu_J = float(jump_sizes.mean())
            self.sigma_J = float(jump_sizes.std())
        
        if verbose:
            print("\n📈 Estimated Jump Parameters:")
            print(f"   λ (jumps per year): {self.lambda_jump:.3f}")
            print(f"   μ_J (jump mean): {self.mu_J:.6f}")
            print(f"   σ_J (jump std): {self.sigma_J:.6f}")
            print(f"   Total jumps detected: {is_jump.sum()}")
        
        return {
            'lambda_jump': self.lambda_jump,
            'mu_J': self.mu_J,
            'sigma_J': self.sigma_J
        }
    
    def _draw_standardized_t(self, size: int) -> np.ndarray:
        z = np.random.standard_t(self.nu, size=size)
        return z * np.sqrt((self.nu - 2.0) / self.nu)
    
    def _forecast_sigma_t_percent(self, horizon: int) -> np.ndarray:
        fc = self.params.forecast(horizon=horizon, reindex=False)
        var_h = fc.variance.values[-1]
        sigma_pct = np.sqrt(np.maximum(var_h, 0.0))
        return sigma_pct
    
    def simulate(self, 
                 horizon: int, 
                 n_simulations: int = 10000,
                 include_jumps: bool = True,
                 initial_price: Optional[float] = None) -> np.ndarray:
        
        if self.params is None:
            raise ValueError("Model not fitted. Run fit() first.")
        
        if initial_price is None:
            if self.data is None:
                raise ValueError("No data loaded.")
            initial_price = self.data['Close'].iloc[-1]
        
        sigma_t_pct = self._forecast_sigma_t_percent(horizon)
        sigma_t = sigma_t_pct / 100.0
        mu = self.mu_pct / 100.0
        
        paths = np.zeros((n_simulations, horizon + 1))
        paths[:, 0] = initial_price
        
        if include_jumps and self.lambda_jump is not None and self.lambda_jump > 0:
            lam_dt = self.lambda_jump / self.trading_days
        else:
            lam_dt = 0
        
        for t in range(1, horizon + 1):
            z = self._draw_standardized_t(n_simulations)
            r_diffusion = mu + sigma_t[t - 1] * z
            
            r_jump = np.zeros(n_simulations)
            if lam_dt > 0:
                K = np.random.poisson(lam_dt, size=n_simulations)
                idx = (K > 0)
                if idx.any():
                    r_jump[idx] = np.random.normal(
                        loc=K[idx] * self.mu_J,
                        scale=np.sqrt(K[idx]) * self.sigma_J
                    )
            
            r_total = r_diffusion + r_jump
            r_total = np.clip(r_total, -0.30, 0.30)
            paths[:, t] = paths[:, t - 1] * np.exp(r_total)
            paths[:, t] = np.clip(paths[:, t], initial_price * 0.10, initial_price * 10.0)
        
        return paths
    
    def calculate_var_forecast(self, horizon: int = 1, n_simulations: int = 10000) -> float:
        """
        Calculate 1-day VaR forecast.
        
        Returns:
            float: VaR as percentage (e.g., 2.5 means 2.5% loss)
        """
        if horizon != 1:
            # For multi-horizon, need more complex calculation
            paths = self.simulate(1, n_simulations, include_jumps=True)
            returns_sim = (paths[:, -1] - paths[:, 0]) / paths[:, 0]
            var = np.percentile(returns_sim, 100 * self.confidence_level)
            return abs(var) * 100
        else:
            # 1-day VaR using conditional distribution
            sigma_t_pct = self._forecast_sigma_t_percent(1)
            sigma_t = sigma_t_pct[0] / 100.0
            mu = self.mu_pct / 100.0
            
            # Student-t quantile
            t_quantile = np.sqrt((self.nu - 2) / self.nu) * np.percentile(
                self._draw_standardized_t(100000), 100 * self.confidence_level)
            
            var = -(mu + sigma_t * t_quantile)
            return max(var * 100, 0)
    
    def backtest_var(self, test_window: int = 500, n_simulations: int = 5000, 
                     verbose: bool = True) -> Dict:
        """
        Backtest VaR forecasts using Kupiec and Christoffersen tests.
        """
        if self.data is None or len(self.data) < test_window + 100:
            raise ValueError(f"Not enough data for backtest. Need at least {test_window + 100} days.")
        
        print(f"\n{'='*80}")
        print(f"📊 VAR BACKTESTING for {self.ticker}")
        print(f"   Test Window: {test_window} days")
        print(f"   VaR Confidence: {int((1 - self.confidence_level)*100)}%")
        print(f"{'='*80}")
        
        # Use last test_window days for testing
        test_data = self.data.iloc[-test_window:].copy()
        
        # Store actual returns and VaR forecasts
        actual_returns = []
        var_forecasts = []
        dates = []
        
        for i in range(50, len(test_data)):
            train_data = test_data.iloc[:i].copy()
            
            # Create temporary model
            temp_model = JumpGARCHModel(
                trading_days_per_year=self.trading_days,
                confidence_level=self.confidence_level,
                random_seed=self.random_seed + i
            )
            
            temp_model.data = train_data
            temp_model.returns = train_data['Return']
            temp_model.ticker = self.ticker
            
            try:
                temp_model.fit(verbose=False)
                temp_model.estimate_jump_params(verbose=False)
                
                # Calculate 1-day VaR forecast
                var_forecast = temp_model.calculate_var_forecast(horizon=1, n_simulations=n_simulations)
                
                actual_returns.append(test_data['Return'].iloc[i])
                var_forecasts.append(var_forecast)
                dates.append(test_data.index[i])
                
                if verbose and i % 100 == 0:
                    print(f"   Processed {i}/{len(test_data)} days...")
                    
            except Exception as e:
                if verbose:
                    print(f"   Warning: Could not process day {i}: {e}")
                continue
        
        if len(actual_returns) < 50:
            raise ValueError(f"Insufficient valid forecasts: {len(actual_returns)}")
        
        # Create series
        actual_returns_series = pd.Series(actual_returns, index=dates)
        var_forecasts_series = pd.Series(var_forecasts, index=dates)
        
        # Run comprehensive backtest
        backtest_results = VaRBacktest.comprehensive_backtest(
            actual_returns_series, var_forecasts_series, 1 - self.confidence_level
        )
        
        if verbose:
            self._print_backtest_results(backtest_results)
            self._plot_backtest_results(actual_returns_series, var_forecasts_series, 
                                        self.confidence_level, dates)
        
        return backtest_results
    
    def _print_backtest_results(self, results: Dict):
        """Print backtest results in formatted table."""
        print(f"\n{'='*80}")
        print("📊 VAR BACKTEST RESULTS")
        print(f"{'='*80}")
        
        # Summary statistics
        print(f"\n📈 SUMMARY STATISTICS:")
        print(f"   Observations:        {results['n_observations']}")
        print(f"   Exceptions:          {results['exceptions']} ({results['exception_rate']*100:.2f}%)")
        print(f"   Expected Rate:       {results['expected_hit_rate']*100:.2f}%")
        print(f"   Hit Rate:            {results['hit_rate']*100:.2f}%")
        
        # Kupiec Test
        print(f"\n📊 KUPIEC TEST (Unconditional Coverage):")
        kupiec = results['kupiec']
        if 'error' not in kupiec:
            print(f"   Expected Exceptions: {kupiec['expected_exceptions']:.1f}")
            print(f"   LR Statistic:       {kupiec['lr_ratio']:.4f}")
            print(f"   P-Value:            {kupiec['p_value']:.4f}")
            print(f"   Result:             {'❌ REJECT H0' if kupiec['reject_h0'] else '✅ FAIL TO REJECT H0'}")
            print(f"   Interpretation:     {kupiec['interpretation']}")
            print(f"   Details:            {kupiec['detailed']}")
        
        # Christoffersen Test
        print(f"\n📊 CHRISTOFFERSEN TEST (Conditional Coverage):")
        christoffersen = results['christoffersen']
        if 'error' not in christoffersen:
            print(f"   Independence Test:")
            print(f"     LR_IND:         {christoffersen['lr_ind']:.4f}")
            print(f"     P-Value:        {christoffersen['p_value_ind']:.4f}")
            print(f"     Result:         {'❌ Clustering detected' if not christoffersen['independence_ok'] else '✅ Independent'}")
            print(f"   Conditional Coverage Test:")
            print(f"     LR_CC:          {christoffersen['lr_cc']:.4f}")
            print(f"     P-Value:        {christoffersen['p_value_cc']:.4f}")
            print(f"     Result:         {'❌ FAIL' if not christoffersen['conditional_coverage_ok'] else '✅ PASS'}")
        
        # Transition matrix
        if 'transition_counts' in christoffersen:
            print(f"\n📊 EXCEPTION TRANSITION MATRIX:")
            tc = christoffersen['transition_counts']
            print(f"             Today: No Exc    Today: Exception")
            print(f"  Yesterday:  {tc['n00']:6d}        {tc['n01']:6d}")
            print(f"  Exception:  {tc['n10']:6d}        {tc['n11']:6d}")
        
        # Final Verdict
        print(f"\n{'='*80}")
        print(f"🏆 FINAL VERDICT: {results['final_verdict']}")
        print(f"{'='*80}")
    
    def _plot_backtest_results(self, actual_returns: pd.Series, var_forecasts: pd.Series, 
                                confidence_level: float, dates: List):
        """Plot backtest results."""
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # Plot 1: Actual returns vs VaR
        ax = axes[0, 0]
        actual_losses = -actual_returns.values * 100
        var_values = var_forecasts.values
        
        ax.plot(dates, actual_losses, 'b-', label='Actual Loss (%)', linewidth=1, alpha=0.7)
        ax.plot(dates, var_values, 'r--', label=f'VaR ({int((1-confidence_level)*100)}%)', linewidth=1.5)
        
        # Mark exceptions
        exceptions = actual_losses > var_values
        exception_dates = [dates[i] for i in range(len(dates)) if exceptions[i]]
        exception_losses = actual_losses[exceptions]
        ax.scatter(exception_dates, exception_losses, color='red', s=30, zorder=5, 
                  label=f'Exceptions ({exceptions.sum()})')
        
        ax.set_title(f'{self.ticker} - VaR Backtest')
        ax.set_xlabel('Date')
        ax.set_ylabel('Loss (%)')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Plot 2: Exception timeline
        ax = axes[0, 1]
        exception_binary = exceptions.astype(int)
        ax.fill_between(dates, exception_binary, 0, where=exception_binary == 1, 
                        color='red', alpha=0.5, label='Exceptions')
        ax.set_title('Exception Timeline')
        ax.set_xlabel('Date')
        ax.set_ylabel('VaR Violation')
        ax.set_ylim(-0.1, 1.1)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Plot 3: Rolling exception rate
        ax = axes[1, 0]
        rolling_window = min(50, len(exception_binary) // 10)
        rolling_rate = pd.Series(exception_binary).rolling(rolling_window).mean() * 100
        ax.plot(dates, rolling_rate, 'b-', linewidth=2)
        ax.axhline(y=(1-confidence_level)*100, color='r', linestyle='--', 
                  label=f'Expected: {(1-confidence_level)*100:.1f}%')
        ax.set_title(f'Rolling Exception Rate (window={rolling_window})')
        ax.set_xlabel('Date')
        ax.set_ylabel('Exception Rate (%)')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Plot 4: Loss distribution
        ax = axes[1, 1]
        ax.hist(actual_losses, bins=50, color='steelblue', edgecolor='black', alpha=0.7, density=True)
        ax.axvline(x=np.mean(var_values), color='r', linestyle='--', 
                  label=f'Mean VaR: {np.mean(var_values):.2f}%')
        ax.axvline(x=np.percentile(var_values, 50), color='orange', linestyle='--', 
                  label=f'Median VaR: {np.percentile(var_values, 50):.2f}%')
        ax.set_title('Loss Distribution with VaR Threshold')
        ax.set_xlabel('Loss (%)')
        ax.set_ylabel('Density')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
    
    def calculate_risk_measures(self, 
                                horizon: int, 
                                n_simulations: int = 10000,
                                include_stress: bool = True,
                                stress_shock: float = 0.9,
                                verbose: bool = True) -> Dict:
        
        if verbose:
            print(f"\n🔄 Simulating {n_simulations} paths for {horizon} days...")
        
        paths_garch = self.simulate(horizon, n_simulations, include_jumps=False)
        paths_jump = self.simulate(horizon, n_simulations, include_jumps=True)
        
        final_garch = paths_garch[:, -1]
        final_jump = paths_jump[:, -1]
        
        var_garch = np.percentile(final_garch, 100 * self.confidence_level)
        var_jump = np.percentile(final_jump, 100 * self.confidence_level)
        
        cvar_garch = final_garch[final_garch <= var_garch].mean()
        cvar_jump = final_jump[final_jump <= var_jump].mean()
        
        results = {
            'horizon': horizon,
            'GARCH': {
                'mean': final_garch.mean(),
                'median': np.median(final_garch),
                'std': final_garch.std(),
                'p5': np.percentile(final_garch, 5),
                'p95': np.percentile(final_garch, 95),
                'VaR': var_garch,
                'CVaR': cvar_garch
            },
            'Jump-GARCH': {
                'mean': final_jump.mean(),
                'median': np.median(final_jump),
                'std': final_jump.std(),
                'p5': np.percentile(final_jump, 5),
                'p95': np.percentile(final_jump, 95),
                'VaR': var_jump,
                'CVaR': cvar_jump
            }
        }
        
        if include_stress and self.data is not None:
            initial_price = self.data['Close'].iloc[-1]
            stress_price = initial_price * stress_shock
            
            paths_stress_garch = self.simulate(horizon, n_simulations, 
                                               include_jumps=False, initial_price=stress_price)
            paths_stress_jump = self.simulate(horizon, n_simulations, 
                                              include_jumps=True, initial_price=stress_price)
            
            results['GARCH']['stress_mean'] = paths_stress_garch[:, -1].mean()
            results['Jump-GARCH']['stress_mean'] = paths_stress_jump[:, -1].mean()
        
        return results
    
    def rolling_backtest(self,
                         window_size: int = 500,
                         forecast_horizon: int = 30,
                         n_simulations: int = 1000,
                         step_size: int = 20,
                         verbose: bool = False) -> pd.DataFrame:
        """Rolling window backtest."""
        if self.data is None:
            raise ValueError("No data loaded.")
        
        data = self.data.copy()
        results = []
        
        n_windows = max(1, (len(data) - window_size - forecast_horizon) // step_size)
        
        print(f"\n🔄 Running rolling backtest with {n_windows} windows...")
        print(f"   Window size: {window_size} days")
        print(f"   Forecast horizon: {forecast_horizon} days")
        print(f"   Step size: {step_size} days")
        print(f"   Simulations per window: {n_simulations}")
        
        for i in range(n_windows):
            start_idx = i * step_size
            end_idx = start_idx + window_size
            
            train_data = data.iloc[start_idx:end_idx].copy()
            actual_future = data.iloc[end_idx:end_idx + forecast_horizon].copy()
            
            if len(actual_future) < forecast_horizon:
                break
            
            temp_model = JumpGARCHModel(
                trading_days_per_year=self.trading_days,
                confidence_level=self.confidence_level,
                random_seed=self.random_seed + i
            )
            
            temp_model.data = train_data
            temp_model.returns = train_data['Return']
            temp_model.ticker = self.ticker
            
            temp_model.fit(verbose=False)
            temp_model.estimate_jump_params(verbose=False)
            
            initial_price = train_data['Close'].iloc[-1]
            paths_jump = temp_model.simulate(forecast_horizon, n_simulations, 
                                             include_jumps=True, initial_price=initial_price)
            
            actual_final = actual_future['Close'].iloc[-1]
            predicted_mean = paths_jump[:, -1].mean()
            predicted_p5 = np.percentile(paths_jump[:, -1], 5)
            predicted_p95 = np.percentile(paths_jump[:, -1], 95)
            
            error_pct = (actual_final - predicted_mean) / actual_final * 100
            
            try:
                one_day_var = temp_model.calculate_var_forecast(horizon=1, n_simulations=n_simulations)
            except:
                one_day_var = np.nan
            
            results.append({
                'window_start': train_data.index[0],
                'window_end': train_data.index[-1],
                'forecast_date': actual_future.index[-1],
                'actual_price': actual_final,
                'predicted_mean': predicted_mean,
                'predicted_p5': predicted_p5,
                'predicted_p95': predicted_p95,
                'error_abs': abs(actual_final - predicted_mean),
                'error_pct': error_pct,
                'abs_error_pct': abs(error_pct),
                'within_interval': int(predicted_p5 <= actual_final <= predicted_p95),
                'one_day_var_95': one_day_var
            })
            
            if (i + 1) % max(1, n_windows // 5) == 0:
                print(f"      Completed {i+1}/{n_windows} windows")
        
        results_df = pd.DataFrame(results)
        
        if len(results_df) > 0:
            print(f"\n📊 Backtest Summary for H={forecast_horizon}:")
            print(f"   Number of tests: {len(results_df)}")
            print(f"   Mean Absolute Error: ${results_df['error_abs'].mean():.2f}")
            print(f"   Mean Absolute Error (%): {results_df['abs_error_pct'].mean():.1f}%")
            print(f"   RMSE: ${np.sqrt((results_df['error_abs']**2).mean()):.2f}")
            print(f"   Within 5-95% Interval: {results_df['within_interval'].mean()*100:.1f}%")
            print(f"   Avg 1-day VaR (95%): {results_df['one_day_var_95'].mean():.2f}%")
        
        self._plot_backtest_results_enhanced(results_df, forecast_horizon)
        
        return results_df
    
    def _plot_backtest_results_enhanced(self, results_df: pd.DataFrame, horizon: int):
        """Plot enhanced backtest results."""
        if len(results_df) == 0:
            return
            
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # Plot 1: Actual vs Predicted
        ax = axes[0, 0]
        ax.plot(results_df['forecast_date'], results_df['actual_price'], 
                'k-', label='Actual Price', linewidth=2)
        ax.plot(results_df['forecast_date'], results_df['predicted_mean'], 
                'r--', label='Predicted Mean', linewidth=1.5)
        ax.fill_between(results_df['forecast_date'], 
                        results_df['predicted_p5'], 
                        results_df['predicted_p95'], 
                        color='red', alpha=0.2, label='5-95% Prediction Interval')
        ax.set_title(f'{self.ticker} - Rolling Backtest (H={horizon}d)')
        ax.set_xlabel('Date')
        ax.set_ylabel('Price ($)')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Plot 2: Prediction Errors
        ax = axes[0, 1]
        colors = ['green' if e >= 0 else 'red' for e in results_df['error_pct']]
        ax.bar(results_df['forecast_date'], results_df['error_pct'], 
               color=colors, alpha=0.7)
        ax.axhline(y=0, color='k', linestyle='-', linewidth=1)
        ax.set_title('Prediction Errors (%)')
        ax.set_xlabel('Date')
        ax.set_ylabel('Error (%)')
        ax.grid(True, alpha=0.3)
        
        # Plot 3: 1-day VaR over time
        ax = axes[1, 0]
        ax.plot(results_df['forecast_date'], results_df['one_day_var_95'], 
                'b-', linewidth=1.5, label='1-day VaR (95%)')
        ax.axhline(y=results_df['one_day_var_95'].mean(), color='r', linestyle='--', 
                  label=f'Mean VaR: {results_df["one_day_var_95"].mean():.2f}%')
        ax.set_title('1-day VaR (95%) Evolution')
        ax.set_xlabel('Date')
        ax.set_ylabel('VaR (%)')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Plot 4: Error distribution
        ax = axes[1, 1]
        ax.hist(results_df['error_pct'], bins=30, color='steelblue', edgecolor='black', alpha=0.7)
        ax.axvline(x=0, color='red', linestyle='-', linewidth=2, label='Zero Error')
        ax.axvline(x=results_df['error_pct'].mean(), color='green', linestyle='--', 
                   label=f'Mean Error: {results_df["error_pct"].mean():.1f}%')
        ax.set_title('Distribution of Prediction Errors (%)')
        ax.set_xlabel('Error (%)')
        ax.set_ylabel('Frequency')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
    
    def create_risk_heatmap(self, horizons: List[int], n_simulations: int = 5000):
        """Create a heatmap of risk measures."""
        print("\n🔥 Generating Risk Heatmap...")
        
        heatmap_data = []
        
        for H in horizons:
            results = self.calculate_risk_measures(H, n_simulations, include_stress=False, verbose=False)
            
            for model in ['GARCH', 'Jump-GARCH']:
                heatmap_data.append({
                    'Horizon': H,
                    'Model': model,
                    'Mean': results[model]['mean'],
                    'VaR_5%': results[model]['VaR'],
                    'CVaR_5%': results[model]['CVaR'],
                    'P5': results[model]['p5'],
                    'P95': results[model]['p95']
                })
        
        heatmap_df = pd.DataFrame(heatmap_data)
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        
        # Heatmap 1: VaR and CVaR
        pivot_var = heatmap_df.pivot(index='Horizon', columns='Model', values='VaR_5%')
        pivot_cvar = heatmap_df.pivot(index='Horizon', columns='Model', values='CVaR_5%')
        
        combined_var = pd.DataFrame({
            'GARCH_VaR': pivot_var['GARCH'],
            'Jump_VaR': pivot_var['Jump-GARCH'],
            'GARCH_CVaR': pivot_cvar['GARCH'],
            'Jump_CVaR': pivot_cvar['Jump-GARCH']
        }, index=horizons)
        
        im = axes[0].imshow(combined_var.T.values, cmap='RdYlGn_r', aspect='auto')
        axes[0].set_xticks(range(len(horizons)))
        axes[0].set_xticklabels([f"{h}d" for h in horizons])
        axes[0].set_yticks(range(len(combined_var.columns)))
        axes[0].set_yticklabels(combined_var.columns)
        axes[0].set_title('Risk Measures: VaR (5%) and CVaR (5%)')
        plt.colorbar(im, ax=axes[0], label='Price ($)')
        
        for i in range(len(combined_var.columns)):
            for j in range(len(horizons)):
                axes[0].text(j, i, f'${combined_var.iloc[i, j]:.0f}',
                           ha="center", va="center", color="black", fontsize=9)
        
        # Heatmap 2: Prediction intervals
        pivot_p5 = heatmap_df.pivot(index='Horizon', columns='Model', values='P5')
        pivot_p95 = heatmap_df.pivot(index='Horizon', columns='Model', values='P95')
        
        combined_interval = pd.DataFrame({
            'GARCH_P5': pivot_p5['GARCH'],
            'Jump_P5': pivot_p5['Jump-GARCH'],
            'GARCH_P95': pivot_p95['GARCH'],
            'Jump_P95': pivot_p95['Jump-GARCH']
        }, index=horizons)
        
        im2 = axes[1].imshow(combined_interval.T.values, cmap='coolwarm', aspect='auto')
        axes[1].set_xticks(range(len(horizons)))
        axes[1].set_xticklabels([f"{h}d" for h in horizons])
        axes[1].set_yticks(range(len(combined_interval.columns)))
        axes[1].set_yticklabels(combined_interval.columns)
        axes[1].set_title('Prediction Intervals: 5th and 95th Percentiles')
        plt.colorbar(im2, ax=axes[1], label='Price ($)')
        
        for i in range(len(combined_interval.columns)):
            for j in range(len(horizons)):
                axes[1].text(j, i, f'${combined_interval.iloc[i, j]:.0f}',
                           ha="center", va="center", color="black", fontsize=9)
        
        plt.tight_layout()
        plt.show()
        
        return heatmap_df
    
    def print_summary_table(self, horizons: List[int], n_simulations: int = 10000):
        """Print a formatted summary table."""
        print("\n" + "="*90)
        print(f"📊 {self.ticker if self.ticker else 'ASSET'} - RISK MANAGEMENT SUMMARY")
        print("="*90)
        
        header = f"{'Horizon':<10} {'Model':<12} {'Mean':<12} {'P5':<12} {'P95':<12} {'VaR(5%)':<12} {'CVaR(5%)':<12}"
        print(header)
        print("-"*90)
        
        for H in horizons:
            results = self.calculate_risk_measures(H, n_simulations, include_stress=False, verbose=False)
            
            print(f"{H:<10} {'GARCH':<12} ${results['GARCH']['mean']:<11.2f} ${results['GARCH']['p5']:<11.2f} ${results['GARCH']['p95']:<11.2f} ${results['GARCH']['VaR']:<11.2f} ${results['GARCH']['CVaR']:<11.2f}")
            print(f"{'':<10} {'Jump-GARCH':<12} ${results['Jump-GARCH']['mean']:<11.2f} ${results['Jump-GARCH']['p5']:<11.2f} ${results['Jump-GARCH']['p95']:<11.2f} ${results['Jump-GARCH']['VaR']:<11.2f} ${results['Jump-GARCH']['CVaR']:<11.2f}")
            print("-"*90)
    
    def plot_forecast(self, horizon: int, n_simulations: int = 5000):
        """Plot fan chart and distribution."""
        paths_garch = self.simulate(horizon, n_simulations, include_jumps=False)
        paths_jump = self.simulate(horizon, n_simulations, include_jumps=True)
        
        days = np.arange(0, horizon + 1)
        
        p5_g = np.percentile(paths_garch, 5, axis=0)
        p95_g = np.percentile(paths_garch, 95, axis=0)
        mean_g = np.mean(paths_garch, axis=0)
        
        p5_j = np.percentile(paths_jump, 5, axis=0)
        p95_j = np.percentile(paths_jump, 95, axis=0)
        mean_j = np.mean(paths_jump, axis=0)
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        ax = axes[0]
        ax.plot(days, mean_g, label='GARCH Mean', color='C0', linewidth=2)
        ax.fill_between(days, p5_g, p95_g, color='C0', alpha=0.2, label='GARCH 5-95%')
        ax.plot(days, mean_j, label='Jump-GARCH Mean', color='C3', linewidth=2)
        ax.fill_between(days, p5_j, p95_j, color='C3', alpha=0.2, label='Jump-GARCH 5-95%')
        
        for i in range(min(20, n_simulations)):
            ax.plot(days, paths_jump[i], color='gray', alpha=0.1)
        
        current_price = self.data['Close'].iloc[-1]
        ax.axhline(y=current_price, color='k', linestyle='--', alpha=0.5, label=f'Current: ${current_price:.2f}')
        ax.set_title(f'{self.ticker} - {horizon} Day Price Forecast')
        ax.set_xlabel('Days Ahead')
        ax.set_ylabel('Price ($)')
        ax.legend(loc='upper left')
        ax.grid(True, alpha=0.3)
        
        ax = axes[1]
        ax.hist(paths_garch[:, -1], bins=80, alpha=0.5, label='GARCH', color='C0')
        ax.hist(paths_jump[:, -1], bins=80, alpha=0.5, label='Jump-GARCH', color='C3')
        ax.axvline(np.percentile(paths_garch[:, -1], 5), color='C0', linestyle='--', alpha=0.7)
        ax.axvline(np.percentile(paths_garch[:, -1], 95), color='C0', linestyle='--', alpha=0.7)
        ax.axvline(np.percentile(paths_jump[:, -1], 5), color='C3', linestyle='--', alpha=0.7)
        ax.axvline(np.percentile(paths_jump[:, -1], 95), color='C3', linestyle='--', alpha=0.7)
        ax.axvline(current_price, color='k', linestyle='-', alpha=0.5, label=f'Current: ${current_price:.2f}')
        ax.set_title(f'Distribution of Final Price (Day {horizon})')
        ax.set_xlabel('Price ($)')
        ax.set_ylabel('Frequency')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
        
        return fig


# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    
    print("\n" + "="*60)
    print("🚀 JUMP-GJR-GARCH MODEL WITH VAR BACKTESTING")
    print("   Kupiec & Christoffersen Tests Included")
    print("="*60)
    
    DB_PATH = Path.home() / "Downloads" / "portfolio.db"
    TICKER = 'NVDA'
    TABLE_NAME = 'nvda_prices'
    
    model = JumpGARCHModel(
        trading_days_per_year=252,
        confidence_level=0.05,
        random_seed=12345
    )
    
    # Load data
    try:
        model.load_from_sqlite(
            db_path=str(DB_PATH),
            ticker=TICKER,
            table_name=TABLE_NAME,
            start_date='2020-01-01',
            end_date='2026-05-03'
        )
    except FileNotFoundError:
        print(f"\n⚠️ SQLite database not found at: {DB_PATH}")
        print("Using sample data generation for demonstration...")
        
        dates = pd.date_range(start='2020-01-01', end='2026-05-03', freq='D')
        np.random.seed(12345)
        returns = np.random.normal(0.0005, 0.02, len(dates))
        prices = 100 * np.exp(np.cumsum(returns))
        df = pd.DataFrame({'Close': prices, 'Return': returns}, index=dates)
        model.data = df
        model.returns = df['Return']
        model.ticker = TICKER
        print(f"✅ Generated {len(df)} sample observations")
    
    # Fit model
    model.fit()
    model.estimate_jump_params(jump_sigma_multiplier=3.0)
    
    # Run VaR backtesting
    print("\n" + "="*60)
    print("🔍 RUNNING VAR BACKTESTING")
    print("="*60)
    
    try:
        var_backtest_results = model.backtest_var(
            test_window=500,
            n_simulations=5000,
            verbose=True
        )
    except ValueError as e:
        print(f"⚠️ Cannot run backtest: {e}")
    
    # Standard analysis
    horizons = [30, 60, 90, 120]
    model.print_summary_table(horizons, n_simulations=10000)
    
    model.create_risk_heatmap(horizons, n_simulations=5000)
    
    print("\n📈 Generating forecast charts...")
    model.plot_forecast(horizon=60, n_simulations=5000)
    model.plot_forecast(horizon=120, n_simulations=5000)
    
    print("\n" + "="*60)
    print("🔄 RUNNING ROLLING BACKTEST")
    print("="*60)
    
    if len(model.data) > 800:
        backtest_results = model.rolling_backtest(
            window_size=500,
            forecast_horizon=30,
            n_simulations=500,
            step_size=20
        )
        
        if len(backtest_results) > 0:
            print("\n📋 Sample Backtest Results (last 5 windows):")
            cols = ['forecast_date', 'actual_price', 'predicted_mean', 'error_pct', 'one_day_var_95', 'within_interval']
            print(backtest_results[cols].tail(5).to_string())
    else:
        print("⚠️ Not enough data for rolling backtest (need >800 observations)")
    
    print("\n" + "="*60)
    print("✅ ANALYSIS COMPLETE")
    print("="*60)
    
    print("\n💡 KEY INSIGHTS WITH VAR BACKTESTING:")
    print("   • Kupiec Test: Checks if exception rate matches expectations")
    print("   • Christoffersen Test: Checks for exception clustering")
    print("   • GJR-GARCH captures asymmetric volatility (leverage effect)")
    print("   • Jump component models sudden price movements")
    print("   • VaR and CVaR provide downside risk measures")

