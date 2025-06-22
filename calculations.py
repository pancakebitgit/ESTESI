import pandas as pd
import numpy as np
from scipy.stats import norm
import datetime # Required for date calculations if DTE is not directly available

def calculate_black_scholes_gamma(S, K, T, r, sigma):
    """
    Calculates the Black-Scholes Gamma for an option.
    S: Current price of the underlying asset (can be scalar or array)
    K: Strike price of the option (array)
    T: Time to expiration (in years) (array)
    r: Risk-free interest rate (annualized) (scalar or array)
    sigma: Volatility of the underlying asset (annualized) (array)
    """
    _S = np.asarray(S)
    _K = np.asarray(K)
    _T = np.asarray(T)
    _r = np.asarray(r)
    _sigma = np.asarray(sigma)

    # Initialize gamma array with NaNs, matching shape of K
    gamma_output = np.full_like(_K, np.nan, dtype=np.float64)

    # Create a mask for valid calculations based on array inputs K, T, sigma
    # and scalar/array S, r
    # S must be > 0. If S is an array, this check is element-wise.
    # If S is scalar, it's a single check.
    s_is_scalar = _S.ndim == 0

    if s_is_scalar:
        if _S <= 1e-9: # If scalar S is invalid, all gammas are NaN (or could be 0)
            return gamma_output
        # Create a boolean array for S matching K's shape for the mask
        s_condition = np.full_like(_K, True, dtype=bool) # Placeholder, S is valid
    else:
        s_condition = (_S > 1e-9)

    # Mask for options that are valid for calculation (T>0, sigma>0, K>0 and S>0)
    # This mask will have the same shape as K, T, sigma
    valid_mask = (_T > 1e-9) & (_sigma > 1e-9) & (_K > 1e-9) & s_condition

    if not np.any(valid_mask):
        return gamma_output # No valid options to calculate

    # Select only valid rows for calculation to avoid warnings/errors
    # If S or r are scalars, they will broadcast correctly.
    # If S or r are arrays, they must be filtered by valid_mask as well.
    S_eff = _S if s_is_scalar else _S[valid_mask]
    K_eff = _K[valid_mask]
    T_eff = _T[valid_mask]
    r_eff = _r if np.isscalar(_r) else (_r[valid_mask] if _r.ndim > 0 else _r) # Handle scalar r or array r
    sigma_eff = _sigma[valid_mask]

    # d1 calculation
    d1_numerator = np.log(S_eff / K_eff) + (r_eff + 0.5 * sigma_eff**2) * T_eff
    d1_denominator = sigma_eff * np.sqrt(T_eff)

    # Avoid division by zero in d1_denominator
    # np.divide handles division by zero by returning inf or nan, which is fine for norm.pdf
    d1 = np.divide(d1_numerator, d1_denominator, out=np.full_like(d1_denominator, np.nan), where=d1_denominator!=0)

    phi_d1 = norm.pdf(d1)

    # Gamma calculation
    gamma_denominator = S_eff * sigma_eff * np.sqrt(T_eff)

    # Calculate gamma for the valid subset
    calculated_gamma_subset = np.divide(phi_d1, gamma_denominator, out=np.full_like(gamma_denominator, np.nan), where=gamma_denominator!=0)

    # Place the calculated gammas back into the original shaped array
    gamma_output[valid_mask] = calculated_gamma_subset

    return gamma_output


def calculate_gex_analytics(df_options_original, current_spot_price, risk_free_rate, calculation_date=None):
    """
    Calculates Gamma Exposure by Strike, Aggregate Gamma Exposure Curve, and Gamma Flip Point.
    """
    if df_options_original is None or df_options_original.empty:
        return pd.DataFrame(columns=['Strike', 'GEX_Signed_Value']), pd.DataFrame(columns=['Simulated_Price', 'Total_GEX']), np.nan, 0

    required_cols = ['Strike', 'Open Int', 'Type', 'IV']
    if not all(col in df_options_original.columns for col in required_cols):
        print(f"GEX Error: Missing one or more required columns: {required_cols}")
        return pd.DataFrame(columns=['Strike', 'GEX_Signed_Value']), pd.DataFrame(columns=['Simulated_Price', 'Total_GEX']), np.nan, 0

    if 'Exp Date' not in df_options_original.columns and 'DTE' not in df_options_original.columns:
        print("GEX Error: Missing 'Exp Date' or 'DTE' column for time to maturity calculation.")
        return pd.DataFrame(columns=['Strike', 'GEX_Signed_Value']), pd.DataFrame(columns=['Simulated_Price', 'Total_GEX']), np.nan, 0

    df = df_options_original.copy()
    df['Type'] = df['Type'].str.lower()

    if calculation_date is None:
        calculation_date = pd.Timestamp('today').normalize()
    else:
        calculation_date = pd.to_datetime(calculation_date).normalize()

    if 'DTE' not in df.columns:
        df['Exp Date'] = pd.to_datetime(df['Exp Date'])
        df['DTE'] = (df['Exp Date'] - calculation_date).dt.days

    df['T_years'] = df['DTE'] / 365.25

    # Filter for options with T_years > 0 (and other conditions handled in calculate_black_scholes_gamma)
    # Ensure IV is numeric and positive
    df['IV'] = pd.to_numeric(df['IV'], errors='coerce')
    df = df[(df['T_years'] > 1e-6) & (df['IV'] > 1e-6) & (df['Strike'] > 1e-6) & (df['Open Int'] >= 0)]

    if df.empty:
        print("GEX Warning: No valid options after filtering for T_years > 0, IV > 0, Strike > 0.")
        return pd.DataFrame(columns=['Strike', 'GEX_Signed_Value']), pd.DataFrame(columns=['Simulated_Price', 'Total_GEX']), np.nan, 0

    # Part A: Calculate Gamma for each contract using current_spot_price
    df['Calculated_Gamma'] = calculate_black_scholes_gamma(
        S=current_spot_price,
        K=df['Strike'].values, # Pass as numpy array
        T=df['T_years'].values,
        r=risk_free_rate,
        sigma=df['IV'].values
    )
    # Drop rows where gamma calculation failed (e.g., due to T=0 or other invalid BS params)
    df.dropna(subset=['Calculated_Gamma'], inplace=True)
    if df.empty:
        print("GEX Warning: No options with valid Calculated_Gamma at current spot price.")
        return pd.DataFrame(columns=['Strike', 'GEX_Signed_Value']), pd.DataFrame(columns=['Simulated_Price', 'Total_GEX']), np.nan, 0

    # Part B: Calculate Gamma Exposure by Strike (for the bars)
    df['GEX_Contract_Value'] = df['Calculated_Gamma'] * df['Open Int'] * 100
    df['GEX_Signed_Value'] = np.where(
        df['Type'] == 'call',
        df['GEX_Contract_Value'],
        -df['GEX_Contract_Value']
    )

    gex_bars_data = df.groupby('Strike')['GEX_Signed_Value'].sum().reset_index()
    net_gex_at_current_price = np.nansum(gex_bars_data['GEX_Signed_Value'])


    # Part C: Calculate Aggregate Gamma Exposure Curve and Flip Point
    min_price_sim = current_spot_price * 0.80
    max_price_sim = current_spot_price * 1.20
    price_step_sim = max(0.1, round((max_price_sim - min_price_sim) / 100, 2)) # Aim for ~100 points
    if price_step_sim == 0: price_step_sim = 0.5
    test_price_range = np.arange(min_price_sim, max_price_sim + price_step_sim, price_step_sim)

    simulated_gex_totals = []

    # Prepare arrays from the DataFrame for faster processing in the loop
    # These are from the already filtered 'df' which has valid T_years, IV etc.
    strikes_arr = df['Strike'].values
    t_years_arr = df['T_years'].values
    iv_arr = df['IV'].values
    oi_arr = df['Open Int'].values
    type_is_call_arr = (df['Type'] == 'call').values

    for s_sim in test_price_range:
        gamma_sim_all = calculate_black_scholes_gamma(
            S=s_sim, K=strikes_arr, T=t_years_arr, r=risk_free_rate, sigma=iv_arr
        )

        gex_contract_sim = gamma_sim_all * oi_arr * 100
        # Ensure gex_signed_sim is calculated correctly, handling NaNs from gamma_sim if any
        gex_signed_sim = np.where(type_is_call_arr, gex_contract_sim, -gex_contract_sim)
        total_gex_for_s_sim = np.nansum(gex_signed_sim)

        simulated_gex_totals.append({'Simulated_Price': s_sim, 'Total_GEX': total_gex_for_s_sim})

    df_aggregate_curve = pd.DataFrame(simulated_gex_totals)
    if df_aggregate_curve.empty: # Should not happen if test_price_range is not empty
        return gex_bars_data, pd.DataFrame(columns=['Simulated_Price', 'Total_GEX']), np.nan, net_gex_at_current_price

    df_aggregate_curve.sort_values(by='Simulated_Price', inplace=True)

    # Find Gamma Flip Point
    gamma_flip_price = np.nan
    # Shift Total_GEX to compare previous with current
    gex_values = df_aggregate_curve['Total_GEX'].values
    prices = df_aggregate_curve['Simulated_Price'].values

    for i in range(1, len(gex_values)):
        if gex_values[i-1] < 0 and gex_values[i] >= 0:
            # Linear interpolation for a more precise flip point
            gex1, gex2 = gex_values[i-1], gex_values[i]
            p1, p2 = prices[i-1], prices[i]
            if (gex2 - gex1) != 0: # Avoid division by zero if GEX values are identical
                gamma_flip_price = p1 - gex1 * (p2 - p1) / (gex2 - gex1)
            else: # GEX values are the same (likely both zero or very close)
                gamma_flip_price = (p1 + p2) / 2
            break

    return gex_bars_data, df_aggregate_curve, gamma_flip_price, net_gex_at_current_price


def calculate_put_call_ratio(df, value_col, type_col='Type', strike_col='Strike', call_label='call', put_label='put'):
    if df is None or df.empty:
        return None, None

    df_copy = df.copy()
    df_copy[type_col] = df_copy[type_col].str.lower()

    total_calls_value = df_copy[df_copy[type_col] == call_label][value_col].sum()
    total_puts_value = df_copy[df_copy[type_col] == put_label][value_col].sum()

    total_pc_ratio = total_puts_value / total_calls_value if total_calls_value > 0 else np.nan

    pc_ratio_by_strike = None
    if strike_col in df_copy.columns:
        calls_by_strike = df_copy[df_copy[type_col] == call_label].groupby(strike_col)[value_col].sum()
        puts_by_strike = df_copy[df_copy[type_col] == put_label].groupby(strike_col)[value_col].sum()

        pc_df = pd.DataFrame({'calls': calls_by_strike, 'puts': puts_by_strike}).fillna(0)
        pc_df['pc_ratio'] = pc_df['puts'] / pc_df['calls'].replace(0, np.nan)
        pc_ratio_by_strike = pc_df.reset_index()

    return total_pc_ratio, pc_ratio_by_strike

def calculate_money_at_risk(df, oi_col='Open Int', mid_price_col='Mid', type_col='Type', strike_col='Strike', multiplier=100):
    if df is None or df.empty or not all(col in df.columns for col in [oi_col, mid_price_col, type_col, strike_col]):
        return None

    df_copy = df.copy()
    df_copy['money_at_risk'] = df_copy[oi_col] * df_copy[mid_price_col] * multiplier

    money_at_risk_summary = df_copy.groupby([strike_col, type_col])['money_at_risk'].sum().unstack(fill_value=0)
    money_at_risk_summary = money_at_risk_summary.rename(
        columns=lambda x: f'{x.lower()}_money_at_risk'
    ).reset_index()

    return money_at_risk_summary

def calculate_max_pain(df_cadena, strike_col='Strike', oi_col='Open Int', type_col='Type', call_label='call', put_label='put', multiplier=100):
    if df_cadena is None or df_cadena.empty or not all(col in df_cadena.columns for col in [strike_col, oi_col, type_col]):
        return None, None # Return None for both if initial check fails

    unique_strikes = sorted(df_cadena[strike_col].astype(float).unique()) # Ensure strikes are float for comparison
    if not unique_strikes:
        return None, None # No strikes to process

    cash_values_at_expiry = []

    for expiry_price in unique_strikes:
        total_value_if_expired_at_strike = 0

        # Calls: intrinsic value = max(0, S - K) * OI * multiplier
        calls_df = df_cadena[df_cadena[type_col].str.lower() == call_label]
        intrinsic_value_calls = np.maximum(0, expiry_price - calls_df[strike_col]) * calls_df[oi_col] * multiplier
        total_value_if_expired_at_strike += intrinsic_value_calls.sum()

        # Puts: intrinsic value = max(0, K - S) * OI * multiplier
        puts_df = df_cadena[df_cadena[type_col].str.lower() == put_label]
        intrinsic_value_puts = np.maximum(0, puts_df[strike_col] - expiry_price) * puts_df[oi_col] * multiplier
        total_value_if_expired_at_strike += intrinsic_value_puts.sum()

        cash_values_at_expiry.append({'strike_price_at_expiry': expiry_price, 'total_option_value': total_value_if_expired_at_strike})

    if not cash_values_at_expiry:
        return None, None # Should not happen if unique_strikes is not empty

    cash_values_df = pd.DataFrame(cash_values_at_expiry)
    if cash_values_df.empty:
        return None, None

    max_pain_strike_info = cash_values_df.loc[cash_values_df['total_option_value'].idxmin()]
    max_pain_strike = max_pain_strike_info['strike_price_at_expiry']

    return max_pain_strike, cash_values_df


# Obsolete GEX function
# def calculate_gex_and_flip(...):
#     pass


def calculate_total_exposure(df_griegas, greek_col, oi_col='Open Int', type_col='Type', strike_col='Strike', multiplier=100):
    if df_griegas is None or df_griegas.empty or not all(col in df_griegas.columns for col in [greek_col, oi_col, type_col, strike_col]):
        return None, None

    df = df_griegas.copy()
    df[greek_col] = pd.to_numeric(df[greek_col], errors='coerce')
    df.dropna(subset=[greek_col, oi_col], inplace=True)

    if df.empty:
        return None, None

    df['exposure_value'] = df[greek_col] * df[oi_col] * multiplier

    exposure_by_strike = df.groupby(strike_col)['exposure_value'].sum().reset_index()
    total_exposure = df['exposure_value'].sum()

    return total_exposure, exposure_by_strike


# Example Usage (for testing, normally called from dashboard.py)
if __name__ == '__main__':
    cadena_dict = {
        'Strike': np.array([50, 55, 60, 65, 70, 50, 55, 60, 65, 70],dtype=float),
        'Mid': [10.5, 6.0, 2.5, 0.8, 0.2, 0.3, 0.7, 2.0, 5.5, 9.8],
        'Volume': [100, 200, 500, 300, 50, 80, 150, 400, 250, 60],
        'Open Int': [1000, 1500, 3000, 2000, 500, 800, 1200, 2500, 1800, 600],
        'Type': ['call', 'call', 'call', 'call', 'call', 'put', 'put', 'put', 'put', 'put'],
        'IV': [0.3, 0.28, 0.25, 0.27, 0.32, 0.31, 0.29, 0.26, 0.28, 0.33],
        'Exp Date': pd.to_datetime(['2025-12-31']*10)
    }
    df_cadena_sample = pd.DataFrame(cadena_dict)

    today_for_calc = pd.Timestamp('2024-07-26') # Fixed date for consistent DTE in test

    print("\n--- GEX Analytics (New Detailed Calculation) ---")
    df_griegas_for_gex = df_cadena_sample[['Strike', 'Open Int', 'Type', 'IV', 'Exp Date']].copy()
    # Calculate DTE for the sample data based on fixed 'today_for_calc'
    df_griegas_for_gex['DTE'] = (df_griegas_for_gex['Exp Date'] - today_for_calc).dt.days

    current_s = 60.0
    risk_free_r = 0.05

    gex_bars, gex_curve, flip_price, net_gex_now = calculate_gex_analytics(
        df_griegas_for_gex,
        current_spot_price=current_s,
        risk_free_rate=risk_free_r,
        calculation_date=today_for_calc
    )

    if gex_bars is not None:
        print("GEX Bars Data (Per Strike):")
        print(gex_bars.head())
    if gex_curve is not None:
        print("\nAggregate GEX Curve Data (Simulated Prices):")
        print(gex_curve.head())
        print(gex_curve.tail())
    print(f"\nGamma Flip Price: {flip_price}")
    print(f"Net GEX at Current Price ({current_s}): {net_gex_now:,.0f}")


    print("\n--- Put/Call Ratio (Volume) ---")
    total_pc_vol, pc_vol_strike = calculate_put_call_ratio(df_cadena_sample, value_col='Volume')
    print(f"Total P/C Ratio (Volume): {total_pc_vol:.2f}")
    if pc_vol_strike is not None: print(pc_vol_strike.head())

    print("\n--- Money at Risk ---")
    mar_df = calculate_money_at_risk(df_cadena_sample)
    if mar_df is not None: print(mar_df.head())

    print("\n--- Max Pain ---")
    max_pain_strike, max_pain_df_viz = calculate_max_pain(df_cadena_sample)
    if max_pain_strike is not None:
        print(f"Max Pain Strike: {max_pain_strike:.2f}")
    if max_pain_df_viz is not None: print(max_pain_df_viz.head())

    df_griegas_for_gex['Vega'] = [0.2, 0.3, 0.4, 0.25, 0.1, 0.18, 0.28, 0.38, 0.23, 0.08]
    df_griegas_for_gex['Theta'] = [-0.05, -0.08, -0.10, -0.06, -0.02, -0.04, -0.07, -0.09, -0.05, -0.01]

    print("\n--- Total Vega Exposure ---")
    if 'Vega' in df_griegas_for_gex.columns:
        total_vega, vega_by_strike = calculate_total_exposure(df_griegas_for_gex, greek_col='Vega')
        if total_vega is not None:
            print(f"Total Vega Exposure: {total_vega:,.0f}")
        if vega_by_strike is not None: print(vega_by_strike.head())
    else:
        print("Vega column not found in sample data for Vega exposure calculation.")

    print("\n--- Total Theta Exposure (Decay) ---")
    if 'Theta' in df_griegas_for_gex.columns:
        total_theta, theta_by_strike = calculate_total_exposure(df_griegas_for_gex, greek_col='Theta')
        if total_theta is not None:
            print(f"Total Theta Exposure: {total_theta:,.0f}")
        if theta_by_strike is not None: print(theta_by_strike.head())
    else:
        print("Theta column not found for Theta exposure calculation.")
