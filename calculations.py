import pandas as pd
import numpy as np
from scipy.stats import norm
import datetime # Required for date calculations if DTE is not directly available

def calculate_black_scholes_gamma(S, K, T, r, sigma):
    """
    Calculates the Black-Scholes Gamma for an option.
    S: Current price of the underlying asset
    K: Strike price of the option
    T: Time to expiration (in years)
    r: Risk-free interest rate (annualized)
    sigma: Volatility of the underlying asset (annualized)
    """
    # Ensure inputs are array-like for vectorized operations
    S = np.asarray(S)
    K = np.asarray(K)
    T = np.asarray(T)
    r = np.asarray(r)
    sigma = np.asarray(sigma)

    # Initialize gamma array with NaNs or zeros
    gamma = np.full_like(S, np.nan, dtype=np.double)

    # Avoid calculations for invalid inputs (T=0, sigma=0, S=0)
    valid_mask = (T > 1e-6) & (sigma > 1e-6) & (S > 1e-6) # Use a small epsilon for T, sigma, S > 0

    if np.any(valid_mask):
        S_valid = S[valid_mask]
        K_valid = K[valid_mask]
        T_valid = T[valid_mask]
        r_valid = r[valid_mask] if np.isscalar(r) else r[valid_mask] # handle scalar r
        sigma_valid = sigma[valid_mask] if np.isscalar(sigma) else sigma[valid_mask] # handle scalar sigma

        d1_numerator = np.log(S_valid / K_valid) + (r_valid + 0.5 * sigma_valid**2) * T_valid
        d1_denominator = sigma_valid * np.sqrt(T_valid)

        # Avoid division by zero in d1_denominator
        d1 = np.full_like(S_valid, np.nan, dtype=np.double)
        denom_valid_mask = (d1_denominator > 1e-9) # Check denominator is not too small

        d1[denom_valid_mask] = d1_numerator[denom_valid_mask] / d1_denominator[denom_valid_mask]
        d1[~denom_valid_mask] = np.inf * np.sign(d1_numerator[~denom_valid_mask]) # Handle case for very small sigma*sqrt(T)

        phi_d1 = norm.pdf(d1)

        gamma_denominator = S_valid * sigma_valid * np.sqrt(T_valid)
        gamma_calc = np.full_like(S_valid, np.nan, dtype=np.double)

        gamma_denom_valid_mask = (gamma_denominator > 1e-9)
        gamma_calc[gamma_denom_valid_mask] = phi_d1[gamma_denom_valid_mask] / gamma_denominator[gamma_denom_valid_mask]

        gamma[valid_mask] = gamma_calc

    return gamma

def calculate_gex_analytics(df_options_original, current_spot_price, risk_free_rate, calculation_date=None):
    """
    Calculates Gamma Exposure by Strike, Aggregate Gamma Exposure Curve, and Gamma Flip Point.
    """
    if df_options_original is None or df_options_original.empty:
        return None, None, np.nan, np.nan

    if not all(col in df_options_original.columns for col in ['Strike', 'Open Int', 'Type', 'IV']):
        print("Missing one or more required columns: 'Strike', 'Open Int', 'Type', 'IV'")
        return None, None, np.nan, np.nan
    if 'Exp Date' not in df_options_original.columns and 'DTE' not in df_options_original.columns:
        print("Missing 'Exp Date' or 'DTE' column for time to maturity calculation.")
        return None, None, np.nan, np.nan

    df = df_options_original.copy()

    # Ensure 'Type' is lowercase
    df['Type'] = df['Type'].str.lower()

    # Part A: Calculate Gamma for each contract using current_spot_price
    if calculation_date is None:
        calculation_date = pd.Timestamp('today').normalize()
    else:
        calculation_date = pd.to_datetime(calculation_date).normalize()

    if 'DTE' not in df.columns:
        if 'Exp Date' not in df.columns:
             raise ValueError("DataFrame must have either 'DTE' or 'Exp Date' column.")
        df['Exp Date'] = pd.to_datetime(df['Exp Date'])
        df['DTE'] = (df['Exp Date'] - calculation_date).dt.days

    df['T_years'] = df['DTE'] / 365.25

    # Filter out expired or invalid options for Gamma calculation
    valid_options_for_gamma = df[df['T_years'] > 1e-6].copy() # Small epsilon to avoid T=0 issues
    if valid_options_for_gamma.empty:
        print("No valid options with T_years > 0 for Gamma calculation.")
        return pd.DataFrame(columns=['Strike', 'GEX_Signed_Value']), pd.DataFrame(columns=['Simulated_Price', 'Total_GEX']), np.nan, 0

    valid_options_for_gamma['Calculated_Gamma'] = calculate_black_scholes_gamma(
        S=current_spot_price,
        K=valid_options_for_gamma['Strike'],
        T=valid_options_for_gamma['T_years'],
        r=risk_free_rate,
        sigma=valid_options_for_gamma['IV'] # Assumes IV is already in decimal form
    )
    valid_options_for_gamma.dropna(subset=['Calculated_Gamma'], inplace=True) # Remove rows where gamma calc failed

    # Part B: Calculate Gamma Exposure by Strike (for the bars)
    valid_options_for_gamma['GEX_Contract_Value'] = valid_options_for_gamma['Calculated_Gamma'] * valid_options_for_gamma['Open Int'] * 100
    valid_options_for_gamma['GEX_Signed_Value'] = np.where(
        valid_options_for_gamma['Type'] == 'call',
        valid_options_for_gamma['GEX_Contract_Value'],
        -valid_options_for_gamma['GEX_Contract_Value']
    )

    if valid_options_for_gamma.empty: # Check again after gamma calculation and signing
        gex_bars_data = pd.DataFrame({'Strike': [], 'GEX_Signed_Value': []})
        net_gex_at_current_price = 0
    else:
        gex_bars_data = valid_options_for_gamma.groupby('Strike')['GEX_Signed_Value'].sum().reset_index()
        net_gex_at_current_price = gex_bars_data['GEX_Signed_Value'].sum()


    # Part C: Calculate Aggregate Gamma Exposure Curve and Flip Point
    # Use all options from the original valid_options_for_gamma for simulation, as T_years remains constant for this part
    # We need df_copy of valid_options_for_gamma to ensure T_years and other necessary columns are present
    sim_df = valid_options_for_gamma.copy()

    min_price = current_spot_price * 0.80
    max_price = current_spot_price * 1.20
    # Ensure step is reasonable, avoid too many points if price range is huge, or too few if small
    price_step = max(0.1, round((max_price - min_price) / 100, 2)) # Aim for around 100-200 points
    if price_step == 0: price_step = 0.5 # Default if range is tiny

    test_price_range = np.arange(min_price, max_price + price_step, price_step) # + price_step to include upper bound

    simulated_gex_totals = []

    for s_sim in test_price_range:
        # Recalculate gamma for all options at this s_sim
        gamma_sim = calculate_black_scholes_gamma(
            S=s_sim,
            K=sim_df['Strike'],
            T=sim_df['T_years'],
            r=risk_free_rate,
            sigma=sim_df['IV']
        )

        gex_contract_sim = gamma_sim * sim_df['Open Int'] * 100
        gex_signed_sim = np.where(sim_df['Type'] == 'call', gex_contract_sim, -gex_contract_sim)
        total_gex_for_s_sim = np.nansum(gex_signed_sim) # Use nansum in case any gamma_sim was NaN

        simulated_gex_totals.append({'Simulated_Price': s_sim, 'Total_GEX': total_gex_for_s_sim})

    df_aggregate_curve = pd.DataFrame(simulated_gex_totals)
    df_aggregate_curve.sort_values(by='Simulated_Price', inplace=True) # Ensure it's sorted for flip point logic

    # Find Gamma Flip Point
    gamma_flip_price = np.nan
    prev_gex = None
    prev_price = None
    for index, row in df_aggregate_curve.iterrows():
        current_gex = row['Total_GEX']
        current_price = row['Simulated_Price']
        if prev_gex is not None:
            if prev_gex < 0 and current_gex >= 0:
                # Linear interpolation for a more precise flip point
                if (current_gex - prev_gex) != 0: # Avoid division by zero
                    gamma_flip_price = prev_price - prev_gex * (current_price - prev_price) / (current_gex - prev_gex)
                else:
                    gamma_flip_price = (prev_price + current_price) / 2 # Fallback to midpoint
                break
        prev_gex = current_gex
        prev_price = current_price

    return gex_bars_data, df_aggregate_curve, gamma_flip_price, net_gex_at_current_price


def calculate_put_call_ratio(df, value_col, type_col='Type', strike_col='Strike', call_label='call', put_label='put'):
    """
    Calculates Put/Call Ratio based on a specified value column (e.g., Volume or Open Interest).
    Can return total P/C ratio and P/C ratio per strike.
    """
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
        return None

    unique_strikes = sorted(df_cadena[strike_col].unique())
    if not unique_strikes:
        return None, None

    cash_values_at_expiry = []

    for expiry_price in unique_strikes:
        total_value_if_expired_at_strike = 0

        calls = df_cadena[df_cadena[type_col].str.lower() == call_label]
        intrinsic_value_calls = np.maximum(0, expiry_price - calls[strike_col]) * calls[oi_col] * multiplier
        total_value_if_expired_at_strike += intrinsic_value_calls.sum()

        puts = df_cadena[df_cadena[type_col].str.lower() == put_label]
        intrinsic_value_puts = np.maximum(0, puts[strike_col] - expiry_price) * puts[oi_col] * multiplier
        total_value_if_expired_at_strike += intrinsic_value_puts.sum()

        cash_values_at_expiry.append({'strike_price_at_expiry': expiry_price, 'total_option_value': total_value_if_expired_at_strike})

    if not cash_values_at_expiry:
        return None, None

    cash_values_df = pd.DataFrame(cash_values_at_expiry)
    max_pain_strike_info = cash_values_df.loc[cash_values_df['total_option_value'].idxmin()]
    max_pain_strike = max_pain_strike_info['strike_price_at_expiry']

    return max_pain_strike, cash_values_df


# This function is now superseded by calculate_gex_analytics
# def calculate_gex_and_flip(df_griegas, underlying_price_col='Underlying_Price', oi_col='Open Int',
#                            gamma_col='Gamma', type_col='Type', strike_col='Strike',
#                            call_label='call', put_label='put', per_share_gex=False):
#     pass # Obsolete


def calculate_total_exposure(df_griegas, greek_col, oi_col='Open Int', type_col='Type', strike_col='Strike', multiplier=100):
    if df_griegas is None or df_griegas.empty or not all(col in df_griegas.columns for col in [greek_col, oi_col, type_col, strike_col]):
        return None, None

    df = df_griegas.copy()
    # Ensure the greek column is numeric, coercing errors. This is important if Gamma is pre-calculated and might have NaNs.
    df[greek_col] = pd.to_numeric(df[greek_col], errors='coerce')
    df.dropna(subset=[greek_col, oi_col], inplace=True) # Drop rows where essential data for calculation is missing

    if df.empty: # Check if df became empty after dropping NaNs
        return None, None

    df['exposure_value'] = df[greek_col] * df[oi_col] * multiplier

    exposure_by_strike = df.groupby(strike_col)['exposure_value'].sum().reset_index()
    total_exposure = df['exposure_value'].sum()

    return total_exposure, exposure_by_strike


# Example Usage (for testing, normally called from dashboard.py)
if __name__ == '__main__':
    # Sample DataFrames
    cadena_dict = {
        'Strike': [50, 55, 60, 65, 70, 50, 55, 60, 65, 70],
        'Mid': [10.5, 6.0, 2.5, 0.8, 0.2, 0.3, 0.7, 2.0, 5.5, 9.8],
        'Volume': [100, 200, 500, 300, 50, 80, 150, 400, 250, 60],
        'Open Int': [1000, 1500, 3000, 2000, 500, 800, 1200, 2500, 1800, 600],
        'Type': ['call', 'call', 'call', 'call', 'call', 'put', 'put', 'put', 'put', 'put'],
        'IV': [0.3, 0.28, 0.25, 0.27, 0.32, 0.31, 0.29, 0.26, 0.28, 0.33],
        'Exp Date': ['2025-12-31']*10 # Example Expiry
    }
    df_cadena_sample = pd.DataFrame(cadena_dict)
    df_cadena_sample['Exp Date'] = pd.to_datetime(df_cadena_sample['Exp Date'])


    # Test for calculate_gex_analytics
    print("\n--- GEX Analytics (New Detailed Calculation) ---")
    # Assuming 'Underlying_Price' would be part of a df_griegas or passed directly
    # For this test, let's use a structure similar to what df_griegas might provide
    # We need 'Strike', 'Open Int', 'Type', 'IV', and 'Exp Date' (or 'DTE')

    # Let's use df_cadena_sample as the base for df_griegas_sample for GEX
    df_griegas_for_gex = df_cadena_sample[['Strike', 'Open Int', 'Type', 'IV', 'Exp Date']].copy()
    current_s = 60.0
    risk_free_r = 0.05
    today = pd.Timestamp('today').normalize()

    # Ensure DTE is calculated for the sample data if not present
    if 'DTE' not in df_griegas_for_gex.columns:
        df_griegas_for_gex['DTE'] = (df_griegas_for_gex['Exp Date'] - today).dt.days

    gex_bars, gex_curve, flip_price, net_gex_now = calculate_gex_analytics(
        df_griegas_for_gex,
        current_spot_price=current_s,
        risk_free_rate=risk_free_r,
        calculation_date=today
    )

    if gex_bars is not None:
        print("GEX Bars Data (Per Strike):")
        print(gex_bars.head())
    if gex_curve is not None:
        print("\nAggregate GEX Curve Data (Simulated Prices):")
        print(gex_curve.head())
    print(f"\nGamma Flip Price: {flip_price}")
    print(f"Net GEX at Current Price ({current_s}): {net_gex_now}")


    print("\n--- Put/Call Ratio (Volume) ---")
    total_pc_vol, pc_vol_strike = calculate_put_call_ratio(df_cadena_sample, value_col='Volume')
    print(f"Total P/C Ratio (Volume): {total_pc_vol}")
    if pc_vol_strike is not None: print(pc_vol_strike.head())

    print("\n--- Money at Risk ---")
    mar_df = calculate_money_at_risk(df_cadena_sample)
    if mar_df is not None: print(mar_df.head())

    print("\n--- Max Pain ---")
    max_pain_strike, max_pain_df_viz = calculate_max_pain(df_cadena_sample) # Renamed to avoid conflict
    print(f"Max Pain Strike: {max_pain_strike}")
    if max_pain_df_viz is not None: print(max_pain_df_viz.head())

    # For Total Exposure, we need a 'Gamma' column if not calculating it via BS
    # Let's assume df_griegas_for_gex now has 'Calculated_Gamma' from the GEX test
    # For testing calculate_total_exposure, we can reuse the structure.
    # If df_griegas_for_gex was modified in-place by calculate_gex_analytics with 'Calculated_Gamma',
    # we might need to be careful or re-fetch/re-create.
    # The current calculate_gex_analytics makes a copy, so df_griegas_for_gex is unchanged.
    # For testing total_exposure for vega/theta, let's add dummy Vega/Theta to df_griegas_for_gex
    df_griegas_for_gex['Vega'] = [0.2, 0.3, 0.4, 0.25, 0.1, 0.18, 0.28, 0.38, 0.23, 0.08] # Example Vega
    df_griegas_for_gex['Theta'] = [-0.05, -0.08, -0.10, -0.06, -0.02, -0.04, -0.07, -0.09, -0.05, -0.01] # Example Theta


    print("\n--- Total Vega Exposure ---")
    # Ensure the greek_col passed exists in the dataframe.
    if 'Vega' in df_griegas_for_gex.columns:
        total_vega, vega_by_strike = calculate_total_exposure(df_griegas_for_gex, greek_col='Vega')
        if total_vega is not None:
            print(f"Total Vega Exposure: {total_vega}")
        if vega_by_strike is not None: print(vega_by_strike.head())
    else:
        print("Vega column not found in sample data for Vega exposure calculation.")

    print("\n--- Total Theta Exposure (Decay) ---")
    if 'Theta' in df_griegas_for_gex.columns:
        total_theta, theta_by_strike = calculate_total_exposure(df_griegas_for_gex, greek_col='Theta')
        if total_theta is not None:
            print(f"Total Theta Exposure: {total_theta}") # Expected to be negative
        if theta_by_strike is not None: print(theta_by_strike.head())
    else:
        print("Theta column not found for Theta exposure calculation.")
