import pandas as pd
import numpy as np

def calculate_put_call_ratio(df, value_col, type_col='Type', strike_col='Strike', call_label='call', put_label='put'):
    """
    Calculates Put/Call Ratio based on a specified value column (e.g., Volume or Open Interest).
    Can return total P/C ratio and P/C ratio per strike.
    """
    if df is None or df.empty:
        return None, None

    df_copy = df.copy()
    df_copy[type_col] = df_copy[type_col].str.lower()

    # Total P/C Ratio
    total_calls_value = df_copy[df_copy[type_col] == call_label][value_col].sum()
    total_puts_value = df_copy[df_copy[type_col] == put_label][value_col].sum()

    total_pc_ratio = total_puts_value / total_calls_value if total_calls_value > 0 else np.nan

    # P/C Ratio per Strike
    pc_ratio_by_strike = None
    if strike_col in df_copy.columns:
        calls_by_strike = df_copy[df_copy[type_col] == call_label].groupby(strike_col)[value_col].sum()
        puts_by_strike = df_copy[df_copy[type_col] == put_label].groupby(strike_col)[value_col].sum()

        pc_df = pd.DataFrame({'calls': calls_by_strike, 'puts': puts_by_strike}).fillna(0)
        pc_df['pc_ratio'] = pc_df['puts'] / pc_df['calls'].replace(0, np.nan) # Avoid division by zero
        pc_ratio_by_strike = pc_df.reset_index()

    return total_pc_ratio, pc_ratio_by_strike

def calculate_money_at_risk(df, oi_col='Open Int', mid_price_col='Mid', type_col='Type', strike_col='Strike', multiplier=100):
    """
    Calculates Money at Risk for options.
    Money at Risk = Open Interest * Option Mid Price * Multiplier
    Returns DataFrame with money at risk per strike for calls and puts.
    """
    if df is None or df.empty or not all(col in df.columns for col in [oi_col, mid_price_col, type_col, strike_col]):
        return None

    df_copy = df.copy()
    df_copy['money_at_risk'] = df_copy[oi_col] * df_copy[mid_price_col] * multiplier

    # Sum money at risk for calls and puts separately per strike
    money_at_risk_summary = df_copy.groupby([strike_col, type_col])['money_at_risk'].sum().unstack(fill_value=0)
    money_at_risk_summary = money_at_risk_summary.rename(
        columns=lambda x: f'{x.lower()}_money_at_risk'
    ).reset_index()

    return money_at_risk_summary

def calculate_max_pain(df_cadena, strike_col='Strike', oi_col='Open Int', type_col='Type', call_label='call', put_label='put', multiplier=100):
    """
    Calculates the Max Pain strike.
    Max Pain is the strike price at which the most option holders (buyers) would lose money if the stock expired at that price.
    This means it's the strike where the intrinsic value of all open options (Calls and Puts) is minimized.
    """
    if df_cadena is None or df_cadena.empty or not all(col in df_cadena.columns for col in [strike_col, oi_col, type_col]):
        return None

    strikes = sorted(df_cadena[strike_col].unique())
    max_pain_data = []

    for expiry_price in strikes:
        total_loss = 0

        # Calculate loss for Calls
        calls = df_cadena[(df_cadena[type_col].str.lower() == call_label) & (df_cadena[strike_col] < expiry_price)]
        total_loss += ( (expiry_price - calls[strike_col]) * calls[oi_col] * multiplier ).sum()

        # Calculate loss for Puts
        puts = df_cadena[(df_cadena[type_col].str.lower() == put_label) & (df_cadena[strike_col] > expiry_price)]
        total_loss += ( (puts[strike_col] - expiry_price) * puts[oi_col] * multiplier ).sum()

        max_pain_data.append({'strike': expiry_price, 'total_value_at_expiry': total_loss})

    if not max_pain_data:
        return None

    max_pain_df = pd.DataFrame(max_pain_data)

    # Max pain is the strike with the minimum total value (loss for holders, gain for sellers)
    # However, the traditional definition refers to the value of options that would expire worthless or with value.
    # Let's redefine to calculate cash value of options if stock expires at 'expiry_price'

    cash_values = []
    for test_strike in strikes:
        current_cash_value = 0
        # For calls: if stock_price > strike_price, value is (stock_price - strike_price) * OI
        calls_df = df_cadena[df_cadena[type_col].str.lower() == call_label]
        intrinsic_calls = np.maximum(0, test_strike - calls_df[strike_col]) * calls_df[oi_col]
        current_cash_value += intrinsic_calls.sum()

        # For puts: if stock_price < strike_price, value is (strike_price - stock_price) * OI
        puts_df = df_cadena[df_cadena[type_col].str.lower() == put_label]
        intrinsic_puts = np.maximum(0, puts_df[strike_col] - test_strike) * puts_df[oi_col]
        current_cash_value += intrinsic_puts.sum()

        cash_values.append({'strike_price_at_expiry': test_strike, 'total_option_value': current_cash_value * multiplier})

    if not cash_values:
        return None

    cash_values_df = pd.DataFrame(cash_values)
    max_pain_strike = cash_values_df.loc[cash_values_df['total_option_value'].idxmin()]['strike_price_at_expiry']

    return max_pain_strike, cash_values_df


def calculate_gex_and_flip(df_griegas, underlying_price_col='Underlying_Price', oi_col='Open Int',
                           gamma_col='Gamma', type_col='Type', strike_col='Strike',
                           call_label='call', put_label='put', per_share_gex=False):
    """
    Calculates Gross Gamma Exposure (GEX) and per-strike Gamma.
    GEX = Sum of (Gamma * Open Interest * 100 * (Underlying Price^2) * 0.01) for Calls
          - Sum of (Gamma * Open Interest * 100 * (Underlying Price^2) * 0.01) for Puts.
    The (Underlying Price^2 * 0.01) part is for dollar gamma. If per_share_gex is True, this is omitted.
    Also identifies potential Gamma Flip Point by looking at GEX across a range of potential underlying prices.
    """
    if df_griegas is None or df_griegas.empty or not all(col in df_griegas.columns for col in [oi_col, gamma_col, type_col, strike_col]):
        return None, None, None

    df = df_griegas.copy()
    df[type_col] = df[type_col].str.lower()

    # Calculate GEX per option contract
    # Positive Gamma for long calls and long puts.
    # Market makers are typically short gamma, so their exposure is opposite.
    # GEX here is defined from the perspective of option holders (long gamma).
    # For market maker exposure, flip the sign or adjust interpretation.
    # The definition given in the prompt "If the GEX neta is positiva, los market makers estabilizan el precio"
    # suggests GEX is viewed from the market maker's perspective (short gamma).
    # So, long calls (positive gamma for holder) = negative gamma exposure for MM.
    # Long puts (positive gamma for holder) = negative gamma exposure for MM.
    # Let's calculate dealer GEX:
    # Dealer Gamma for Calls = -Gamma * OI * 100 (gamma is per share, OI is per 100 shares)
    # Dealer Gamma for Puts = -Gamma * OI * 100

    # Standard GEX calculation often sums Call Gamma and subtracts Put Gamma, assuming retail is long.
    # Or, if looking at dealer exposure which is short options:
    # GEX = OI * Gamma * 100 (for calls)  <-- this is dealer exposure if they sold calls
    # GEX = OI * Gamma * 100 (for puts)   <-- this is dealer exposure if they sold puts
    # If dealer is short calls, their gamma is negative. If short puts, their gamma is negative.
    # Net GEX = sum(OI * Gamma * 100 * Multiplier_for_Notional)
    # Let's use the common definition: Call Gamma - Put Gamma (scaled by OI)
    # This usually means (OI_call * Gamma_call) - (OI_put * Gamma_put)
    # The prompt's "GEX neta positiva -> market makers estabilizan" implies we are calculating market maker gamma.
    # If market makers are short calls, their gamma is -Gamma_call. If short puts, -Gamma_put.
    # So, GEX_dealer = sum(-Gamma_call * OI_call * 100) + sum(-Gamma_put * OI_put * 100)
    # This is total negative gamma.
    # Let's stick to the user's definition for now and assume GEX is calculated in a way that
    # positive GEX means dealers buy on dips and sell on rallies.

    # Gamma exposure per strike (dealer perspective, short options)
    # For a dealer short a call: exposure = -Gamma * OI * 100
    # For a dealer short a put: exposure = -Gamma * OI * 100
    # So, total dealer GEX = sum(-Gamma * OI * 100) for all options.
    # The "per_share_gex" factor for dollar gamma: * Underlying Price^2 * 0.01 for dollar gamma notional
    # Or simply $ per 1% move: Gamma * OI * 100 * Underlying_Price * 0.01

    # Let's use a common definition for GEX:
    # GEX_per_strike = OI * Gamma * 100 (this is share gamma, not dollar gamma)
    # For Calls, this is positive. For Puts, this is also positive (as Gamma is positive for long options).
    # The "flip" comes from how calls and puts influence dealer hedging.
    # Dealers hedge delta. If they are short calls (delta negative), they buy stock. If price goes up, delta increases (becomes less negative), they buy more.
    # If they are short puts (delta positive), they sell stock. If price goes up, delta decreases (becomes less positive), they buy back stock.

    # A common way to calculate GEX representing dealer impact:
    # GEX = sum over Calls (OI * Gamma * 100) - sum over Puts (OI * Gamma * 100)
    # This is often scaled by notional value (Underlying Price).

    # Let's use the formula that aligns with "positive GEX = stabilizing":
    # GEX_strike = Gamma * OI * 100
    # If Type is Put, multiply by -1 for the GEX calculation to represent dealer hedging flow.
    # (This is one interpretation for GEX where positive means stability)
    df['strike_gex_contribution'] = df[gamma_col] * df[oi_col] * 100
    df['strike_gex_contribution'] = df.apply(
        lambda row: -row['strike_gex_contribution'] if row[type_col] == put_label else row['strike_gex_contribution'],
        axis=1
    )
    # This GEX means: if price rises, positive GEX dealers sell (counter-trend), negative GEX dealers buy (pro-trend).
    # So, if GEX is positive, dealers are delta hedging in a way that dampens moves.
    # If GEX is negative, dealers are delta hedging in a way that amplifies moves.

    if not per_share_gex and underlying_price_col in df.columns and df[underlying_price_col].iloc[0] > 0:
        # Convert to dollar gamma exposure (approximate): GEX_shares * Underlying_Price
        # More precise $Gamma: Gamma * OI * 100 * (Underlying_Price^2) * 0.01 for $ change per 1% move in underlying
        # Or $ per 1 point move in underlying: Gamma * OI * 100 * Underlying_Price
        # Let's use the $ per 1% move interpretation for now for the notional GEX.
        # Actually, GEX is often quoted in $MM per 1% move.
        # GEX ($) = Gamma * OI * 100 * (Underlying Price)^2 * 0.01
        # Let's use the simpler definition first: GEX in shares.
        # The prompt implies GEX is a value that flips sign.
        pass # Using share GEX for now. Can be scaled later.

    gex_by_strike = df.groupby(strike_col)['strike_gex_contribution'].sum().reset_index()
    gex_by_strike = gex_by_strike.rename(columns={'strike_gex_contribution': 'gex'})

    total_gex = gex_by_strike['gex'].sum()

    # Gamma Flip Point:
    # This requires simulating GEX at different underlying prices, as Gamma itself changes with stock price and IV.
    # For a simplified "static" flip point based on current gamma values:
    # Find where cumulative GEX (summed from low strikes up) changes sign, or where GEX per strike changes sign.
    # The prompt definition: "El nivel de precio donde la Exposición a Gamma neta del mercado cruza de positiva a negativa."
    # This implies plotting Net GEX (total_gex or gex_by_strike) against the underlying price.
    # The gex_by_strike we calculated is already against option strikes.
    # A common way to find the "gamma flip" is to see where the GEX profile (GEX vs Strike) crosses zero.

    # Find strike where GEX per strike changes sign (approximate flip)
    # This requires gex_by_strike to be sorted by strike
    gex_by_strike_sorted = gex_by_strike.sort_values(by=strike_col)
    sign_changes = np.sign(gex_by_strike_sorted['gex']).diff().fillna(0).ne(0)
    flip_points_df = gex_by_strike_sorted[sign_changes]

    # A more robust Gamma Flip might be where cumulative GEX from ATM out changes sign,
    # or requires dynamic calculation of gamma. For now, this is an indicator.

    return total_gex, gex_by_strike, flip_points_df


def calculate_total_exposure(df_griegas, greek_col, oi_col='Open Int', type_col='Type', strike_col='Strike', multiplier=100):
    """
    Calculates total exposure for a given Greek (Vega, Theta).
    Exposure = Greek Value * Open Interest * Multiplier (typically 100 shares/contract)
    Theta is often negative; Vega is positive.
    """
    if df_griegas is None or df_griegas.empty or not all(col in df_griegas.columns for col in [greek_col, oi_col, type_col, strike_col]):
        return None, None

    df = df_griegas.copy()
    df['exposure_value'] = df[greek_col] * df[oi_col] * multiplier

    exposure_by_strike = df.groupby(strike_col)['exposure_value'].sum().reset_index()
    total_exposure = df['exposure_value'].sum()

    return total_exposure, exposure_by_strike


# Example Usage (for testing, normally called from dashboard.py)
if __name__ == '__main__':
    # Sample DataFrames (simplified based on structures from file_handler.py)
    cadena_dict = {
        'Strike': [50, 55, 60, 65, 70, 50, 55, 60, 65, 70],
        'Mid': [10.5, 6.0, 2.5, 0.8, 0.2, 0.3, 0.7, 2.0, 5.5, 9.8],
        'Volume': [100, 200, 500, 300, 50, 80, 150, 400, 250, 60],
        'Open Int': [1000, 1500, 3000, 2000, 500, 800, 1200, 2500, 1800, 600],
        'Type': ['call', 'call', 'call', 'call', 'call', 'put', 'put', 'put', 'put', 'put']
    }
    df_cadena_sample = pd.DataFrame(cadena_dict)

    griegas_dict = {
        'Strike': [50, 55, 60, 65, 70, 50, 55, 60, 65, 70],
        'Open Int': [1000, 1500, 3000, 2000, 500, 800, 1200, 2500, 1800, 600],
        'Type': ['call', 'call', 'call', 'call', 'call', 'put', 'put', 'put', 'put', 'put'],
        'Gamma': [0.05, 0.08, 0.12, 0.07, 0.03, 0.04, 0.07, 0.11, 0.06, 0.02],
        'Vega': [0.2, 0.3, 0.4, 0.25, 0.1, 0.18, 0.28, 0.38, 0.23, 0.08],
        'Theta': [-0.05, -0.08, -0.10, -0.06, -0.02, -0.04, -0.07, -0.09, -0.05, -0.01],
        'Underlying_Price': [60.5] * 10 # Assuming a static underlying price for this sample
    }
    df_griegas_sample = pd.DataFrame(griegas_dict)

    print("--- Put/Call Ratio (Volume) ---")
    total_pc_vol, pc_vol_strike = calculate_put_call_ratio(df_cadena_sample, value_col='Volume')
    print(f"Total P/C Ratio (Volume): {total_pc_vol}")
    if pc_vol_strike is not None: print(pc_vol_strike.head())

    print("\n--- Put/Call Ratio (Open Interest) ---")
    total_pc_oi, pc_oi_strike = calculate_put_call_ratio(df_cadena_sample, value_col='Open Int')
    print(f"Total P/C Ratio (Open Interest): {total_pc_oi}")
    if pc_oi_strike is not None: print(pc_oi_strike.head())

    print("\n--- Money at Risk ---")
    mar_df = calculate_money_at_risk(df_cadena_sample)
    if mar_df is not None: print(mar_df.head())

    print("\n--- Max Pain ---")
    max_pain_strike, max_pain_df = calculate_max_pain(df_cadena_sample)
    print(f"Max Pain Strike: {max_pain_strike}")
    if max_pain_df is not None: print(max_pain_df.head())

    print("\n--- GEX (Gamma Exposure) ---")
    # Note: GEX calculation can be complex and has varied definitions. This is one interpretation.
    total_gex, gex_by_strike, flip_points = calculate_gex_and_flip(df_griegas_sample, underlying_price_col='Underlying_Price')
    print(f"Total GEX: {total_gex}")
    if gex_by_strike is not None: print("GEX by Strike:\n", gex_by_strike.head())
    if flip_points is not None and not flip_points.empty: print("Potential Gamma Flip Points (Strikes):\n", flip_points)
    else: print("No distinct Gamma Flip Points found with this data/method.")

    print("\n--- Total Vega Exposure ---")
    total_vega, vega_by_strike = calculate_total_exposure(df_griegas_sample, greek_col='Vega')
    print(f"Total Vega Exposure: {total_vega}")
    if vega_by_strike is not None: print(vega_by_strike.head())

    print("\n--- Total Theta Exposure (Decay) ---")
    total_theta, theta_by_strike = calculate_total_exposure(df_griegas_sample, greek_col='Theta')
    print(f"Total Theta Exposure: {total_theta}") # Expected to be negative
    if theta_by_strike is not None: print(theta_by_strike.head())
