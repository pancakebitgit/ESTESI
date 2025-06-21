import pandas as pd
import io

def clean_numeric_column(series, remove_chars=None):
    """Helper function to clean numeric columns."""
    if remove_chars is None:
        remove_chars = [',', '%', '+']
    series_str = series.astype(str)
    for char in remove_chars:
        series_str = series_str.str.replace(char, '', regex=False)
    return pd.to_numeric(series_str, errors='coerce')

def load_and_clean_cadena(uploaded_file):
    """Loads and cleans data from CADENA.csv."""
    if uploaded_file is None:
        return None
    try:
        # Read the uploaded file
        df = pd.read_csv(uploaded_file)

        # Clean numeric columns
        cols_to_clean = ['Strike', 'Bid', 'Mid', 'Ask', 'Last', 'Change', 'Volume', 'Open Int', 'OI Chg', 'IV', 'Delta']
        for col in cols_to_clean:
            if col in df.columns:
                df[col] = clean_numeric_column(df[col])

        if '%Chg' in df.columns:
            df['%Chg'] = clean_numeric_column(df['%Chg'], remove_chars=['%'])
        if 'Moneyness' in df.columns: # Example: +15.62%
             df['Moneyness'] = clean_numeric_column(df['Moneyness'], remove_chars=['%','+']) / 100


        # Convert 'Last Trade' to datetime if it exists
        if 'Last Trade' in df.columns:
            df['Last Trade'] = pd.to_datetime(df['Last Trade'], errors='coerce')

        # Ensure 'Type' column is string
        if 'Type' in df.columns:
            df['Type'] = df['Type'].astype(str).str.lower()


        return df
    except Exception as e:
        print(f"Error loading or cleaning CADENA.csv: {e}")
        return None

def load_and_clean_griegas(uploaded_file):
    """Loads and cleans data from Griegas.csv."""
    if uploaded_file is None:
        return None
    try:
        df = pd.read_csv(uploaded_file)

        # Clean numeric columns (many might have % or commas)
        # Price~ might be Price_subyacente o Underlying_Price
        if 'Price~' in df.columns: # Assuming this is underlying price
            df['Underlying_Price'] = clean_numeric_column(df['Price~'])
            df.drop(columns=['Price~'], inplace=True)

        cols_to_clean = ['Strike', 'Bid', 'Ask', 'Volume', 'Open Int',
                         'IV', 'Delta', 'Gamma', 'Theta', 'Vega', 'ITM Prob']
        for col in cols_to_clean:
            if col in df.columns:
                # IV, Delta, ITM Prob are often percentages
                if col in ['IV', 'Delta', 'ITM Prob']:
                     df[col] = clean_numeric_column(df[col], remove_chars=['%']) / 100.0
                else:
                    df[col] = clean_numeric_column(df[col])

        # Dates
        if 'Exp Date' in df.columns:
            df['Exp Date'] = pd.to_datetime(df['Exp Date'], errors='coerce')
        if 'Time' in df.columns: # This might be last trade time for the option
            pass # Keep as string or parse if needed for specific logic

        # Ensure 'Type' column is string and lowercased
        if 'Type' in df.columns:
            df['Type'] = df['Type'].astype(str).str.lower()

        return df
    except Exception as e:
        print(f"Error loading or cleaning Griegas.csv: {e}")
        return None

def load_and_clean_inusual(uploaded_file):
    """Loads and cleans data from Inusual.csv."""
    if uploaded_file is None:
        return None
    try:
        df = pd.read_csv(uploaded_file)

        if 'Price~' in df.columns: # Assuming this is underlying price at trade time
            df['Underlying_Price_at_Trade'] = clean_numeric_column(df['Price~'])
            df.drop(columns=['Price~'], inplace=True)

        cols_to_clean = ['Strike', 'DTE', 'Trade', 'Size', 'Premium', 'Volume', 'Open Int', 'IV', 'Delta']
        for col in cols_to_clean:
            if col in df.columns:
                if col in ['IV', 'Delta']: # Often percentages
                    df[col] = clean_numeric_column(df[col], remove_chars=['%']) / 100.0
                else:
                    df[col] = clean_numeric_column(df[col])

        # Dates
        if 'Expires' in df.columns:
            df['Expires'] = pd.to_datetime(df['Expires'], errors='coerce')

        # Time - keep as string for now, might need parsing if used in time-based analysis
        if 'Time' in df.columns:
            df['Trade_Time'] = df['Time'] # Rename for clarity
            # df.drop(columns=['Time'], inplace=True) # Optional: drop original if renamed

        # Categorical
        if 'Type' in df.columns:
            df['Type'] = df['Type'].astype(str).str.lower()
        if 'Side' in df.columns:
            df['Side'] = df['Side'].astype(str).str.lower()

        # 'Bid x Size', 'Ask x Size' are complex, leave as string for now
        # unless specific parsing is requested later.

        return df
    except Exception as e:
        print(f"Error loading or cleaning Inusual.csv: {e}")
        return None

# Example usage (for testing purposes, normally called from dashboard.py)
if __name__ == '__main__':
    # Create dummy CSV data for testing
    cadena_data = """Strike,Moneyness,Bid,Mid,Ask,Last,Change,%Chg,Volume,Open Int,OI Chg,IV,Delta,Type,Last Trade
55.00,+9.00%,4.50,5.15,5.80,5.55,+0.60,+12.12%,"2,000",279,-18,46.45%,0.9475,Call,06/20/25
60.00,+0.73%,1.41,1.51,1.60,1.43,+0.23,+19.17%,218,"10,328",-616,38.29%,0.5731,Call,06/20/25
55.00,+9.00%,0.50,0.55,0.60,0.58,-0.02,-3.33%,100,500,+10,50.00%,-0.1000,Put,06/20/25"""

    griegas_data = """Symbol,Price~,Type,Exp Date,Strike,Bid,Ask,Volume,Open Int,IV,Delta,Gamma,Theta,Vega,ITM Prob,Time
EQT,60.44,Call,2025-06-27,60.00,1.41,1.6,218,10328,38.29%,0.573068,0.132181,-0.099921,0.030387,55.92%,2025-06-20
EQT,60.44,Put,2025-06-27,55.00,0.50,0.60,100,500,50.00%,-0.1000,0.05,-0.05,0.02,10.00%,2025-06-20"""

    inusual_data = """Symbol,Price~,Type,Strike,Expires,DTE,Bid x Size,Ask x Size,Trade,Size,Side,Premium,Volume,Open Int,IV,Delta,Code,*,Time
EQT,60.37,Put,57,2025-06-27T16:30:00-05:00,7,"0.16 x 1","0.34 x 2",0.2,22543,mid,450800,22545,137,41.73%,-0.150790699,SLFT,ToOpen,"11:21:59 ET"
EQT,60.72,Call,64,2025-06-27T16:30:00-05:00,7,"0.25 x 224","0.36 x 416",0.25,6800,bid,170000,6801,47,36.57%,0.15837508057708,SLFT,SellToOpen,"11:58:57 ET" """

    # Simulate file upload
    cadena_file = io.StringIO(cadena_data)
    griegas_file = io.StringIO(griegas_data)
    inusual_file = io.StringIO(inusual_data)

    df_cadena = load_and_clean_cadena(cadena_file)
    if df_cadena is not None:
        print("--- Cleaned CADENA.csv ---")
        print(df_cadena.head())
        print(df_cadena.info())

    df_griegas = load_and_clean_griegas(griegas_file)
    if df_griegas is not None:
        print("\n--- Cleaned Griegas.csv ---")
        print(df_griegas.head())
        print(df_griegas.info())

    df_inusual = load_and_clean_inusual(inusual_file)
    if df_inusual is not None:
        print("\n--- Cleaned Inusual.csv ---")
        print(df_inusual.head())
        print(df_inusual.info())
