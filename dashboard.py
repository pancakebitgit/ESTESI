import streamlit as st
import pandas as pd

# Import functions from other modules
import file_handler
import calculations
import plotting

# --- Page Configuration ---
st.set_page_config(
    page_title="Interactive Options Analysis Dashboard",
    layout="wide",
    initial_sidebar_state="expanded",
    # Streamlit's native theme options are 'light' or 'dark'.
    # For Plotly charts, we set template='plotly_dark' individually.
)
# Apply dark theme using Streamlit's capabilities if available,
# or custom CSS. For now, relying on Plotly's dark theme for charts.
# To enable Streamlit's dark theme by default if user OS is dark:
# theme = {"base": "dark"} # This would go into set_page_config if supported directly like this
# Or, use st.markdown to inject CSS for a dark theme.
# For simplicity, we'll ensure Plotly charts use their dark theme.

st.title("📊 Interactive Options Analysis Dashboard")
st.markdown("Upload your options data CSV files to begin analysis.")

# --- File Uploaders in Sidebar ---
st.sidebar.header("Upload Data Files")
cadena_file = st.sidebar.file_uploader("Upload CADENA.csv (Option Chain)", type="csv")
griegas_file = st.sidebar.file_uploader("Upload Griegas.csv (Options with Greeks)", type="csv")
inusual_file = st.sidebar.file_uploader("Upload Inusual.csv (Unusual Option Flow)", type="csv")

# --- Data Loading and Caching ---
@st.cache_data # Use st.cache_data for dataframes
def load_data(file, handler_func):
    if file:
        df = handler_func(file)
        if df is not None and not df.empty:
            return df
    return None

df_cadena = load_data(cadena_file, file_handler.load_and_clean_cadena)
df_griegas = load_data(griegas_file, file_handler.load_and_clean_griegas)
df_inusual = load_data(inusual_file, file_handler.load_and_clean_inusual)

# --- Main Dashboard Area ---
if df_cadena is None and df_griegas is None and df_inusual is None:
    st.info("Please upload at least one CSV file to start the analysis.")
else:
    tab_titles = []
    if df_cadena is not None:
        tab_titles.extend(["Overall Chain", "P/C Ratios", "Money & Max Pain"])
    if df_griegas is not None:
        tab_titles.extend(["Greek Exposures (GEX, Vega, Theta)"])
    if df_inusual is not None:
        tab_titles.extend(["Unusual Flow"])

    if not tab_titles: # Should not happen if previous block is correct
        st.warning("No data loaded successfully for any analysis.")

    tabs = st.tabs(tab_titles)
    tab_idx = 0

    # --- Tab 1: Overall Chain View (Requires CADENA.csv) ---
    if df_cadena is not None:
        with tabs[tab_idx]:
            st.header("⛓️ Option Chain Overview (CADENA.csv)")
            st.dataframe(df_cadena, height=300)

            st.subheader("Volume and Open Interest by Strike")
            # Use 'Type' from df_cadena if available, else it might raise error if not present
            # Defaulting to 'Type' as it's expected from the CSV structure.
            fig_vol_oi = plotting.plot_volume_oi(df_cadena, type_col='Type')
            st.plotly_chart(fig_vol_oi, use_container_width=True)
        tab_idx += 1

        # --- Tab 2: Put/Call Ratios (Requires CADENA.csv) ---
        with tabs[tab_idx]:
            st.header(" احساس Sentiment: Put/Call Ratios (CADENA.csv)")

            col1, col2 = st.columns(2)
            with col1:
                st.subheader("Based on Volume")
                pc_vol_total, pc_vol_strike = calculations.calculate_put_call_ratio(df_cadena, value_col='Volume', type_col='Type')
                if pc_vol_total is not None:
                    st.metric(label="Total P/C Ratio (Volume)", value=f"{pc_vol_total:.2f}")
                if pc_vol_strike is not None:
                    fig_pc_vol = plotting.plot_put_call_ratio(pc_vol_strike, total_pcr=pc_vol_total, strike_col='Strike', pcr_col='pc_ratio')
                    st.plotly_chart(fig_pc_vol, use_container_width=True)
                else:
                    st.info("Not enough data for P/C Volume analysis.")

            with col2:
                st.subheader("Based on Open Interest")
                pc_oi_total, pc_oi_strike = calculations.calculate_put_call_ratio(df_cadena, value_col='Open Int', type_col='Type')
                if pc_oi_total is not None:
                    st.metric(label="Total P/C Ratio (Open Interest)", value=f"{pc_oi_total:.2f}")
                if pc_oi_strike is not None:
                    fig_pc_oi = plotting.plot_put_call_ratio(pc_oi_strike, total_pcr=pc_oi_total, strike_col='Strike', pcr_col='pc_ratio')
                    st.plotly_chart(fig_pc_oi, use_container_width=True)
                else:
                    st.info("Not enough data for P/C Open Interest analysis.")
        tab_idx += 1

        # --- Tab 3: Money at Risk & Max Pain (Requires CADENA.csv) ---
        with tabs[tab_idx]:
            st.header("💰 Key Levels: Money at Risk & Max Pain (CADENA.csv)")

            st.subheader("Money at Risk")
            # Ensure 'Mid' column exists, as it's used by calculate_money_at_risk
            if 'Mid' in df_cadena.columns:
                mar_df = calculations.calculate_money_at_risk(df_cadena, mid_price_col='Mid', type_col='Type', strike_col='Strike')
                if mar_df is not None:
                    fig_mar = plotting.plot_money_at_risk(mar_df, strike_col='Strike')
                    st.plotly_chart(fig_mar, use_container_width=True)
                else:
                    st.info("Could not calculate Money at Risk. Check 'Mid', 'Open Int', 'Type', 'Strike' columns.")
            else:
                st.warning("Column 'Mid' not found in CADENA.csv, cannot calculate Money at Risk.")

            st.subheader("Max Pain")
            # Max pain calculation needs Open Interest and Strike prices.
            max_pain_strike, max_pain_df = calculations.calculate_max_pain(df_cadena, type_col='Type', strike_col='Strike')
            if max_pain_strike is not None and max_pain_df is not None:
                st.metric(label="Calculated Max Pain Strike", value=f"${max_pain_strike}")
                fig_max_pain = plotting.plot_max_pain(max_pain_strike, max_pain_df, strike_price_col='strike_price_at_expiry', total_option_value_col='total_option_value')
                st.plotly_chart(fig_max_pain, use_container_width=True)
            else:
                st.info("Could not calculate Max Pain. Ensure 'Strike', 'Open Int', 'Type' columns are present and valid.")
        tab_idx += 1

    # --- Tab 4: Greek Exposures (Requires Griegas.csv) ---
    if df_griegas is not None:
        with tabs[tab_idx]:
            st.header("🇬🇷 Greek Exposures (Griegas.csv)")
            st.dataframe(df_griegas.head(), height=200)

            # Assuming 'Underlying_Price' is available in df_griegas after cleaning
            underlying_price_for_gex = df_griegas['Underlying_Price'].iloc[0] if 'Underlying_Price' in df_griegas.columns and not df_griegas.empty else None

            if underlying_price_for_gex:
                st.sidebar.info(f"Using Underlying Price: ${underlying_price_for_gex:.2f} for some Greek calculations (from Griegas.csv first row).")
            else:
                st.sidebar.warning("Underlying price not found in Griegas.csv. Some Greek calculations might be affected or use defaults.")


            st.subheader("Gamma Exposure (GEX) & Potential Flip Points")
            # GEX calculation needs Gamma, OI, Type, Strike.
            # The 'Underlying_Price' from Griegas.csv (Price~) is used if per_share_gex=False in calc and plotting
            total_gex, gex_by_strike, flip_points = calculations.calculate_gex_and_flip(
                df_griegas,
                underlying_price_col='Underlying_Price', # Make sure this column exists from file_handler
                oi_col='Open Int',
                gamma_col='Gamma',
                type_col='Type',
                strike_col='Strike'
            )
            if total_gex is not None and gex_by_strike is not None:
                st.metric(label="Total Net GEX (approx. shares)", value=f"{total_gex:,.0f}")
                fig_gex = plotting.plot_gex(gex_by_strike, total_gex, strike_col='Strike', gex_col='gex', flip_points_df=flip_points)
                st.plotly_chart(fig_gex, use_container_width=True)
                if flip_points is not None and not flip_points.empty:
                    st.write("Potential Gamma Flip Strikes:", flip_points[['Strike', 'gex']])
                else:
                    st.write("No distinct gamma flip points identified based on current GEX profile.")
            else:
                st.info("Could not calculate GEX. Check 'Gamma', 'Open Int', 'Type', 'Strike' columns in Griegas.csv.")

            col_vega, col_theta = st.columns(2)
            with col_vega:
                st.subheader("Vega Exposure (Sensitivity to IV)")
                total_vega, vega_by_strike = calculations.calculate_total_exposure(df_griegas, greek_col='Vega', oi_col='Open Int', type_col='Type', strike_col='Strike')
                if total_vega is not None and vega_by_strike is not None:
                    st.metric(label="Total Vega Exposure", value=f"${total_vega:,.0f}")
                    fig_vega = plotting.plot_greek_exposure(vega_by_strike, total_vega, 'Vega', strike_col='Strike', exposure_col='exposure_value')
                    st.plotly_chart(fig_vega, use_container_width=True)
                else:
                    st.info("Could not calculate Vega exposure. Check 'Vega' column.")

            with col_theta:
                st.subheader("Theta Exposure (Time Decay)")
                total_theta, theta_by_strike = calculations.calculate_total_exposure(df_griegas, greek_col='Theta', oi_col='Open Int', type_col='Type', strike_col='Strike')
                if total_theta is not None and theta_by_strike is not None:
                    st.metric(label="Total Theta Exposure (Daily Decay)", value=f"${total_theta:,.0f}")
                    fig_theta = plotting.plot_greek_exposure(theta_by_strike, total_theta, 'Theta', strike_col='Strike', exposure_col='exposure_value')
                    st.plotly_chart(fig_theta, use_container_width=True)
                else:
                    st.info("Could not calculate Theta exposure. Check 'Theta' column.")
        tab_idx += 1

    # --- Tab 5: Unusual Option Flow (Requires Inusual.csv) ---
    if df_inusual is not None:
        with tabs[tab_idx]:
            st.header("🌊 Unusual Option Flow (Inusual.csv)")

            # Display a sample of the data
            st.dataframe(df_inusual.head(), height=200)

            # Plot unusual flow
            # Ensure necessary columns like 'Symbol' (or provide a default in plotting if missing)
            # 'Trade_Time' is used in hover, renamed from 'Time' in file_handler
            fig_flow = plotting.plot_unusual_flow(
                df_inusual,
                strike_col='Strike',
                premium_col='Trade', # 'Trade' column is the price of the option trade
                size_col='Size',
                side_col='Side',
                type_col='Type',
                time_col='Trade_Time', # Renamed in file_handler
                symbol_col='Symbol' # Ensure this is present or handled in plotting
            )
            st.plotly_chart(fig_flow, use_container_width=True)

            st.subheader("Top Unusual Trades by Premium")
            # Sort by 'Premium' (total dollar value of the trade)
            if 'Premium' in df_inusual.columns:
                top_trades = df_inusual.sort_values(by='Premium', ascending=False).head(10)
                st.dataframe(top_trades)
            else:
                st.info("Column 'Premium' (total value) not found for sorting top trades.")
        tab_idx += 1

st.sidebar.markdown("---")
st.sidebar.info("This dashboard provides an interactive analysis of options data. Calculations and interpretations are based on common financial models and may require domain expertise for full understanding.")

# To run this dashboard:
# 1. Save this code as dashboard.py
# 2. Ensure file_handler.py, calculations.py, plotting.py are in the same directory.
# 3. Ensure requirements.txt lists streamlit, pandas, plotly. Install with `pip install -r requirements.txt`.
# 4. Run `streamlit run dashboard.py` in your terminal.
