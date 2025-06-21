import streamlit as st
import pandas as pd
import datetime # For date input default

# Import functions from other modules
import file_handler
import calculations
import plotting

# --- Page Configuration ---
st.set_page_config(
    page_title="Interactive Options Analysis Dashboard",
    layout="wide",
    initial_sidebar_state="expanded",
)
st.title("📊 Interactive Options Analysis Dashboard")
st.markdown("Upload your options data CSV files to begin analysis. Ensure `Griegas.csv` contains `IV` (Implied Volatility) and `Exp Date` or `DTE` for GEX calculations.")

# --- Sidebar Inputs ---
st.sidebar.header("Global Parameters")
risk_free_rate = st.sidebar.number_input("Risk-Free Interest Rate (e.g., 0.05 for 5%)",
                                         min_value=0.0, max_value=1.0, value=0.05, step=0.001, format="%.3f")
calculation_date = st.sidebar.date_input("Calculation Date (for DTE)", value=datetime.date.today())
# Convert calculation_date to pandas Timestamp for consistency with datetime operations in pandas
calculation_date_ts = pd.Timestamp(calculation_date)


st.sidebar.header("Upload Data Files")
cadena_file = st.sidebar.file_uploader("Upload CADENA.csv (Option Chain)", type="csv")
griegas_file = st.sidebar.file_uploader("Upload Griegas.csv (Options with Greeks)", type="csv")
inusual_file = st.sidebar.file_uploader("Upload Inusual.csv (Unusual Option Flow)", type="csv")

# --- Data Loading and Caching ---
@st.cache_data # Use st.cache_data for dataframes
def load_data(file, _handler_func):
    if file:
        df = _handler_func(file)
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
    # Determine active tabs based on uploaded files
    active_tabs = []
    if df_cadena is not None:
        active_tabs.extend(["Chain Overview & Sentiment", "Key Levels (CADENA)"])
    if df_griegas is not None:
        active_tabs.extend(["Greek Exposures"])
    if df_inusual is not None:
        active_tabs.extend(["Unusual Flow"])

    if not active_tabs:
        st.warning("No data loaded successfully for any analysis. Please check file formats or content.")
        st.stop()

    tabs = st.tabs(active_tabs)
    current_tab_idx = 0

    # --- Tab: Chain Overview & Sentiment (Requires CADENA.csv) ---
    if df_cadena is not None:
        with tabs[current_tab_idx]:
            st.header("⛓️ Option Chain Overview & Sentiment (CADENA.csv)")

            st.subheader("Raw Chain Data (Sample)")
            st.dataframe(df_cadena.head(), height=200)

            st.subheader("Volume & Open Interest by Strike")
            col_vol, col_oi = st.columns(2)
            with col_vol:
                fig_vol = plotting.plot_volume_by_strike(df_cadena, type_col='Type')
                st.plotly_chart(fig_vol, use_container_width=True)
            with col_oi:
                fig_oi = plotting.plot_oi_by_strike(df_cadena, type_col='Type')
                st.plotly_chart(fig_oi, use_container_width=True)

            st.subheader("Put/Call Ratios")
            col_pcr_vol, col_pcr_oi = st.columns(2)
            with col_pcr_vol:
                pc_vol_total, pc_vol_strike = calculations.calculate_put_call_ratio(df_cadena, value_col='Volume', type_col='Type')
                if pc_vol_total is not None:
                    st.metric(label="Total P/C Ratio (Volume)", value=f"{pc_vol_total:.2f}")
                if pc_vol_strike is not None:
                    fig_pc_vol = plotting.plot_put_call_ratio(pc_vol_strike, total_pcr=pc_vol_total)
                    st.plotly_chart(fig_pc_vol, use_container_width=True)
                else:
                    st.info("Not enough data for P/C Volume analysis.")

            with col_pcr_oi:
                pc_oi_total, pc_oi_strike = calculations.calculate_put_call_ratio(df_cadena, value_col='Open Int', type_col='Type')
                if pc_oi_total is not None:
                    st.metric(label="Total P/C Ratio (Open Interest)", value=f"{pc_oi_total:.2f}")
                if pc_oi_strike is not None:
                    fig_pc_oi = plotting.plot_put_call_ratio(pc_oi_strike, total_pcr=pc_oi_total)
                    st.plotly_chart(fig_pc_oi, use_container_width=True)
                else:
                    st.info("Not enough data for P/C Open Interest analysis.")
        current_tab_idx += 1

        # --- Tab: Key Levels (Requires CADENA.csv) ---
        with tabs[current_tab_idx]:
            st.header("💰 Key Levels Analysis (CADENA.csv)")

            st.subheader("Money at Risk")
            if 'Mid' in df_cadena.columns and 'Open Int' in df_cadena.columns and 'Strike' in df_cadena.columns and 'Type' in df_cadena.columns:
                mar_df = calculations.calculate_money_at_risk(df_cadena, mid_price_col='Mid')
                if mar_df is not None:
                    fig_mar = plotting.plot_money_at_risk(mar_df)
                    st.plotly_chart(fig_mar, use_container_width=True)
                else:
                    st.info("Could not calculate Money at Risk.")
            else:
                st.warning("Missing one or more columns ('Mid', 'Open Int', 'Strike', 'Type') needed for Money at Risk in CADENA.csv.")

            st.subheader("Max Pain")
            if 'Open Int' in df_cadena.columns and 'Strike' in df_cadena.columns and 'Type' in df_cadena.columns:
                max_pain_strike, max_pain_df = calculations.calculate_max_pain(df_cadena)
                if max_pain_strike is not None and max_pain_df is not None:
                    st.metric(label="Calculated Max Pain Strike", value=f"${max_pain_strike:.2f}")
                    fig_max_pain = plotting.plot_max_pain(max_pain_strike, max_pain_df)
                    st.plotly_chart(fig_max_pain, use_container_width=True)
                else:
                    st.info("Could not calculate Max Pain.")
            else:
                st.warning("Missing one or more columns ('Open Int', 'Strike', 'Type') needed for Max Pain in CADENA.csv.")
        current_tab_idx += 1

    # --- Tab: Greek Exposures (Requires Griegas.csv) ---
    if df_griegas is not None:
        with tabs[current_tab_idx]:
            st.header("🇬🇷 Greek Exposures (Griegas.csv)")

            st.subheader("Raw Griegas Data (Sample)")
            st.dataframe(df_griegas.head(), height=200)

            current_spot_price = None
            if 'Underlying_Price' in df_griegas.columns and not df_griegas.empty:
                current_spot_price = df_griegas['Underlying_Price'].iloc[0]
                st.sidebar.info(f"Using Underlying Price for GEX: ${current_spot_price:.2f} (from Griegas.csv first row).")
            else:
                st.sidebar.error("Column 'Underlying_Price' not found in Griegas.csv or file is empty. Cannot calculate GEX.")
                st.error("GEX calculation requires 'Underlying_Price' in Griegas.csv.")

            if current_spot_price is not None:
                st.subheader("Gamma Exposure (GEX) Analytics (Black-Scholes based)")
                # Ensure all required columns for calculate_gex_analytics are present
                required_gex_cols = ['Strike', 'Open Int', 'Type', 'IV']
                # DTE or Exp Date is also checked inside calculate_gex_analytics
                if not all(col in df_griegas.columns for col in required_gex_cols) or \
                   ('Exp Date' not in df_griegas.columns and 'DTE' not in df_griegas.columns):
                    st.error("Griegas.csv is missing one or more required columns for GEX: 'Strike', 'Open Int', 'Type', 'IV', and ('Exp Date' or 'DTE').")
                else:
                    try:
                        gex_bars, gex_curve, flip_price, net_gex_now = calculations.calculate_gex_analytics(
                            df_griegas,
                            current_spot_price=current_spot_price,
                            risk_free_rate=risk_free_rate,
                            calculation_date=calculation_date_ts # Use the pandas Timestamp
                        )
                        if gex_bars is not None and gex_curve is not None:
                            st.metric(label="Net GEX at Current Price (approx. shares)", value=f"{net_gex_now:,.0f}")
                            fig_gex_detailed = plotting.plot_gex_dashboard_view(
                                gex_bars_data=gex_bars,
                                df_aggregate_curve=gex_curve,
                                gamma_flip_price=flip_price,
                                net_gex_at_current_price=net_gex_now,
                                current_spot_price=current_spot_price
                            )
                            st.plotly_chart(fig_gex_detailed, use_container_width=True)
                            if flip_price is not None and not pd.isna(flip_price):
                                st.metric(label="Calculated Gamma Flip Price", value=f"${flip_price:.2f}")
                            else:
                                st.info("Gamma Flip Price could not be determined (e.g., GEX curve does not cross zero or data insufficient).")
                        else:
                            st.warning("GEX Analytics could not be computed. Check data quality or console for errors from calculation module.")
                    except Exception as e:
                        st.error(f"Error during GEX calculation: {e}")
                        st.exception(e)


            col_vega, col_theta = st.columns(2)
            # Vega Exposure
            if 'Vega' in df_griegas.columns:
                with col_vega:
                    st.subheader("Vega Exposure")
                    total_vega, vega_by_strike = calculations.calculate_total_exposure(df_griegas, greek_col='Vega')
                    if total_vega is not None and vega_by_strike is not None:
                        st.metric(label="Total Vega Exposure", value=f"${total_vega:,.0f}")
                        fig_vega = plotting.plot_greek_exposure(vega_by_strike, total_vega, 'Vega', strike_col='Strike')
                        st.plotly_chart(fig_vega, use_container_width=True)
                    else:
                        st.info("Could not calculate Vega exposure.")
            else:
                 with col_vega:
                    st.subheader("Vega Exposure")
                    st.warning("Column 'Vega' not found in Griegas.csv.")

            # Theta Exposure
            if 'Theta' in df_griegas.columns:
                with col_theta:
                    st.subheader("Theta Exposure")
                    total_theta, theta_by_strike = calculations.calculate_total_exposure(df_griegas, greek_col='Theta')
                    if total_theta is not None and theta_by_strike is not None:
                        st.metric(label="Total Theta Exposure (Daily Decay)", value=f"${total_theta:,.0f}")
                        fig_theta = plotting.plot_greek_exposure(theta_by_strike, total_theta, 'Theta', strike_col='Strike')
                        st.plotly_chart(fig_theta, use_container_width=True)
                    else:
                        st.info("Could not calculate Theta exposure.")
            else:
                with col_theta:
                    st.subheader("Theta Exposure")
                    st.warning("Column 'Theta' not found in Griegas.csv.")
        current_tab_idx += 1

    # --- Tab: Unusual Option Flow (Requires Inusual.csv) ---
    if df_inusual is not None:
        with tabs[current_tab_idx]:
            st.header("🌊 Unusual Option Flow (Inusual.csv)")

            st.subheader("Raw Unusual Flow Data (Sample)")
            st.dataframe(df_inusual.head(), height=200)

            if not all(col in df_inusual.columns for col in ['Strike', 'Trade', 'Size', 'Side']):
                st.warning("Missing one or more required columns for Unusual Flow plot: 'Strike', 'Trade', 'Size', 'Side'.")
            else:
                fig_flow = plotting.plot_unusual_flow(
                    df_inusual,
                    premium_col='Trade', # 'Trade' is the trade price per contract
                )
                st.plotly_chart(fig_flow, use_container_width=True)

            st.subheader("Top Unusual Trades by Total Premium")
            if 'Premium' in df_inusual.columns: # 'Premium' column is total dollar value of trade
                top_trades = df_inusual.sort_values(by='Premium', ascending=False).head(10)
                st.dataframe(top_trades)
            elif 'Trade' in df_inusual.columns and 'Size' in df_inusual.columns:
                 # Attempt to calculate total premium if 'Premium' column is missing
                df_inusual['Calculated_Total_Premium'] = df_inusual['Trade'] * df_inusual['Size'] * 100 # Assuming 100 shares/contract
                top_trades = df_inusual.sort_values(by='Calculated_Total_Premium', ascending=False).head(10)
                st.dataframe(top_trades)
                st.caption("Note: 'Premium' column (total value) was missing, calculated as Trade * Size * 100.")
            else:
                st.info("Column 'Premium' (total value) not found for sorting top trades, and cannot calculate it from 'Trade' and 'Size'.")
        current_tab_idx += 1

st.sidebar.markdown("---")
st.sidebar.info("This dashboard provides an interactive analysis of options data. Calculations and interpretations are based on common financial models and may require domain expertise for full understanding.")

# To run this dashboard:
# 1. Save this code as dashboard.py
# 2. Ensure file_handler.py, calculations.py, plotting.py are in the same directory.
# 3. Ensure requirements.txt lists streamlit, pandas, plotly, scipy. Install with `pip install -r requirements.txt`.
# 4. Run `streamlit run dashboard.py` in your terminal.
