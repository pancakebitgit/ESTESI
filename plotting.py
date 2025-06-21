import plotly.express as px
import plotly.graph_objects as go
import pandas as pd

# Template for dark theme for plotly express figs
THEME_DARK = "plotly_dark"

def plot_volume_oi(df, strike_col='Strike', vol_col='Volume', oi_col='Open Int', type_col='Type', call_label='call', put_label='put'):
    """Plots Volume and Open Interest by Strike for Calls and Puts."""
    if df is None or df.empty:
        return go.Figure()

    df_plot = df.copy()
    df_plot[type_col] = df_plot[type_col].str.lower()

    fig = go.Figure()

    # Calls
    df_calls = df_plot[df_plot[type_col] == call_label].sort_values(by=strike_col)
    fig.add_trace(go.Bar(x=df_calls[strike_col], y=df_calls[vol_col], name='Call Volume', marker_color='green'))
    fig.add_trace(go.Bar(x=df_calls[strike_col], y=df_calls[oi_col], name='Call Open Interest', marker_color='lightgreen', visible='legendonly'))

    # Puts
    df_puts = df_plot[df_plot[type_col] == put_label].sort_values(by=strike_col)
    fig.add_trace(go.Bar(x=df_puts[strike_col], y=df_puts[vol_col], name='Put Volume', marker_color='red'))
    fig.add_trace(go.Bar(x=df_puts[strike_col], y=df_puts[oi_col], name='Put Open Interest', marker_color='salmon', visible='legendonly'))

    fig.update_layout(
        title_text='Volume and Open Interest by Strike',
        xaxis_title='Strike Price',
        yaxis_title='Amount',
        barmode='group',
        template=THEME_DARK
    )
    return fig

def plot_put_call_ratio(pcr_data_by_strike, strike_col='Strike', pcr_col='pc_ratio', total_pcr=None):
    """Plots Put/Call Ratio by Strike and optionally displays total P/C Ratio."""
    if pcr_data_by_strike is None or pcr_data_by_strike.empty:
        fig = go.Figure()
        fig.update_layout(title_text='Put/Call Ratio (No Data)', template=THEME_DARK)
        if total_pcr is not None:
            fig.add_annotation(text=f"Total P/C Ratio: {total_pcr:.2f}", showarrow=False, xref="paper", yref="paper", x=0.5, y=0.9)
        return fig

    fig = px.bar(pcr_data_by_strike, x=strike_col, y=pcr_col, title='Put/Call Ratio by Strike', template=THEME_DARK)
    fig.update_yaxes(title_text='P/C Ratio')
    if total_pcr is not None:
        fig.add_annotation(
            text=f"Overall P/C Ratio: {total_pcr:.2f}",
            align='left',
            showarrow=False,
            xref='paper',
            yref='paper',
            x=0.05,
            y=0.95,
            bgcolor="rgba(0,0,0,0.5)",
            bordercolor="rgba(255,255,255,0.5)",
            borderwidth=1
        )
    return fig

def plot_money_at_risk(df_mar, strike_col='Strike'):
    """Plots Money at Risk by Strike for Calls and Puts."""
    if df_mar is None or df_mar.empty:
        return go.Figure().update_layout(title_text='Money at Risk (No Data)', template=THEME_DARK)

    fig = go.Figure()
    if 'call_money_at_risk' in df_mar.columns:
        fig.add_trace(go.Bar(x=df_mar[strike_col], y=df_mar['call_money_at_risk'], name='Call Money at Risk', marker_color='blue'))
    if 'put_money_at_risk' in df_mar.columns:
        fig.add_trace(go.Bar(x=df_mar[strike_col], y=df_mar['put_money_at_risk'], name='Put Money at Risk', marker_color='orange'))

    fig.update_layout(
        title_text='Money at Risk by Strike',
        xaxis_title='Strike Price',
        yaxis_title='Dollar Value at Risk',
        barmode='group',
        template=THEME_DARK
    )
    return fig

def plot_max_pain(max_pain_strike, cash_values_df, strike_price_col='strike_price_at_expiry', total_option_value_col='total_option_value'):
    """Plots the total option value at different expiry prices to illustrate Max Pain."""
    if cash_values_df is None or cash_values_df.empty or max_pain_strike is None:
        return go.Figure().update_layout(title_text='Max Pain Analysis (No Data)', template=THEME_DARK)

    fig = px.line(cash_values_df, x=strike_price_col, y=total_option_value_col,
                  title=f'Max Pain Analysis - Minimized Option Value at: ${max_pain_strike}',
                  labels={strike_price_col: 'Potential Expiry Price', total_option_value_col: 'Total Value of Options that would have Value'},
                  template=THEME_DARK)
    fig.add_vline(x=max_pain_strike, line_width=2, line_dash="dash", line_color="red",
                  annotation_text=f"Max Pain: {max_pain_strike}", annotation_position="top right")
    return fig

def plot_gex(gex_by_strike_df, total_gex, strike_col='strike', gex_col='gex', flip_points_df=None):
    """Plots Gamma Exposure (GEX) by strike and indicates total GEX and flip points."""
    if gex_by_strike_df is None or gex_by_strike_df.empty:
        fig = go.Figure().update_layout(title_text='Gamma Exposure (GEX) (No Data)', template=THEME_DARK)
        if total_gex is not None:
             fig.add_annotation(text=f"Total GEX: {total_gex:,.0f} shares (approx)", showarrow=False, xref="paper", yref="paper", x=0.5, y=0.9)
        return fig

    fig = px.bar(gex_by_strike_df, x=strike_col, y=gex_col, title=f'Gamma Exposure (GEX) by Strike (Total GEX: {total_gex:,.0f} shares approx.)',
                 template=THEME_DARK)
    fig.update_yaxes(title_text='GEX (Shares)')
    fig.add_hline(y=0, line_width=1, line_dash="dash", line_color="grey")

    if flip_points_df is not None and not flip_points_df.empty:
        for idx, row in flip_points_df.iterrows():
            fig.add_vline(x=row[strike_col], line_width=1, line_dash="longdash", line_color="yellow",
                          annotation_text=f"Potential Flip: {row[strike_col]}", annotation_position="bottom right")
    return fig

def plot_greek_exposure(exposure_df, total_exposure, greek_name, strike_col='strike', exposure_col='exposure_value'):
    """Plots total exposure for a given Greek (Vega, Theta) by strike."""
    if exposure_df is None or exposure_df.empty:
        return go.Figure().update_layout(title_text=f'{greek_name} Exposure (No Data)', template=THEME_DARK)

    title = f'Total {greek_name} Exposure by Strike (Overall: {total_exposure:,.2f})'
    fig = px.bar(exposure_df, x=strike_col, y=exposure_col, title=title, template=THEME_DARK)
    fig.update_yaxes(title_text=f'{greek_name} Exposure')
    return fig

def plot_unusual_flow(df_inusual, strike_col='Strike', premium_col='Premium', size_col='Size',
                      side_col='Side', type_col='Type', time_col='Trade_Time', symbol_col='Symbol'):
    """Plots unusual option flow activity."""
    if df_inusual is None or df_inusual.empty:
        return go.Figure().update_layout(title_text='Unusual Option Flow (No Data)', template=THEME_DARK)

    # Ensure required columns exist, or provide defaults
    df_plot = df_inusual.copy()
    if symbol_col not in df_plot.columns and 'Symbol' in df_plot.columns: # Check if 'Symbol' was not passed as symbol_col
        df_plot[symbol_col] = df_plot['Symbol'] # Use default if available
    elif symbol_col not in df_plot.columns:
         df_plot[symbol_col] = "N/A"


    hover_data = [strike_col, premium_col, size_col, side_col, type_col, time_col, symbol_col]
    # Filter out columns not present in df_plot for hover_data
    hover_data = [col for col in hover_data if col in df_plot.columns]


    fig = px.scatter(df_plot, x=strike_col, y=premium_col,
                     size=size_col, color=side_col, # Or type_col
                     hover_name=symbol_col if symbol_col in df_plot.columns else None,
                     hover_data=hover_data,
                     title='Unusual Option Flow Activity',
                     size_max=50,
                     template=THEME_DARK)

    fig.update_layout(
        xaxis_title='Strike Price',
        yaxis_title='Premium Paid/Received per Contract (Trade Price)',
        coloraxis_colorbar=dict(title="Trade Side")
    )
    return fig


if __name__ == '__main__':
    # Create dummy data for testing plots
    sample_strikes = list(range(50, 75, 5))

    # For plot_volume_oi
    df_vol_oi_sample = pd.DataFrame({
        'Strike': sample_strikes * 2,
        'Volume': [100, 200, 500, 300, 50] * 2,
        'Open Int': [1000, 1500, 3000, 2000, 500] * 2,
        'Type': ['call']*len(sample_strikes) + ['put']*len(sample_strikes)
    })
    fig_vol_oi = plot_volume_oi(df_vol_oi_sample)
    # fig_vol_oi.show() # Uncomment to display locally if running this file

    # For plot_put_call_ratio
    pcr_data_sample = pd.DataFrame({
        'Strike': sample_strikes,
        'pc_ratio': [0.5, 0.8, 1.0, 1.2, 0.9]
    })
    fig_pcr = plot_put_call_ratio(pcr_data_sample, total_pcr=0.95)
    # fig_pcr.show()

    # For plot_money_at_risk
    mar_sample = pd.DataFrame({
        'Strike': sample_strikes,
        'call_money_at_risk': [100000, 150000, 200000, 120000, 40000],
        'put_money_at_risk': [80000, 130000, 180000, 100000, 30000]
    })
    fig_mar = plot_money_at_risk(mar_sample)
    # fig_mar.show()

    # For plot_max_pain
    max_pain_df_sample = pd.DataFrame({
        'strike_price_at_expiry': sample_strikes,
        'total_option_value': [500000, 400000, 350000, 420000, 520000]
    })
    fig_max_pain = plot_max_pain(max_pain_strike=60, cash_values_df=max_pain_df_sample)
    # fig_max_pain.show()

    # For plot_gex
    gex_sample_df = pd.DataFrame({
        'strike': sample_strikes,
        'gex': [-10000, -5000, 2000, 8000, 3000] # Shares
    })
    flip_sample_df = pd.DataFrame({'strike': [57.5], 'gex': [0]}) # Dummy flip point
    fig_gex = plot_gex(gex_sample_df, total_gex=sum(gex_sample_df['gex']), flip_points_df=flip_sample_df)
    # fig_gex.show()

    # For plot_greek_exposure (Vega)
    vega_exposure_sample = pd.DataFrame({
        'strike': sample_strikes,
        'exposure_value': [5000, 8000, 12000, 7000, 2000]
    })
    fig_vega = plot_greek_exposure(vega_exposure_sample, total_exposure=sum(vega_exposure_sample['exposure_value']), greek_name='Vega')
    # fig_vega.show()

    # For plot_unusual_flow
    inusual_flow_sample = pd.DataFrame({
        'Symbol': ['XYZ'] * 5,
        'Strike': sample_strikes,
        'Premium': [1.5, 2.0, 2.5, 1.0, 0.5], # Trade price
        'Size': [100, 50, 200, 150, 75], # Number of contracts
        'Side': ['buy', 'buy', 'sell', 'buy', 'sell'],
        'Type': ['call', 'call', 'put', 'call', 'put'],
        'Trade_Time': ['10:00', '10:05', '10:15', '10:20', '10:30']
    })
    fig_flow = plot_unusual_flow(inusual_flow_sample)
    # fig_flow.show()

    print("Plotting functions created with dummy examples. Uncomment '.show()' lines to display plots if running locally.")
