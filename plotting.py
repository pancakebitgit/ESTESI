import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np

# Template for dark theme for plotly express figs
THEME_DARK = "plotly_dark"

def plot_volume_by_strike(df, strike_col='Strike', vol_col='Volume', type_col='Type', call_label='call', put_label='put'):
    """Plots Volume by Strike for Calls and Puts."""
    if df is None or df.empty or not all(c in df.columns for c in [strike_col, vol_col, type_col]):
        return go.Figure().update_layout(title_text='Volume by Strike (No Data or Missing Columns)', template=THEME_DARK)

    df_plot = df.copy()
    df_plot[type_col] = df_plot[type_col].str.lower()

    fig = go.Figure()
    df_calls = df_plot[df_plot[type_col] == call_label].groupby(strike_col)[vol_col].sum().reset_index()
    fig.add_trace(go.Bar(x=df_calls[strike_col], y=df_calls[vol_col], name='Call Volume', marker_color='green'))

    df_puts = df_plot[df_plot[type_col] == put_label].groupby(strike_col)[vol_col].sum().reset_index()
    fig.add_trace(go.Bar(x=df_puts[strike_col], y=df_puts[vol_col], name='Put Volume', marker_color='red'))

    fig.update_layout(
        title_text='Volume by Strike',
        xaxis_title='Strike Price',
        yaxis_title='Volume',
        barmode='group',
        template=THEME_DARK
    )
    return fig

def plot_oi_by_strike(df, strike_col='Strike', oi_col='Open Int', type_col='Type', call_label='call', put_label='put'):
    """Plots Open Interest by Strike for Calls and Puts."""
    if df is None or df.empty or not all(c in df.columns for c in [strike_col, oi_col, type_col]):
        return go.Figure().update_layout(title_text='Open Interest by Strike (No Data or Missing Columns)', template=THEME_DARK)

    df_plot = df.copy()
    df_plot[type_col] = df_plot[type_col].str.lower()

    fig = go.Figure()
    df_calls = df_plot[df_plot[type_col] == call_label].groupby(strike_col)[oi_col].sum().reset_index()
    fig.add_trace(go.Bar(x=df_calls[strike_col], y=df_calls[oi_col], name='Call Open Interest', marker_color='lightgreen'))

    df_puts = df_plot[df_plot[type_col] == put_label].groupby(strike_col)[oi_col].sum().reset_index()
    fig.add_trace(go.Bar(x=df_puts[strike_col], y=df_puts[oi_col], name='Put Open Interest', marker_color='salmon'))

    fig.update_layout(
        title_text='Open Interest by Strike',
        xaxis_title='Strike Price',
        yaxis_title='Open Interest',
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
                  title=f'Max Pain Analysis - Minimized Option Value at: ${max_pain_strike:.2f}',
                  labels={strike_price_col: 'Potential Expiry Price', total_option_value_col: 'Total Value of Options that would have Value'},
                  template=THEME_DARK)
    fig.add_vline(x=max_pain_strike, line_width=2, line_dash="dash", line_color="red",
                  annotation_text=f"Max Pain: {max_pain_strike:.2f}", annotation_position="top right")
    return fig

def plot_gex_dashboard_view(gex_bars_data, df_aggregate_curve, gamma_flip_price,
                            net_gex_at_current_price, current_spot_price):
    """
    Plots Gamma Exposure (GEX) by Strike (bars) and Aggregate GEX Curve (line).
    Highlights current spot price and Gamma Flip Point.
    """
    if gex_bars_data is None or gex_bars_data.empty or df_aggregate_curve is None or df_aggregate_curve.empty:
        fig = go.Figure().update_layout(title_text='Gamma Exposure (GEX) Analysis (No Data)', template=THEME_DARK)
        if net_gex_at_current_price is not None:
             fig.add_annotation(text=f"Net GEX at current price: {net_gex_at_current_price:,.0f}",
                                showarrow=False, xref="paper", yref="paper", x=0.5, y=0.9)
        return fig

    fig = make_subplots(specs=[[{"secondary_y": True}]])

    # Bar chart for GEX per strike
    colors = ['green' if val >= 0 else 'red' for val in gex_bars_data['GEX_Signed_Value']]
    fig.add_trace(
        go.Bar(x=gex_bars_data['Strike'], y=gex_bars_data['GEX_Signed_Value'],
               name='GEX per Strike', marker_color=colors),
        secondary_y=False,
    )

    # Line chart for Aggregate GEX curve
    fig.add_trace(
        go.Scatter(x=df_aggregate_curve['Simulated_Price'], y=df_aggregate_curve['Total_GEX'],
                   name='Aggregate GEX Curve', line=dict(color='deepskyblue', width=2)),
        secondary_y=True, # Plotting on a secondary y-axis if scales differ significantly
    )

    # Line for current spot price
    if current_spot_price is not None:
        fig.add_vline(x=current_spot_price, line_width=2, line_dash="dash", line_color="yellow",
                      annotation_text=f"Current Spot: {current_spot_price:.2f}",
                      annotation_position="bottom right")

    # Line for Gamma Flip Point
    if gamma_flip_price is not None and not np.isnan(gamma_flip_price):
        fig.add_vline(x=gamma_flip_price, line_width=2, line_dash="dot", line_color="magenta",
                      annotation_text=f"Gamma Flip: {gamma_flip_price:.2f}",
                      annotation_position="bottom left")

    title_text = f'Gamma Exposure Analysis<br>Net GEX at Current Price: {net_gex_at_current_price:,.0f}'
    fig.update_layout(
        title_text=title_text,
        xaxis_title='Underlying Price / Strike Price',
        template=THEME_DARK,
        legend_title_text='Legend',
        barmode='relative' # Ensures bars are drawn from zero
    )
    fig.update_yaxes(title_text="GEX per Strike", secondary_y=False)
    fig.update_yaxes(title_text="Total Aggregate GEX", secondary_y=True, showgrid=False) # Hide grid for secondary y-axis if desired

    return fig


def plot_greek_exposure(exposure_df, total_exposure, greek_name, strike_col='strike', exposure_col='exposure_value'):
    """Plots total exposure for a given Greek (Vega, Theta) by strike."""
    if exposure_df is None or exposure_df.empty:
        return go.Figure().update_layout(title_text=f'{greek_name} Exposure (No Data)', template=THEME_DARK)

    title = f'Total {greek_name} Exposure by Strike (Overall: {total_exposure:,.2f})'
    fig = px.bar(exposure_df, x=strike_col, y=exposure_col, title=title, template=THEME_DARK)
    fig.update_yaxes(title_text=f'{greek_name} Exposure ($)') # Assuming exposure is in $
    return fig

def plot_unusual_flow(df_inusual, strike_col='Strike', premium_col='Premium', size_col='Size',
                      side_col='Side', type_col='Type', time_col='Trade_Time', symbol_col='Symbol'):
    """Plots unusual option flow activity."""
    if df_inusual is None or df_inusual.empty:
        return go.Figure().update_layout(title_text='Unusual Option Flow (No Data)', template=THEME_DARK)

    df_plot = df_inusual.copy()
    if symbol_col not in df_plot.columns and 'Symbol' in df_plot.columns:
        df_plot[symbol_col] = df_plot['Symbol']
    elif symbol_col not in df_plot.columns:
         df_plot[symbol_col] = "N/A"

    hover_data = [strike_col, premium_col, size_col, side_col, type_col, time_col, symbol_col]
    hover_data = [col for col in hover_data if col in df_plot.columns]

    fig = px.scatter(df_plot, x=strike_col, y=premium_col, # 'premium_col' here is trade price
                     size=size_col, color=side_col,
                     hover_name=symbol_col if symbol_col in df_plot.columns else None,
                     hover_data=hover_data,
                     title='Unusual Option Flow Activity',
                     size_max=50,
                     template=THEME_DARK)

    fig.update_layout(
        xaxis_title='Strike Price',
        yaxis_title='Trade Price per Contract', # Changed from 'Premium' to 'Trade Price'
        coloraxis_colorbar=dict(title="Trade Side")
    )
    return fig


if __name__ == '__main__':
    # Create dummy data for testing plots
    sample_strikes = list(range(50, 75, 5))

    # For plot_volume_by_strike and plot_oi_by_strike
    df_chain_sample = pd.DataFrame({
        'Strike': sample_strikes * 2,
        'Volume': [100, 200, 500, 300, 50] * 2,
        'Open Int': [1000, 1500, 3000, 2000, 500] * 2,
        'Type': ['call']*len(sample_strikes) + ['put']*len(sample_strikes)
    })
    fig_vol = plot_volume_by_strike(df_chain_sample)
    # fig_vol.show()
    fig_oi = plot_oi_by_strike(df_chain_sample)
    # fig_oi.show()

    # For plot_gex_dashboard_view
    gex_bars_sample = pd.DataFrame({
        'Strike': sample_strikes,
        'GEX_Signed_Value': [-100000, 50000, 200000, -80000, 30000]
    })
    aggregate_curve_sample = pd.DataFrame({
        'Simulated_Price': np.linspace(min(sample_strikes)*0.9, max(sample_strikes)*1.1, 50),
        'Total_GEX': np.sin(np.linspace(0, 2*np.pi, 50)) * 500000
    })
    aggregate_curve_sample.loc[aggregate_curve_sample['Simulated_Price'] < 60, 'Total_GEX'] *= -1 # Make some negative for flip

    flip_price_sample = 58.0
    net_gex_current_sample = 100000
    current_spot_sample = 62.5

    fig_gex_detailed = plot_gex_dashboard_view(
        gex_bars_sample,
        aggregate_curve_sample,
        flip_price_sample,
        net_gex_current_sample,
        current_spot_sample
    )
    # fig_gex_detailed.show()

    print("Plotting functions updated/created with dummy examples. Uncomment '.show()' lines to display plots if running locally.")
