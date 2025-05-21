"""
pages/8_🤖_AI_and_ML.py
AI and ML insights using a tabbed interface.
Includes Time Series Forecasting, Anomaly Detection, and Trade Outcome Classification.
"""
import streamlit as st
import pandas as pd
import numpy as np
import logging
import plotly.graph_objects as go
import plotly.express as px # For feature importance plot

try:
    from config import APP_TITLE, EXPECTED_COLUMNS, FORECAST_HORIZON, COLORS, CONCEPTUAL_COLUMNS
    from utils.common_utils import display_custom_message
    from services.ai_model_service import AIModelService
    from plotting import _apply_custom_theme
    from ai_models import PMDARIMA_AVAILABLE, PROPHET_AVAILABLE, ISOLATION_FOREST_AVAILABLE, SKLEARN_AVAILABLE
except ImportError as e:
    st.error(f"AI & ML Page Error: Critical module import failed: {e}.")
    APP_TITLE = "TradingDashboard_Error_AIML" 
    logger = logging.getLogger(APP_TITLE)
    logger.error(f"CRITICAL IMPORT ERROR in 8_🤖_AI_and_ML.py: {e}", exc_info=True)
    COLORS = {"primary_color": "#4169E1", "orange": "#FFA500", "red": "#FF0000", "green": "#28a745", "blue": "#007bff"} 
    PMDARIMA_AVAILABLE = False; PROPHET_AVAILABLE = False; ISOLATION_FOREST_AVAILABLE = False; SKLEARN_AVAILABLE = False
    EXPECTED_COLUMNS = {} ; FORECAST_HORIZON = 30; CONCEPTUAL_COLUMNS = {}
    def display_custom_message(msg, type="error"): st.error(msg)
    class AIModelService:
        def get_arima_forecast(self, *args, **kwargs): return {"error": "Service not loaded"}
        def get_prophet_forecast(self, *args, **kwargs): return {"error": "Service not loaded"}
        def perform_anomaly_detection(self, *args, **kwargs): return {"error": "Service not loaded"}
        def train_trade_outcome_classifier(self, *args, **kwargs): return {"error": "Service not loaded"}
    def _apply_custom_theme(fig, theme): return fig
    st.stop()

logger = logging.getLogger(APP_TITLE)
ai_model_service = AIModelService()

# --- Helper Function for Time Series Forecasting Tab ---
def render_forecasting_tab(filtered_df: pd.DataFrame, plot_theme: str):
    # ... (Existing forecasting tab code - no changes needed from previous version) ...
    st.markdown("<div class='page-content-container ai-ml-task-container'>", unsafe_allow_html=True) 
    pnl_col = EXPECTED_COLUMNS.get('pnl')
    date_col = EXPECTED_COLUMNS.get('date')
    with st.expander("Configure & Run Forecast Model", expanded=True):
        forecast_series_options = {}
        if pnl_col and date_col and pnl_col in filtered_df.columns and date_col in filtered_df.columns:
            try:
                daily_pnl_series = filtered_df.groupby(pd.to_datetime(filtered_df[date_col]).dt.normalize())[pnl_col].sum().dropna()
                if not daily_pnl_series.empty:
                    if not isinstance(daily_pnl_series.index, pd.DatetimeIndex): daily_pnl_series.index = pd.to_datetime(daily_pnl_series.index)
                    daily_pnl_series = daily_pnl_series.asfreq('D') 
                    forecast_series_options["Daily PnL"] = daily_pnl_series
            except Exception as e: logger.error(f"Error preparing Daily PnL for forecast: {e}", exc_info=True)

        if 'cumulative_pnl' in filtered_df.columns and date_col and date_col in filtered_df.columns:
            try:
                equity_curve_series_raw = filtered_df.set_index(pd.to_datetime(filtered_df[date_col]))['cumulative_pnl'].dropna()
                if not equity_curve_series_raw.empty:
                    if not equity_curve_series_raw.index.is_monotonic_increasing: equity_curve_series_raw = equity_curve_series_raw.sort_index()
                    equity_curve_daily = equity_curve_series_raw.resample('D').last().ffill() 
                    if not equity_curve_daily.empty: forecast_series_options["Equity Curve (Daily)"] = equity_curve_daily
            except Exception as e: logger.error(f"Error preparing Equity Curve for forecast: {e}", exc_info=True)
        
        if not forecast_series_options:
            st.warning("No suitable time series data found for forecasting."); st.markdown("</div>", unsafe_allow_html=True); return
        
        with st.form("forecasting_form_tab"): 
            sel_fc_series_name = st.selectbox("Series to Forecast:", list(forecast_series_options.keys()), key="fc_series_tab")
            model_opts_desc = {}
            if PMDARIMA_AVAILABLE: model_opts_desc["ARIMA (Auto - pmdarima)"] = "Auto-selects ARIMA order."
            model_opts_desc["ARIMA (Manual Order)"] = "Specify ARIMA(p,d,q) & SARIMA orders."
            if PROPHET_AVAILABLE: model_opts_desc["Prophet"] = "Handles seasonality & trend."
            available_fc_model_names = list(model_opts_desc.keys())
            if not available_fc_model_names: st.error("No forecast models available."); st.form_submit_button("Run", disabled=True)
            else:
                sel_fc_model = st.selectbox("Forecast Model:", available_fc_model_names, key="fc_model_tab_select")
                if sel_fc_model in model_opts_desc: st.caption(model_opts_desc[sel_fc_model])
                n_fc_periods = st.number_input("Periods (days):", 1, 365, min(FORECAST_HORIZON, 90), key="fc_periods_tab")
                arima_order_manual = None; arima_seasonal_order_manual = None
                if sel_fc_model == "ARIMA (Manual Order)": # ARIMA manual inputs
                    st.markdown("###### ARIMA Order (p,d,q)"); arima_cols = st.columns(3)
                    p,d,q = arima_cols[0].number_input("p",0,5,1,key="p_tab"), arima_cols[1].number_input("d",0,2,1,key="d_tab"), arima_cols[2].number_input("q",0,5,1,key="q_tab")
                    arima_order_manual = (p,d,q)
                    if st.checkbox("Seasonal ARIMA (SARIMA)?", key="sarima_tab"):
                        st.markdown("###### SARIMA Order (P,D,Q,s)"); sarima_cols = st.columns(4)
                        P,D,Q,s = sarima_cols[0].number_input("P",0,2,1,key="P_tab"), sarima_cols[1].number_input("D",0,1,0,key="D_tab"), sarima_cols[2].number_input("Q",0,2,1,key="Q_tab"), sarima_cols[3].number_input("s",1,365,7,key="s_tab")
                        arima_seasonal_order_manual = (P,D,Q,s)
                submit_fc_btn = st.form_submit_button(f"Generate {sel_fc_model} Forecast")

    if 'submit_fc_btn' in locals() and submit_fc_btn and forecast_series_options and sel_fc_series_name and sel_fc_model:
        ts_to_forecast = forecast_series_options[sel_fc_series_name]
        if len(ts_to_forecast.dropna()) < 20: display_custom_message("Need >= 20 data points.", "warning")
        else:
            with st.spinner(f"Running {sel_fc_model}..."): # Forecast logic
                forecast_output = None
                if "ARIMA" in sel_fc_model:
                    order_to_pass = arima_order_manual if sel_fc_model == "ARIMA (Manual Order)" else None
                    forecast_output = ai_model_service.get_arima_forecast(ts_to_forecast, order=order_to_pass, seasonal_order=arima_seasonal_order_manual, n_periods=n_fc_periods)
                elif sel_fc_model == "Prophet":
                    prophet_df_in = ts_to_forecast.reset_index(); prophet_df_in.columns = ['ds', 'y']; prophet_df_in['ds'] = pd.to_datetime(prophet_df_in['ds'])
                    forecast_output = ai_model_service.get_prophet_forecast(prophet_df_in, n_periods=n_fc_periods)
            st.markdown("---"); st.markdown(f"<h3 class='section-subheader'>Results: {sel_fc_model} for {sel_fc_series_name}</h3>", unsafe_allow_html=True)
            if forecast_output and 'error' not in forecast_output: # Plotting logic
                st.success("Forecast generated!")
                fv, cil, ciu = None, None, None
                if "ARIMA" in sel_fc_model: fv, cil, ciu = forecast_output.get('forecast'), forecast_output.get('conf_int_lower'), forecast_output.get('conf_int_upper')
                elif sel_fc_model == "Prophet":
                    f_df = forecast_output.get('forecast_df')
                    if f_df is not None:
                        f_seg = f_df[pd.to_datetime(f_df['ds']) > pd.to_datetime(ts_to_forecast.index.max())]
                        fv, cil, ciu = f_seg.set_index('ds')['yhat'], f_seg.set_index('ds')['yhat_lower'], f_seg.set_index('ds')['yhat_upper']
                if fv is not None and not fv.empty:
                    fig = go.Figure(); fig.add_trace(go.Scatter(x=ts_to_forecast.index,y=ts_to_forecast.values,name=sel_fc_series_name,line=dict(color=COLORS.get("royal_blue"))))
                    fig.add_trace(go.Scatter(x=fv.index,y=fv.values,name=f'{sel_fc_model} Forecast',line=dict(color=COLORS.get("orange"),dash='dash')))
                    if cil is not None and ciu is not None: fig.add_trace(go.Scatter(x=cil.index,y=cil.values,line=dict(width=0),showlegend=False,hoverinfo='skip')); fig.add_trace(go.Scatter(x=ciu.index,y=ciu.values,line=dict(width=0),fill='tonexty',fillcolor='rgba(255,165,0,0.2)',name='CI',hoverinfo='skip'))
                    fig.update_layout(title_text=f"{sel_fc_model} Forecast: {sel_fc_series_name}", legend=dict(orientation="h",yanchor="bottom",y=1.02,xanchor="right",x=1)); st.plotly_chart(_apply_custom_theme(fig,plot_theme),use_container_width=True)
                    if "ARIMA" in sel_fc_model and forecast_output.get('model_summary'):
                        with st.expander("ARIMA Summary",False): st.code(str(forecast_output.get('model_summary')))
                else: display_custom_message("No forecast values to plot.", "warning")
            elif forecast_output and 'error' in forecast_output: display_custom_message(f"Error: {forecast_output.get('error')}", "error")
            else: display_custom_message("Forecast failed.", "error")
    st.markdown("</div>", unsafe_allow_html=True)


# --- Helper Function for Anomaly Detection Tab ---
def render_anomaly_detection_tab(filtered_df: pd.DataFrame, plot_theme: str):
    # ... (Existing anomaly detection tab code - no changes needed from previous version) ...
    st.markdown("<div class='page-content-container ai-ml-task-container'>", unsafe_allow_html=True)
    pnl_col = EXPECTED_COLUMNS.get('pnl'); date_col = EXPECTED_COLUMNS.get('date'); duration_col = EXPECTED_COLUMNS.get('duration_minutes')
    if not ISOLATION_FOREST_AVAILABLE: display_custom_message("Anomaly Detection (Isolation Forest) not available.", "warning"); st.markdown("</div>", unsafe_allow_html=True); return
    with st.expander("Configure & Run Anomaly Detection", expanded=True): # Keep expanded
        ad_series_options = {}
        if pnl_col and date_col and pnl_col in filtered_df: ad_series_options["Trade PnL"] = filtered_df[[date_col, pnl_col]].copy().dropna(subset=[pnl_col]).set_index(date_col)[pnl_col]
        if duration_col and date_col and duration_col in filtered_df: ad_series_options["Trade Duration (minutes)"] = filtered_df[[date_col, duration_col]].copy().dropna(subset=[duration_col]).set_index(date_col)[duration_col]
        if not ad_series_options: st.warning("No suitable series for anomaly detection."); st.markdown("</div>", unsafe_allow_html=True); return
        with st.form("anomaly_detection_form_tab"):
            sel_ad_series_name = st.selectbox("Series for Anomaly Detection:", list(ad_series_options.keys()), key="ad_series_tab")
            st.caption("Using Isolation Forest method.")
            contamination = st.slider("Expected Anomaly Rate:", 0.01,0.5,0.05,0.01,key="ad_contamination_tab",help="Proportion of outliers.")
            submit_ad_btn = st.form_submit_button("Detect Anomalies")
    if 'submit_ad_btn' in locals() and submit_ad_btn and ad_series_options and sel_ad_series_name:
        series_for_ad = ad_series_options[sel_ad_series_name]
        if len(series_for_ad.dropna()) < 10: display_custom_message("Need >= 10 data points.", "warning")
        else:
            with st.spinner(f"Detecting anomalies in '{sel_ad_series_name}'..."):
                data_for_model = series_for_ad.dropna().values.reshape(-1,1); dt_index_plot = series_for_ad.dropna().index
                ad_output = ai_model_service.perform_anomaly_detection(data=data_for_model,method='isolation_forest',contamination=contamination)
            st.markdown("---"); st.markdown(f"<h3 class='section-subheader'>Results: {sel_ad_series_name}</h3>", unsafe_allow_html=True)
            if ad_output and 'error' not in ad_output:
                anom_indices = ad_output.get('anomalies_indices')
                if anom_indices is not None and len(anom_indices) > 0:
                    st.success(f"{len(anom_indices)} anomalies detected."); plot_df = pd.DataFrame({'date':dt_index_plot,'value':series_for_ad.dropna().values}); plot_df['anomaly']=0; plot_df.loc[anom_indices,'anomaly']=1
                    anom_pts_df = plot_df[plot_df['anomaly']==1]
                    fig = go.Figure(); fig.add_trace(go.Scatter(x=plot_df['date'],y=plot_df['value'],mode='lines+markers',name='Normal',marker=dict(color=COLORS.get("royal_blue"),size=5)))
                    fig.add_trace(go.Scatter(x=anom_pts_df['date'],y=anom_pts_df['value'],mode='markers',name='Anomaly',marker=dict(color=COLORS.get("red"),size=10,symbol='x')))
                    fig.update_layout(title_text=f"Anomalies in {sel_ad_series_name}",legend=dict(orientation="h",yanchor="bottom",y=1.02,xanchor="right",x=1)); st.plotly_chart(_apply_custom_theme(fig,plot_theme),use_container_width=True)
                    with st.expander("View Anomalous Data",False): st.dataframe(anom_pts_df[['date','value']].rename(columns={'value':sel_ad_series_name}),use_container_width=True)
                else: st.info("No anomalies detected.")
            elif ad_output and 'error' in ad_output: display_custom_message(f"Error: {ad_output.get('error')}", "error")
            else: display_custom_message("Anomaly detection failed.", "error")
    st.markdown("</div>", unsafe_allow_html=True)

# --- Helper Function for Trade Outcome Classification Tab ---
def render_classification_tab(filtered_df: pd.DataFrame, plot_theme: str):
    st.markdown("<div class='page-content-container ai-ml-task-container'>", unsafe_allow_html=True)
    
    if not SKLEARN_AVAILABLE:
        display_custom_message("Trade Outcome Classification requires scikit-learn. Please ensure it is installed.", "warning")
        st.markdown("</div>", unsafe_allow_html=True)
        return

    st.markdown("<h3 class='section-subheader'>Configure Trade Outcome Classifier (Random Forest)</h3>", unsafe_allow_html=True)

    # --- Feature and Target Selection ---
    # Exclude non-numeric/non-boolean columns that are hard to auto-process for this simple model
    # Also exclude target-leaking columns like 'pnl' itself if target is derived from it.
    pnl_col = EXPECTED_COLUMNS.get('pnl')
    date_col = EXPECTED_COLUMNS.get('date')
    
    # Create a binary target 'is_win' from PnL
    if pnl_col and pnl_col in filtered_df.columns:
        df_for_classification = filtered_df.copy()
        df_for_classification['is_win'] = (df_for_classification[pnl_col] > 0).astype(int)
        target_col_name = 'is_win'
        
        # Potential features: select numeric columns, exclude PnL and highly correlated ones if known
        # For simplicity, let's offer a selection of numeric columns
        # More advanced: allow selection of categorical and handle encoding
        potential_feature_cols = df_for_classification.select_dtypes(include=np.number).columns.tolist()
        # Remove pnl, cumulative_pnl, and the target itself from potential features
        cols_to_exclude_from_features = [pnl_col, 'cumulative_pnl', target_col_name, 
                                         'win_indicator_num', 'loss_indicator_num', # These are direct indicators of win/loss
                                         # Potentially drawdown columns if they are too correlated with outcome
                                         'drawdown_abs', 'drawdown_pct' 
                                         ]
        
        available_feature_cols = [col for col in potential_feature_cols if col not in cols_to_exclude_from_features and col != date_col] # Exclude date_col if it was somehow numeric
        
        if not available_feature_cols:
            st.warning("No suitable numeric features found for classification after excluding PnL-related columns. Please ensure your data has relevant numeric features.")
            st.markdown("</div>", unsafe_allow_html=True); return

        with st.form("classification_form_tab"):
            st.write(f"Target variable: **{target_col_name}** (1 if PnL > 0, else 0)")
            
            selected_features = st.multiselect(
                "Select features for classification:",
                options=available_feature_cols,
                default=available_feature_cols[:min(5, len(available_feature_cols))], # Default to first 5 or all if less
                key="clf_features_tab",
                help="Select numeric columns to use as predictors. Categorical features are not yet supported in this simple model."
            )
            
            # Model parameters (simplified for now)
            n_estimators = st.slider("Number of Trees (Random Forest):", 50, 200, 100, 10, key="clf_n_estimators_tab")
            max_depth_option = st.select_slider("Max Tree Depth:", options=[None, 5, 10, 15, 20], value=10, key="clf_max_depth_tab", help="'None' means nodes are expanded until all leaves are pure.")
            
            submit_clf_btn = st.form_submit_button("Train & Evaluate Classifier")

    else:
        st.warning(f"PnL column ('{pnl_col}') not found. Cannot create target variable for classification.")
        st.markdown("</div>", unsafe_allow_html=True); return

    # Classification Training and Display Logic
    if 'submit_clf_btn' in locals() and submit_clf_btn and selected_features:
        if len(selected_features) < 1:
            display_custom_message("Please select at least one feature for classification.", "warning")
        else:
            with st.spinner("Training classifier and evaluating..."):
                # The service method expects the original DataFrame and feature/target names
                classification_results = ai_model_service.train_trade_outcome_classifier(
                    df=df_for_classification, # DataFrame with 'is_win' target and original features
                    features=selected_features,
                    target=target_col_name,
                    n_estimators=n_estimators,
                    max_depth=max_depth_option if max_depth_option != "None" else None # Handle None string from select_slider
                )

            st.markdown("---")
            st.markdown(f"<h3 class='section-subheader'>Classification Results (Random Forest)</h3>", unsafe_allow_html=True)

            if classification_results and 'error' not in classification_results:
                st.success("Classifier trained and evaluated successfully!")
                
                acc = classification_results.get('accuracy')
                report = classification_results.get('classification_report')
                importances = classification_results.get('feature_importances')

                if acc is not None:
                    st.metric(label="Model Accuracy", value=f"{acc:.2%}")

                if report:
                    st.markdown("##### Classification Report")
                    # Convert report dict to DataFrame for better display
                    report_df = pd.DataFrame(report).transpose()
                    st.dataframe(report_df.style.format("{:.2f}"), use_container_width=True)
                
                if importances is not None and not importances.empty:
                    st.markdown("##### Feature Importances")
                    # Ensure importances Series has a name for the y-axis in bar plot
                    importances.name = 'Importance' 
                    fig_imp = px.bar(importances.head(15), x=importances.head(15).values, y=importances.head(15).index, orientation='h',
                                     labels={'x': 'Importance', 'y': 'Feature'}, title="Top 15 Feature Importances")
                    fig_imp.update_layout(yaxis={'categoryorder':'total ascending'})
                    st.plotly_chart(_apply_custom_theme(fig_imp, plot_theme), use_container_width=True)

            elif classification_results and 'error' in classification_results:
                display_custom_message(f"Classification Error: {classification_results.get('error')}", "error")
            else:
                display_custom_message("Classification training/evaluation failed to return results.", "error")
    
    st.markdown("</div>", unsafe_allow_html=True) # Close container


# --- Main Page Function ---
def show_ai_ml_page():
    st.title("🤖 AI & Machine Learning Insights") 
    st.markdown(
        "<p class='page-subtitle'>Explore a suite of AI-powered tools to analyze your trading data, forecast trends, detect anomalies, and gain deeper insights into market dynamics and strategy performance.</p>",
        unsafe_allow_html=True
    )

    if 'filtered_data' not in st.session_state or st.session_state.filtered_data is None:
        display_custom_message("Please upload and process your trading data to access AI/ML tools.", "info")
        return
    
    filtered_df = st.session_state.filtered_data
    plot_theme = st.session_state.get('current_theme', 'dark')

    if filtered_df.empty:
        display_custom_message("No data matches the current filters. AI/ML tools require filtered data.", "info")
        return

    tab_titles = ["📈 Time Series Forecasting", "⚠️ Anomaly Detection", "🎯 Outcome Classification"]
    tab1, tab2, tab3 = st.tabs(tab_titles)

    with tab1:
        render_forecasting_tab(filtered_df, plot_theme)
    with tab2:
        render_anomaly_detection_tab(filtered_df, plot_theme)
    with tab3:
        render_classification_tab(filtered_df, plot_theme)

if __name__ == "__main__":
    if 'app_initialized' not in st.session_state:
        st.warning("This page is part of a multi-page app. Please run the main app.py script.")
    show_ai_ml_page()
