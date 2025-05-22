"""
components/calendar_view.py

This component aims to provide a P&L calendar heatmap view.
Due to Streamlit's native limitations for complex calendar heatmaps like GitHub's
contribution graph, this implementation will use Plotly to create a
heatmap of daily PnL over a year, which is a common way to visualize this.
A true interactive day-by-day calendar might require custom HTML/JS components.
"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from typing import Optional, Dict, Any, List

# Attempt to import from project structure, fallback to placeholders
try:
    from config import EXPECTED_COLUMNS, COLORS, APP_TITLE
    from utils.common_utils import format_currency # For tooltips
except ImportError:
    print("Warning (calendar_view.py): Could not import from root config/utils. Using placeholders.")
    APP_TITLE = "TradingDashboard_Default"
    EXPECTED_COLUMNS = {"date": "date", "pnl": "pnl"}
    # Define default COLORS dictionary if import fails
    COLORS = {
        "green": "#2ECC71", "red": "#E74C3C", "gray": "#BDC3C7", "neutral_dark": "#4A4A4A", "neutral_light": "#F0F0F0",
        "royal_blue": "#3498DB", "dark_background": "#2C3E50", "text_dark": "#ECF0F1",
        "light_background": "#FFFFFF", "text_light": "#2C3E50"
    }
    def format_currency(value, currency_symbol="$", decimals=2):
        """Placeholder format_currency function."""
        if pd.isna(value) or np.isnan(value):
            return "N/A" # Consistent with "No Data" or can be changed
        return f"{currency_symbol}{value:.{decimals}f}"

import logging
logger = logging.getLogger(APP_TITLE if 'APP_TITLE' in globals() else __name__)

class PnLCalendarComponent:
    """
    Component to display a P&L calendar-style heatmap.
    This creates a heatmap of daily PnL, typically for a selected year.
    """
    def __init__(
        self,
        daily_pnl_df: pd.DataFrame,
        year: Optional[int] = None,
        date_col: str = 'date',
        pnl_col: str = 'pnl',
        plot_theme: str = 'dark',
        calendar_colors: Optional[Dict[str, str]] = None, # e.g., {"positive": "green", "negative": "red", "neutral": "gray"}
        custom_colorscale: Optional[List[List[Any]]] = None # e.g., [[0.0, "red"], [0.5, "white"], [1.0, "green"]]
    ):
        """
        Initializes the PnLCalendarComponent.

        Args:
            daily_pnl_df (pd.DataFrame): DataFrame with daily P&L data.
                                         Must contain date_col and pnl_col.
            year (Optional[int]): The year to display. Defaults to the latest year in data.
            date_col (str): Name of the date column in daily_pnl_df.
            pnl_col (str): Name of the P&L column in daily_pnl_df.
            plot_theme (str): 'dark' or 'light'.
            calendar_colors (Optional[Dict[str, str]]): Dictionary to override default
                                                        positive, negative, and neutral colors.
                                                        Keys: 'positive_pnl', 'negative_pnl', 'neutral_color'.
            custom_colorscale (Optional[List[List[Any]]]): A Plotly colorscale.
                                                            Overrides calendar_colors if provided.
        """
        self.daily_pnl_df = daily_pnl_df
        self.year = year
        self.date_col = date_col
        self.pnl_col = pnl_col
        self.plot_theme = plot_theme
        self.calendar_colors = calendar_colors if calendar_colors else {}
        self.custom_colorscale = custom_colorscale
        
        # Validate required columns early
        if self.daily_pnl_df is not None:
            if self.date_col not in self.daily_pnl_df.columns:
                msg = f"PnLCalendar Error: Date column '{self.date_col}' not found in DataFrame."
                logger.error(msg)
                raise ValueError(msg)
            if self.pnl_col not in self.daily_pnl_df.columns:
                msg = f"PnLCalendar Error: P&L column '{self.pnl_col}' not found in DataFrame."
                logger.error(msg)
                raise ValueError(msg)
        
        self.calendar_data = self._prepare_calendar_data()
        logger.debug(f"PnLCalendarComponent initialized for year {self.year} with theme {self.plot_theme}.")

    def _prepare_calendar_data(self) -> Optional[pd.DataFrame]:
        """
        Prepares the data for the calendar heatmap.
        Fills missing days with np.nan for P&L.
        """
        if self.daily_pnl_df is None or self.daily_pnl_df.empty:
            logger.warning("PnLCalendar: daily_pnl_df is empty or None.")
            return None

        df = self.daily_pnl_df.copy()
        try:
            df[self.date_col] = pd.to_datetime(df[self.date_col])
        except Exception as e:
            logger.error(f"PnLCalendar: Could not convert date column '{self.date_col}' to datetime: {e}")
            # Optionally, raise the error or return None if conversion is critical
            # For now, we'll return None to prevent further processing with incorrect data types.
            st.error(f"Error processing date column '{self.date_col}'. Please ensure it's in a recognizable date format.")
            return None

        df = df.set_index(self.date_col)

        if self.year is None:
            if not df.index.empty:
                self.year = df.index.max().year
            else: # Handle case where df is empty after filtering or initially
                self.year = pd.Timestamp.now().year
                logger.info(f"PnLCalendar: No data to infer year, defaulting to current year: {self.year}")
        
        start_date = pd.Timestamp(f'{self.year}-01-01')
        end_date = pd.Timestamp(f'{self.year}-12-31')
        all_days_in_year = pd.date_range(start_date, end_date, freq='D')
        
        # Reindex to include all days in the year, filling missing P&L with np.nan
        # Ensure only relevant columns (pnl_col) are present before reindexing, or select after
        if self.pnl_col in df.columns:
            calendar_df = df[[self.pnl_col]].reindex(all_days_in_year, fill_value=np.nan).reset_index()
        else:
            # This case should ideally be caught by the __init__ checks, but as a safeguard:
            logger.error(f"PnLCalendar: P&L column '{self.pnl_col}' unexpectedly missing during reindex preparation.")
            return None

        calendar_df = calendar_df.rename(columns={'index': 'date', self.pnl_col: 'pnl'}) # Standardize column name to 'pnl'

        # Calculate calendar-specific date parts
        calendar_df['week_of_year'] = calendar_df['date'].dt.isocalendar().week.astype(int)
        calendar_df['day_of_week'] = calendar_df['date'].dt.dayofweek # Monday=0, Sunday=6
        calendar_df['month_text'] = calendar_df['date'].dt.strftime('%b')
        calendar_df['day_of_month'] = calendar_df['date'].dt.day

        # Adjust week numbers at year boundaries for consistent plotting within one year
        # This logic helps ensure that days at the very start/end of a year that might
        # belong to week 53 of the previous year or week 1 of the next year are
        # mapped to week 0 or week 52/53 of the *current* display year.
        if not calendar_df.empty:
            # If Jan 1st is in week 52 or 53 (of previous year), map it to week 0 of current year
            first_day_week = calendar_df['week_of_year'].iloc[0]
            if calendar_df['date'].iloc[0].month == 1 and first_day_week >= 52:
                calendar_df.loc[calendar_df['week_of_year'] == first_day_week, 'week_of_year'] = 0
            
            # If last days of Dec fall into week 1 of next year, map them to the last week of current year (e.g., 52 or 53)
            last_day_week = calendar_df['week_of_year'].iloc[-1]
            if calendar_df['date'].iloc[-1].month == 12 and last_day_week == 1:
                 # Find the max week number for December excluding week 1 (which belongs to next year)
                 max_week_for_dec = calendar_df.loc[(calendar_df['date'].dt.month == 12) & (calendar_df['week_of_year'] != 1), 'week_of_year'].max()
                 if pd.isna(max_week_for_dec): # If all Dec days were in week 1 (e.g. Dec 31 is Monday of week 1)
                     max_week_for_dec = calendar_df.loc[calendar_df['date'].dt.month < 12, 'week_of_year'].max() # Use max week of previous months
                 
                 target_week = max_week_for_dec + 1 if pd.notna(max_week_for_dec) else 52 # Default to 52 if still NaN
                 calendar_df.loc[(calendar_df['date'].dt.month == 12) & (calendar_df['week_of_year'] == 1), 'week_of_year'] = target_week
        
        # Ensure week_of_year is 0-indexed for array population later (0-52)
        # isocalendar().week is 1-53. We need to adjust this for 0-indexed array.
        # Let's map week 1 to index 0, week 53 to index 52.
        # The previous logic for week 0 (Jan 1st in week 52/53) and week 53 (Dec 31st in week 1)
        # might need slight adjustment if we strictly 0-index from isoweek.
        # For simplicity, we'll use the isoweek directly for now and adjust indexing in render.
        # Or, more robustly, create a 0-52 mapping.
        # Let's assume week_of_year will be used to map to 0-52 columns.
        # If week_of_year can be 53, our z_data needs 54 columns or we cap it.
        # Standard approach: 53 weeks max.
        # Map week 53 to index 52. Week 1 to index 0.
        calendar_df['plot_week_idx'] = calendar_df['week_of_year'] - calendar_df['week_of_year'].min()


        logger.debug(f"Calendar data prepared for {self.year}:\n{calendar_df.head()}")
        return calendar_df

    def _get_plotly_theme_layout(self) -> Dict:
        """Gets Plotly layout settings based on the theme."""
        bg_color = COLORS.get('dark_background', '#2C3E50') if self.plot_theme == 'dark' else COLORS.get('light_background', '#FFFFFF')
        font_color = COLORS.get('text_dark', '#ECF0F1') if self.plot_theme == 'dark' else COLORS.get('text_light', '#2C3E50')
        
        return {
            'plot_bgcolor': bg_color,
            'paper_bgcolor': bg_color,
            'font_color': font_color,
            'xaxis': dict(showgrid=False, zeroline=False, dtick=4), # Show tick every 4 weeks
            'yaxis': dict(showgrid=False, zeroline=False),
            'modebar_remove': ['select2d', 'lasso2d', 'zoomIn2d', 'zoomOut2d', 'autoScale2d', 'toggleSpikelines']
        }

    def _get_colorscale(self) -> List[List[Any]]:
        """Determines the colorscale for the heatmap."""
        if self.custom_colorscale:
            return self.custom_colorscale

        # Define colors, allowing overrides from self.calendar_colors
        neg_color = self.calendar_colors.get('negative_pnl', COLORS.get('red', '#E74C3C'))
        pos_color = self.calendar_colors.get('positive_pnl', COLORS.get('green', '#2ECC71'))
        
        # Neutral color depends on theme if not specified
        default_neutral = COLORS.get('neutral_dark', '#4A4A4A') if self.plot_theme == 'dark' else COLORS.get('neutral_light', '#F0F0F0')
        neutral_color = self.calendar_colors.get('neutral_color', default_neutral)

        return [
            [0.0, neg_color],    # Negative P&L
            [0.5, neutral_color], # Zero or No Data (midpoint)
            [1.0, pos_color]     # Positive P&L
        ]

    def render(self) -> None:
        """Renders the P&L calendar heatmap using Plotly."""
        st.subheader(f"P&L Calendar Heatmap for {self.year}")
        if self.calendar_data is None or self.calendar_data.empty:
            st.info(f"No P&L data available to display for the year {self.year}.")
            logger.info(f"Render: No calendar data for year {self.year}.")
            return

        # Determine the number of weeks for the heatmap grid.
        # Typically 53 weeks can cover a year. Using min/max of plot_week_idx ensures it fits the data.
        # Ensure plot_week_idx exists, otherwise fallback
        if 'plot_week_idx' not in self.calendar_data.columns:
            logger.error("Render: 'plot_week_idx' is missing from calendar_data. This should not happen.")
            st.error("An internal error occurred preparing calendar week indices.")
            # Fallback: use week_of_year and assume it's 1-based for now, adjust to 0-based
            self.calendar_data['plot_week_idx'] = self.calendar_data['week_of_year'] -1


        num_weeks = self.calendar_data['plot_week_idx'].max() + 1 if not self.calendar_data['plot_week_idx'].empty else 53
        num_weeks = max(num_weeks, 53) # Ensure at least 53 weeks for standard layout

        z_data = np.full((7, num_weeks), np.nan) # 7 days, num_weeks
        hover_text_data = [[None for _ in range(num_weeks)] for _ in range(7)]

        for _, row in self.calendar_data.iterrows():
            # Defensive checks for indices
            day_idx = int(row['day_of_week']) # Monday=0 to Sunday=6
            week_idx = int(row['plot_week_idx']) # Should be 0-indexed

            if not (0 <= day_idx < 7):
                logger.warning(f"Render: Invalid day_idx {day_idx} for date {row['date']}. Skipping.")
                continue
            if not (0 <= week_idx < num_weeks):
                logger.warning(f"Render: Invalid week_idx {week_idx} (num_weeks: {num_weeks}) for date {row['date']}. Skipping.")
                continue
            
            pnl_value = row['pnl']
            z_data[day_idx, week_idx] = pnl_value

            # Handle np.nan for P&L display in hover text
            if pd.isna(pnl_value) or np.isnan(pnl_value): # Check both pandas NA and numpy NaN
                pnl_display = "No Data"
            else:
                pnl_display = format_currency(pnl_value) # Use the imported/defined format_currency

            hover_text_data[day_idx][week_idx] = (
                f"<b>Date:</b> {row['date'].strftime('%Y-%m-%d')} ({row['date'].strftime('%a')})<br>"
                f"<b>P&L:</b> {pnl_display}<br>"
                f"Day of Month: {row['day_of_month']}<br>"
                f"ISO Week: {row['week_of_year']}" # Display original ISO week
            )
        
        # Determine colorscale and zmid
        colorscale = self._get_colorscale()
        
        # Calculate min/max for colorscale, ignoring NaNs
        # If all data is NaN, zmin/zmax will be NaN. Set a default range.
        valid_pnl_values = z_data[~np.isnan(z_data)]
        if valid_pnl_values.size == 0: # All P&L is NaN or no data points
            min_pnl, max_pnl = 0, 0 # Default range if no valid P&L
            abs_max = 0.01 # Avoid division by zero if min_pnl=max_pnl=0
            zmid_val = 0
        else:
            min_pnl, max_pnl = np.nanmin(z_data), np.nanmax(z_data)
            abs_max = max(abs(min_pnl), abs(max_pnl), 0.01) # Ensure abs_max is at least 0.01
            zmid_val = 0 # Center colorscale at 0 P&L

        fig = go.Figure(data=go.Heatmap(
            z=z_data,
            x=list(range(1, num_weeks + 1)), # Weeks 1 to num_weeks for x-axis labels
            y=['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'],
            colorscale=colorscale,
            zmin=-abs_max if min_pnl < 0 else 0, # Adjust zmin if all PnL is positive
            zmax=abs_max if max_pnl > 0 else 0, # Adjust zmax if all PnL is negative
            zmid=zmid_val, # Center the colorscale around zero
            xgap=2, ygap=2,
            showscale=True,
            colorbar=dict(
                title=dict(text='Daily P&L', side='right'),
                thickness=15,
                len=0.75,
                y=0.5,
                yanchor='middle'
            ),
            customdata=hover_text_data,
            hovertemplate="%{customdata}<extra></extra>" # Use customdata for full control over hover
        ))
        
        # Month labels logic
        month_positions = []
        last_month_seen = ""
        # Ensure calendar_data is not empty and plot_week_idx exists
        if not self.calendar_data.empty and 'plot_week_idx' in self.calendar_data.columns:
            # Group by plot_week_idx and get the first month_text for that week
            # Sort by date to ensure month transitions are correctly identified
            sorted_calendar_data = self.calendar_data.sort_values(by='date')
            
            for week_plot_idx in range(num_weeks): # Iterate through 0-indexed plot weeks
                # Find data for this specific plot week index
                week_data = sorted_calendar_data[sorted_calendar_data['plot_week_idx'] == week_plot_idx]
                if not week_data.empty:
                    # Get the month of the first day in this plot week
                    current_month_in_week = week_data['month_text'].iloc[0]
                    if current_month_in_week != last_month_seen:
                        # Store 1-indexed week for display on plot's x-axis
                        month_positions.append({'week_label': week_plot_idx + 1, 'month': current_month_in_week})
                        last_month_seen = current_month_in_week
        
        theme_layout = self._get_plotly_theme_layout()
        for pos in month_positions:
             fig.add_annotation(
                x=pos['week_label'], y=-0.8, # Adjusted y for better spacing below x-axis labels
                text=f"<b>{pos['month']}</b>", showarrow=False,
                font=dict(size=10, color=theme_layout['font_color']),
                xanchor='left' # Anchor to the left of the week tick
            )

        fig.update_layout(
            title_text=f'Daily P&L Heatmap - {self.year}',
            xaxis_title=None, # Remove "Week of Year" as months are shown
            yaxis_title='Day of Week',
            yaxis_autorange='reversed',
            height=380, # Slightly increased height for month labels
            margin=dict(t=60, l=60, b=100, r=40), # Adjusted margins
            **theme_layout
        )
        # Position x-axis ticks and labels at the top
        fig.update_xaxes(
            side="top",
            tickvals=[w for w in range(1, num_weeks + 1, 4)], # Tick every 4 weeks
            ticktext=[str(w) for w in range(1, num_weeks + 1, 4)], # Display week numbers
            showline=True, linewidth=1, linecolor=theme_layout['font_color'],
            showgrid=False
        )
        fig.update_yaxes(
            showline=True, linewidth=1, linecolor=theme_layout['font_color'],
            showgrid=False
        )
        
        st.plotly_chart(fig, use_container_width=True)
        logger.debug(f"PnL Calendar rendered for year {self.year}.")

# Example Usage (Optional - for testing or direct script execution)
if __name__ == '__main__':
    # Create dummy data for testing
    np.random.seed(42)
    num_days = 700
    start_test_date = pd.Timestamp('2022-01-01')
    dates = pd.date_range(start_test_date, periods=num_days, freq='D')
    
    # Introduce some NaNs and a wider range of P&L
    pnl_data = np.random.randn(num_days) * 1000
    nan_indices = np.random.choice(num_days, size=num_days//10, replace=False)
    pnl_data[nan_indices] = np.nan
    # Add some zero P&L days
    zero_indices = np.random.choice(np.setdiff1d(np.arange(num_days), nan_indices), size=num_days//20, replace=False)
    pnl_data[zero_indices] = 0


    test_df = pd.DataFrame({
        'trade_date': dates,
        'daily_profit_loss': pnl_data
    })

    st.set_page_config(layout="wide", page_title="P&L Calendar Test")
    
    st.title("P&L Calendar Component Test")

    # --- Test Section ---
    selected_year = st.sidebar.selectbox("Select Year", options=sorted(test_df['trade_date'].dt.year.unique(), reverse=True), index=0)
    selected_theme = st.sidebar.radio("Select Theme", options=['dark', 'light'], index=0)

    # Filter data for the selected year for the component
    df_for_year = test_df[test_df['trade_date'].dt.year == selected_year]

    st.header(f"Displaying Calendar for {selected_year} (Theme: {selected_theme})")

    # Custom colors example
    custom_color_options = {
        "Default": None,
        "Ocean": {"positive_pnl": "dodgerblue", "negative_pnl": "orangered", "neutral_color": "lightgrey"},
        "Forest": {"positive_pnl": "forestgreen", "negative_pnl": "saddlebrown", "neutral_color": "beige"}
    }
    selected_custom_colors_key = st.sidebar.selectbox("Custom P&L Colors", options=list(custom_color_options.keys()))
    custom_colors = custom_color_options[selected_custom_colors_key]

    # Custom colorscale example
    custom_colorscale_options = {
        "Default": None,
        "PurpleRed_Green": [[0.0, "purple"], [0.25, "red"], [0.5, "lightgrey"], [0.75, "lightgreen"], [1.0, "green"]],
        "Blue_Yellow_Red": [[0.0, "blue"], [0.5, "yellow"], [1.0, "red"]]
    }
    selected_colorscale_key = st.sidebar.selectbox("Custom Colorscale (Overrides P&L Colors)", options=list(custom_colorscale_options.keys()))
    custom_scale = custom_colorscale_options[selected_colorscale_key]


    if df_for_year.empty:
        st.warning(f"No data for the year {selected_year}.")
    else:
        try:
            # Instantiate the component
            calendar_view = PnLCalendarComponent(
                daily_pnl_df=df_for_year,
                year=selected_year, # Explicitly pass year
                date_col='trade_date',
                pnl_col='daily_profit_loss',
                plot_theme=selected_theme,
                calendar_colors=custom_colors,
                custom_colorscale=custom_scale
            )
            # Render the component
            calendar_view.render()
        except ValueError as ve:
            st.error(f"Configuration Error: {ve}")
            logger.exception("ValueError during PnLCalendarComponent instantiation or rendering.")
        except Exception as e:
            st.error(f"An unexpected error occurred: {e}")
            logger.exception("Unexpected error during PnLCalendarComponent instantiation or rendering.")

    st.sidebar.markdown("---")
    st.sidebar.markdown("### Test Data Sample:")
    st.sidebar.dataframe(df_for_year.head())
