from mftool import Mftool
import numpy as np
import streamlit as st
import pandas as pd
import plotly.express as px

#First initie with
mf = Mftool()

st.title('🎈Welcome to Suriya Mutual Fund Dashboard🎈')

option = st.sidebar.selectbox(
    "Choose an action",
    ["View Available Schemes", "Scheme Details", "Historical NAV", "Compare NAVs",
     "Average AUM", "Performance Heatmap", "Risk and Volatility Analysis"]
)

# Fetch all scheme codes for dropdowns
scheme_names = {v: K for K, v in mf.get_scheme_codes().items()}

if option == 'View Available Schemes' :
    st.header('View Available Schemes')
    amc = st.sidebar.text_input("Enter AMC Name", "ICICI")
    schemes = mf.get_available_schemes(amc)
    st.write(pd.DataFrame(schemes.items(), columns=["Scheme Code", "Scheme Name"]) if schemes else "No schemes found")

if option == 'Scheme Details' :
    st.header("Scheme Details")
    scheme_code = scheme_names[st.sidebar.selectbox("Select a Scheme", scheme_names.keys())]
    details = pd.DataFrame (mf.get_scheme_details(scheme_code)).iloc[0]
    st.write(details)

if option == 'Historical NAV' :
    st.header("Historical NAV")
    scheme_code = scheme_names[st.sidebar.selectbox("Select a Scheme", scheme_names.keys())]
    NAV_DATA = mf.get_scheme_historical_nav(scheme_code, as_Dataframe=True)
    st.write(NAV_DATA)

if option == 'Compare NAVs' :
    st.header("Compare NAVs")
    selected_schemes = st.sidebar.multiselect("Select schemes to compare", options=list(scheme_names.keys()))
    if selected_schemes:
        comparison_df = pd.DataFrame()  # Create an empty DataFrame to store NAV data for comparison

        for scheme in selected_schemes:
            code = scheme_names[scheme]  # Get the scheme code
            data = mf.get_scheme_historical_nav(code, as_Dataframe=True)  # Fetch NAV data
            data = data.reset_index().rename(columns={"index": "date"})  # Rename index to 'date'
            data["date"] = pd.to_datetime(data["date"], dayfirst=True).sort_values()  # Format dates
            data["nav"] = data["nav"].replace(0, None).interpolate()  # Replace 0s and interpolate missing values
            comparison_df[scheme] = data.set_index("date")["nav"]  # Add NAV data to comparison DataFrame

         # Plot NAV comparison
        fig = px.line(comparison_df, title="Comparison of NAVs")
        st.plotly_chart(fig)  # Display the chart using Streamlit
    else:
        st.info("Select at least one scheme.")  # Prompt user to select schemes

if option == 'Average AUM' :
    st.header("Average AUM")
    aum_data = mf.get_average_aum('July - september 2024', False)
    if aum_data:
        aum_df = pd.DataFrame(aum_data)
        aum_df["Total AUM"] = aum_df[["AAUM Overseas", "AAUM Domestic"]].astype(float).sum(axis=1)
        st.write(aum_df[["Fund Name", "Total AUM"]])
    else:
        st.write("No AUM is available.")

elif option == "Performance Heatmap":
    st.header("Performance Heatmap")
    # User selects a scheme from the sidebar
    scheme_code = scheme_names[st.sidebar.selectbox("Select a Scheme", scheme_names.keys())]
    # Fetch historical NAV data for the selected scheme
    nav_data = mf.get_scheme_historical_nav(scheme_code, as_Dataframe=True)

    if not nav_data.empty:  # Check if data is available
        # Preprocess the data
        nav_data = nav_data.reset_index().rename(columns={"index": "date"})  # Rename 'index' to 'date'
        nav_data["month"] = pd.DatetimeIndex(nav_data["date"]).month  # Extract month from 'date'
        nav_data["nav"] = nav_data["nav"].astype(float)  # Convert NAV to float for calculations

        # Group data by month and calculate the average daily change
        heatmap_data = nav_data.groupby("month")["dayChange"].mean().reset_index()
        heatmap_data["month"] = heatmap_data["month"].astype(str)  # Convert month to string

        # Create a density heatmap using Plotly
        fig = px.density_heatmap(heatmap_data,x="month",y="dayChange",title="NAV Performance Heatmap",color_continuous_scale="Viridis")# You can customize the color scale
        st.plotly_chart(fig)  # Display the heatmap in Streamlit
    else:
        st.write("No historical NAV data available.")  # Display a message if no data is available

if option == "Risk and Volatility Analysis":
    st.header("Risk and Volatility Analysis")
    # User selects a scheme from the sidebar
    scheme_name = st.sidebar.selectbox("Select a Scheme", scheme_names.keys())
    scheme_code = scheme_names[scheme_name]  # Retrieve the scheme code based on the selected name

    # Fetch historical NAV data for the selected scheme
    nav_data = mf.get_scheme_historical_nav(scheme_code, as_Dataframe=True)
    
    if not nav_data.empty:
        # Reset and clean data
        nav_data = nav_data.reset_index().rename(columns={"index": "date"})  # Rename 'index' to 'date'
        nav_data["date"] = pd.to_datetime(nav_data["date"], dayfirst=True)  # Convert 'date' to datetime

        # Convert NAV column to numeric
        nav_data["nav"] = pd.to_numeric(nav_data["nav"], errors="coerce")  # Handle invalid NAV values
        nav_data = nav_data.dropna(subset=["nav"])  # Remove rows with missing or invalid NAVs

        # Calculate daily returns
        nav_data["returns"] = nav_data["nav"] / nav_data["nav"].shift(-1) - 1  # Daily percentage change
        nav_data = nav_data.dropna(subset=["returns"])  # Remove rows with NaN returns
       
        # Calculate Metrics
        annualized_volatility = nav_data["returns"].std() * np.sqrt(252)  # Assuming 252 trading days
        annualized_return = (1 + nav_data["returns"].mean())**252 - 1
        risk_free_rate = 0.06  # Example: 6% annual risk-free rate
        sharpe_ratio = (annualized_return - risk_free_rate) / annualized_volatility

        # Display Metrics in Streamlit
        st.write(f"### Metrics for {scheme_name}")
        st.metric("Annualized Volatility", f"{annualized_volatility:.2%}")
        st.metric("Annualized Return", f"{annualized_return:.2%}")
        st.metric("Sharpe Ratio", f"{sharpe_ratio:.2f}")

        # Plot Risk-Return Distribution
        fig = px.scatter(
            nav_data, x="date", y="returns", 
            title=f"Risk-Return Scatter for {scheme_name}",
            labels={"returns": "Daily Returns", "date": "Date"},
        )
        st.plotly_chart(fig)  # Display the chart in Streamlit

        # Add Monte Carlo Simulation For Future NAV Projection
        st.write('### Monte Carlo Simulation For Future NAV Projection')
        
        # Simulation parameters
        num_simulations = st.slider("Number of Simulation", min_value=1, max_value=5000, value=1000)
        num_days = st.slider("Projection Period (Days)", min_value=30, max_value=365, value=252)
        
        # Monte carlo simulation
        last_nav = nav_data["nav"].iloc[-1]
        daily_volatility = nav_data["returns"].std()
        daily_mean_return= nav_data["returns"].mean()
        
        simulation_results = []
        for _ in range(num_simulations):
            prices = [last_nav]
            for _ in range(num_days):
                simulated_return = np.random.normal(daily_mean_return, daily_volatility)
                prices.append(prices[-1] * (1 + simulated_return))
            simulation_results.append(prices)

        # Create DataFrame for visualization
        simulation_df = pd.DataFrame(simulation_results).T
        simulation_df.index.name = "Day"
        simulation_df.columns = [f"Simulation {i+1}" for i in range(num_simulations)]
       
        # Plot simulations
        fig_simulation = px.line(
            simulation_df,
            title=f"Monte Carlo Simulation for {scheme_name} NAV Projection",
            labels={"value": "Projected NAV", "index": "Day"},
            template="plotly_dark"
        )
        st.plotly_chart(fig_simulation)

        # Show Summary Statistics
        final_prices = simulation_df.iloc[-1]
        st.write(f"### Simulation Summary for {scheme_name}")
        st.metric("Expected Final NAV", f"{final_prices.mean():.2f}")
        st.metric("Minimum Final NAV", f"{final_prices.min():.2f}")
        st.metric("Maximum Final NAV", f"{final_prices.max():.2f}")

    else:
        st.write("No historical NAV data available.")