import streamlit as st
import plotly.express as px
import pandas as pd
from backend.database import get_joined_data, get_experiments
from backend.forecasting import train_and_evaluate

def render():
    st.title("🔬 Research Laboratory")
    
    if st.button("🚀 Train Random Forest Model"):
        with st.spinner("Training models..."):
            df = get_joined_data()
            results = train_and_evaluate(df)
            
            if results:
                st.success("Training Complete!")
                col1, col2 = st.columns(2)
                col1.metric("Baseline MAE", f"{results['baseline_mae']:.2f}")
                col2.metric("RF MAE", f"{results['rf_mae']:.2f}", delta=f"{results['baseline_mae'] - results['rf_mae']:.2f}")
                
                # Feature Importance
                imp_df = pd.DataFrame({'Feature': results['features'], 'Importance': results['importances']})
                fig = px.bar(imp_df, x='Importance', y='Feature', orientation='h', template="plotly_dark")
                st.plotly_chart(fig)
            else:
                st.error("Not enough data to train.")
                
    st.divider()
    st.subheader("Experiment History")
    exp = get_experiments()
    if not exp.empty:
        st.dataframe(exp, use_container_width=True)
