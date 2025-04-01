import streamlit as st
import pandas as pd
from openai import OpenAI
import re
from prompts import (
    CHURN_FACTORS_PROMPT,
    CHURN_MODEL_PROMPT,
    RISK_SEGMENTS_ACTIONS_PROMPT,
    AUTOMATION_IDEAS_PROMPT,
)

st.set_page_config(page_title="Churn Prediction Prototype", layout="wide")
st.title("🔍 AI-Powered Churn Prediction Prototype")

client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"], timeout=60.0)

if "step" not in st.session_state:
    st.session_state.step = 1

step_names = [
    "Upload Data",
    "Unify Data",
    "Identify Churn Factors",
    "Build Prediction Model",
    "Retention Actions",
    "Automation Ideas",
]

progress_percent = (st.session_state.step / len(step_names))
st.progress(progress_percent)

step_indicator = " > ".join(
    [f"**{name}**" if i+1 == st.session_state.step else name for i, name in enumerate(step_names)]
)
st.markdown(f"### 🧭 Workflow Progress: {step_indicator}")

def ai_call(prompt, system_message="You are an expert assistant."):
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": system_message},
            {"role": "user", "content": prompt},
        ],
    )
    return response.choices[0].message.content

# STEP 1: Upload Datasets
if st.session_state.step == 1:
    st.header("📂 Upload Your Datasets")
    st.markdown("""
    Welcome to my entry for the Prompt Masters Challenge — an interactive "proof of concept" prototype demonstrating how AI could help proactively identify members at risk of churning. 

    Please upload the provided synthetic datasets (formatted to mirror real-world structures from Stripe, Braze, and Zendesk). AI-powered insights await you in the following steps!

    """)
    braze_file = st.file_uploader("📩 Upload Braze CSV", type="csv")
    stripe_file = st.file_uploader("💳 Upload Stripe CSV", type="csv")
    zendesk_file = st.file_uploader("🎟️ Upload Zendesk CSV", type="csv")

    if braze_file and stripe_file and zendesk_file:
        st.session_state.braze_df = pd.read_csv(braze_file)
        st.session_state.stripe_df = pd.read_csv(stripe_file)
        st.session_state.zendesk_df = pd.read_csv(zendesk_file)
        st.success("✅ Files uploaded successfully!")
        if st.button("Proceed to Data Unification →"):
            st.session_state.step = 2
            st.rerun()

# STEP 2: Unify Data using Pandas
elif st.session_state.step == 2:
    st.header("🔗 Dataset Unification using Pandas")
    st.markdown("""
    In this step, the uploaded datasets are unified into a single dataset using pandas.  

    Initially, I planned to use GPT-4o for this step. However, careful testing made me realise something important: effective use of AI also means knowing when **not** to use it. For large-scale, precise data integration tasks, traditional tools like pandas offered more reliability, scalability, and efficiency.  
    Using pandas in this step ensured a robust data foundation, allowing AI to shine in subsequent steps — analyzing patterns and providing actionable insights.
    """)

    if "unified_df" not in st.session_state:
        if st.button("Unify Datasets Now"):
            with st.spinner("🛠️ Unifying datasets using pandas..."):
                zendesk_df = st.session_state.zendesk_df.rename(columns={"Requester email": "email"})

                unified_df = pd.merge(
                    st.session_state.stripe_df[[
                        "customer_id", "email", "subscription_status",
                        "subscription_type", "total_payments", "payment_failures"
                    ]],
                    st.session_state.braze_df[[
                        "email", "percent_emails_clicked", "days_since_last_email_click"
                    ]],
                    on="email", how="outer"
                )

                unified_df = pd.merge(
                    unified_df,
                    zendesk_df[["email", "Number of tickets", "Tags"]],
                    on="email", how="outer"
                )

                unified_df.rename(columns={
                    "Number of tickets": "number_of_tickets",
                    "Tags": "recent_ticket_issue"
                }, inplace=True)

                unified_df.fillna({
                    "total_payments": 0,
                    "payment_failures": 0,
                    "percent_emails_clicked": 0,
                    "days_since_last_email_click": 999,
                    "number_of_tickets": 0,
                    "recent_ticket_issue": "unknown",
                    "subscription_status": "unknown",
                    "subscription_type": "unknown"
                }, inplace=True)

                unified_df["churn_status"] = unified_df["subscription_status"].apply(
                    lambda x: "Churned" if x in ["canceled", "past_due"] else "Active"
                )

                st.session_state.unified_df = unified_df
                st.success("✅ Datasets unified successfully!")

    if "unified_df" in st.session_state:
        st.dataframe(st.session_state.unified_df, use_container_width=True)
        csv = st.session_state.unified_df.to_csv(index=False)
        st.download_button("📥 Download Full Unified Dataset", data=csv,
                           file_name="unified_dataset.csv", mime="text/csv")


        col1, col2 = st.columns([1, 1])
        with col1:
            if st.button("⬅️ Back to Data Upload"):
                st.session_state.step = 1
                st.rerun()

        with col2:
            if st.button("➡️ Proceed to Churn Factor Analysis →"):
                st.session_state.step = 3
                st.rerun()



# STEP 3: Identify Churn Factors
elif st.session_state.step == 3:
    st.header("📊 Churn Factor Identification")
    st.markdown("""
    In this step, we utilise AI to analyse the unified dataset to clearly identify and rank key factors that predict churn, providing explicit thresholds that indicate increased risk.
    """)
    with st.expander("🔍 View the actual AI prompt powering this step"):
        st.code(CHURN_FACTORS_PROMPT, language='markdown')

    if "churn_factors_analysis" not in st.session_state:
        if st.button("Identify Churn Factors Now"):
            with st.spinner("🤖 Analyzing churn factors..."):
                prompt = f"{CHURN_FACTORS_PROMPT}\n\n### Unified Dataset (CSV):\n{st.session_state.unified_df.to_csv(index=False)}"
                st.session_state.churn_factors_analysis = ai_call(prompt, "You are a world-class churn analyst with deep expertise in behavioral analytics and customer psychology. Your job is to uncover the hidden patterns that drive member churn, explain your reasoning clearly, and suggest practical insights that can guide real-world retention strategies. Always think step-by-step, prioritize human-understandable insights, and highlight anything unexpected that may be worth further exploration.")
                st.success("✅ Churn factors identified successfully!")

    if "churn_factors_analysis" in st.session_state:
        st.markdown(st.session_state.churn_factors_analysis)

        col1, col2 = st.columns([1, 1])
        with col1:
            if st.button("⬅️ Back to Data Unification"):
                st.session_state.step = 2
                st.rerun()

        with col2:
            if st.button("➡️ Proceed to Prediction Model →"):
                st.session_state.step = 4
                st.rerun()

# STEP 4: Build Prediction Model
elif st.session_state.step == 4:
    st.header("🧮 Build Churn Prediction Model")
    st.markdown("""
    In this step, GPT-4o generates Python code to create an interpretable churn scoring model tailored specifically to your dataset.

    **Workflow Explained Clearly:**
    1. Click **"🛠️ Generate Scoring Logic"**: GPT-4o produces Python code designed specifically for the provided data.
    2. Click **"▶️ Apply Generated Scoring Logic to Data"**: Executes the displayed Python code to calculate churn risk scores and assign clear risk segments.

    **Note:** The Python code displayed is only provided for transparency, allowing verification before application.
    """)

    with st.expander("🔍 View the actual AI prompt powering this step"):
        st.code(CHURN_MODEL_PROMPT, language='markdown')

    if "ai_generated_scoring_code" not in st.session_state:
        if st.button("🛠️ Generate Scoring Logic"):
            with st.spinner("🤖 Generating scoring logic using GPT-4o..."):
                prompt = f"{CHURN_MODEL_PROMPT}\n\n### Unified Dataset (CSV):\n{st.session_state.unified_df.to_csv(index=False)}"

                system_msg = (
                    "You are a senior data scientist with deep expertise in customer churn analytics and behavioral modeling. "
                    "Your job is to design a practical, interpretable, and human-readable scoring model that classifies customers by churn risk. "
                    "Provide production-ready Python code with inline comments clearly explaining each step. Explicitly state thresholds chosen based on realistic customer behaviors derived from provided data. "
                    "Provide ONLY executable Python code, without markdown fences or explanations."
                )

                ai_code = ai_call(prompt, system_msg)
                st.session_state.ai_generated_scoring_code = ai_code

    if "ai_generated_scoring_code" in st.session_state:
        st.subheader("🔧 AI-Generated Churn Scoring Code")
        st.code(st.session_state.ai_generated_scoring_code, language='python')

        if st.button("▶️ Apply Generated Scoring Logic to Data"):
            with st.spinner("🔄 Applying scoring logic to dataset..."):
                df = st.session_state.unified_df.copy()

                numeric_cols = ['payment_failures', 'percent_emails_clicked',
                                'days_since_last_email_click', 'number_of_tickets',
                                'total_payments']
                for col in numeric_cols:
                    df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)

                # Extract clean Python code (remove markdown if present)
                import re

                code_match = re.search(r'```python\n(.*?)\n```', st.session_state.ai_generated_scoring_code, re.DOTALL)
                executable_code = code_match.group(1) if code_match else st.session_state.ai_generated_scoring_code

                try:
                    exec(executable_code, {'df': df, 'pd': pd})
                except Exception as e:
                    st.error(f"❗ Error applying scoring logic: {e}")
                    with st.expander("📄 Debugging - View AI-generated code"):
                        st.code(executable_code, language='python')
                    st.warning("🔁 Please regenerate scoring logic or manually correct the displayed Python code.")
                    st.stop()

                required_cols = ['churn_risk_score', 'churn_risk_segment']
                for col in required_cols:
                    if col not in df.columns:
                        st.error(f"❗ Missing required column: {col}")
                        st.stop()

                st.session_state.scored_df = df[['customer_id', 'email', 'churn_status',
                                                 'churn_risk_score', 'churn_risk_segment']]
                st.success("✅ Churn scoring logic applied successfully!")

    if "scored_df" in st.session_state:
        active_customers_df = st.session_state.scored_df[
            st.session_state.scored_df['churn_status'] == 'Active'
        ].copy().reset_index(drop=True)

        st.subheader("🔍 Customer Churn Scores (Active Customers Only)")
        st.dataframe(active_customers_df, use_container_width=True)

        csv = active_customers_df.to_csv(index=False)
        st.download_button(
            "📥 Download Active Customers Churn Scores",
            data=csv,
            file_name="active_customers_churn_scores.csv",
            mime="text/csv"
        )



        col1, col2 = st.columns([1, 1])
        with col1:
            if st.button("⬅️ Back to Data Unification"):
                st.session_state.step = 3
                st.rerun()

        with col2:
            if st.button("➡️ Proceed to Retention Actions →"):
                st.session_state.step = 5
                st.rerun()

# STEP 5: Retention Actions
elif st.session_state.step == 5:
    st.header("📌 Tailored Retention Actions by Risk Segment")
    st.markdown("""
    In this step, GPT-4o takes the churn risk segments you've identified and generates personalized, psychology-driven retention strategies tailored specifically for each risk category (High, Moderate, and Low Risk).

    ### 🔍 **Here's exactly what's happening:**
    1. **Risk Segment Classification:** GPT-4o first confirms each member’s churn risk segment (High, Moderate, or Low).
    2. **Personalized Retention Strategies:** AI generates tailored retention actions that leverage behavioral psychology, clearly addressing why each segment might churn and what can persuade them to remain engaged.
    3. **Downloadable Retention Plan:** After generation, you'll get a downloadable, actionable CSV that clearly matches every member to their recommended retention action.

    Click **"🚀 Generate Tailored Retention Strategies"** to proceed. You'll receive actionable strategies ready for immediate use.
    """)

    with st.expander("🔍 View the actual AI prompt powering this step"):
        st.code(RISK_SEGMENTS_ACTIONS_PROMPT, language='markdown')

    if "retention_strategies" not in st.session_state:
        if st.button("🚀 Generate Tailored Retention Strategies"):
            with st.spinner("✨ Generating tailored retention strategies..."):
                active_customers_df = st.session_state.scored_df[
                    st.session_state.scored_df['churn_status'] == 'Active'
                ]

                prompt = (
                    f"{RISK_SEGMENTS_ACTIONS_PROMPT}\n\n"
                    f"### Scored Dataset (Active Customers Only) (CSV):\n"
                    f"{active_customers_df[['email', 'churn_risk_segment']].to_csv(index=False)}"
                )

                ai_response = ai_call(
                    prompt,
                    "You are a senior customer retention strategist with expertise in behavioral psychology, customer engagement, and churn prevention. Provide detailed, psychologically informed retention actions tailored precisely to each customer's churn risk segment. Prioritize actionable, personalized strategies clearly differentiated by risk level."
                )

                try:
                    from io import StringIO
                    retention_df = pd.read_csv(StringIO(ai_response))
                    st.session_state.retention_strategies = retention_df
                    st.success("✅ Retention strategies generated successfully!")
                except Exception as e:
                    st.error(
                        "❗ The retention strategies generated by AI could not be parsed. "
                        "Please regenerate or review the raw output below for troubleshooting."
                    )
                    with st.expander("📄 View Raw AI Response for Debugging"):
                        st.text(ai_response)
                    st.stop()

    if "retention_strategies" in st.session_state:
        st.subheader("📋 Retention Strategies Table")
        st.dataframe(st.session_state.retention_strategies, use_container_width=True)

        csv = st.session_state.retention_strategies.to_csv(index=False)
        st.download_button(
            "📥 Download Retention Strategies CSV",
            data=csv,
            file_name="retention_strategies.csv",
            mime="text/csv"
        )


        col1, col2 = st.columns([1, 1])
        with col1:
            if st.button("⬅️ Back to Prediction Model"):
                st.session_state.step = 4
                st.rerun()

        with col2:
            if st.button("➡️ Proceed to Automation Ideas →"):
                st.session_state.step = 6
                st.rerun()



# STEP 6: Automation Ideas
elif st.session_state.step == 6:
    st.header("⚙️ Automation Recommendations")
    st.markdown("""
    In this final step, GPT-4o generates practical, actionable recommendations to automate and scale your churn prediction and retention workflow from a prototype into a robust, fully operational system.

    ### 🚀 **What you'll receive:**
    - **Automated Data Ingestion**: Clear recommendations for seamlessly integrating data from Stripe, Braze, and Zendesk using reliable API solutions.
    - **Scheduled Predictions & Updates**: Guidance on automating churn predictions, regularly updating models, and ensuring ongoing accuracy.
    - **Automated Retention Actions**: Specific strategies to automatically deploy personalized retention campaigns based on churn segments.
    - **Reporting & Alerts**: Suggestions for creating engaging dashboards, monitoring performance, and alerting your team to critical issues in real-time.
    - **Scalable Infrastructure**: Clear technology recommendations for scaling this workflow in a robust, cost-effective manner.

    ### 📌 **Instructions:**
    Click the **"Generate Automation Recommendations"** button to see GPT-4o's suggestions. 

    """)

    with st.expander("🔍 View the actual AI prompt powering this step"):
        st.code(AUTOMATION_IDEAS_PROMPT, language='markdown')

    if "automation_plan" not in st.session_state:
        if st.button("Generate Automation Recommendations"):
            with st.spinner("🤖 Generating automation solutions..."):
                context = (
                    f"Churn Factors Analysis:\n{st.session_state.churn_factors_analysis}\n\n"
                    f"Retention Strategies:\n{st.session_state.retention_strategies}"
                )
                prompt = f"{AUTOMATION_IDEAS_PROMPT}\n\n### Context:\n{context}"
                st.session_state.automation_plan = ai_call(prompt, "You are a senior automation architect with deep expertise in designing scalable and practical workflow automation solutions. Your task is to carefully recommend detailed, actionable strategies that clearly address technical implementation steps, scalability, reliability, and seamless integration within existing infrastructures. Provide structured, practical, and clear recommendations suitable for immediate consideration and deployment.")
                st.success("✅ Automation strategies generated successfully!")

    if "automation_plan" in st.session_state:
        st.markdown(st.session_state.automation_plan)
        final_report = f"""
Unified Dataset (Sample):\n{st.session_state.unified_df.head().to_csv(index=False)}\n\n
Churn Factors Analysis:\n{st.session_state.churn_factors_analysis}\n\n
Prediction Model Results (Sample):\n{st.session_state.scored_df.head().to_csv(index=False)}\n\n
Retention Strategies:\n{st.session_state.retention_strategies}\n\n
Automation Plan:\n{st.session_state.automation_plan}
"""
        st.download_button("📥 Download Complete Report", final_report, "churn_prediction_report.txt")
