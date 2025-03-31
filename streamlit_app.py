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
    "Unify Data (Pandas)",
    "Identify Churn Factors",
    "Build Prediction Model",
    "Retention Actions",
    "Automation Ideas",
]

st.markdown(f"### 🧭 Step {st.session_state.step}: **{step_names[st.session_state.step - 1]}**")

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
    Welcome to my entry for the Prompt Masters Challenge—an interactive prototype demonstrating how AI can proactively identify at-risk members by intelligently unifying data from multiple sources (Stripe, Braze, and Zendesk).

    This tool demonstrates how we can use AI to help proactively identify members at risk of churning by intelligently unifying data from disparate sources - in this case: subscription, engagement, and support data (from Stripe, Braze, and Zendesk). Where AI has been employed, you’ll see the actual AI prompts powering the analysis and have the opportunity to execute AI-driven tasks to produce actionable insights for reducing churn.
    
    First, you'll need to upload the dummy data sets you have been provided. Due to data privacy considerations, I obviously did not want to use actual data extracts from our Braze, Stripe and Zendesk instances. For that reason, I have employed synthetic datasets for demonstration purposes. However, it should be noted that these datasets have specifically been formatted to accurately reflect our real-world data structures and typical customer behaviors recorded in those systems.
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
    In this step, the datasets you've uploaded are unified into a single dataset using pandasto provide a robust and scalable approach to data preparation for AI-driven churn analysis. In my initial planning stages, I anticipated using AI to unify our datasets. However, through careful experimentation I realised an important point: effective use of AI means knowing when **not** to use it, too. Given the complexity, scale, and precision required for data unification, I realised that for this demo, traditional tools like pandas offered superior reliability and efficiency.
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
        if st.button("Proceed to Churn Factor Analysis →"):
            st.session_state.step = 3
            st.rerun()

# STEP 3: Identify Churn Factors
elif st.session_state.step == 3:
    st.header("📊 Churn Factor Identification")
    st.markdown("""
    In this step, we utlise AI to analyse the unified dataset to clearly identify and rank key factors that predict churn, providing explicit thresholds that indicate increased risk.
    """)
    with st.expander("🔍 View the actual AI prompt powering this step"):
        st.code(CHURN_FACTORS_PROMPT, language='markdown')

    if "churn_factors_analysis" not in st.session_state:
        if st.button("Identify Churn Factors Now"):
            with st.spinner("🤖 Analyzing churn factors..."):
                prompt = f"{CHURN_FACTORS_PROMPT}\n\n### Unified Dataset (CSV):\n{st.session_state.unified_df.to_csv(index=False)}"
                st.session_state.churn_factors_analysis = ai_call(prompt, "You are an expert churn analyst.")
                st.success("✅ Churn factors identified successfully!")

    if "churn_factors_analysis" in st.session_state:
        st.markdown(st.session_state.churn_factors_analysis)
        if st.button("Proceed to Prediction Model →"):
            st.session_state.step = 4
            st.rerun()


# STEP 4: Build Prediction Model
elif st.session_state.step == 4:
    st.header("🧮 Build Churn Prediction Model")
    st.markdown("""
    Here, AI creates a practical, easy-to-understand scoring model based on previously identified churn factors. This model calculates each member’s risk of churn.
    """)
    with st.expander("🔍 View the actual AI prompt powering this step"):
        st.code(CHURN_MODEL_PROMPT, language='markdown')

    if "ai_generated_scoring_code" not in st.session_state:
        if st.button("🛠️ Generate and Apply AI-Based Scoring Logic"):
            with st.spinner("🤖 Generating and applying scoring logic from AI..."):
                churn_analysis = st.session_state.churn_factors_analysis

                try:
                    payment_failures_thresh_match = re.search(r'Payment Failures.*?≥\s*(\d+)', churn_analysis, re.I)
                    email_click_thresh_match = re.search(r'Days Since Last Email Click.*?≥\s*(\d+)', churn_analysis, re.I)
                    total_payments_thresh_match = re.search(r'Total Payments.*?≤\s*(\d+)', churn_analysis, re.I)
                    percent_emails_clicked_thresh_match = re.search(r'Percent Emails Clicked.*?<\s*([\d.]+)', churn_analysis, re.I)
                    tickets_thresh_match = re.search(r'Number of Tickets.*?≥\s*(\d+)', churn_analysis, re.I)

                    payment_failures_thresh = int(payment_failures_thresh_match.group(1)) if payment_failures_thresh_match else 2
                    email_click_thresh = int(email_click_thresh_match.group(1)) if email_click_thresh_match else 90
                    total_payments_thresh = int(total_payments_thresh_match.group(1)) if total_payments_thresh_match else 1
                    percent_emails_clicked_thresh = float(percent_emails_clicked_thresh_match.group(1)) if percent_emails_clicked_thresh_match else 0.2
                    tickets_thresh = int(tickets_thresh_match.group(1)) if tickets_thresh_match else 3

                except AttributeError as e:
                    st.error(f"Regex extraction failed. Please review churn analysis text: {e}")
                    st.stop()

                scoring_code = f'''
import pandas as pd

def calculate_churn_risk(row):
    risk_score = 0

    if row['payment_failures'] >= {payment_failures_thresh}:
        risk_score += 0.3
    if row['days_since_last_email_click'] >= {email_click_thresh}:
        risk_score += 0.25
    if row['total_payments'] <= {total_payments_thresh}:
        risk_score += 0.15
    if row['percent_emails_clicked'] < {percent_emails_clicked_thresh}:
        risk_score += 0.2
    if row['number_of_tickets'] >= {tickets_thresh}:
        risk_score += 0.1

    risk_score = min(risk_score, 1)

    if risk_score >= 0.75:
        risk_segment = 'High Risk'
    elif risk_score >= 0.4:
        risk_segment = 'Moderate Risk'
    else:
        risk_segment = 'Low Risk'

    return pd.Series([risk_score, risk_segment])

df[['churn_risk_score', 'churn_risk_segment']] = df.apply(calculate_churn_risk, axis=1)
'''
                st.session_state.ai_generated_scoring_code = scoring_code
                st.write("🔧 Dynamically Generated Scoring Logic:")
                st.code(scoring_code, language='python')

                df = st.session_state.unified_df.copy()

                numeric_cols = ['payment_failures', 'percent_emails_clicked', 'days_since_last_email_click', 'number_of_tickets', 'total_payments']
                for col in numeric_cols:
                    df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)

                original_churn_status = df['churn_status'].copy()

                try:
                    exec(scoring_code, {'df': df, 'pd': pd})

                    required_cols = ['churn_risk_score', 'churn_risk_segment']
                    for col in required_cols:
                        if col not in df.columns:
                            st.error(f"AI logic failed to produce required column: {col}")
                            st.stop()

                    df['churn_status'] = original_churn_status
                    st.session_state.scored_df = df[['customer_id', 'email', 'churn_status', 'churn_risk_score', 'churn_risk_segment']]
                    st.success("✅ Churn scoring logic applied successfully!")

                    active_df = df[df['churn_status'] == 'Active']
                    st.subheader("🚨 Debugging: Score Distribution Check (Active Only)")
                    st.write(active_df[['churn_risk_score']].describe())

                    # FIXED: Robust segment counting explicitly avoiding column name clashes
                    segment_counts = active_df['churn_risk_segment'].value_counts().reset_index()
                    segment_counts.columns = ['churn_risk_segment', 'segment_count']
                    st.write(segment_counts)

                except Exception as e:
                    st.error(f"Error executing AI-generated scoring logic: {e}")
                    st.stop()

    if "scored_df" in st.session_state:
        active_customers_df = st.session_state.scored_df[st.session_state.scored_df['churn_status'] == 'Active'].copy()
        active_customers_df.reset_index(drop=True, inplace=True)

        st.subheader("🔍 Customer Churn Scores (Active Customers Only)")
        st.dataframe(active_customers_df, use_container_width=True)

        csv = active_customers_df.to_csv(index=False)
        st.download_button(
            "📥 Download Active Customers Churn Scores",
            data=csv,
            file_name="active_customers_churn_scores.csv",
            mime="text/csv"
        )

        if st.button("Proceed to Retention Actions →"):
            st.session_state.step = 5
            st.rerun()




# STEP 5: Retention Actions
elif st.session_state.step == 5:
    st.header("📌 Recommended Retention Actions")
    st.markdown("""
    In this stage, AI categorizes members into clear risk segments and recommends tailored, actionable retention strategies for each segment.
    """)
    with st.expander("🔍 View the actual AI prompt powering this step"):
        st.code(RISK_SEGMENTS_ACTIONS_PROMPT, language='markdown')

    if "retention_strategies" not in st.session_state:
        if st.button("Generate Retention Strategies"):
            with st.spinner("🤖 Crafting strategies..."):
                # Explicitly ensure using ONLY active customers
                active_customers_df = st.session_state.scored_df[
                    st.session_state.scored_df['churn_status'] == 'Active'
                ]

                prompt = (
                    f"{RISK_SEGMENTS_ACTIONS_PROMPT}\n\n"
                    f"### Scored Dataset (Active Customers Only) (CSV):\n"
                    f"{active_customers_df[['email', 'churn_risk_segment']].to_csv(index=False)}"
                )

                ai_response = ai_call(prompt, "You are a retention specialist.")

                try:
                    from io import StringIO
                    retention_df = pd.read_csv(StringIO(ai_response))
                    st.session_state.retention_strategies = retention_df
                    st.success("✅ Retention strategies generated successfully!")
                except Exception as e:
                    st.error(f"Error parsing AI response: {e}")
                    st.stop()

    if "retention_strategies" in st.session_state:
        st.subheader("Retention Strategies Table")
        st.dataframe(st.session_state.retention_strategies, use_container_width=True)

        # Provide CSV download
        csv = st.session_state.retention_strategies.to_csv(index=False)
        st.download_button(
            "📥 Download Retention Strategies CSV",
            data=csv,
            file_name="retention_strategies.csv",
            mime="text/csv"
        )

        if st.button("Proceed to Automation Ideas →"):
            st.session_state.step = 6
            st.rerun()



# STEP 6: Automation Ideas
elif st.session_state.step == 6:
    st.header("⚙️ Automation Recommendations")
    st.markdown("""
    Finally, AI proposes practical automation solutions to scale this churn prediction and retention workflow, outlining how future enhancements could further improve predictive accuracy and efficiency.
    """)
    with st.expander("🔍 View the actual AI prompt powering this step"):
        st.code(AUTOMATION_IDEAS_PROMPT, language='markdown')

    if "automation_plan" not in st.session_state:
        if st.button("Suggest Automation Solutions"):
            with st.spinner("🤖 Generating automation solutions..."):
                context = (
                    f"Churn Factors Analysis:\n{st.session_state.churn_factors_analysis}\n\n"
                    f"Retention Strategies:\n{st.session_state.retention_strategies}"
                )
                prompt = f"{AUTOMATION_IDEAS_PROMPT}\n\n### Context:\n{context}"
                st.session_state.automation_plan = ai_call(prompt, "You are an automation expert.")
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
