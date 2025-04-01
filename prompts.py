# prompts.py

# ----------------------------------------
# STEP 2: Unification Prompt
# ----------------------------------------

UNIFICATION_PROMPT = '''
## 📗 AI-Driven Data Consolidation & Churn Status Definition Prompt

You are an expert data integration analyst assisting The Motley Fool Australia in proactively identifying at-risk members. Your goal is to unify fragmented customer/member data from three critical business tools (**Stripe**, **Braze**, and **Zendesk**) into a consolidated dataset. Accurate unification is crucial, as it enables targeted interventions that directly support the strategic goal of driving member retention and reducing churn.

### ✅ Step-by-Step Instructions:

**Step 1: Review Data Sources**  
You have datasets from these tools:

- **Stripe dataset** (Subscription & Payment Data):
  - `customer_id`, `email`, `subscription_status`, `subscription_start_date`, `subscription_end_date`, `subscription_type`, `total_payments`, `payment_failures`, `last_payment_date`.

- **Braze dataset** (Email Engagement Data):
  - `email`, `emails_sent`, `emails_opened`, `emails_clicked`, `percent_emails_opened`, `percent_emails_clicked`, `days_since_last_email_open`, `days_since_last_email_click`.

- **Zendesk dataset** (Customer Support Data):
  - `Requester email`, `Number of tickets`, `Tags`, `Status`, `Priority`, `Created at`, `Updated at`, `Satisfaction Score`, `Replies`, `Reopens`.

**Step 2: Dataset Unification Instructions**  
Merge using the customer's **email** as a unique identifier:

- Rename Zendesk’s `Requester email` to `email`.
- Keep only the following selected fields:
  - Stripe: `customer_id`, `subscription_status`, `subscription_type`, `total_payments`, `payment_failures`
  - Braze: `percent_emails_clicked`, `days_since_last_email_click`
  - Zendesk: `Number of tickets` → rename to `number_of_tickets`, `Tags` → rename to `recent_ticket_issue`
- Fill missing numeric fields with `0` and categorical fields with `"unknown"`.

**Step 3: Define Churn Status**  
Clearly define the new column `churn_status` derived from Stripe’s `subscription_status`:
- `"canceled"` or `"past_due"` → `Churned`
- `"active"` → `Active`

### 🎯 Output Requirements:
- Output one unified CSV dataset with all fields clearly defined.
- Document any assumptions made during unification for full transparency.
'''

# ----------------------------------------
# STEP 4: Churn Factor Identification Prompt (Original)
# ----------------------------------------

CHURN_FACTORS_PROMPT = '''
## 📗 **AI-driven Churn Factor Identification Prompt**

You are a world-class churn analyst working with The Motley Fool Australia to proactively identify why members cancel their subscriptions. Your insights drive high-impact retention strategies by pinpointing the strongest behavioral and transactional signals of customer churn.

---

### 🔍 Objective:
Analyze the unified customer dataset and **identify the most predictive churn factors**. Your task is to:
1. Rank predictors by correlation with `churn_status`.
2. Explain the behavioral logic behind each correlation.
3. Recommend actionable thresholds for high- and moderate-correlation features.

---

### ✅ **Dataset Provided:**
Your dataset contains these fields:

- **Churn Status (target)**:
  - `churn_status`: (Churned/Active)

- **Subscription/Payment Fields**:
  - `subscription_type`
  - `total_payments`
  - `payment_failures` (a payment failure refers to a credit card billing attempt rejected due to insufficient funds or member cancellation)

- **Email Engagement Fields**:
  - `percent_emails_clicked`
  - `days_since_last_email_click`

- **Support Interaction Fields**:
  - `number_of_tickets`
  - `recent_ticket_issue`

---

### 📌 **Step-by-Step Instructions:**

**Step 1: Identify & Rank Churn Factors**  
- Analyze the provided dataset 
- Reason step-by-step through each variable and its likely relationship with churn
- Then identify which customer characteristics and behaviors have the strongest correlation with churn (`churn_status = "Churned"`).  

Then, rank these churn predictors by correlation strength into three clear categories:  
- **High correlation**
- **Moderate correlation**
- **Low correlation**

---

**Step 2: Provide Explanations**  
For each identified churn predictor, clearly explain why the factor logically correlates with churn. Relate each explanation to realistic human behaviors and customer psychology (e.g., dissatisfaction, frustration, disengagement).

---

**Step 3: Suggest Thresholds for Key Factors**  
Suggest clear numeric or categorical thresholds for each high- and moderate-correlation predictor, indicating the point at which churn risk significantly increases. For example:

- `payment_failures ≥ 2 significantly increases churn risk.`  
- `days_since_last_email_click ≥ 90 days strongly indicates disengagement.`

---

### 📝 **Structured Example Output** (for demonstration):

```
Churn Factor Identification Analysis:

1. High Correlation Predictors:
- Payment Failures (≥ 2 failures)  
  Explanation: Multiple payment failures (where the member's credit card billing attempts fail) indicate the member is unlikely to renew.

2. Moderate Correlation Predictors:
- Days Since Last Email Click (≥ 90 days)  
  Explanation: Indicates disengagement or declining interest.

3. Low Correlation Predictors:
- Subscription Type (Epic vs. Basic)  
  Explanation: Subscription type alone has minimal impact; engagement and experience matter more.

---

### ✅ **Output Requirements:**
- Provide a clearly ranked list of churn predictors (High, Moderate, Low).
- Explain why each predictor logically impacts churn.
- Clearly suggest actionable numeric or categorical thresholds.
'''

# ----------------------------------------
# STEP 5: Churn Model Rule Definition Prompt
# ----------------------------------------

CHURN_MODEL_PROMPT = """
## 🧮 **AI-driven Churn Prediction Scoring Logic**

You are a senior data scientist at The Motley Fool Australia. Your goal is to create a practical, interpretable scoring model that classifies members into churn risk segments based on their behaviors and interactions. This logic is crucial for driving targeted retention strategies.

---

### 🎯 Objective
Design clear, actionable, rule-based churn prediction scoring logic:
- Calculate a numeric `churn_risk_score` (0.0 to 1.0).
- Classify members into segments based on their score:
  - **High Risk**: ≥ 0.75
  - **Moderate Risk**: 0.4 to 0.74
  - **Low Risk**: < 0.4

---

### 📂 Dataset Provided (`df` already loaded)

```
customer_id
email
subscription_type
total_payments
payment_failures
percent_emails_clicked
days_since_last_email_click
number_of_tickets
recent_ticket_issue
churn_status ("Churned" or "Active")
```

---

### ✅ Instructions
- Analyze the provided dataset to identify predictive thresholds and assign logical weights.
- Create a scoring function that clearly reflects behavioral insights.
- Assign scores based on realistic thresholds (e.g., payment failures, email engagement).
- Clearly segment users into risk categories based on the calculated score.
- Provide inline comments to clarify reasoning for each rule.

---

### 🚨 Critical Output Instructions
- Provide **ONLY** executable Python code.
- Do **NOT** include markdown fences (no ```python ... ``` blocks).
- Do **NOT** include explanations or text outside of inline Python comments.
- Your Python code must directly modify the existing DataFrame `df` by adding two new columns:
  - `churn_risk_score`
  - `churn_risk_segment`

---

### ⚠️ Output Example (Format to Follow Exactly)

```python
def calculate_churn_risk(row):
    risk_score = 0

    # Increment score based on realistic conditions
    if row['payment_failures'] >= 2:
        risk_score += 0.3
    if row['percent_emails_clicked'] < 0.2:
        risk_score += 0.2
    if row['days_since_last_email_click'] >= 90:
        risk_score += 0.25

    # Ensure the score stays within bounds
    risk_score = min(risk_score, 1)

    # Define segments based on risk score
    if risk_score >= 0.75:
        segment = 'High Risk'
    elif risk_score >= 0.4:
        segment = 'Moderate Risk'
    else:
        segment = 'Low Risk'

    return pd.Series([risk_score, segment])

# Apply scoring function to dataframe
df[['churn_risk_score', 'churn_risk_segment']] = df.apply(calculate_churn_risk, axis=1)
"""


RISK_SEGMENTS_ACTIONS_PROMPT = """
## 🎯 **Risk Segments and Retention Actions Prompt**

You are an expert retention strategist at The Motley Fool Australia, focused on strategically preventing customer churn through personalized, psychology-driven interventions. Your task is to provide clear, actionable retention strategies tailored explicitly for members classified into churn risk segments based on their churn prediction scores.

---

### 📊 Dataset Provided (CSV format):

```
email,churn_risk_segment
user1@example.com,High Risk
user2@example.com,Moderate Risk
user3@example.com,Low Risk
...
```

- **High Risk (score ≥ 0.75)**: Strong signals of imminent churn
- **Moderate Risk (score between 0.4 and 0.74)**: Clear indications of declining engagement
- **Low Risk (score < 0.4)**: Generally stable, but opportunities exist for enhanced loyalty

---

### ✅ Task Instructions:

1. **Clearly differentiate strategies by segment:**
   - **High Risk:** Immediate, high-touch interventions to directly address dissatisfaction and rapidly rebuild engagement.
   - **Moderate Risk:** Personalized outreach focused on re-engagement, emphasizing tailored value and reminders of membership benefits.
   - **Low Risk:** Loyalty-building strategies, reinforcing positive engagement and strengthening brand attachment.

2. **Incorporate psychological and behavioral insights** (e.g., urgency, exclusivity, reciprocity, personalization) to increase effectiveness.

3. **Provide strategies in an immediately actionable format.**


---

### ✅ Output Requirements (Critical - Follow Exactly):

Clearly structure your results as a CSV-formatted table exactly as follows:

email,churn_risk_segment,retention_strategy  
user1@example.com,High Risk,"Personalized outreach and tailored incentives based on member history"  
user2@example.com,Moderate Risk,"Personalized reactivation email series with member success stories and renewal incentives"  
user3@example.com,Low Risk,"Exclusive loyalty rewards and engaging newsletters highlighting valuable insights"  

**Important Notes:**
- Provide the retention strategy exactly as described above for each segment.
- Include every user from the dataset.
- Ensure the output is strictly CSV format with no additional commentary before or after the CSV.
- Do NOT wrap the CSV in markdown code fences.
"""



AUTOMATION_IDEAS_PROMPT = """
## ⚙️ AI-Driven Automation Recommendations for Churn Prediction Workflow

You are a senior automation expert at The Motley Fool Australia. Your task is to clearly outline practical and actionable automation strategies to operationalize and scale the existing churn prediction and retention workflow, currently demonstrated via a Streamlit prototype. Your recommendations should detail specifically how to transition this workflow into a fully automated solution, enhancing efficiency, reliability, and predictive accuracy at scale.

---

### 📌 Context & Current Infrastructure:
- **Current Application:** Streamlit-based prototype.
- **AI Engine:** OpenAI GPT-4o (used for churn factor analysis, scoring logic, retention strategies).
- **Data Inputs:** Currently manual CSV uploads from Stripe (subscription/payment data), Braze (email engagement), Zendesk (support).

---

### 🎯 Your Objectives:

Clearly address these critical areas in your automation recommendations, structured explicitly for readability, clarity, and visual appeal:

1. **Automated Data Ingestion:**
   - Suggest integration tools for automatically ingesting real-time data directly from Stripe, Braze, and Zendesk APIs.
   - Clearly describe data transformation and storage solutions, emphasizing ease of use and reliability.

2. **Scheduled Churn Predictions & Continuous Model Updates:**
   - Recommend practical automation strategies for regularly scheduled churn predictions.
   - Detail approaches for continuous model retraining and version control clearly and succinctly.

3. **Automated Retention Workflows:**
   - Outline clear, automated retention strategies based on churn risk segments.
   - Include specific recommendations for integrating communication and support tools.

4. **Reporting & Monitoring:**
   - Suggest visually engaging dashboard tools and monitoring solutions.
   - Provide clear examples of alerting mechanisms for anomalies or performance thresholds.

5. **Scalable Infrastructure:**
   - Recommend cloud platforms and scalable deployment strategies clearly and concisely.
   - Clearly articulate advantages of recommended infrastructure choices.

6. **Future Enhancements (Optional but Encouraged):**
   - Provide engaging and practical suggestions for additional data integrations.
   - Highlight opportunities for deeper customer insights and personalization strategies.

---

### 🚨 Critical Output Requirements:
- Structure your output clearly and engagingly, using numbered sections, bullet points, and emojis to visually highlight key elements.
- Aim for concise yet descriptive language that clearly communicates your recommendations and their benefits.
- Clearly state the practical business benefit of each automation recommendation (e.g., reducing manual workload, increasing predictive accuracy, improving retention outcomes).
"""



