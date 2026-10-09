
import os
import json
import re
import uuid
import smtplib
from email.message import EmailMessage
from datetime import datetime, timezone

import pandas as pd
import streamlit as st
from groq import Groq


# =========================================================
# 1. PAGE CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="AI Complaint Resolution Agent",
    page_icon="📩",
    layout="wide",
)

st.title("📩 AI Complaint Resolution Agent")
st.caption(
    "Analyze customer complaints, verify orders, "
    "recommend resolutions, and prepare support tickets."
)

POLICY_FILE = "company_policies.csv"
ORDERS_FILE = "orders.csv"


# =========================================================
# 2. API KEY AND DATABASE HELPERS
# =========================================================

def get_secret(name, default=""):
    try:
        value = st.secrets.get(name, default)
        if value:
            return str(value)
    except Exception:
        pass

    return os.getenv(name, default)


def load_csv(path):
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Missing file: {path}. "
            "Upload it to the same GitHub folder as app.py."
        )

    return pd.read_csv(path).fillna("")


def get_groq_client():
    api_key = get_secret("GROQ_API_KEY")

    if not api_key:
        raise ValueError(
            "GROQ_API_KEY is missing. Add it in "
            "Streamlit Cloud → App settings → Secrets."
        )

    return Groq(api_key=api_key)


def ask_groq(client, prompt, temperature=0.3):
    response = client.chat.completions.create(
        model="openai/gpt-oss-120b",
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a careful customer-support assistant. "
                    "Never invent verified order facts, policy rules, "
                    "or completed actions."
                ),
            },
            {"role": "user", "content": prompt},
        ],
        temperature=temperature,
    )

    return (response.choices[0].message.content or "").strip()


def extract_json(text):
    """Extract a JSON object from an AI response."""
    text = text.strip()

    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text)

    try:
        result = json.loads(text)
        if isinstance(result, dict):
            return result
    except json.JSONDecodeError:
        pass

    match = re.search(r"\{.*\}", text, re.DOTALL)

    if match:
        try:
            result = json.loads(match.group(0))
            if isinstance(result, dict):
                return result
        except json.JSONDecodeError:
            pass

    raise ValueError("The AI did not return valid JSON. Please try again.")


def find_column(df, possible_names):
    """Find a CSV column without depending on capitalization."""
    normalized = {
        str(column).strip().lower().replace(" ", "_"): column
        for column in df.columns
    }

    for name in possible_names:
        if name in normalized:
            return normalized[name]

    return None


def get_order(order_id, orders_df):
    id_column = find_column(
        orders_df,
        ["order_id", "orderid", "order_number", "id"],
    )

    if id_column is None:
        raise ValueError(
            "orders.csv needs an order_id column."
        )

    matches = orders_df[
        orders_df[id_column].astype(str).str.strip().str.lower()
        == order_id.strip().lower()
    ]

    if matches.empty:
        return None

    return matches.iloc[0].to_dict()


def get_policy(complaint, policies_df):
    """Select a relevant policy using basic keyword matching."""
    text = complaint.lower()

    policy_column = find_column(
        policies_df,
        ["policy", "description", "policy_details", "rules"],
    )
    category_column = find_column(
        policies_df,
        ["category", "issue_type", "complaint_type", "topic"],
    )

    if policy_column is None:
        policy_column = policies_df.columns[-1]

    if category_column is None:
        category_column = policies_df.columns[0]

    keywords = {
        "Delivery": [
            "delivery", "delayed", "late", "shipping",
            "tracking", "courier",
        ],
        "Refund": [
            "refund", "money back", "reimbursement",
        ],
        "Damaged Product": [
            "damaged", "broken", "defective", "faulty",
        ],
        "Missing Order": [
            "missing order", "not received", "lost order",
            "never arrived",
        ],
    }

    selected_category = None

    for category, terms in keywords.items():
        if any(term in text for term in terms):
            selected_category = category
            break

    if selected_category:
        matching = policies_df[
            policies_df[category_column].astype(str).str.strip().str.lower()
            == selected_category.lower()
        ]

        if not matching.empty:
            row = matching.iloc[0]
            return (
                str(row[category_column]),
                str(row[policy_column]),
            )

    # Fallback: ask the AI to use the available policy text.
    available = policies_df.to_dict(orient="records")

    return (
        "General",
        json.dumps(available, ensure_ascii=False, default=str),
    )


# =========================================================
# 3. OPTIONAL EMAIL NOTIFICATION
# =========================================================

def send_ticket_email(ticket_id, customer_name, complaint, resolution):
    """
    Optional SMTP notification.

    Configure SMTP_EMAIL, SMTP_APP_PASSWORD, and SUPPORT_EMAIL
    in Streamlit Cloud Secrets to enable this feature.
    """
    smtp_email = get_secret("SMTP_EMAIL")
    smtp_password = get_secret("SMTP_APP_PASSWORD")
    support_email = get_secret("SUPPORT_EMAIL")

    if not (smtp_email and smtp_password and support_email):
        return False, (
            "Email is not configured. The ticket was still created."
        )

    message = EmailMessage()
    message["Subject"] = f"New Complaint Ticket: {ticket_id}"
    message["From"] = smtp_email
    message["To"] = support_email

    message.set_content(
        f"""A new customer complaint requires review.

Ticket ID: {ticket_id}
Customer: {customer_name}

Complaint:
{complaint}

Recommended resolution:
{resolution.get("recommended_action", "Review required")}

Reason:
{resolution.get("reason", "Please review the complaint")}

This is an automated notification. Review the case before taking action.
"""
    )

    try:
        with smtplib.SMTP("smtp.gmail.com", 587, timeout=20) as server:
            server.starttls()
            server.login(smtp_email, smtp_password)
            server.send_message(message)

        return True, "Support notification email sent successfully."

    except Exception as exc:
        return False, (
            "The ticket was created, but the email could not be sent. "
            f"Check your SMTP settings. Details: {exc}"
        )


# =========================================================
# 4. LOAD DATABASES
# =========================================================

try:
    policies_df = load_csv(POLICY_FILE)
    orders_df = load_csv(ORDERS_FILE)

except Exception as exc:
    st.error(f"Could not load company databases: {exc}")
    st.info(
        "Upload company_policies.csv and orders.csv "
        "to the GitHub repository root, alongside app.py."
    )
    st.stop()


# =========================================================
# 5. COMPLAINT FORM
# =========================================================

st.subheader("Submit a Customer Complaint")

with st.form("complaint_form"):
    customer_name = st.text_input(
        "Customer Name",
        placeholder="e.g. Ali",
    )

    order_id = st.text_input(
        "Order ID",
        placeholder="e.g. ORD101",
    )

    complaint = st.text_area(
        "Describe the Complaint",
        placeholder=(
            "Example: My order is 10 days late and I contacted "
            "support twice but received no response."
        ),
        height=150,
    )

    analyze_button = st.form_submit_button(
        "Analyze Complaint",
        type="primary",
        use_container_width=True,
    )


# =========================================================
# 6. ANALYZE COMPLAINT AND CREATE TICKET
# =========================================================

if analyze_button:
    if not complaint.strip():
        st.warning("Please enter a complaint before continuing.")
        st.stop()

    try:
        with st.spinner(
            "Analyzing complaint and preparing the support ticket..."
        ):
            client = get_groq_client()

            # Verify order from the local CSV database.
            order_info = get_order(order_id, orders_df) if order_id.strip() else None

            # Match a company policy.
            policy_category, company_policy = get_policy(
                complaint,
                policies_df,
            )

            # Analyze complaint.
            analysis_prompt = f"""
Analyze this customer complaint.

Complaint:
{complaint}

Verified order data:
{json.dumps(order_info, ensure_ascii=False, default=str)}

Return ONLY a JSON object with these keys:
{{
  "category": "complaint category",
  "priority": "Low, Medium, or High",
  "sentiment": "Positive, Neutral, or Negative",
  "summary": "One concise sentence",
  "requires_human_review": true
}}

Set priority to High when there is a significant delay,
repeated unanswered support requests, or a serious unresolved issue.
Do not invent order facts.
"""

            analysis = extract_json(
                ask_groq(client, analysis_prompt)
            )

            # Prepare resolution recommendation.
            resolution_prompt = f"""
Recommend a resolution for this customer complaint.

Complaint:
{complaint}

Verified order information:
{json.dumps(order_info, ensure_ascii=False, default=str)}

Relevant company policy:
{company_policy}

Analysis:
{json.dumps(analysis, ensure_ascii=False)}

Return ONLY a JSON object with these keys:
{{
  "recommended_action": "Next recommended action",
  "reason": "Why this action is appropriate",
  "policy_compliance": "Compliant, Non-compliant, or Needs verification",
  "human_review_required": true,
  "review_reason": "Reason for human review"
}}

Important:
- Recommend actions; do not claim they have been completed.
- Do not promise an unapproved refund or replacement.
- If the order cannot be verified, say verification is needed.
- Require human review before consequential action.
"""

            resolution = extract_json(
                ask_groq(client, resolution_prompt)
            )

            # Generate a customer-facing reply.
            reply_prompt = f"""
Write a polite, empathetic, professional customer support reply.

Customer name:
{customer_name.strip() or "Customer"}

Complaint:
{complaint}

Verified order information:
{json.dumps(order_info, ensure_ascii=False, default=str)}

Relevant company policy:
{company_policy}

Recommended resolution:
{json.dumps(resolution, ensure_ascii=False)}

Instructions:
- Acknowledge the customer's concern.
- Be concise and professional.
- Explain the proposed next step.
- Do not claim an action has already been completed.
- Do not promise an unapproved refund or replacement.
- Do not disclose internal AI analysis or ticket notes.
- If the order is not verified, explain that verification is needed.

Return only the customer-facing reply, without a heading.
"""

            customer_reply = ask_groq(
                client,
                reply_prompt,
                temperature=0.4,
            )

            if not customer_reply:
                customer_reply = (
                    "Thank you for contacting us. We apologize for "
                    "the inconvenience. Your complaint requires review "
                    "by our support team, who will verify the details "
                    "and determine the appropriate next steps."
                )

            # Create support ticket.
            ticket_id = (
                "TICKET-"
                + datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
                + "-"
                + uuid.uuid4().hex[:6].upper()
            )

            ticket = {
                "ticket_id": ticket_id,
                "created_at": datetime.now(timezone.utc).isoformat(),
                "customer_name": customer_name.strip() or "Not provided",
                "customer_complaint": complaint.strip(),
                "analysis": analysis,
                "order_information": order_info,
                "policy_category": policy_category,
                "company_policy": company_policy,
                "resolution": resolution,
                "customer_reply": customer_reply,
                "ticket_status": "Open",
                "human_review_required": True,
            }

            # Save result to session state so the output remains visible.
            st.session_state["complaint_result"] = ticket

    except Exception as exc:
        st.error("The complaint could not be processed.")
        st.exception(exc)


# =========================================================
# 7. DISPLAY RESULTS
# =========================================================

ticket = st.session_state.get("complaint_result")

if ticket:
    st.success("Complaint analysis completed!")

    st.divider()
    st.subheader("Complaint Analysis")

    analysis = ticket["analysis"]

    col1, col2, col3 = st.columns(3)

    col1.metric(
        "Category",
        str(analysis.get("category", "Not specified")),
    )
    col2.metric(
        "Priority",
        str(analysis.get("priority", "Not specified")),
    )
    col3.metric(
        "Sentiment",
        str(analysis.get("sentiment", "Not specified")),
    )

    st.write("**Summary:**", analysis.get("summary", "No summary provided."))

    with st.expander("View full complaint analysis"):
        st.json(analysis)

    st.divider()
    st.subheader("Verified Order Information")

    if ticket["order_information"]:
        st.json(ticket["order_information"])
    else:
        st.warning(
            "No matching order was found. Verify the order ID "
            "before taking action."
        )

    st.divider()
    st.subheader("Relevant Company Policy")
    st.write(ticket["company_policy"])

    st.divider()
    st.subheader("Recommended Resolution")
    st.json(ticket["resolution"])

    st.warning(
        "Human review is required before taking action. "
        "No refund, replacement, or escalation has been automatically completed."
    )

    st.divider()
    st.subheader("Customer Support Reply")

    customer_reply = ticket.get("customer_reply", "")

    if customer_reply and customer_reply.strip():
        st.success("Customer reply generated successfully.")
        st.text_area(
            "Reply to customer",
            value=customer_reply,
            height=160,
            key="display_customer_reply",
        )
        st.caption(
            "Review this draft before sending it to the customer."
        )
    else:
        st.warning(
            "The customer reply is empty. Please analyze the complaint again."
        )

    st.divider()
    st.subheader("Support Ticket")

    st.write("**Ticket ID:**", ticket["ticket_id"])
    st.write("**Status:**", ticket["ticket_status"])
    st.write(
        "**Created at (UTC):**",
        ticket["created_at"],
    )

    ticket_json = json.dumps(
        ticket,
        indent=2,
        ensure_ascii=False,
        default=str,
    )

    st.download_button(
        label="Download Support Ticket (JSON)",
        data=ticket_json,
        file_name=f"{ticket['ticket_id']}.json",
        mime="application/json",
        use_container_width=True,
    )

    st.divider()
    st.subheader("Optional Email Notification")

    st.caption(
        "This sends a ticket notification to your configured support "
        "email. It does not automatically email the customer."
    )

    if st.button("Send Support Ticket Email"):
        with st.spinner("Sending notification..."):
            success, message = send_ticket_email(
                ticket["ticket_id"],
                ticket["customer_name"],
                ticket["customer_complaint"],
                ticket["resolution"],
            )

        if success:
            st.success(message)
        else:
            st.warning(message)

st.divider()

st.caption(
    "Prototype notice: This app uses sample order and policy data. "
    "Verify information and obtain appropriate human approval "
    "before acting on any recommendation."
)
