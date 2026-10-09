
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


# ======================================================
# 1. PAGE CONFIGURATION
# ======================================================

st.set_page_config(
    page_title="AI Complaint Resolution Agent",
    page_icon="📩",
    layout="wide",
)

st.title("📩 AI Complaint Resolution Agent")
st.caption("AI-powered complaint analysis, resolution and email support.")

POLICY_FILE = "company_policies.csv"
ORDERS_FILE = "orders.csv"


# ======================================================
# 2. SECRETS AND EMAIL CONFIGURATION
# ======================================================

def get_secret(name, default=""):
    try:
        value = st.secrets.get(name, default)
        if value:
            return str(value)
    except Exception:
        pass

    return os.getenv(name, default)


def send_email(to_address, subject, body):
    """Send an email using configured SMTP credentials."""

    smtp_email = get_secret("SMTP_EMAIL")
    smtp_password = get_secret("SMTP_APP_PASSWORD")

    if not smtp_email or not smtp_password:
        return False, (
            "Email is not configured. Add SMTP_EMAIL and "
            "SMTP_APP_PASSWORD to Streamlit Secrets."
        )

    message = EmailMessage()
    message["Subject"] = subject
    message["From"] = smtp_email
    message["To"] = to_address
    message.set_content(body)

    try:
        with smtplib.SMTP("smtp.gmail.com", 587, timeout=20) as server:
            server.starttls()
            server.login(smtp_email, smtp_password)
            server.send_message(message)

        return True, f"Email sent successfully to {to_address}."

    except Exception as exc:
        return False, (
            "Email could not be sent. Check your SMTP settings. "
            f"Details: {exc}"
        )


# ======================================================
# 3. AI AND DATABASE HELPERS
# ======================================================

def get_groq_client():
    api_key = get_secret("GROQ_API_KEY")

    if not api_key:
        raise ValueError(
            "GROQ_API_KEY is missing from Streamlit Secrets."
        )

    return Groq(api_key=api_key)


def ask_groq(client, prompt, temperature=0.3):
    response = client.chat.completions.create(
        model="openai/gpt-oss-120b",
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a careful customer support assistant. "
                    "Never invent verified facts or claim actions "
                    "have been completed when they have not."
                ),
            },
            {"role": "user", "content": prompt},
        ],
        temperature=temperature,
    )

    return (response.choices[0].message.content or "").strip()


def extract_json(text):
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
        result = json.loads(match.group(0))
        if isinstance(result, dict):
            return result

    raise ValueError("The AI response was not valid JSON. Try again.")


def load_csv(path):
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"{path} was not found. Upload it beside app.py on GitHub."
        )

    return pd.read_csv(path).fillna("")


def find_column(df, names):
    columns = {
        str(column).strip().lower().replace(" ", "_"): column
        for column in df.columns
    }

    for name in names:
        if name in columns:
            return columns[name]

    return None


def get_order(order_id, orders_df):
    id_column = find_column(
        orders_df,
        ["order_id", "orderid", "order_number", "id"],
    )

    if id_column is None:
        raise ValueError("orders.csv must contain an order_id column.")

    matches = orders_df[
        orders_df[id_column].astype(str).str.strip().str.lower()
        == order_id.strip().lower()
    ]

    if matches.empty:
        return None

    return matches.iloc[0].to_dict()


def get_policy(complaint, policies_df):
    category_column = find_column(
        policies_df,
        ["category", "issue_type", "complaint_type", "topic"],
    )
    policy_column = find_column(
        policies_df,
        ["policy", "description", "policy_details", "rules"],
    )

    if category_column is None:
        category_column = policies_df.columns[0]

    if policy_column is None:
        policy_column = policies_df.columns[-1]

    text = complaint.lower()

    keyword_map = {
        "Delivery": [
            "delivery", "delayed", "late", "shipping", "tracking",
        ],
        "Refund": ["refund", "money back", "reimbursement"],
        "Damaged Product": [
            "damaged", "broken", "defective", "faulty",
        ],
        "Missing Order": [
            "missing order", "not received", "lost order",
            "never arrived",
        ],
    }

    selected = None

    for category, keywords in keyword_map.items():
        if any(keyword in text for keyword in keywords):
            selected = category
            break

    if selected:
        rows = policies_df[
            policies_df[category_column].astype(str).str.strip().str.lower()
            == selected.lower()
        ]

        if not rows.empty:
            row = rows.iloc[0]
            return str(row[category_column]), str(row[policy_column])

    return (
        "General",
        json.dumps(
            policies_df.to_dict(orient="records"),
            ensure_ascii=False,
            default=str,
        ),
    )


# ======================================================
# 4. LOAD DATABASES
# ======================================================

try:
    policies_df = load_csv(POLICY_FILE)
    orders_df = load_csv(ORDERS_FILE)
except Exception as exc:
    st.error(f"Could not load company databases: {exc}")
    st.stop()


# ======================================================
# 5. COMPLAINT FORM
# ======================================================

st.subheader("Submit a Customer Complaint")

with st.form("complaint_form"):
    customer_name = st.text_input(
        "Customer Name",
        placeholder="e.g. Ali",
    )

    customer_email = st.text_input(
        "Customer Email Address",
        placeholder="customer@example.com",
    )

    order_id = st.text_input(
        "Order ID",
        placeholder="e.g. ORD101",
    )

    complaint = st.text_area(
        "Describe the Complaint",
        placeholder=(
            "My order is 10 days late. I contacted support twice "
            "but have not received a response."
        ),
        height=150,
    )

    analyze_button = st.form_submit_button(
        "Analyze Complaint",
        type="primary",
        use_container_width=True,
    )


# ======================================================
# 6. ANALYZE AND CREATE TICKET
# ======================================================

if analyze_button:
    if not complaint.strip():
        st.warning("Please enter the customer's complaint.")
        st.stop()

    try:
        with st.spinner("Analyzing complaint and creating ticket..."):
            client = get_groq_client()

            order_info = (
                get_order(order_id, orders_df)
                if order_id.strip()
                else None
            )

            policy_category, company_policy = get_policy(
                complaint,
                policies_df,
            )

            analysis_prompt = f"""
Analyze this complaint.

Complaint:
{complaint}

Verified order information:
{json.dumps(order_info, ensure_ascii=False, default=str)}

Return only a JSON object with these keys:
{{
  "category": "complaint category",
  "priority": "Low, Medium, or High",
  "sentiment": "Positive, Neutral, or Negative",
  "summary": "One concise sentence",
  "requires_human_review": true
}}

Do not invent order facts.
"""

            analysis = extract_json(
                ask_groq(client, analysis_prompt)
            )

            resolution_prompt = f"""
Recommend a resolution for this complaint.

Complaint:
{complaint}

Verified order:
{json.dumps(order_info, ensure_ascii=False, default=str)}

Relevant company policy:
{company_policy}

Analysis:
{json.dumps(analysis, ensure_ascii=False)}

Return only JSON with these keys:
{{
  "recommended_action": "Next recommended action",
  "reason": "Why this action is appropriate",
  "policy_compliance": "Compliant, Non-compliant, or Needs verification",
  "human_review_required": true,
  "review_reason": "Reason for human review"
}}

Do not claim any action has already been completed.
Do not promise an unapproved refund or replacement.
Require human review before consequential action.
"""

            resolution = extract_json(
                ask_groq(client, resolution_prompt)
            )

            reply_prompt = f"""
Write a polite, empathetic customer support reply.

Customer name:
{customer_name.strip() or "Customer"}

Complaint:
{complaint}

Verified order:
{json.dumps(order_info, ensure_ascii=False, default=str)}

Company policy:
{company_policy}

Recommended resolution:
{json.dumps(resolution, ensure_ascii=False)}

Acknowledge the concern, apologize where appropriate, and explain
the proposed next step. Do not claim an action has been completed.
Do not promise an unapproved refund or replacement.
Return only the customer-facing message.
"""

            customer_reply = ask_groq(
                client,
                reply_prompt,
                temperature=0.4,
            )

            if not customer_reply:
                customer_reply = (
                    "Thank you for contacting us. We apologize for "
                    "the inconvenience. Our support team needs to "
                    "review your complaint and verify the details "
                    "before confirming the next steps."
                )

            ticket_id = (
                "TICKET-"
                + datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
                + "-"
                + uuid.uuid4().hex[:6].upper()
            )

            ticket = {
                "ticket_id": ticket_id,
                "created_at": datetime.now(timezone.utc).isoformat(),
                "customer_name": customer_name.strip() or "Customer",
                "customer_email": customer_email.strip(),
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

            # Save results before rendering the page.
            st.session_state["complaint_result"] = ticket

            # Notify support email automatically when a ticket is created.
            support_email = get_secret("SUPPORT_EMAIL")

            if support_email:
                notification_body = f"""
A new customer complaint has been submitted.

Ticket ID: {ticket_id}
Customer: {ticket["customer_name"]}
Customer email: {customer_email.strip() or "Not provided"}
Order ID: {order_id.strip() or "Not provided"}

Complaint:
{complaint}

Recommended resolution:
{resolution.get("recommended_action", "Review required")}

Reason:
{resolution.get("reason", "Please review this complaint.")}

Ticket status: Open
Human review required: Yes

This is an automated notification. Review the case before taking action.
"""

                ok, message = send_email(
                    support_email,
                    f"New Complaint Ticket: {ticket_id}",
                    notification_body,
                )

                ticket["admin_email_status"] = (
                    message if ok else f"Notification failed: {message}"
                )
            else:
                ticket["admin_email_status"] = (
                    "Not configured: add SUPPORT_EMAIL to Streamlit Secrets."
                )

            st.session_state["complaint_result"] = ticket

    except Exception as exc:
        st.error("The complaint could not be processed.")
        st.exception(exc)


# ======================================================
# 7. DISPLAY RESULTS AND EMAIL OPTIONS
# ======================================================

ticket = st.session_state.get("complaint_result")

if ticket:
    st.success("Complaint analysis completed!")

    st.divider()
    st.subheader("Complaint Analysis")

    analysis = ticket["analysis"]

    col1, col2, col3 = st.columns(3)

    col1.metric("Category", str(analysis.get("category", "Unknown")))
    col2.metric("Priority", str(analysis.get("priority", "Unknown")))
    col3.metric("Sentiment", str(analysis.get("sentiment", "Unknown")))

    st.write("**Summary:**", analysis.get("summary", ""))

    with st.expander("View full complaint analysis"):
        st.json(analysis)

    st.divider()
    st.subheader("Verified Order Information")

    if ticket["order_information"]:
        st.json(ticket["order_information"])
    else:
        st.warning("No matching order was found. Verify the order ID.")

    st.divider()
    st.subheader("Relevant Company Policy")
    st.write(ticket["company_policy"])

    st.divider()
    st.subheader("Recommended Resolution")
    st.json(ticket["resolution"])

    st.warning(
        "Human review is required. No refund, replacement, or escalation "
        "has been automatically completed."
    )

    st.divider()
    st.subheader("Customer Support Reply")

    st.text_area(
        "Review the email message",
        value=ticket["customer_reply"],
        height=170,
        key=f"reply_{ticket['ticket_id']}",
    )

    st.subheader("📧 Email the Customer")

    st.write(
        "The message below will be sent to the customer when you "
        "click the send button."
    )

    recipient = st.text_input(
        "Customer email recipient",
        value=ticket.get("customer_email", ""),
        key=f"recipient_{ticket['ticket_id']}",
    )

    confirm_customer_email = st.checkbox(
        "I have checked the recipient and reviewed the message.",
        key=f"confirm_customer_{ticket['ticket_id']}",
    )

    if st.button(
        "Send Support Reply to Customer",
        key=f"send_customer_{ticket['ticket_id']}",
        type="primary",
    ):
        if not recipient.strip() or "@" not in recipient:
            st.error("Enter a valid customer email address.")
        elif not confirm_customer_email:
            st.warning("Confirm the recipient and message before sending.")
        else:
            reply = ticket["customer_reply"]

            with st.spinner("Sending customer email..."):
                ok, message = send_email(
                    recipient.strip(),
                    f"Customer Support Update - {ticket['ticket_id']}",
                    reply,
                )

            if ok:
                st.success(message)
            else:
                st.error(message)

    st.divider()
    st.subheader("📬 Your Support Notification")

    st.write("**Ticket ID:**", ticket["ticket_id"])
    st.write("**Status:**", ticket["ticket_status"])
    st.write(
        "**Notification status:**",
        ticket.get("admin_email_status", "Not available"),
    )

    st.caption(
        "The support notification is attempted automatically when a "
        "ticket is created. If it fails, check the email settings."
    )

    st.divider()
    st.subheader("Support Ticket Download")

    ticket_json = json.dumps(
        ticket,
        indent=2,
        ensure_ascii=False,
        default=str,
    )

    st.download_button(
        "Download Support Ticket (JSON)",
        data=ticket_json,
        file_name=f"{ticket['ticket_id']}.json",
        mime="application/json",
        use_container_width=True,
    )

st.divider()

st.caption(
    "Prototype: Uses sample order and policy data. Verify all details "
    "and obtain human approval before taking action."
)
