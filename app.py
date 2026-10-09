
import os
import json
import uuid
import re
import smtplib
from email.message import EmailMessage
from datetime import datetime, timezone

import pandas as pd
import streamlit as st
from groq import Groq


# ---------------- PAGE CONFIG ----------------

st.set_page_config(
    page_title="AI Complaint Resolution Agent",
    page_icon="📩",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown("""
<style>
.block-container {
    max-width: 1000px;
    padding-top: 2rem;
    padding-bottom: 3rem;
}
.main-title {
    font-size: clamp(2rem, 5vw, 3.8rem);
    font-weight: 800;
    line-height: 1.15;
    letter-spacing: -1px;
}
.subtitle {
    color: #64748b;
    font-size: 1.1rem;
    margin-bottom: 2rem;
}
div[data-testid="stForm"] {
    border: 1px solid #dedede;
    border-radius: 16px;
    padding: 1.4rem;
}
div.stButton > button,
div.stFormSubmitButton > button {
    border-radius: 10px;
    min-height: 46px;
    font-weight: 600;
}
div[data-baseweb="input"] input,
div[data-baseweb="textarea"] textarea {
    border-radius: 9px;
}
</style>

<div class="main-title">📩 AI Complaint Resolution Agent</div>
<div class="subtitle">
Submit a complaint for AI-assisted analysis, urgency assessment,
and resolution planning.
</div>
""", unsafe_allow_html=True)


# ---------------- SESSION STATE ----------------

if "complaints" not in st.session_state:
    st.session_state.complaints = []

if "current_ticket" not in st.session_state:
    st.session_state.current_ticket = None


# ---------------- HELPERS ----------------

def secret(name, default=""):
    try:
        value = st.secrets.get(name, default)
        if value:
            return str(value)
    except Exception:
        pass
    return os.getenv(name, default)


def load_csv(filename):
    if not os.path.exists(filename):
        raise FileNotFoundError(
            f"{filename} is missing from the GitHub repository."
        )
    return pd.read_csv(filename).fillna("")


def send_email(recipient, subject, body):
    sender = secret("SMTP_EMAIL")
    password = secret("SMTP_APP_PASSWORD")

    if not sender or not password:
        return False, "Email is not configured in Streamlit Secrets."

    message = EmailMessage()
    message["Subject"] = subject
    message["From"] = sender
    message["To"] = recipient
    message.set_content(body)

    try:
        with smtplib.SMTP("smtp.gmail.com", 587, timeout=20) as server:
            server.starttls()
            server.login(sender, password)
            server.send_message(message)
        return True, f"Email sent to {recipient}."
    except Exception as exc:
        return False, f"Email failed: {exc}"


def get_client():
    key = secret("GROQ_API_KEY")
    if not key:
        raise ValueError("GROQ_API_KEY is missing from Streamlit Secrets.")
    return Groq(api_key=key)


def ask_ai(client, prompt, temperature=0.3):
    response = client.chat.completions.create(
        model="openai/gpt-oss-120b",
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a careful customer support assistant. "
                    "Do not invent order facts or claim actions are complete."
                ),
            },
            {"role": "user", "content": prompt},
        ],
        temperature=temperature,
    )
    return (response.choices[0].message.content or "").strip()


def parse_json(text):
    text = re.sub(r"^```(?:json)?\s*", "", text.strip())
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

    raise ValueError("AI did not return valid JSON. Please try again.")


def find_column(df, names):
    normalized = {
        str(col).strip().lower().replace(" ", "_"): col
        for col in df.columns
    }
    for name in names:
        if name in normalized:
            return normalized[name]
    return None


def find_order(order_id, orders):
    column = find_column(
        orders, ["order_id", "orderid", "order_number", "id"]
    )
    if not column or not order_id.strip():
        return None

    rows = orders[
        orders[column].astype(str).str.strip().str.lower()
        == order_id.strip().lower()
    ]
    return None if rows.empty else rows.iloc[0].to_dict()


def find_policy(complaint, policies):
    category_col = find_column(
        policies, ["category", "issue_type", "complaint_type", "topic"]
    )
    policy_col = find_column(
        policies, ["policy", "description", "policy_details", "rules"]
    )

    if category_col is None:
        category_col = policies.columns[0]
    if policy_col is None:
        policy_col = policies.columns[-1]

    text = complaint.lower()
    keyword_map = {
        "Delivery": ["delivery", "delayed", "late", "shipping", "tracking"],
        "Refund": ["refund", "money back", "reimbursement"],
        "Damaged Product": ["damaged", "broken", "defective", "faulty"],
        "Missing Order": ["missing order", "not received", "lost order"],
    }

    for category, keywords in keyword_map.items():
        if any(word in text for word in keywords):
            rows = policies[
                policies[category_col].astype(str).str.strip().str.lower()
                == category.lower()
            ]
            if not rows.empty:
                row = rows.iloc[0]
                return str(row[category_col]), str(row[policy_col])

    return "General", json.dumps(
        policies.to_dict(orient="records"), ensure_ascii=False
    )


# ---------------- LOAD DATA ----------------

try:
    policies_df = load_csv("company_policies.csv")
    orders_df = load_csv("orders.csv")
except Exception as exc:
    st.error(f"Database error: {exc}")
    st.stop()


# ---------------- TABS ----------------

submit_tab, dashboard_tab = st.tabs(
    ["Submit Complaint", "Complaint Dashboard"]
)


# ---------------- SUBMIT COMPLAINT ----------------

with submit_tab:
    with st.form("complaint_form", clear_on_submit=False):
        customer_name = st.text_input("Your name")
        customer_email = st.text_input("Your email address")
        subject = st.text_input("Complaint subject")
        complaint = st.text_area(
            "Describe your complaint",
            height=150,
        )
        order_id = st.text_input(
            "Order ID (optional)",
            placeholder="e.g. ORD101",
        )

        submitted = st.form_submit_button(
            "Submit Complaint",
            use_container_width=False,
        )

    if submitted:
        if not customer_name.strip():
            st.warning("Please enter your name.")
        elif not subject.strip():
            st.warning("Please enter a complaint subject.")
        elif not complaint.strip():
            st.warning("Please describe your complaint.")
        else:
            try:
                with st.spinner("Analyzing your complaint..."):
                    client = get_client()
                    order_info = find_order(order_id, orders_df)
                    policy_category, policy = find_policy(
                        subject + " " + complaint, policies_df
                    )

                    analysis = parse_json(ask_ai(client, f"""
Analyze this complaint.

Subject: {subject}
Complaint: {complaint}
Verified order: {json.dumps(order_info, default=str)}

Return only JSON with:
{{
 "category": "complaint category",
 "priority": "Low, Medium, or High",
 "sentiment": "Positive, Neutral, or Negative",
 "summary": "One sentence summary",
 "requires_human_review": true
}}
"""))

                    resolution = parse_json(ask_ai(client, f"""
Recommend a resolution for this complaint.

Subject: {subject}
Complaint: {complaint}
Verified order: {json.dumps(order_info, default=str)}
Relevant policy: {policy}
Analysis: {json.dumps(analysis)}

Return only JSON with:
{{
 "recommended_action": "Next action",
 "reason": "Reason",
 "policy_compliance": "Compliant, Non-compliant, or Needs verification",
 "human_review_required": true,
 "review_reason": "Reason for review"
}}

Do not claim actions have been completed or promise an unapproved refund.
"""))

                    reply = ask_ai(client, f"""
Write a polite, empathetic customer support reply.

Customer: {customer_name}
Subject: {subject}
Complaint: {complaint}
Verified order: {json.dumps(order_info, default=str)}
Recommended resolution: {json.dumps(resolution)}

Explain the proposed next step. Do not claim an action has already
been completed. Do not promise an unapproved refund or replacement.
Return only the customer-facing reply.
""", temperature=0.4)

                    if not reply:
                        reply = (
                            f"Dear {customer_name}, thank you for contacting us. "
                            "We apologize for the inconvenience. Our support "
                            "team will review your complaint and verify the "
                            "details before confirming the next steps."
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
                        "customer_name": customer_name.strip(),
                        "customer_email": customer_email.strip(),
                        "subject": subject.strip(),
                        "complaint": complaint.strip(),
                        "order_id": order_id.strip(),
                        "order_information": order_info,
                        "policy_category": policy_category,
                        "company_policy": policy,
                        "analysis": analysis,
                        "resolution": resolution,
                        "customer_reply": reply,
                        "status": "Open",
                        "human_review_required": True,
                    }

                    st.session_state.complaints.append(ticket)
                    st.session_state.current_ticket = ticket

                    # Notify support when configured.
                    support_email = secret("SUPPORT_EMAIL")
                    if support_email:
                        body = f"""
New complaint submitted.

Ticket: {ticket_id}
Customer: {customer_name}
Customer email: {customer_email or "Not provided"}
Subject: {subject}

Complaint:
{complaint}

Recommended resolution:
{resolution.get("recommended_action", "Review required")}

Reason:
{resolution.get("reason", "Please review this complaint.")}

Status: Open
Human review required: Yes
"""
                        ok, message = send_email(
                            support_email,
                            f"New Complaint: {ticket_id}",
                            body,
                        )
                        ticket["admin_email_status"] = message
                    else:
                        ticket["admin_email_status"] = (
                            "Not configured: add SUPPORT_EMAIL to Secrets."
                        )

                    st.session_state.current_ticket = ticket
                    st.success(
                        f"Complaint submitted successfully! Ticket: {ticket_id}"
                    )

            except Exception as exc:
                st.error("Unable to process the complaint.")
                st.exception(exc)


# ---------------- RESULTS AND CUSTOMER EMAIL ----------------

ticket = st.session_state.current_ticket

if ticket:
    st.divider()
    st.header("Complaint Analysis")

    a = ticket["analysis"]
    c1, c2, c3 = st.columns(3)
    c1.metric("Category", str(a.get("category", "Unknown")))
    c2.metric("Priority", str(a.get("priority", "Unknown")))
    c3.metric("Sentiment", str(a.get("sentiment", "Unknown")))

    st.write("**Summary:**", a.get("summary", ""))

    with st.expander("View full complaint analysis"):
        st.json(a)

    st.subheader("Verified Order Information")
    if ticket["order_information"]:
        st.json(ticket["order_information"])
    else:
        st.info("No matching order was found or no order ID was provided.")

    st.subheader("Relevant Company Policy")
    st.write(ticket["company_policy"])

    st.subheader("Recommended Resolution")
    st.json(ticket["resolution"])

    st.warning(
        "Human review is required. No refund, replacement, or escalation "
        "has been automatically completed."
    )

    st.subheader("Customer Support Reply")
    st.text_area(
        "Review the message before sending",
        value=ticket["customer_reply"],
        height=170,
        key=f"reply_{ticket['ticket_id']}",
    )

    st.subheader("📧 Email the Customer")
    recipient = st.text_input(
        "Customer email recipient",
        value=ticket["customer_email"],
        key=f"recipient_{ticket['ticket_id']}",
    )
    confirmed = st.checkbox(
        "I checked the recipient and reviewed the message.",
        key=f"confirm_{ticket['ticket_id']}",
    )

    if st.button(
        "Send Support Reply to Customer",
        key=f"send_{ticket['ticket_id']}",
    ):
        if "@" not in recipient or "." not in recipient:
            st.error("Please enter a valid email address.")
        elif not confirmed:
            st.warning("Please confirm the recipient and message.")
        else:
            ok, message = send_email(
                recipient.strip(),
                f"Support Update - {ticket['ticket_id']}",
                ticket["customer_reply"],
            )
            if ok:
                st.success(message)
            else:
                st.error(message)

    st.subheader("📬 Your Support Notification")
    st.write("**Ticket ID:**", ticket["ticket_id"])
    st.write("**Status:**", ticket["status"])
    st.write(
        "**Notification status:**",
        ticket.get("admin_email_status", "Not available"),
    )

    st.download_button(
        "Download Support Ticket (JSON)",
        data=json.dumps(ticket, indent=2, ensure_ascii=False, default=str),
        file_name=f"{ticket['ticket_id']}.json",
        mime="application/json",
    )


# ---------------- DASHBOARD ----------------

with dashboard_tab:
    st.subheader("Complaint Dashboard")

    complaints = st.session_state.complaints

    if not complaints:
        st.info(
            "No complaints have been submitted in this session yet. "
            "Submit a complaint to see it here."
        )
    else:
        c1, c2 = st.columns(2)
        c1.metric("Total Complaints", len(complaints))
        c2.metric(
            "Open Tickets",
            sum(1 for item in complaints if item["status"] == "Open"),
        )

        dashboard_rows = []
        for item in complaints:
            dashboard_rows.append({
                "Ticket ID": item["ticket_id"],
                "Customer": item["customer_name"],
                "Subject": item["subject"],
                "Category": item["analysis"].get("category", ""),
                "Priority": item["analysis"].get("priority", ""),
                "Status": item["status"],
            })

        st.dataframe(
            pd.DataFrame(dashboard_rows),
            use_container_width=True,
            hide_index=True,
        )

        selected_ticket = st.selectbox(
            "View a ticket",
            options=[item["ticket_id"] for item in complaints],
        )

        selected = next(
            item for item in complaints
            if item["ticket_id"] == selected_ticket
        )

        with st.expander("Selected complaint details"):
            st.write("**Customer:**", selected["customer_name"])
            st.write("**Email:**", selected["customer_email"])
            st.write("**Subject:**", selected["subject"])
            st.write("**Complaint:**", selected["complaint"])
            st.write("**Resolution:**")
            st.json(selected["resolution"])


st.divider()
st.caption(
    "Prototype uses sample order and policy data. AI recommendations "
    "may be incorrect and require human review."
)
