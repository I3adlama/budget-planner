import os
import math
import datetime as dt

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# =============================
# Config & storage
# =============================
DATA_DIR = "budget_data"
DEBTS_FILE = os.path.join(DATA_DIR, "debts.csv")
EXPENSES_FILE = os.path.join(DATA_DIR, "expenses.csv")
ONETIME_FILE = os.path.join(DATA_DIR, "onetime.csv")
INVEST_FILE = os.path.join(DATA_DIR, "investments.csv")


def ensure_data_dir():
    os.makedirs(DATA_DIR, exist_ok=True)


def load_df(path, columns):
    """Load CSV; guarantee required columns exist (fills missing with NaN)."""
    if os.path.exists(path):
        df = pd.read_csv(path)
        for col in columns:
            if col not in df.columns:
                df[col] = np.nan
        return df[columns]
    else:
        return pd.DataFrame(columns=columns)


def save_df(df, path):
    df.to_csv(path, index=False)


# =============================
# Financial helpers
# =============================
def payoff_stats_fixed(balance, apr_pct, monthly_payment):
    """
    More realistic fixed-payment payoff:
      • Simulates month-by-month
      • Uses a smaller final payment instead of assuming N full payments
    Returns (months_to_payoff, total_interest) or (None, None) if payment is too small.
    """
    balance = float(balance)
    pmt = float(monthly_payment)
    r = float(apr_pct) / 100.0 / 12.0  # monthly rate

    if balance <= 0 or pmt <= 0:
        return 0, 0.0

    # If no interest, it's just balance / pmt
    if r == 0:
        months = int(math.ceil(balance / pmt))
        total_paid = months * pmt
        interest_paid = total_paid - balance
        return months, max(0.0, interest_paid)

    months = 0
    total_interest = 0.0

    # Safety cap: 50 years of months (should never hit)
    for _ in range(50 * 12):
        if balance <= 0:
            break

        # monthly interest
        interest = balance * r
        total_interest += interest

        # Default planned payment
        payment = pmt

        # If this payment would overshoot, just pay what's left
        if payment > balance + interest:
            payment = balance + interest

        principal = payment - interest
        if principal <= 0:
            # Payment doesn't cover interest → never pays off
            return None, None

        balance -= principal
        months += 1

    if balance > 0:
        # Hit safety cap, treat as non-payoff
        return None, None

    return months, total_interest


def payoff_stats_minimum_style(
    balance, apr_pct, min_pct=0.01, min_dollar=25.0, max_months=50 * 12
):
    """
    Approximate "credit-card minimum payment only" behavior.

    Each month:
      min_payment = max(balance * min_pct + interest, min_dollar)
    (You can tweak min_pct / min_dollar to better match your card.)

    Returns (months_to_payoff, total_interest) or (None, None) if the
    minimum formula never pays the balance off.
    """
    balance = float(balance)
    r = float(apr_pct) / 100.0 / 12.0

    if balance <= 0 or r < 0:
        return 0, 0.0

    months = 0
    total_interest = 0.0

    for _ in range(max_months):
        if balance <= 0:
            break

        interest = balance * r
        total_interest += interest

        # Minimum payment per disclosure style
        min_payment = balance * min_pct + interest
        if min_payment < min_dollar:
            min_payment = min_dollar

        # Don't pay more than what's left
        if min_payment > balance + interest:
            min_payment = balance + interest

        principal = min_payment - interest
        if principal <= 0:
            # minimum never reduces principal → infinite
            return None, None

        balance -= principal
        months += 1

    if balance > 0:
        # Hit safety cap → essentially never paid
        return None, None

    return months, total_interest

def simulate_debt_payoff(
    debts_df,
    total_monthly_budget,
    months,
    strategy="Snowball",
    allocate_extra=True,
    track_history=False,
    schedule_start_date=None,
):
    """
    Simulate monthly payments against debts.

    Parameters
    ----------
    debts_df : DataFrame
        Columns: balance, apr (as %), min_payment, debt_type, name, due_day (optional).
    total_monthly_budget : float
        Amount available for debts each month during the period.
    months : int
        Number of months to simulate.
    strategy : {"Snowball", "Avalanche"}
        Snowball  = target smallest remaining balance first with extra.
        Avalanche = target highest APR first with extra.
    allocate_extra : bool
        If True, any amount above the sum of minimum payments is
        applied to debts according to the chosen strategy.
        If False, we pay ONLY the minimums (scaled if budget is too low)
        and treat the rest of the monthly budget as leftover.
    track_history : bool
        If True, returns a per-month history DataFrame and detailed per-debt schedule.
    schedule_start_date : date or None
        If provided and track_history=True, used as the first month for the schedule.
        Otherwise uses today's date.

    Returns
    -------
    dict with keys:
        - "final_df": DataFrame of ending balances (same shape as input)
        - "leftover_total": float, total unused budget over the period
        - "total_interest": float, total interest paid during the period
        - "total_paid": float, total dollars actually paid to debts
        - "total_principal": float, total principal reduction
        - "history": DataFrame or None (month-level summary)
        - "schedule": DataFrame or None (per-debt payment schedule)
    """
    if debts_df.empty:
        empty_hist = pd.DataFrame(
            columns=["month", "interest", "principal", "paid", "leftover"]
        )
        return {
            "final_df": debts_df.copy(),
            "leftover_total": total_monthly_budget * months,
            "total_interest": 0.0,
            "total_paid": 0.0,
            "total_principal": 0.0,
            "history": empty_hist if track_history else None,
            "schedule": pd.DataFrame(
                columns=[
                    "month",
                    "date",
                    "debt_name",
                    "debt_type",
                    "payment",
                    "min_component",
                    "extra_component",
                    "interest",
                    "principal",
                    "balance_after",
                ]
            )
            if track_history
            else None,
        }

    df = debts_df.copy().reset_index(drop=True)
    df["balance"] = df["balance"].astype(float)
    df["apr"] = df["apr"].astype(float) / 100.0
    df["min_payment"] = df["min_payment"].astype(float)
    if "debt_type" not in df.columns:
        df["debt_type"] = ""
    if "name" not in df.columns:
        df["name"] = df.index.astype(str)

    df["r_month"] = df["apr"] / 12.0

    leftover_total = 0.0
    total_interest = 0.0
    total_paid = 0.0
    history_rows = []
    schedule_rows = []

    months = int(months)

    # Helper for month → date
    if schedule_start_date is None:
        base_date = dt.date.today()
    else:
        base_date = schedule_start_date

    def month_to_date(m_index: int) -> dt.date:
        """Return a representative date for month m_index (0-based)."""
        year = base_date.year + (base_date.month - 1 + m_index) // 12
        month = (base_date.month - 1 + m_index) % 12 + 1
        return dt.date(year, month, 1)

    for m in range(months):
        # If everything is already paid off, just accumulate leftover budget
        if (df["balance"] <= 0).all():
            leftover_total += total_monthly_budget * (months - m)
            break

        # 1) Interest accrues on current balances
        interest = df["balance"] * df["r_month"]
        df["balance"] = df["balance"] + interest
        interest_this_month = float(interest.sum())
        total_interest += interest_this_month

        # 2) Compute minimum payments (can't exceed current balance)
        min_payments = df["min_payment"].copy()
        min_payments = np.minimum(min_payments, df["balance"])
        total_min = float(min_payments.sum())

        # This will track actual payments applied per debt this month
        payments_this_month = pd.Series(0.0, index=df.index)
        min_component = pd.Series(0.0, index=df.index)
        extra_component = pd.Series(0.0, index=df.index)

        leftover_this_month = 0.0

        if total_min > total_monthly_budget:
            # Can't cover all minimums → scale them proportionally
            ratio = total_monthly_budget / total_min if total_min > 0 else 0.0
            scaled = min_payments * ratio
            df["balance"] = df["balance"] - scaled
            payments_this_month = scaled
            min_component = scaled
        else:
            # Pay all minimums
            df["balance"] = df["balance"] - min_payments
            payments_this_month = min_payments.copy()
            min_component = min_payments.copy()

            extra = total_monthly_budget - total_min

            if allocate_extra and extra > 0:
                # 3) Allocate extra according to strategy
                if strategy == "Snowball":
                    order = df.sort_values("balance").index
                else:  # Avalanche
                    order = df.sort_values("apr", ascending=False).index

                for idx in order:
                    bal = df.at[idx, "balance"]
                    if bal <= 0:
                        continue
                    pay = min(extra, bal)
                    if pay <= 0:
                        continue
                    df.at[idx, "balance"] = bal - pay
                    payments_this_month.loc[idx] += pay
                    extra_component.loc[idx] += pay
                    extra -= pay
                    if extra <= 0:
                        break

                leftover_this_month = max(0.0, extra)
            else:
                leftover_this_month = max(0.0, extra)

        # Do not allow negative balances
        df["balance"] = df["balance"].clip(lower=0.0)

        paid_this_month = float(payments_this_month.sum())
        principal_this_month = paid_this_month - interest_this_month

        total_paid += paid_this_month
        leftover_total += leftover_this_month

        if track_history:
            history_rows.append(
                {
                    "month": m + 1,
                    "interest": interest_this_month,
                    "principal": principal_this_month,
                    "paid": paid_this_month,
                    "leftover": leftover_this_month,
                }
            )
            # Build per-debt schedule rows
            current_date = month_to_date(m)
            for idx in df.index:
                pay = float(payments_this_month.loc[idx])
                if abs(pay) < 1e-9:
                    continue
                row = {
                    "month": m + 1,
                    "date": current_date,
                    "debt_name": df.at[idx, "name"],
                    "debt_type": df.at[idx, "debt_type"],
                    "payment": pay,
                    "min_component": float(min_component.loc[idx]),
                    "extra_component": float(extra_component.loc[idx]),
                    "interest": float(interest.loc[idx]),
                    "principal": float(payments_this_month.loc[idx] - interest.loc[idx]),
                    "balance_after": float(df.at[idx, "balance"]),
                }
                schedule_rows.append(row)

    total_principal = total_paid - total_interest

    history_df = pd.DataFrame(history_rows) if track_history else None
    schedule_df = pd.DataFrame(schedule_rows) if track_history else None

    return {
        "final_df": df,
        "leftover_total": leftover_total,
        "total_interest": total_interest,
        "total_paid": total_paid,
        "total_principal": total_principal,
        "history": history_df,
        "schedule": schedule_df,
    }



def simulate_portfolio_growth(total_initial, weighted_yield_pct, monthly_contrib, months):
    """Simple projection: portfolio-level yield, monthly compounding."""
    balance = float(total_initial)
    r_month = weighted_yield_pct / 100.0 / 12.0
    history = []
    for m in range(months + 1):
        history.append((m, balance))
        # next month
        balance += monthly_contrib
        balance += balance * r_month
    history_df = pd.DataFrame(history, columns=["Month", "Balance"])
    return history_df


# =============================
# Chart helpers
# =============================
def debt_chart(df):
    if df.empty:
        st.info("No debts yet.")
        return
    # Bar by balance, colored by debt_type. APR in hover.
    fig = px.bar(
        df,
        x="name",
        y="balance",
        color="debt_type",
        hover_data=["apr", "min_payment"],
        labels={"balance": "Balance", "name": "Debt", "debt_type": "Type", "apr": "APR %"},
        title="Debts by Balance (colored by Type)",
    )
    fig.update_layout(xaxis_title="Debt", yaxis_title="Balance ($)")
    st.plotly_chart(fig, use_container_width=True)


def expenses_chart(df):
    if df.empty:
        st.info("No living expenses yet.")
        return
    monthly = df.copy()
    monthly["monthly_amount"] = np.where(
        monthly["frequency"] == "Weekly",
        monthly["amount"] * 4.33,
        monthly["amount"],
    )
    fig = px.bar(
        monthly,
        x="name",
        y="monthly_amount",
        color="subcategory",
        hover_data=["frequency"],
        labels={"name": "Expense", "monthly_amount": "Monthly Amount", "subcategory": "Subcategory"},
        title="Living Expenses (normalized to monthly, colored by subcategory)",
    )
    fig.update_layout(xaxis_title="Expense", yaxis_title="Monthly amount ($)")
    st.plotly_chart(fig, use_container_width=True)


def onetime_chart(df):
    if df.empty:
        st.info("No one-time bills yet.")
        return
    fig = px.bar(
        df,
        x="name",
        y="amount",
        color="due_date",
        labels={"amount": "Amount", "name": "Bill"},
        title="One-time Bills by Due Date",
    )
    fig.update_layout(xaxis_title="Bill", yaxis_title="Amount ($)")
    st.plotly_chart(fig, use_container_width=True)


def portfolio_projection_chart(history_df):
    if history_df.empty:
        return
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=history_df["Month"],
            y=history_df["Balance"],
            mode="lines+markers",
            name="Portfolio value",
        )
    )
    for m, label in [(6, "6 mo"), (12, "12 mo"), (24, "24 mo")]:
        fig.add_vline(
            x=m,
            line=dict(color="#9ca3af", width=1, dash="dash"),
            annotation_text=label,
            annotation_position="top left",
        )

    fig.update_layout(
        title="Portfolio Projection (24 months)",
        xaxis_title="Month",
        yaxis_title="Balance ($)",
    )
    st.plotly_chart(fig, use_container_width=True)


# =============================
# UI helper: progress card
# =============================
def progress_card(title, subtitle, current, start_or_goal, positive_good=True, color="#00c853"):
    """
    Custom progress card:
      - positive_good=True: progress = (start_or_goal - current) / start_or_goal (e.g., paying debt down).
      - positive_good=False: progress = current / start_or_goal (e.g., savings toward a goal).
    """
    if start_or_goal is None or start_or_goal <= 0:
        progress = 0.0
    else:
        if positive_good:
            progress = (start_or_goal - current) / start_or_goal
        else:
            progress = current / start_or_goal
    progress = max(0.0, min(1.0, progress))

    bar_color = color
    bg = "#111827"
    text_main = "#e5e7eb"
    text_sub = "#9ca3af"
    border = "#374151"

    html = f"""
    <div style="
        padding: 1rem 1.2rem;
        border-radius: 0.8rem;
        background: {bg};
        border: 1px solid {border};
        box-shadow: 0 8px 20px rgba(0,0,0,0.35);
        ">
        <div style="font-size:0.8rem;color:{text_sub};margin-bottom:0.25rem;">
            {title}
        </div>
        <div style="font-size:1.4rem;font-weight:600;color:{text_main};">
            ${current:,.2f}
        </div>
        <div style="font-size:0.75rem;color:{text_sub};margin-top:0.1rem;margin-bottom:0.5rem;">
            {subtitle}
        </div>
        <div style="
            width:100%;
            height:0.5rem;
            border-radius:999px;
            background:#1f2933;
            overflow:hidden;
        ">
            <div style="
                width:{progress*100:.1f}%;
                height:100%;
                background:{bar_color};
                border-radius:999px;
                transition:width 0.3s ease-out;
            "></div>
        </div>
        <div style="font-size:0.7rem;color:{text_sub};margin-top:0.25rem;">
            Progress: {progress*100:.1f}%
        </div>
    </div>
    """
    st.markdown(html, unsafe_allow_html=True)

# =============================
# Formatting helpers
# =============================
def format_currency(value) -> str:
    """Format a number like 1234.5 as '$1,234.50'. Empty string if invalid."""
    try:
        v = float(value)
    except (TypeError, ValueError):
        return ""
    return f"${v:,.2f}"


def format_months_to_ym(m) -> str:
    """Convert a month count into 'Xy Ym' (e.g., 18 -> '1y 6m')."""
    if m is None or pd.isna(m):
        return ""
    try:
        m = int(round(m))
    except (TypeError, ValueError):
        return ""
    if m <= 0:
        return "0 mo"
    years = m // 12
    months = m % 12
    parts = []
    if years:
        parts.append(f"{years}y")
    if months:
        parts.append(f"{months}m")
    return " ".join(parts)

def compute_portfolio_scenario_fixed(debts_df, extra_per_month=0.0, round_up_to_10=False, strategy="Proportional"):
    """
    Portfolio-level 'what-if' scenario:

    - Base plan: each debt keeps paying its current min_payment as a fixed payment.
    - Scenario: we spread `extra_per_month` across debts according to `strategy`
      and optionally round each payment up to the next $10.

    strategy:
      - "Proportional" (default): extra spread proportional to starting balances.
      - "Snowball": extra biased toward smaller balances.
      - "Avalanche": extra biased toward higher APRs.

    We then use payoff_stats_fixed() per debt to estimate payoff time & interest.

    Returns:
      {
        "base_total_min": float,
        "base_months": int or None,
        "base_interest": float,
        "scenario_total_payment": float,
        "scenario_months": int or None,
        "scenario_interest": float,
        "per_debt": DataFrame with scenario payments, months, interest,
      }
    """
    import numpy as np

    if debts_df.empty:
        return None

    df = debts_df.copy()
    df["balance"] = df["balance"].astype(float)
    df["apr"] = df["apr"].astype(float)
    df["min_payment"] = df["min_payment"].astype(float)

    # --- Base plan (today's minimums fixed) ---
    base_total_min = float(df["min_payment"].sum())

    base_months_list = []
    base_interest_list = []

    for _, row in df.iterrows():
        m, i = payoff_stats_fixed(row["balance"], row["apr"], row["min_payment"])
        base_months_list.append(m)
        base_interest_list.append(i)

    base_months = max([m for m in base_months_list if m is not None], default=None)
    base_interest = float(sum(i for i in base_interest_list if i is not None))

    # --- Scenario payments (min + strategy-based extra, optional round-up) ---
    balances = df["balance"].to_numpy()
    aprs = df["apr"].to_numpy()

    if strategy == "Snowball":
        # Smaller balances get larger share of extra (inverse balance)
        with np.errstate(divide="ignore"):
            inv_bal = np.where(balances > 0, 1.0 / balances, 0.0)
        weights = inv_bal
    elif strategy == "Avalanche":
        # Debts with higher APR × balance get more weight
        weights = aprs * balances
    else:  # "Proportional"
        weights = balances

    total_weight = float(weights.sum())
    scenario_payments = []

    for bal, min_pmt, w in zip(balances, df["min_payment"], weights):
        if total_weight > 0:
            share = w / total_weight
        else:
            share = 0.0
        per_debt_extra = extra_per_month * share
        p = min_pmt + per_debt_extra
        if round_up_to_10:
            p = math.ceil(p / 10.0) * 10.0
        scenario_payments.append(p)

    df["scenario_payment"] = scenario_payments
    scenario_total_payment = float(df["scenario_payment"].sum())

    scen_months_list = []
    scen_interest_list = []

    for _, row in df.iterrows():
        m, i = payoff_stats_fixed(row["balance"], row["apr"], row["scenario_payment"])
        scen_months_list.append(m)
        scen_interest_list.append(i)

    scenario_months = max([m for m in scen_months_list if m is not None], default=None)
    scenario_interest = float(sum(i for i in scen_interest_list if i is not None))

    df["base_months"] = base_months_list
    df["base_interest"] = base_interest_list
    df["scenario_months"] = scen_months_list
    df["scenario_interest"] = scen_interest_list

    return {
        "base_total_min": base_total_min,
        "base_months": base_months,
        "base_interest": base_interest,
        "scenario_total_payment": scenario_total_payment,
        "scenario_months": scenario_months,
        "scenario_interest": scenario_interest,
        "per_debt": df,
    }

# =============================
# Pages
# =============================
def page_overview(debts_df, expenses_df, onetime_df, invest_df):
    st.header("Budget Overview & Data Entry")

    tabs = st.tabs(
        [
            "💳 Debts",
            "🏠 Living Expenses",
            "💥 One-time Bills",
            "📊 Summary",
            "📋 Itemization",
        ]
    )
    tab_debts, tab_expenses, tab_onetime, tab_summary, tab_details = tabs

    # ---------- Debts ----------
    with tab_debts:
        st.subheader("Debts")

        col_form, col_pay = st.columns([2, 1])

        # ----------------- Add / Update Debt (with selector) -----------------
        with col_form:
            st.markdown("**Add or Edit Debt**")

            # Choose existing or new
            if not debts_df.empty:
                debt_names = list(debts_df["name"].astype(str))
                debt_selector_options = ["New"] + debt_names
                selected_debt = st.selectbox(
                    "Select a debt to load into the form (or choose 'New')",
                    options=debt_selector_options,
                )
            else:
                st.info("No debts yet. Start by adding a new one.")
                selected_debt = "New"

            # Defaults based on selected debt
            if selected_debt != "New" and not debts_df.empty:
                drow = debts_df.loc[debts_df["name"] == selected_debt].iloc[0]
                default_debt_type = drow["debt_type"] if pd.notna(drow["debt_type"]) else ""
                default_name = drow["name"]
                default_balance = float(drow["balance"]) if pd.notna(drow["balance"]) else 0.0
                default_apr = float(drow["apr"]) if pd.notna(drow["apr"]) else 0.0
                default_min = float(drow["min_payment"]) if pd.notna(drow["min_payment"]) else 0.0
                default_due = int(drow["due_day"]) if pd.notna(drow["due_day"]) else 15
            else:
                default_debt_type = ""
                default_name = ""
                default_balance = 0.0
                default_apr = 0.0
                default_min = 0.0
                default_due = 15

            with st.form("add_debt"):
                debt_type = st.text_input(
                    "Debt type (e.g., Credit card, Loan)",
                    value=default_debt_type,
                )
                name = st.text_input(
                    "Debt name (e.g., 'Chase Sapphire')",
                    value=default_name,
                )
                balance = st.number_input(
                    "Current balance",
                    min_value=0.0,
                    step=10.0,
                    value=float(default_balance),
                )
                apr = st.number_input(
                    "APR (%)",
                    min_value=0.0,
                    step=0.1,
                    value=float(default_apr),
                )
                min_payment = st.number_input(
                    "Minimum monthly payment",
                    min_value=0.0,
                    step=5.0,
                    value=float(default_min),
                )
                due_day = st.number_input(
                    "Due day of month (1–31)",
                    min_value=1,
                    max_value=31,
                    value=int(default_due),
                )
                submitted = st.form_submit_button("Save Debt")

            if submitted and name:
                idx_list = debts_df.index[debts_df["name"] == name].tolist()
                if idx_list:
                    idx = idx_list[0]
                    starting_balance = debts_df.at[idx, "starting_balance"]
                else:
                    starting_balance = balance

                row = {
                    "debt_type": debt_type or "Uncategorized",
                    "name": name,
                    "balance": balance,
                    "starting_balance": starting_balance,
                    "apr": apr,
                    "min_payment": min_payment,
                    "due_day": int(due_day),
                }

                if idx_list:
                    debts_df.loc[idx_list[0]] = row
                else:
                    debts_df = pd.concat(
                        [debts_df, pd.DataFrame([row])], ignore_index=True
                    )

                save_df(debts_df, DEBTS_FILE)
                st.success("Debt saved / updated.")
                st.rerun()

            # Optional delete
            if selected_debt != "New" and not debts_df.empty:
                if st.button(f"Delete '{selected_debt}'"):
                    debts_df = debts_df[debts_df["name"] != selected_debt].reset_index(drop=True)
                    save_df(debts_df, DEBTS_FILE)
                    st.success(f"Deleted debt '{selected_debt}'.")
                    st.rerun()

        # ----------------- Payment UI -----------------
        with col_pay:
            st.markdown("**Make a Payment**")
            if debts_df.empty:
                st.info("Add a debt first.")
            else:
                display_labels = debts_df["debt_type"].fillna("") + " - " + debts_df[
                    "name"
                ].astype(str)
                choice = st.selectbox(
                    "Select debt", options=debts_df.index, format_func=lambda i: display_labels.iloc[i]
                )
                payment = st.number_input(
                    "Payment amount", min_value=0.0, step=10.0, value=0.0
                )
                if st.button("Apply Payment"):
                    old_bal = float(debts_df.at[choice, "balance"])
                    new_bal = max(0.0, old_bal - payment)
                    debts_df.at[choice, "balance"] = new_bal
                    save_df(debts_df, DEBTS_FILE)
                    st.success(
                        f"Payment of ${payment:,.2f} applied to {debts_df.at[choice, 'name']} (new balance: ${new_bal:,.2f})."
                    )

        # ----------------- Payoff stats & tables -----------------
        if not debts_df.empty:
            stats_rows = []
            stats_rows_min = []

            today = dt.date.today()

            for _, row in debts_df.iterrows():
                bal = row["balance"]
                apr = row["apr"]
                pmt = row["min_payment"]

                # Scenario 1: fixed payment = current minimum
                m_fixed, i_fixed = payoff_stats_fixed(bal, apr, pmt)
                if m_fixed is None:
                    payoff_fixed_str = "Payment too low"
                else:
                    payoff_fixed_date = today + dt.timedelta(days=30 * m_fixed)
                    payoff_fixed_str = payoff_fixed_date.strftime("%b %Y")

                stats_rows.append((m_fixed, i_fixed, payoff_fixed_str))

                # Scenario 2: card-style "minimum only" behavior
                m_min, i_min = payoff_stats_minimum_style(bal, apr)
                if m_min is None:
                    payoff_min_str = "Never (min too low)"
                else:
                    payoff_min_date = today + dt.timedelta(days=30 * m_min)
                    payoff_min_str = payoff_min_date.strftime("%b %Y")

                stats_rows_min.append((m_min, i_min, payoff_min_str))

            (
                months_fixed_list,
                interest_fixed_list,
                payoff_fixed_dates,
            ) = zip(*stats_rows)
            (
                months_min_list,
                interest_min_list,
                payoff_min_dates,
            ) = zip(*stats_rows_min)

            debts_df_display = debts_df.copy()
            # Fixed-payment metrics (optimistic scenario)
            debts_df_display["fixed_months"] = months_fixed_list
            debts_df_display["fixed_interest"] = interest_fixed_list
            debts_df_display["fixed_payoff_date"] = payoff_fixed_dates

            # Minimum-payment metrics (card disclosure–style scenario)
            debts_df_display["minpay_months"] = months_min_list
            debts_df_display["minpay_interest"] = interest_min_list
            debts_df_display["minpay_payoff_date"] = payoff_min_dates

            # Aggregate base & min stats for later use
            fixed_months_ser = pd.to_numeric(
                debts_df_display["fixed_months"], errors="coerce"
            )
            minpay_months_ser = pd.to_numeric(
                debts_df_display["minpay_months"], errors="coerce"
            )

            base_max_months = fixed_months_ser.max()
            base_max_months = int(base_max_months) if not np.isnan(base_max_months) else None
            minpay_max_months = minpay_months_ser.max()
            minpay_max_months = int(minpay_max_months) if not np.isnan(minpay_max_months) else None

            base_total_interest = float(debts_df_display["fixed_interest"].sum())
            minpay_total_interest = float(debts_df_display["minpay_interest"].sum())
        else:
            debts_df_display = debts_df.copy()
            base_max_months = None
            minpay_max_months = None
            base_total_interest = 0.0
            minpay_total_interest = 0.0

        debt_chart(debts_df)

        if debts_df_display.empty:
            st.info("No debts yet.")
        else:
            # ---------- Scenario A: fixed payment = today's minimum ----------
            st.markdown("#### Scenario A: Keep paying **today's minimum** every month")
            st.caption(
                "Assumes you never let your payment drop below today's minimum. "
                "This is the faster, lower-interest payoff plan."
            )

            fixed_rows = []
            for _, row in debts_df_display.iterrows():
                fixed_rows.append(
                    {
                        "Type": row["debt_type"],
                        "Debt": row["name"],
                        "Balance ($)": format_currency(row["balance"]),
                        "APR (%)": f"{row['apr']:.2f}%" if pd.notna(row["apr"]) else "",
                        "Monthly payment ($)": format_currency(row["min_payment"]),
                        "Months to payoff": format_months_to_ym(row["fixed_months"]),
                        "Total interest ($)": format_currency(row["fixed_interest"]),
                        "Est. payoff date": row["fixed_payoff_date"],
                    }
                )

            total_fixed_balance = float(debts_df_display["balance"].sum())
            total_fixed_min_payment = float(debts_df_display["min_payment"].sum())
            total_fixed_interest = float(debts_df_display["fixed_interest"].sum())

            fixed_rows.append(
                {
                    "Type": "TOTAL",
                    "Debt": "",
                    "Balance ($)": format_currency(total_fixed_balance),
                    "APR (%)": "",
                    "Monthly payment ($)": format_currency(total_fixed_min_payment),
                    "Months to payoff": "",
                    "Total interest ($)": format_currency(total_fixed_interest),
                    "Est. payoff date": "",
                }
            )

            fixed_view = pd.DataFrame(fixed_rows)
            st.dataframe(fixed_view, use_container_width=True)

            st.markdown("---")

            # ---------- Scenario B: pay only the card's minimum ----------
            st.markdown("#### Scenario B: Pay **only the card's minimum** each month")
            st.caption(
                "Approximates the disclosure box on your statements — "
                "the minimum payment shrinks as the balance shrinks, "
                "so payoff takes much longer and costs more interest."
            )

            minpay_rows = []
            for _, row in debts_df_display.iterrows():
                minpay_rows.append(
                    {
                        "Type": row["debt_type"],
                        "Debt": row["name"],
                        "Balance ($)": format_currency(row["balance"]),
                        "APR (%)": f"{row['apr']:.2f}%" if pd.notna(row["apr"]) else "",
                        "Starting minimum ($)": format_currency(row["min_payment"]),
                        "Months to payoff": format_months_to_ym(row["minpay_months"]),
                        "Total interest ($)": format_currency(row["minpay_interest"]),
                        "Est. payoff date": row["minpay_payoff_date"],
                    }
                )

            total_minpay_balance = float(debts_df_display["balance"].sum())
            total_minpay_start_min = float(debts_df_display["min_payment"].sum())
            total_minpay_interest = float(debts_df_display["minpay_interest"].sum())

            minpay_rows.append(
                {
                    "Type": "TOTAL",
                    "Debt": "",
                    "Balance ($)": format_currency(total_minpay_balance),
                    "APR (%)": "",
                    "Starting minimum ($)": format_currency(total_minpay_start_min),
                    "Months to payoff": "",
                    "Total interest ($)": format_currency(total_minpay_interest),
                    "Est. payoff date": "",
                }
            )

            minpay_view = pd.DataFrame(minpay_rows)
            st.dataframe(minpay_view, use_container_width=True)

            # ----- Savings vs minimum payments chip -----
            try:
                interest_saved = minpay_total_interest - base_total_interest
                years_saved = None
                if base_max_months is not None and minpay_max_months is not None:
                    months_saved = minpay_max_months - base_max_months
                    years_saved = months_saved / 12.0

                if interest_saved > 0:
                    msg_interest = f"≈ ${interest_saved:,.0f} less interest"
                else:
                    msg_interest = "no interest savings vs minimums"

                if years_saved is not None and years_saved > 0:
                    msg_time = f"and about {years_saved:,.1f} years faster"
                else:
                    msg_time = ""

                chip_html = f"""
                <div style="
                    margin-top:1rem;
                    display:inline-flex;
                    align-items:center;
                    padding:0.6rem 0.9rem;
                    border-radius:999px;
                    background:linear-gradient(135deg,#facc15,#eab308);
                    color:#000;
                    font-size:0.85rem;
                    font-weight:600;
                    box-shadow:0 8px 20px rgba(0,0,0,0.25);
                ">
                    <span style="margin-right:0.4rem;">🔥</span>
                    <span>
                        Sticking with today's payments could save you {msg_interest}
                        {msg_time} compared to paying only card minimums.
                    </span>
                </div>
                """
                st.markdown(chip_html, unsafe_allow_html=True)
            except Exception:
                pass

            # ---------- Debt Payoff Calculator (extra + round-up) ----------
            st.markdown("---")
            st.markdown("### 🧮 Debt Payoff Calculator")

            calc_col1, calc_col2, calc_col3 = st.columns([1.4, 1, 1])
            with calc_col1:
                extra_per_month = st.number_input(
                    "Extra you could add across all debts each month",
                    min_value=0.0,
                    step=10.0,
                    value=25.0,
                    help="This is on top of today's minimums and will be spread across debts.",
                )
            with calc_col2:
                round_up = st.checkbox(
                    "Round each payment up to next $10",
                    value=False,
                    help="If a payment is $103, it becomes $110, etc.",
                )
                strategy_choice = st.selectbox(
                    "Extra payment strategy",
                    options=[
                        "Proportional to balance",
                        "Snowball (smallest balance first)",
                        "Avalanche (highest APR first)",
                    ],
                )
                if strategy_choice.startswith("Snowball"):
                    strategy_key_calc = "Snowball"
                elif strategy_choice.startswith("Avalanche"):
                    strategy_key_calc = "Avalanche"
                else:
                    strategy_key_calc = "Proportional"
            with calc_col3:
                st.caption(
                    "This calculator uses a 'fixed payment' plan based on today's minimums, "
                    "then layers in your extra and optional rounding using the chosen strategy."
                )

            if st.button("Run payoff scenario"):
                scenario = compute_portfolio_scenario_fixed(
                    debts_df,
                    extra_per_month=extra_per_month,
                    round_up_to_10=round_up,
                    strategy=strategy_key_calc,
                )

                if scenario is None:
                    st.warning("No debts to simulate.")
                else:
                    base_total_min = scenario["base_total_min"]
                    base_months = scenario["base_months"]
                    base_interest = scenario["base_interest"]
                    scen_total_pmt = scenario["scenario_total_payment"]
                    scen_months = scenario["scenario_months"]
                    scen_interest = scenario["scenario_interest"]
                    per_debt_df = scenario["per_debt"]

                    interest_delta = base_interest - scen_interest
                    months_delta = (base_months - scen_months) if (base_months and scen_months) else None

                    helper_cols = st.columns(3)
                    helper_cols[0].metric(
                        "Base minimums (today's plan)",
                        f"${base_total_min:,.2f}/mo",
                    )
                    helper_cols[1].metric(
                        "Scenario payment total",
                        f"${scen_total_pmt:,.2f}/mo",
                        delta=f"${scen_total_pmt - base_total_min:,.2f}/mo",
                    )
                    helper_cols[2].metric(
                        "Interest saved (scenario vs base)",
                        f"${interest_delta:,.0f}",
                        delta=f"-{months_delta} months" if months_delta and months_delta > 0 else None,
                    )

                    # Little helper suggestion guy
                    try:
                        per_debt_df["weight_cost"] = per_debt_df["apr"] * per_debt_df["balance"]
                        target = per_debt_df.sort_values("weight_cost", ascending=False).iloc[0]
                        helper_name = target["name"]
                        helper_pmt = target["scenario_payment"]
                        helper_html = f"""
                        <div style="
                            margin-top:1rem;
                            padding:0.9rem 1rem;
                            border-radius:0.9rem;
                            background:rgba(15,23,42,0.95);
                            border:1px solid #4b5563;
                            color:#e5e7eb;
                            display:flex;
                            align-items:flex-start;
                            gap:0.7rem;
                            box-shadow:0 10px 30px rgba(0,0,0,0.5);
                        ">
                            <div style="font-size:1.8rem;">🤖</div>
                            <div style="font-size:0.9rem;line-height:1.4;">
                                <strong>Helper tip:</strong><br/>
                                With the {strategy_key_calc.lower()} strategy, your extra is leaning toward
                                <strong>{helper_name}</strong> right now.<br/>
                                In this scenario, I'm modeling it as paying about
                                <strong>{format_currency(helper_pmt)}</strong> per month toward it.
                            </div>
                        </div>
                        """
                        st.markdown(helper_html, unsafe_allow_html=True)
                    except Exception:
                        pass

                    # Per-debt scenario table + totals row
                    with st.expander("See per-debt scenario details"):
                        show_cols = [
                            "debt_type",
                            "name",
                            "balance",
                            "apr",
                            "min_payment",
                            "scenario_payment",
                            "base_months",
                            "base_interest",
                            "scenario_months",
                            "scenario_interest",
                        ]
                        view = per_debt_df[show_cols].copy()
                        view["balance"] = view["balance"].apply(format_currency)
                        view["min_payment"] = view["min_payment"].apply(format_currency)
                        view["scenario_payment"] = view["scenario_payment"].apply(format_currency)
                        view["base_interest"] = view["base_interest"].apply(format_currency)
                        view["scenario_interest"] = view["scenario_interest"].apply(format_currency)
                        view["base_months"] = view["base_months"].apply(format_months_to_ym)
                        view["scenario_months"] = view["scenario_months"].apply(format_months_to_ym)

                        # Totals row for payments & interest
                        totals_row = {
                            "debt_type": "TOTAL",
                            "name": "",
                            "balance": format_currency(per_debt_df["balance"].sum()),
                            "apr": "",
                            "min_payment": format_currency(per_debt_df["min_payment"].sum()),
                            "scenario_payment": format_currency(per_debt_df["scenario_payment"].sum()),
                            "base_months": "",
                            "base_interest": format_currency(
                                sum(i for i in per_debt_df["base_interest"] if pd.notna(i))
                            ),
                            "scenario_months": "",
                            "scenario_interest": format_currency(
                                sum(i for i in per_debt_df["scenario_interest"] if pd.notna(i))
                            ),
                        }
                        view = pd.concat([view, pd.DataFrame([totals_row])], ignore_index=True)

                        st.dataframe(view, use_container_width=True)

    # ---------- Living Expenses ----------
    with tab_expenses:
        st.subheader("Living Expenses")

        st.markdown("**Add or Edit Expense**")

        # --- Choose an existing expense to edit (or add new) ---
        if not expenses_df.empty:
            expense_names = list(expenses_df["name"].astype(str))
            selector_options = ["New"] + expense_names
            selected_label = st.selectbox(
                "Select an expense to load into the form (or choose 'New')",
                options=selector_options,
            )
        else:
            st.info("No expenses yet. Start by adding a new one.")
            selected_label = "New"

        # Determine default values for the form based on selection
        if selected_label != "New" and not expenses_df.empty:
            row = expenses_df.loc[expenses_df["name"] == selected_label].iloc[0]
            default_name = row["name"]
            default_amount = float(row["amount"]) if pd.notna(row["amount"]) else 0.0
            default_freq = row["frequency"] if pd.notna(row["frequency"]) else "Monthly"
            default_subcat = row["subcategory"] if pd.notna(row["subcategory"]) else ""
            default_due = int(row["due_day"]) if pd.notna(row["due_day"]) else 1
        else:
            default_name = ""
            default_amount = 0.0
            default_freq = "Monthly"
            default_subcat = ""
            default_due = 1

        # --- Form: add or update expense ---
        with st.form("add_expense"):
            name = st.text_input("Expense name (e.g., Rent)", value=default_name)

            amount = st.number_input(
                "Amount",
                min_value=0.0,
                step=10.0,
                value=float(default_amount),
            )

            freq_options = ["Monthly", "Weekly"]
            if default_freq in freq_options:
                freq_index = freq_options.index(default_freq)
            else:
                freq_index = 0

            frequency = st.selectbox(
                "Frequency",
                freq_options,
                index=freq_index,
            )

            subcategory = st.text_input(
                "Subcategory (e.g., Housing, Utilities)",
                value=default_subcat,
            )

            due_day = st.number_input(
                "Due day of month (1–31 for monthly, any for weekly)",
                min_value=1,
                max_value=31,
                value=int(default_due),
            )

            submitted = st.form_submit_button("Save Expense")

        if submitted and name:
            idx_list = expenses_df.index[expenses_df["name"] == name].tolist()
            row = {
                "name": name,
                "amount": amount,
                "frequency": frequency,
                "subcategory": subcategory or "Uncategorized",
                "due_day": int(due_day),
            }
            if idx_list:
                expenses_df.loc[idx_list[0]] = row
            else:
                expenses_df = pd.concat(
                    [expenses_df, pd.DataFrame([row])],
                    ignore_index=True,
                )

            save_df(expenses_df, EXPENSES_FILE)
            st.success("Expense saved / updated.")
            st.rerun()

        # Optional delete
        if selected_label != "New" and not expenses_df.empty:
            if st.button(f"Delete '{selected_label}'"):
                expenses_df = expenses_df[expenses_df["name"] != selected_label].reset_index(drop=True)
                save_df(expenses_df, EXPENSES_FILE)
                st.success(f"Deleted expense '{selected_label}'.")
                st.rerun()

        expenses_chart(expenses_df)
        st.markdown("**Current expenses**")
        st.dataframe(expenses_df, use_container_width=True)

    # ---------- One-time Bills ----------
    with tab_onetime:
        st.subheader("One-time Bills")

        st.markdown("**Add or Edit One-time Bill**")

        if not onetime_df.empty:
            bill_names = list(onetime_df["name"].astype(str))
            bill_selector_options = ["New"] + bill_names
            selected_bill = st.selectbox(
                "Select a bill to load into the form (or choose 'New')",
                options=bill_selector_options,
            )
        else:
            st.info("No one-time bills yet. Start by adding a new one.")
            selected_bill = "New"

        if selected_bill != "New" and not onetime_df.empty:
            orow = onetime_df.loc[onetime_df["name"] == selected_bill].iloc[0]
            default_name = orow["name"]
            default_amount = float(orow["amount"]) if pd.notna(orow["amount"]) else 0.0

            if pd.notna(orow["due_date"]):
                try:
                    default_due_date = pd.to_datetime(orow["due_date"]).date()
                except Exception:
                    default_due_date = dt.date.today()
            else:
                default_due_date = dt.date.today()
        else:
            default_name = ""
            default_amount = 0.0
            default_due_date = dt.date.today()

        with st.form("add_onetime"):
            name = st.text_input("Bill name (e.g., Car repair)", value=default_name)
            amount = st.number_input(
                "Amount",
                min_value=0.0,
                step=10.0,
                value=float(default_amount),
            )
            due_date = st.date_input("Due date", default_due_date)

            submitted = st.form_submit_button("Save Bill")

        if submitted and name:
            idx_list = onetime_df.index[onetime_df["name"] == name].tolist()
            row = {
                "name": name,
                "amount": amount,
                "due_date": str(due_date),
            }
            if idx_list:
                onetime_df.loc[idx_list[0]] = row
            else:
                onetime_df = pd.concat(
                    [onetime_df, pd.DataFrame([row])], ignore_index=True
                )
            save_df(onetime_df, ONETIME_FILE)
            st.success("Bill saved / updated.")
            st.rerun()

        if selected_bill != "New" and not onetime_df.empty:
            if st.button(f"Delete '{selected_bill}'"):
                onetime_df = onetime_df[onetime_df["name"] != selected_bill].reset_index(drop=True)
                save_df(onetime_df, ONETIME_FILE)
                st.success(f"Deleted bill '{selected_bill}'.")
                st.rerun()

        onetime_chart(onetime_df)
        st.dataframe(onetime_df, use_container_width=True)

    # ---------- Summary with cards ----------
    with tab_summary:
        # First row: controls inline near title
        top_col1, top_col2, top_col3 = st.columns([2, 1, 1])
        with top_col1:
            st.markdown("### Summary & Progress")
        with top_col2:
            planned_income_6m = st.number_input(
                "Planned income over next 6 months",
                min_value=0.0,
                step=500.0,
                value=22000.0,
            )
        with top_col3:
            emergency_goal = st.number_input(
                "Emergency fund goal",
                min_value=0.0,
                step=500.0,
                value=3000.0,
            )

        st.markdown("#### Progress Cards")

        # Totals
        total_debt = float(debts_df["balance"].sum()) if not debts_df.empty else 0.0
        if not debts_df.empty and "starting_balance" in debts_df.columns:
            starting_total_debt = float(debts_df["starting_balance"].sum())
        else:
            starting_total_debt = total_debt

        if not expenses_df.empty:
            monthly_exp = expenses_df.copy()
            monthly_exp["monthly_amount"] = np.where(
                monthly_exp["frequency"] == "Weekly",
                monthly_exp["amount"] * 4.33,
                monthly_exp["amount"],
            )
            total_monthly_expenses = float(monthly_exp["monthly_amount"].sum())
        else:
            total_monthly_expenses = 0.0

        total_onetime = float(onetime_df["amount"].sum()) if not onetime_df.empty else 0.0
        six_month_need = total_monthly_expenses * 6 + total_onetime

        card_col1, card_col2, card_col3 = st.columns(3)

        with card_col1:
            progress_card(
                "Total Debt",
                "Goal: $0 (progress from starting total)",
                current=total_debt,
                start_or_goal=starting_total_debt,
                positive_good=True,
                color="#ef4444",
            )

        with card_col2:
            progress_card(
                "6-month Expense Need",
                "Need = 6×monthly expenses + one-time bills",
                current=six_month_need,
                start_or_goal=planned_income_6m,
                positive_good=False,
                color="#3b82f6",
            )

        with card_col3:
            current_emergency = 0.0  # placeholder
            progress_card(
                "Emergency Fund (placeholder)",
                "Current vs goal (manual tracking)",
                current=current_emergency,
                start_or_goal=emergency_goal,
                positive_good=False,
                color="#22c55e",
            )

        st.markdown("#### Category Breakdown")

        summary_df = pd.DataFrame(
            {
                "Category": ["Debt", "Living (monthly equiv.)", "One-time"],
                "Amount": [total_debt, total_monthly_expenses, total_onetime],
            }
        )
        fig = px.pie(summary_df, names="Category", values="Amount", title="Overall Mix")
        st.plotly_chart(fig, use_container_width=True)

        st.markdown("#### Key Metrics")
        st.metric("Total Debt", f"${total_debt:,.2f}")
        st.metric("Total Monthly Living Expenses", f"${total_monthly_expenses:,.2f}")
        st.metric("Total One-time Bills", f"${total_onetime:,.2f}")

    # ---------- Itemization ----------
    with tab_details:
        st.subheader("Itemization")

        item_tabs = st.tabs(
            ["Debts", "Living Expenses", "One-time Bills", "Investments"]
        )

        # Debts
        with item_tabs[0]:
            if debts_df.empty:
                st.info("No debts yet.")
            else:
                st.markdown("**Debts**")
                st.dataframe(debts_df)

        # Living
        with item_tabs[1]:
            if expenses_df.empty:
                st.info("No living expenses yet.")
            else:
                st.markdown("**Living expenses (with subcategories)**")
                st.dataframe(expenses_df)

        # One-time
        with item_tabs[2]:
            if onetime_df.empty:
                st.info("No one-time bills yet.")
            else:
                st.markdown("**One-time bills**")
                st.dataframe(onetime_df)

        # Investments
        with item_tabs[3]:
            if invest_df.empty:
                st.info("No investments yet. Add them on the Investment Planner page.")
            else:
                st.markdown("**Investments**")
                st.dataframe(invest_df)

def page_debt_snowball(debts_df, expenses_df, onetime_df):
    import plotly.express as px
    import streamlit as st
    import datetime as dt
    import numpy as np
    import pandas as pd

    st.header("Debt Payoff Simulator (Snowball / Avalanche)")

    if debts_df.empty:
        st.info("Add some debts on the Overview page first.")
        return

    top = st.columns(2)
    with top[0]:
        total_budget = st.number_input(
            "Total cash available over entire period ($)",
            min_value=0.0,
            value=22000.0,
            step=500.0,
        )
        months = st.number_input(
            "Period length (months)", min_value=1, max_value=60, value=6, step=1
        )
    with top[1]:
        strategy_label = st.selectbox(
            "Strategy",
            ["Snowball (smallest balance first)", "Avalanche (highest APR first)"],
        )
        strategy_key = "Snowball" if strategy_label.startswith("Snowball") else "Avalanche"

        st.caption(
            "We spread that total evenly across the period, subtract estimated "
            "living expenses and one-time bills, then apply the remainder to debts."
        )

    # Monthly living expenses
    if not expenses_df.empty:
        monthly_exp = expenses_df.copy()
        monthly_exp["monthly_amount"] = np.where(
            monthly_exp["frequency"] == "Weekly",
            monthly_exp["amount"] * 4.33,
            monthly_exp["amount"],
        )
        total_monthly_expenses = float(monthly_exp["monthly_amount"].sum())
    else:
        total_monthly_expenses = 0.0

    # One-time within horizon
    if not onetime_df.empty:
        horizon_end = dt.date.today() + dt.timedelta(days=int(months * 30))
        temp = onetime_df.copy()
        temp["due_date"] = pd.to_datetime(temp["due_date"]).dt.date
        within = temp[temp["due_date"] <= horizon_end]
        total_onetime = float(within["amount"].sum())
    else:
        total_onetime = 0.0

    monthly_budget = total_budget / months
    monthly_for_debt = monthly_budget - total_monthly_expenses - (total_onetime / months)

    info_cols = st.columns(4)
    info_cols[0].metric("Monthly budget", f"${monthly_budget:,.2f}")
    info_cols[1].metric("Monthly living expenses", f"${total_monthly_expenses:,.2f}")
    info_cols[2].metric(
        "One-time (avg per month)", f"${(total_onetime / months) if months else 0:,.2f}"
    )
    info_cols[3].metric("Available for debts / month", f"${monthly_for_debt:,.2f}")

    if monthly_for_debt <= 0:
        st.error(
            "Budget is not enough to cover living expenses and one-time bills. "
            "Adjust inputs or horizon."
        )
        return

    total_min_payments = float(debts_df["min_payment"].sum())
    st.markdown(
        f"**Total minimum payments:** ${total_min_payments:,.2f}  "
        + (
            "(below available amount ✅)"
            if total_min_payments <= monthly_for_debt
            else "(above available amount ⚠️ — minimums scaled proportionally)"
        )
    )

    # --- Main plan: use ALL available debt budget + chosen strategy ---
    sim_plan = simulate_debt_payoff(
        debts_df,
        monthly_for_debt,
        months,
        strategy=strategy_key,
        allocate_extra=True,
        track_history=True,
        schedule_start_date=dt.date.today(),
    )

    final_df = sim_plan["final_df"].copy()
    final_df["balance"] = final_df["balance"].clip(lower=0.0)
    remaining_total = float(final_df["balance"].sum())
    leftover = sim_plan["leftover_total"]

    # --- Comparison: "minimums only" (no extra sent to debts) over this horizon ---
    sim_min_only = simulate_debt_payoff(
        debts_df,
        monthly_for_debt,
        months,
        strategy=strategy_key,     # irrelevant here; no extra is used
        allocate_extra=False,
        track_history=False,
    )

    st.subheader("Results")

    result_cols = st.columns(3)
    result_cols[0].metric("Remaining total debt", f"${remaining_total:,.2f}")
    result_cols[1].metric("Estimated leftover unallocated cash", f"${leftover:,.2f}")
    result_cols[2].metric(
        "Debts fully paid off",
        f"{int((final_df['balance'] <= 0).sum())} / {len(final_df)}",
    )

    st.markdown("**Debt balances after simulation**")
    st.dataframe(final_df)

    st.markdown("**Debt distribution after simulation**")
    fig_bal = px.bar(
        final_df,
        x="name",
        y="balance",
        color="debt_type",
        labels={"name": "Debt", "balance": "Balance ($)", "debt_type": "Type"},
        title="Balances by Debt after Simulation",
    )
    st.plotly_chart(fig_bal, use_container_width=True)

    # ---------- Horizon Interest & Principal Breakdown ----------
    st.markdown(f"### {int(months)}-month Interest & Principal Breakdown")

    plan_interest = sim_plan["total_interest"]
    plan_paid = sim_plan["total_paid"]
    plan_principal = sim_plan["total_principal"]

    min_interest = sim_min_only["total_interest"]
    min_paid = sim_min_only["total_paid"]
    min_principal = sim_min_only["total_principal"]

    interest_saved = max(0.0, min_interest - plan_interest)
    extra_principal = max(0.0, plan_principal - min_principal)

    cols_top = st.columns(3)
    cols_top[0].metric(
        "Total paid to debts (this plan)",
        f"${plan_paid:,.2f}",
    )
    cols_top[1].metric(
        "Interest paid (this plan)",
        f"${plan_interest:,.2f}",
    )
    cols_top[2].metric(
        "Principal reduced (this plan)",
        f"${plan_principal:,.2f}",
    )

    st.markdown("#### Compared to paying only minimums over this period")

    cols_cmp = st.columns(3)
    cols_cmp[0].metric(
        "Interest paid with minimums only",
        f"${min_interest:,.2f}",
    )
    cols_cmp[1].metric(
        "Interest avoided by using this plan",
        f"${interest_saved:,.2f}",
    )
    cols_cmp[2].metric(
        "Extra principal paid vs minimums",
        f"${extra_principal:,.2f}",
    )

    explainer_html = f"""
    <div style="
        margin-top:1rem;
        padding:0.9rem 1.1rem;
        border-radius:999px;
        background:linear-gradient(135deg,#facc15,#eab308);
        color:#1f2933;
        font-size:0.85rem;
        font-weight:600;
        display:inline-flex;
        align-items:center;
        box-shadow:0 10px 25px rgba(0,0,0,0.4);
    ">
        <span style="margin-right:0.5rem;">✅</span>
        <span>
            Over these {int(months)} months, using your full debt budget with
            the {strategy_key.lower()} strategy sends about
            <strong>${extra_principal:,.0f}</strong> more to principal and avoids roughly
            <strong>${interest_saved:,.0f}</strong> in interest compared to paying only minimums.
        </span>
    </div>
    """
    st.markdown(explainer_html, unsafe_allow_html=True)

    # ---------- Lifetime impact: minimums forever vs this plan for N months ----------
    st.markdown("### Lifetime impact of this plan")

    base_months_list = []
    base_interest_list = []
    for _, row in debts_df.iterrows():
        m0, i0 = payoff_stats_minimum_style(row["balance"], row["apr"])
        base_months_list.append(m0)
        base_interest_list.append(i0)

    if any(m is None for m in base_months_list):
        st.warning(
            "At least one debt would never be paid off using only minimum payments. "
            "Lifetime comparison is not meaningful in that case."
        )
    else:
        base_lifetime_months = max(base_months_list) if base_months_list else 0
        base_lifetime_interest = float(sum(base_interest_list))

        scen_extra_months_list = []
        scen_extra_interest_list = []
        for _, row in final_df.iterrows():
            bal = row["balance"]
            if bal <= 0:
                scen_extra_months_list.append(0)
                scen_extra_interest_list.append(0.0)
            else:
                m1, i1 = payoff_stats_minimum_style(bal, row["apr"])
                if m1 is None:
                    scen_extra_months_list.append(None)
                    scen_extra_interest_list.append(None)
                else:
                    scen_extra_months_list.append(m1)
                    scen_extra_interest_list.append(i1)

        if any(m is None for m in scen_extra_months_list):
            st.warning(
                "After this plan, at least one debt still wouldn't pay off with minimums only. "
                "Lifetime comparison is approximate."
            )
        else:
            scen_lifetime_months = int(months + max(scen_extra_months_list))
            scen_lifetime_interest = float(
                sim_plan["total_interest"] + sum(scen_extra_interest_list)
            )

            lifetime_interest_saved = max(0.0, base_lifetime_interest - scen_lifetime_interest)
            lifetime_months_saved = max(0, base_lifetime_months - scen_lifetime_months)

            cols_life = st.columns(3)
            cols_life[0].metric(
                "Lifetime interest (minimums only)",
                f"${base_lifetime_interest:,.0f}",
            )
            cols_life[1].metric(
                "Lifetime interest (this plan for "
                f"{int(months)} mo, then minimums)",
                f"${scen_lifetime_interest:,.0f}",
            )
            cols_life[2].metric(
                "Lifetime impact",
                f"${lifetime_interest_saved:,.0f} interest avoided",
                delta=f"-{lifetime_months_saved} months to debt-free"
                if lifetime_months_saved > 0
                else None,
            )

    # ---------- Trajectory chart (total balance over time) ----------
    st.markdown("### Debt payoff trajectory")

    schedule_df = sim_plan["schedule"]
    if schedule_df is not None and not schedule_df.empty:
        schedule_df = schedule_df.copy()
        total_balance_by_month = schedule_df.groupby("month")["balance_after"].sum()
        traj_df = total_balance_by_month.reset_index()
        traj_df.columns = ["Month", "Total balance"]

        fig_traj = px.line(
            traj_df,
            x="Month",
            y="Total balance",
            markers=True,
            labels={"Month": "Month", "Total balance": "Total balance ($)"},
            title="Total debt balance over simulation horizon",
        )
        st.plotly_chart(fig_traj, use_container_width=True)

        # ---------- Calendar-style payment planner ----------
        st.markdown("### Payment calendar")

        schedule_df["date"] = pd.to_datetime(schedule_df["date"])
        schedule_df["MonthPeriod"] = schedule_df["date"].dt.to_period("M")

        unique_periods = schedule_df["MonthPeriod"].sort_values().unique()
        period_labels = [p.strftime("%b %Y") for p in unique_periods]
        selected_label = st.selectbox(
            "Select month to view",
            options=list(range(len(unique_periods))),
            format_func=lambda i: period_labels[i],
        )
        selected_period = unique_periods[selected_label]

        month_view = schedule_df[schedule_df["MonthPeriod"] == selected_period].copy()
        month_view = month_view.sort_values(["date", "debt_name"])

        for col in ["payment", "min_component", "extra_component", "interest", "principal", "balance_after"]:
            month_view[col] = month_view[col].apply(format_currency)

        st.markdown(
            "This calendar shows each payment for the selected month, including "
            "how much is minimum vs extra, and the balance right after that month's payment."
        )
        st.dataframe(
            month_view[
                [
                    "date",
                    "debt_name",
                    "debt_type",
                    "payment",
                    "min_component",
                    "extra_component",
                    "interest",
                    "principal",
                    "balance_after",
                ]
            ],
            use_container_width=True,
        )
    else:
        st.info("No schedule available for this simulation.")

def page_investment(invest_df):
    st.header("Investment Planner & Portfolio")

    # ----- Portfolio entry -----
    st.subheader("Portfolio Holdings")

    with st.form("add_investment"):
        col1, col2 = st.columns(2)
        with col1:
            ticker = st.text_input("Ticker (e.g., VTI)")
            name = st.text_input("Name (e.g., Vanguard Total Market)")
            category = st.text_input("Category (Core, Income, Growth, etc.)")
            industry = st.text_input("Industry / Sector (Tech, REIT, etc.)")
        with col2:
            yield_pct = st.number_input(
                "Annual dividend yield (%)", min_value=0.0, step=0.1, value=3.0
            )
            amount = st.number_input(
                "Amount invested ($)", min_value=0.0, step=100.0, value=1000.0
            )

        submitted = st.form_submit_button("Save Holding")

    if submitted and ticker:
        idx_list = invest_df.index[invest_df["ticker"] == ticker].tolist()
        row = {
            "ticker": ticker.upper(),
            "name": name or ticker.upper(),
            "category": category or "Uncategorized",
            "industry": industry or "Unspecified",
            "yield_pct": yield_pct,
            "amount": amount,
        }
        if idx_list:
            invest_df.loc[idx_list[0]] = row
        else:
            invest_df = pd.concat(
                [invest_df, pd.DataFrame([row])], ignore_index=True
            )
        save_df(invest_df, INVEST_FILE)
        st.success("Holding saved / updated.")

    if invest_df.empty:
        st.info("No holdings yet. Add some above.")
        return

    # ----- Portfolio stats -----
    df = invest_df.copy()
    df["amount"] = df["amount"].astype(float)
    total_invested = float(df["amount"].sum())
    df["weight_pct"] = df["amount"] / total_invested * 100.0
    df["monthly_dividend"] = df["amount"] * (df["yield_pct"] / 100.0) / 12.0

    weighted_yield = float(
        (df["yield_pct"] * df["amount"]).sum() / total_invested
    ) if total_invested > 0 else 0.0
    monthly_dividends_total = float(df["monthly_dividend"].sum())

    stat_cols = st.columns(3)
    stat_cols[0].metric("Total invested", f"${total_invested:,.2f}")
    stat_cols[1].metric(
        "Weighted annual yield", f"{weighted_yield:,.2f}%"
    )
    stat_cols[2].metric(
        "Expected monthly dividends", f"${monthly_dividends_total:,.2f}"
    )

    st.markdown("**Holdings with weights & dividends**")
    st.dataframe(df)

    # Exposure by category / industry
    exp_tabs = st.tabs(["By Category", "By Industry"])
    with exp_tabs[0]:
        cat = df.groupby("category", as_index=False)["amount"].sum()
        cat["weight_pct"] = cat["amount"] / total_invested * 100.0
        fig = px.bar(
            cat,
            x="category",
            y="weight_pct",
            labels={"category": "Category", "weight_pct": "Portfolio %"},
            title="Exposure by Category",
        )
        st.plotly_chart(fig, use_container_width=True)
    with exp_tabs[1]:
        ind = df.groupby("industry", as_index=False)["amount"].sum()
        ind["weight_pct"] = ind["amount"] / total_invested * 100.0
        fig = px.bar(
            ind,
            x="industry",
            y="weight_pct",
            labels={"industry": "Industry", "weight_pct": "Portfolio %"},
            title="Exposure by Industry",
        )
        st.plotly_chart(fig, use_container_width=True)

    # ----- Rebalancing scenario -----
    st.subheader("Rebalancing Scenario")

    st.caption(
        "We start with an even-weight scenario, which you can edit. "
        "If you apply the scenario, the portfolio amounts will be updated."
    )

    scenario_df = pd.DataFrame(
        {
            "ticker": df["ticker"],
            "current_weight_pct": df["weight_pct"],
            "target_weight_pct": 100.0 / len(df),
        }
    )

    edited = st.data_editor(
        scenario_df,
        num_rows="dynamic",
        key="rebalance_editor",
    )

    target_sum = float(edited["target_weight_pct"].sum())
    if abs(target_sum - 100.0) > 1e-3:
        st.warning(
            f"Target weights sum to {target_sum:.2f}%. "
            "You may want them to total 100%."
        )

    # Preview new amounts
    preview = edited.copy()
    preview["new_amount"] = preview["target_weight_pct"] / 100.0 * total_invested
    st.markdown("**Preview of rebalanced amounts (same total invested)**")
    st.dataframe(preview)

    if st.button("Apply scenario to portfolio"):
        # Map new amounts back
        new_amounts = dict(zip(preview["ticker"], preview["new_amount"]))
        for i, row in invest_df.iterrows():
            t = row["ticker"]
            if t in new_amounts:
                invest_df.at[i, "amount"] = new_amounts[t]
        save_df(invest_df, INVEST_FILE)
        st.success("Scenario applied. Portfolio amounts updated.")

    # ----- Projection (line chart) -----
    st.subheader("Portfolio Projection (24 months)")

    col_proj1, col_proj2 = st.columns(2)
    with col_proj1:
        monthly_contrib = st.number_input(
            "Additional monthly contribution",
            min_value=0.0,
            step=50.0,
            value=300.0,
        )
    with col_proj2:
        projection_years = st.slider(
            "Projection horizon (years)", min_value=1, max_value=3, value=2
        )
    months = projection_years * 12

    history_df = simulate_portfolio_growth(
        total_invested, weighted_yield, monthly_contrib, months
    )
    portfolio_projection_chart(history_df)

def page_financial_projection(debts_df, expenses_df, onetime_df, invest_df):
    st.header("12-Month Financial Projection (Debt → Freedom → Build)")

    # -------- Phase settings --------
    col_phase1, col_phase2 = st.columns(2)
    with col_phase1:
        phase1_months = st.number_input(
            "Phase 1: months focused on debt payoff / unemployment",
            min_value=1,
            max_value=36,
            value=6,
            step=1,
        )
        phase1_income = st.number_input(
            "Phase 1 monthly income (after tax, e.g. unemployment, side income)",
            min_value=0.0,
            value=3300.0,
            step=100.0,
        )
    with col_phase2:
        phase2_months = st.number_input(
            "Phase 2: months after new job (growth mode)",
            min_value=0,
            max_value=60,
            value=6,
            step=1,
        )
        phase2_income = st.number_input(
            "Phase 2 monthly income (after tax, e.g. software engineer salary)",
            min_value=0.0,
            value=6500.0,
            step=100.0,
        )

    total_months = phase1_months + phase2_months

    # -------- Baseline living costs (reuse your logic) --------
    if not expenses_df.empty:
        monthly_exp = expenses_df.copy()
        monthly_exp["monthly_amount"] = np.where(
            monthly_exp["frequency"] == "Weekly",
            monthly_exp["amount"] * 4.33,
            monthly_exp["amount"],
        )
        base_monthly_expenses = float(monthly_exp["monthly_amount"].sum())
    else:
        base_monthly_expenses = 0.0

    if not onetime_df.empty and total_months > 0:
        # Amortize one-time bills across the whole projection horizon
        try:
            temp = onetime_df.copy()
            temp["amount"] = temp["amount"].astype(float)
            total_onetime = float(temp["amount"].sum())
        except Exception:
            total_onetime = 0.0
        monthly_onetime = total_onetime / float(total_months)
    else:
        monthly_onetime = 0.0

    base_cols = st.columns(3)
    base_cols[0].metric("Baseline monthly expenses", f"${base_monthly_expenses:,.2f}")
    base_cols[1].metric("Avg monthly one-time allocation", f"${monthly_onetime:,.2f}")
    base_cols[2].metric("Projection length", f"{int(total_months)} months")

    st.markdown("---")

    # -------- Fund goals / caps --------
    goal_col1, goal_col2, goal_col3 = st.columns(3)
    with goal_col1:
        emergency_goal = st.number_input(
            "Emergency fund goal (3–6 months expenses)",
            min_value=0.0,
            value=6000.0,
            step=500.0,
        )
    with goal_col2:
        rainy_cap = st.number_input(
            "Rainy-day fund soft cap",
            min_value=0.0,
            value=3000.0,
            step=250.0,
        )
    with goal_col3:
        savings_cap = st.number_input(
            "Savings fund soft cap",
            min_value=0.0,
            value=3000.0,
            step=250.0,
        )

    st.caption(
        "Rainy-day and Savings caps act as soft ceilings — once reached, "
        "extra dollars that *would* go there get redirected into Investments instead."
    )

    # -------- Allocation sliders --------
    st.markdown("### Phase 1 allocation (while paying off debt)")

    p1c = st.columns(5)
    with p1c[0]:
        p1_debt = st.slider("Debt extra %", 0, 100, 70, step=5)
    with p1c[1]:
        p1_em = st.slider("Emergency %", 0, 100, 10, step=5)
    with p1c[2]:
        p1_rain = st.slider("Rainy-day %", 0, 100, 5, step=5)
    with p1c[3]:
        p1_sav = st.slider("Savings %", 0, 100, 5, step=5)
    with p1c[4]:
        p1_inv = st.slider("Investments %", 0, 100, 10, step=5)

    p1_sum = p1_debt + p1_em + p1_rain + p1_sav + p1_inv
    if p1_sum == 0:
        st.error("Phase 1 allocations must sum to more than 0%.")
        return
    p1_weights = {
        "debt": p1_debt / p1_sum,
        "emergency": p1_em / p1_sum,
        "rainy": p1_rain / p1_sum,
        "savings": p1_sav / p1_sum,
        "invest": p1_inv / p1_sum,
    }
    if p1_sum != 100:
        st.warning(f"Phase 1 sliders sum to {p1_sum}%. Normalizing internally to 100%.")

    st.markdown("### Phase 2 allocation (after you’re working)")

    p2c = st.columns(5)
    with p2c[0]:
        p2_debt = st.slider("Debt % (likely 0)", 0, 100, 0, step=5)
    with p2c[1]:
        p2_em = st.slider("Emergency %", 0, 100, 20, step=5)
    with p2c[2]:
        p2_rain = st.slider("Rainy-day %", 0, 100, 10, step=5)
    with p2c[3]:
        p2_sav = st.slider("Savings %", 0, 100, 20, step=5)
    with p2c[4]:
        p2_inv = st.slider("Investments %", 0, 100, 50, step=5)

    p2_sum = p2_debt + p2_em + p2_rain + p2_sav + p2_inv
    if p2_sum == 0 and phase2_months > 0:
        st.error("Phase 2 allocations must sum to more than 0% if Phase 2 months > 0.")
        return
    p2_weights = {
        "debt": (p2_debt / p2_sum) if p2_sum > 0 else 0.0,
        "emergency": (p2_em / p2_sum) if p2_sum > 0 else 0.0,
        "rainy": (p2_rain / p2_sum) if p2_sum > 0 else 0.0,
        "savings": (p2_sav / p2_sum) if p2_sum > 0 else 0.0,
        "invest": (p2_inv / p2_sum) if p2_sum > 0 else 0.0,
    }
    if p2_sum != 100 and phase2_months > 0:
        st.warning(f"Phase 2 sliders sum to {p2_sum}%. Normalizing internally to 100%.")

    st.markdown("### Investment assumptions")
    col_yield1, col_yield2 = st.columns(2)
    with col_yield1:
        expected_yield = st.number_input(
            "Expected portfolio yield (%)",
            min_value=0.0,
            max_value=20.0,
            value=5.0,
            step=0.5,
            help="Use ~4–6% for a blended dividend-focused, relatively conservative portfolio.",
        )
    with col_yield2:
        reinvest_dividends = st.checkbox(
            "Reinvest dividends into the portfolio",
            value=True,
        )

    if st.button("Run 12-month projection"):
        if total_months <= 0:
            st.error("Total months must be > 0.")
            return

        monthly_yield = expected_yield / 100.0 / 12.0

        # State variables
        emergency = 0.0
        rainy = 0.0
        savings = 0.0
        invest_balance = 0.0
        total_debt_extra = 0.0

        rows = []

        for m in range(1, total_months + 1):
            if m <= phase1_months:
                income = phase1_income
                w = p1_weights
                phase_label = "Phase 1 (debt/unemployed)"
            else:
                income = phase2_income
                w = p2_weights
                phase_label = "Phase 2 (employed)"

            # Free cash after covering baseline expenses + one-time amortization
            free_cash = income - base_monthly_expenses - monthly_onetime
            if free_cash < 0:
                free_cash = 0.0  # You cover essentials, but can't invest/save that month

            # Planned allocations
            alloc_debt = free_cash * w["debt"]
            alloc_em = free_cash * w["emergency"]
            alloc_rain = free_cash * w["rainy"]
            alloc_sav = free_cash * w["savings"]
            alloc_inv = free_cash * w["invest"]

            # If we assume you're debt-free after Phase 1,
            # any "debt" allocation in Phase 2 gets auto-routed into investments.
            if m > phase1_months and alloc_debt > 0:
                alloc_inv += alloc_debt
                alloc_debt = 0.0

            # Apply caps: rainy & savings redirect overflow into investments
            # Rainy
            if rainy + alloc_rain > rainy_cap:
                overflow = (rainy + alloc_rain) - rainy_cap
                rainy = rainy_cap
                alloc_inv += overflow
            else:
                rainy += alloc_rain

            # Savings
            if savings + alloc_sav > savings_cap:
                overflow = (savings + alloc_sav) - savings_cap
                savings = savings_cap
                alloc_inv += overflow
            else:
                savings += alloc_sav

            # Emergency has only a goal (not hard stop), so just add
            emergency += alloc_em

            # Debt extra (we're not recomputing payoff here, just tracking total directed)
            if m <= phase1_months:
                total_debt_extra += alloc_debt

            # Dividends on starting investment balance this month
            dividends = invest_balance * monthly_yield
            if reinvest_dividends:
                invest_balance += dividends

            # New contributions into investments
            invest_balance += alloc_inv

            net_worth = emergency + rainy + savings + invest_balance  # ignoring debt as a negative here

            rows.append(
                {
                    "Month": m,
                    "Phase": phase_label,
                    "Free cash": free_cash,
                    "To debt (extra)": alloc_debt,
                    "To emergency": alloc_em,
                    "To rainy": alloc_rain,
                    "To savings": alloc_sav,
                    "To investments": alloc_inv,
                    "Dividends": dividends,
                    "Emergency balance": emergency,
                    "Rainy balance": rainy,
                    "Savings balance": savings,
                    "Investment balance": invest_balance,
                    "Net worth": net_worth,
                }
            )

        proj_df = pd.DataFrame(rows)

        # --- Summary at 6 months and at end ---
        m6 = phase1_months
        row_6 = proj_df.loc[proj_df["Month"] == m6].iloc[0] if m6 in proj_df["Month"].values else None
        row_end = proj_df.iloc[-1]

        st.markdown("## Milestones")

        sum_cols = st.columns(4)
        if row_6 is not None:
            sum_cols[0].metric(
                "Month 6 net worth",
                f"${row_6['Net worth']:,.2f}",
            )
            sum_cols[1].metric(
                "Month 6 investment balance",
                f"${row_6['Investment balance']:,.2f}",
            )
        else:
            sum_cols[0].metric("Month 6 net worth", "n/a")
            sum_cols[1].metric("Month 6 investment balance", "n/a")

        sum_cols[2].metric(
            "Month 12 net worth" if total_months >= 12 else "Final net worth",
            f"${row_end['Net worth']:,.2f}",
        )
        sum_cols[3].metric(
            "Final investment balance",
            f"${row_end['Investment balance']:,.2f}",
        )

        # Approx passive income at end
        final_monthly_div = row_end["Investment balance"] * monthly_yield
        st.metric(
            "Estimated monthly dividend income at end of projection",
            f"${final_monthly_div:,.2f}",
        )

        st.markdown("### Fund balances over time")
        fig_bal = px.line(
            proj_df,
            x="Month",
            y=[
                "Net worth",
                "Investment balance",
                "Emergency balance",
                "Savings balance",
                "Rainy balance",
            ],
            labels={"value": "Balance ($)", "variable": "Fund"},
        )
        st.plotly_chart(fig_bal, use_container_width=True)

        st.markdown("### Monthly dividends over time")
        fig_div = px.bar(
            proj_df,
            x="Month",
            y="Dividends",
            labels={"Dividends": "Dividends ($/mo)"},
        )
        st.plotly_chart(fig_div, use_container_width=True)

        with st.expander("See full month-by-month table"):
            st.dataframe(proj_df, use_container_width=True)

        # --- Helper: how many tickers if you cap at 1.25%? ---
        st.markdown("### Diversification helper (max % per ticker)")

        max_pos_pct = st.number_input(
            "Max allocation per ticker for this final portfolio (%)",
            min_value=0.1,
            max_value=10.0,
            value=1.25,
            step=0.25,
        )
        if row_end["Investment balance"] > 0 and max_pos_pct > 0:
            max_positions = int(100.0 // max_pos_pct)
            per_position_dollars = row_end["Investment balance"] * (max_pos_pct / 100.0)
            st.info(
                f"If you cap each holding at **{max_pos_pct:.2f}%**, you can own up to "
                f"**{max_positions} positions** before you're fully allocated.\n\n"
                f"At the final projected investment balance of "
                f"{format_currency(row_end['Investment balance'])}, "
                f"each full-sized position would be about **{format_currency(per_position_dollars)}**."
            )
        else:
            st.info("Set a non-zero final investment balance and max % to see diversification guidance.")


# =============================
# Main
# =============================
def main():
    st.set_page_config(
        page_title="Personal Budget & Planning",
        layout="wide",
        page_icon="💸",
    )

    ensure_data_dir()

    debts_df = load_df(
        DEBTS_FILE,
        ["debt_type", "name", "balance", "starting_balance", "apr", "min_payment", "due_day"],
    )
    expenses_df = load_df(
        EXPENSES_FILE,
        ["name", "amount", "frequency", "subcategory", "due_day"],
    )
    onetime_df = load_df(
        ONETIME_FILE,
        ["name", "amount", "due_date"],
    )
    invest_df = load_df(
        INVEST_FILE,
        ["ticker", "name", "category", "industry", "yield_pct", "amount"],
    )

    st.sidebar.title("📂 Navigation")
    page = st.sidebar.radio(
        "Go to",
        [
            "Overview & Data Entry",
            "Debt Snowball Simulator",
            "Investment Planner",
            "12-Month Projection",
        ],
    )

    if page == "Overview & Data Entry":
        page_overview(debts_df, expenses_df, onetime_df, invest_df)
    elif page == "Debt Snowball Simulator":
        page_debt_snowball(debts_df, expenses_df, onetime_df)
    elif page == "Investment Planner":
        page_investment(invest_df)
    elif page == "12-Month Projection":
        page_financial_projection(debts_df, expenses_df, onetime_df, invest_df)

if __name__ == "__main__":
    main()
