"""
UPI Fraud Analysis — Data Processing & EDA
==========================================
Reads raw Excel → cleans → engineers features → exports cleaned CSVs
for Power BI ingestion.
"""

import pandas as pd
import numpy as np
import os
import warnings
warnings.filterwarnings("ignore")

BASE   = os.path.dirname(os.path.abspath(__file__))
RAW    = os.path.join(BASE, "..", "data", "UPI_Fraud_Demo_Data.xlsx")
OUT    = os.path.join(BASE, "..", "data")


# ────────────────────────────────────────────────────────────────────────────
# 1. LOAD
# ────────────────────────────────────────────────────────────────────────────
print("=" * 60)
print("  UPI FRAUD ANALYSIS — DATA PROCESSING PIPELINE")
print("=" * 60)

df = pd.read_excel(RAW, sheet_name="Raw_Transactions")
print(f"\n✅  Loaded {len(df):,} rows × {df.shape[1]} columns")


# ────────────────────────────────────────────────────────────────────────────
# 2. BASIC EDA
# ────────────────────────────────────────────────────────────────────────────
print("\n── DATASET INFO ──────────────────────────────────────")
print(df.dtypes.to_string())

print("\n── NULL COUNTS ───────────────────────────────────────")
nulls = df.isnull().sum()
print(nulls[nulls > 0] if nulls.sum() else "  No nulls found ✅")

print("\n── NUMERIC SUMMARY ───────────────────────────────────")
print(df[["Amount_INR", "Response_Time_Sec", "Login_Attempts", "Risk_Score"]]
      .describe().round(2))


# ────────────────────────────────────────────────────────────────────────────
# 3. CLEANING
# ────────────────────────────────────────────────────────────────────────────
print("\n── CLEANING ──────────────────────────────────────────")

# Parse datetime
df["DateTime"] = pd.to_datetime(df["Date"] + " " + df["Time"])
df["Date"]     = pd.to_datetime(df["Date"])
df = df.sort_values("DateTime").reset_index(drop=True)

# Remove duplicates
before = len(df)
df = df.drop_duplicates(subset="Transaction_ID")
print(f"  Duplicates removed : {before - len(df)}")

# Cap extreme amounts at 99.9th percentile
p999 = df["Amount_INR"].quantile(0.999)
df["Amount_INR_Capped"] = df["Amount_INR"].clip(upper=p999)
print(f"  Amount capped at   : ₹{p999:,.0f}")

# Standardise categorical columns
for col in ["Sender_Bank", "Receiver_Bank", "Transaction_Type",
            "UPI_App", "Device_Type", "Sender_State", "Receiver_State"]:
    df[col] = df[col].str.strip().str.title()

print(f"  Final clean rows   : {len(df):,}")


# ────────────────────────────────────────────────────────────────────────────
# 4. FEATURE ENGINEERING
# ────────────────────────────────────────────────────────────────────────────
print("\n── FEATURE ENGINEERING ──────────────────────────────")

# Time features
df["Hour"]          = df["DateTime"].dt.hour
df["WeekDay"]       = df["DateTime"].dt.day_name()
df["WeekNumber"]    = df["DateTime"].dt.isocalendar().week.astype(int)
df["Is_Weekend"]    = df["DateTime"].dt.dayofweek.isin([5, 6]).astype(int)
df["Is_Night_Txn"]  = df["Hour"].between(0, 5).astype(int)

# Amount buckets
bins   = [0, 500, 2000, 10000, 50000, 200000, np.inf]
labels = ["Micro (<₹500)", "Small (₹500-2K)", "Medium (₹2K-10K)",
          "Large (₹10K-50K)", "High (₹50K-2L)", "Very High (>₹2L)"]
df["Amount_Category"] = pd.cut(df["Amount_INR"], bins=bins, labels=labels)

# Risk label
df["Risk_Label"] = pd.cut(
    df["Risk_Score"],
    bins=[0, 25, 50, 75, 100],
    labels=["Low", "Medium", "High", "Critical"]
)

# Cross-state flag
df["Cross_State_Txn"] = (df["Sender_State"] != df["Receiver_State"]).astype(int)

# Fraud velocity: count of fraud transactions per sender UPI
fraud_velocity = (
    df[df["Is_Fraud"] == 1]
    .groupby("Sender_UPI")["Is_Fraud"]
    .count()
    .rename("Fraud_Velocity")
)
df = df.merge(fraud_velocity, on="Sender_UPI", how="left")
df["Fraud_Velocity"] = df["Fraud_Velocity"].fillna(0).astype(int)

print("  Features created   : Hour, WeekDay, Is_Weekend, Is_Night_Txn,")
print("                       Amount_Category, Risk_Label, Cross_State_Txn,")
print("                       Fraud_Velocity")


# ────────────────────────────────────────────────────────────────────────────
# 5. ANALYSIS SUMMARIES
# ────────────────────────────────────────────────────────────────────────────
print("\n── FRAUD SUMMARY ────────────────────────────────────")
total  = len(df)
frauds = df["Is_Fraud"].sum()
print(f"  Total Transactions : {total:,}")
print(f"  Fraud Transactions : {frauds:,}")
print(f"  Fraud Rate         : {frauds/total*100:.2f}%")
print(f"  Total Amount       : ₹{df['Amount_INR'].sum():,.0f}")
print(f"  Fraud Amount       : ₹{df[df['Is_Fraud']==1]['Amount_INR'].sum():,.0f}")
print(f"  Avg Fraud Amount   : ₹{df[df['Is_Fraud']==1]['Amount_INR'].mean():,.0f}")
print(f"  Avg Legit Amount   : ₹{df[df['Is_Fraud']==0]['Amount_INR'].mean():,.0f}")

print("\n── TOP FRAUD TYPES ──────────────────────────────────")
print(df[df["Is_Fraud"]==1]["Fraud_Type"].value_counts().head())

print("\n── FRAUD BY UPI APP ─────────────────────────────────")
app_fraud = (
    df.groupby("UPI_App")
    .agg(Total=("Is_Fraud","count"), Fraud=("Is_Fraud","sum"))
    .assign(Rate=lambda x: (x.Fraud/x.Total*100).round(2))
    .sort_values("Fraud", ascending=False)
)
print(app_fraud)

print("\n── FRAUD BY HOUR ─────────────────────────────────────")
hour_fraud = (
    df.groupby("Hour")
    .agg(Total=("Is_Fraud","count"), Fraud=("Is_Fraud","sum"))
    .assign(Rate=lambda x: (x.Fraud/x.Total*100).round(2))
)
peak = hour_fraud["Fraud"].idxmax()
print(f"  Peak fraud hour    : {peak}:00 ({hour_fraud.loc[peak,'Fraud']} cases)")


# ────────────────────────────────────────────────────────────────────────────
# 6. EXPORT CLEANED DATA FOR POWER BI
# ────────────────────────────────────────────────────────────────────────────
print("\n── EXPORTING CSVs ───────────────────────────────────")

# Main transactions table
main_cols = [
    "Transaction_ID","Date","Time","DateTime","Month","Quarter",
    "Day_of_Week","Hour","WeekDay","WeekNumber","Is_Weekend","Is_Night_Txn",
    "Sender_UPI","Receiver_UPI","Sender_Bank","Receiver_Bank",
    "Sender_State","Receiver_State","Cross_State_Txn",
    "Amount_INR","Amount_INR_Capped","Amount_Category",
    "Transaction_Type","UPI_App","Device_Type",
    "Is_Fraud","Fraud_Type","Fraud_Flag",
    "Response_Time_Sec","Login_Attempts","New_Device_Used",
    "Odd_Hour_Txn","Risk_Score","Risk_Label","Fraud_Velocity"
]
df[main_cols].to_csv(os.path.join(OUT, "cleaned_transactions.csv"), index=False)
print(f"  cleaned_transactions.csv     → {len(df):,} rows")

# Monthly trend
monthly = (
    df.groupby(["Month","Quarter"])
    .agg(
        Total_Txns=("Transaction_ID","count"),
        Fraud_Txns=("Is_Fraud","sum"),
        Total_Amount=("Amount_INR","sum"),
        Avg_Amount=("Amount_INR","mean"),
        Avg_Risk=("Risk_Score","mean"),
    )
    .reset_index()
)
monthly["Fraud_Rate_Pct"] = (monthly["Fraud_Txns"] / monthly["Total_Txns"] * 100).round(2)
monthly["Legit_Txns"] = monthly["Total_Txns"] - monthly["Fraud_Txns"]
month_order = ["January","February","March","April","May","June",
               "July","August","September","October","November","December"]
monthly["Month_Num"] = monthly["Month"].map({m:i for i,m in enumerate(month_order,1)})
monthly = monthly.sort_values("Month_Num")
monthly.to_csv(os.path.join(OUT, "monthly_trends.csv"), index=False)
print(f"  monthly_trends.csv           → {len(monthly)} rows")

# Bank analysis
bank_df = (
    df.groupby("Sender_Bank")
    .agg(
        Total_Txns=("Transaction_ID","count"),
        Fraud_Txns=("Is_Fraud","sum"),
        Total_Amount=("Amount_INR","sum"),
        Avg_Amount=("Amount_INR","mean"),
        Avg_Risk=("Risk_Score","mean"),
    )
    .reset_index()
)
bank_df["Fraud_Rate_Pct"] = (bank_df["Fraud_Txns"] / bank_df["Total_Txns"] * 100).round(2)
bank_df.to_csv(os.path.join(OUT, "bank_analysis.csv"), index=False)
print(f"  bank_analysis.csv            → {len(bank_df)} rows")

# State analysis
state_df = (
    df.groupby("Sender_State")
    .agg(
        Total_Txns=("Transaction_ID","count"),
        Fraud_Txns=("Is_Fraud","sum"),
        Total_Amount=("Amount_INR","sum"),
        Avg_Risk=("Risk_Score","mean"),
    )
    .reset_index()
)
state_df["Fraud_Rate_Pct"] = (state_df["Fraud_Txns"] / state_df["Total_Txns"] * 100).round(2)
state_df.to_csv(os.path.join(OUT, "state_analysis.csv"), index=False)
print(f"  state_analysis.csv           → {len(state_df)} rows")

# Hourly pattern
hourly_df = (
    df.groupby("Hour")
    .agg(
        Total_Txns=("Transaction_ID","count"),
        Fraud_Txns=("Is_Fraud","sum"),
        Total_Amount=("Amount_INR","sum"),
    )
    .reset_index()
)
hourly_df["Fraud_Rate_Pct"] = (hourly_df["Fraud_Txns"] / hourly_df["Total_Txns"] * 100).round(2)
hourly_df.to_csv(os.path.join(OUT, "hourly_patterns.csv"), index=False)
print(f"  hourly_patterns.csv          → {len(hourly_df)} rows")

# UPI App analysis
app_df = (
    df.groupby("UPI_App")
    .agg(
        Total_Txns=("Transaction_ID","count"),
        Fraud_Txns=("Is_Fraud","sum"),
        Total_Amount=("Amount_INR","sum"),
        Avg_Risk=("Risk_Score","mean"),
    )
    .reset_index()
)
app_df["Fraud_Rate_Pct"] = (app_df["Fraud_Txns"] / app_df["Total_Txns"] * 100).round(2)
app_df.to_csv(os.path.join(OUT, "upi_app_analysis.csv"), index=False)
print(f"  upi_app_analysis.csv         → {len(app_df)} rows")

# Fraud type breakdown
fraud_type_df = (
    df[df["Is_Fraud"]==1]
    .groupby("Fraud_Type")
    .agg(
        Fraud_Count=("Transaction_ID","count"),
        Total_Fraud_Amount=("Amount_INR","sum"),
        Avg_Fraud_Amount=("Amount_INR","mean"),
        Avg_Risk_Score=("Risk_Score","mean"),
    )
    .reset_index()
    .sort_values("Fraud_Count", ascending=False)
)
fraud_type_df.to_csv(os.path.join(OUT, "fraud_type_breakdown.csv"), index=False)
print(f"  fraud_type_breakdown.csv     → {len(fraud_type_df)} rows")

print("\n✅  All files exported to /data/")
print("=" * 60)
print("  PIPELINE COMPLETE — Ready for Power BI")
print("=" * 60)
