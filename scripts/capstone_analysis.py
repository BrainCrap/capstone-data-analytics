import pandas as pd
import statsmodels.formula.api as smf
import PyPDF2
import re

#Load PowerBI Export Dataset from M4
core = pd.read_excel("Capstone_M4_PowerBI.xlsx")

#Clean State names
core["State"] = core["State"].astype(str).str.replace("\xa0", "").str.strip()

#Build enrollment dataset from IPEDS exports
enroll_files = [
    "TrendGenerator-Table-Trend-q2-11-21-25 11-43.xlsx",
    "TrendGenerator-Table-Trend-q2-11-21-25 11-57.xlsx",
    "TrendGenerator-Table-Trend-q2-11-21-25 11-57(1).xlsx",
    "TrendGenerator-Table-Trend-q2-11-21-25 11-57(2).xlsx",
    "TrendGenerator-Table-Trend-q2-11-21-25 11-58.xlsx",
]

def load_enrollment(files):
    records = []
    for path in files:
        df = pd.read_excel(path)
        # Row 3: "Year, Total, Alabama, Alaska, ..."
        header = df.iloc[3].values
        # Row 4: "2023-24, 7665750, 390617, ..."
        data = df.iloc[4].values
        year = str(data[0])
        # Skip "Year" and "Total", take each state and value
        for state, val in zip(header[2:], data[2:]):
            if isinstance(state, str) and state.strip():
                if isinstance(val, (int, float)) and not pd.isna(val):
                    records.append(
                        {
                            "State": state.strip(),
                            "Year": year,
                            "Enrollment_2023_24": int(val),
                        }
                    )
    return pd.DataFrame(records)

enroll = load_enrollment(enroll_files)

#Build student retention dataset
ret_raw = pd.read_excel("TrendGenerator-Table-Trend-q32-11-21-25 11-30.xlsx")

#Row 3: State Names
header = ret_raw.iloc[3].values
row_2023 = ret_raw[ret_raw.iloc[:, 0] == 2023].iloc[0].values

ret_records = []
#Skip the non-state first entry ("Fall")
for state, val in zip(header[1:], row_2023[1:]):
    if isinstance(state, str) and state.strip():
        if isinstance(val, (int, float)) and not pd.isna(val):
            ret_records.append(
                {
                    "State": state.strip(),
                    "Retention_2023": float(val),
                }
            )

retention = pd.DataFrame(ret_records)

#Parse ACT scores using PyPDF2
reader = PyPDF2.PdfReader("6ed16a.pdf")
page_text = reader.pages[0].extract_text()

# Pattern of score is [State Abbreviation] 6 decimal scores (2019-2024) + integer percent
pattern = re.compile(
    r"([A-Z][A-Za-z ]+?)\s+(\d+\.\d)\s+(\d+\.\d)\s+(\d+\.\d)\s+(\d+\.\d)\s+(\d+\.\d)\s+(\d+\.\d)\s+(\d+)",
    re.M,
)

rows = [] 
for m in pattern.findall(page_text): #map any matched patterns
    state = m[0].strip()
    vals = list(map(float, m[1:7]))
    pct = int(m[7])
    rows.append(
        {
            "State": state,
            "ACT_2019": vals[0],
            "ACT_2020": vals[1],
            "ACT_2021": vals[2],
            "ACT_2022": vals[3],
            "ACT_2023": vals[4],
            "ACT_2024": vals[5],
            "Pct_Grads_Taking_ACT_2024": pct,
        }
    )

act = pd.DataFrame(rows)
#drop any non-us rows
act = act[act["State"] != "United States"]

#read academic integrity trends data
ai_int = pd.read_excel("academic_integrity_trends.xlsx")
ai_int["Delta_Instructors"] = (
    ai_int["Pct_Instructors_Cheating_More"]
    - ai_int["Pct_Instructors_Cheating_More"].shift(1)
)
ai_int["Delta_Students_Easier"] = (
    ai_int["Pct_Students_Cheating_Easier"]
    - ai_int["Pct_Students_Cheating_Easier"].shift(1)
)

#mereg all state level datasets
merged = (
    core
    .merge(enroll[["State", "Enrollment_2023_24"]], on="State", how="left")
    .merge(
        act[["State", "ACT_2019", "ACT_2024", "Pct_Grads_Taking_ACT_2024"]],
        on="State",
        how="left",
    )
    .merge(retention, on="State", how="left")
)

merged["ACT_change_19_24"] = merged["ACT_2024"] - merged["ACT_2019"]

#Hypothesis - H01
h01_df = merged.dropna(subset=["AI Usage Rate", "Graduation Rate (2024)"])

model_h01 = smf.ols(
    'Q("Graduation Rate (2024)") ~ Q("AI Usage Rate") '
    '+ Median_Income + BroadbandAccessRate_Pct '
    '+ Q("K-12 Spending Per Pupil") + Q("GDP Per Capita (USD)")',
    data=h01_df,
).fit()

print("\nH01: Graduation Rate model")
print(model_h01.summary())

#Hypothesis - H02
h02_enroll_df = merged.dropna(subset=["AI Usage Rate", "Enrollment_2023_24"])

model_h02_enroll = smf.ols(
    "Enrollment_2023_24 ~ Q('AI Usage Rate') "
    "+ Median_Income + BroadbandAccessRate_Pct "
    "+ Q('K-12 Spending Per Pupil') + Q('GDP Per Capita (USD)')",
    data=h02_enroll_df,
).fit()

print("\nH02a: Enrollment model")
print(model_h02_enroll.summary())

#Hypothesis - H02 (b)
h02_ret_df = merged.dropna(subset=["AI Usage Rate", "Retention_2023"])

model_h02_ret = smf.ols(
    "Retention_2023 ~ Q('AI Usage Rate')",
    data=h02_ret_df,
).fit()

print("\nH02b: Retention model (subset only)")
print(h02_ret_df[["State", "AI Usage Rate", "Retention_2023"]])
print(model_h02_ret.summary())

#Hypothesis - H03
print("\H03: Academic integrity trends")
print(ai_int)

#Hypothesis - H04
h04_df = merged.dropna(subset=["AI Usage Rate", "ACT_change_19_24"])

model_h04 = smf.ols(
    "ACT_change_19_24 ~ Q('AI Usage Rate') + Median_Income + BroadbandAccessRate_Pct",
    data=h04_df,
).fit()

print("\nH04: ACT score change model")
print(model_h04.summary())

#ACT_2024 on AI usage only
model_h04_simple = smf.ols(
    "ACT_2024 ~ Q('AI Usage Rate')",
    data=h04_df,
).fit()

print("\nH04 (simple): ACT_2024 on AI usage")
print(model_h04_simple.summary())
