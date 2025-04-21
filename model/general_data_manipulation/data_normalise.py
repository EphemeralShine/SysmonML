import pandas as pd
import ast

# Input/output paths
INPUT_CSV = "sysmon_labeled_full.csv"
OUTPUT_CSV = "sysmon_normalized_full.csv"

def normalize_sysmon(df):
    # Parse Message field into actual dictionaries
    parsed_msgs = df["Message"].apply(lambda x: ast.literal_eval(x) if pd.notnull(x) else {})

    # Convert list of dicts into DataFrame
    message_df = pd.json_normalize(parsed_msgs)
    # Avoid duplication, as those fields are already parsed
    message_df = message_df.drop(columns=["ProcessId", "ParentProcessId", "ProcessGuid", "Image", "ParentProcessGuid", "ParentImage"])

    # Clean up float-like values in message_df (e.g., "11328.0" -> "11328")
    for col in ["ProcessId", "ParentProcessId"]:
        if col in message_df.columns:
            message_df[col] = message_df[col].astype(str).str.rstrip(".0")

    # Merge original columns avoiding duplication
    df_meta = df.drop(columns=["Message"])

    final_df = pd.concat([df_meta, message_df], axis=1)

    # Replace NaNs with "NULL"
    final_df = final_df.fillna("NULL")
    
    return final_df


if __name__ == "__main__":
    print("[*] Reading labeled Sysmon CSV...")
    df = pd.read_csv(INPUT_CSV)

    print("[*] Normalizing 'Message' column into structured fields...")
    df_normalized = normalize_sysmon(df)

    print("[*] Saving full normalized dataset...")
    df_normalized.to_csv(OUTPUT_CSV, index=False)
    print(f"[✓] Saved normalized Sysmon logs to {OUTPUT_CSV}")