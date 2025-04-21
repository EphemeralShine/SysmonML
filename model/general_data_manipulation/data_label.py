import pandas as pd
import xmltodict

sysmon_xml_path = "sysmon_logs.xml"
art_log_csv_path = "art_log.csv"
output_csv_path = "sysmon_labeled_full.csv"

# Load known malicious PIDs
def extract_malicious_pids(csv_path):
    df = pd.read_csv(csv_path)
    df['ProcessId'] = df['ProcessId'].astype(str).str.strip()
    df = df[df['ProcessId'].str.isdigit()]
    return set(df['ProcessId'].astype(int))

# Parse Sysmon XML into DataFrame
def parse_sysmon_raw(xml_path):
    with open(xml_path, 'r', encoding='utf-8') as f:
        data = xmltodict.parse(f.read())

    events = data['Events']['Event']
    parsed = []

    for evt in events:
        try:
            system = evt['System']
            event_id = int(system['EventID']['#text']) if isinstance(system['EventID'], dict) else int(system['EventID'])

            event_data = evt.get('EventData', {}).get('Data', [])
            event_dict = {}
            if isinstance(event_data, list):
                for item in event_data:
                    key = item.get('@Name')
                    value = item.get('#text', '')
                    if key:
                        event_dict[key] = value
            elif isinstance(event_data, dict):
                key = event_data.get('@Name')
                value = event_data.get('#text', '')
                if key:
                    event_dict[key] = value

            # Build row
            record = {
                "EventID": event_id,
                "Message": str(event_dict),
                "ProcessId": int(event_dict.get("ProcessId")) if event_dict.get("ProcessId", "").isdigit() else None,
                "ParentProcessId": int(event_dict.get("ParentProcessId")) if event_dict.get("ParentProcessId", "").isdigit() else None,
                "Image": event_dict.get("Image", "").lower(),
                "ParentImage": event_dict.get("ParentImage", "").lower(),
                "ProcessGuid": event_dict.get("ProcessGuid", "").lower(),
                "ParentProcessGuid": event_dict.get("ParentProcessGuid", "").lower()
            }

            parsed.append(record)

        except Exception:
            continue

    return pd.DataFrame(parsed)

# Labeling
def label_logs(df, malicious_pids):
    def is_powershell(path):
        return isinstance(path, str) and path.lower().endswith("powershell.exe")

    def assign_label(row):
        pid_match = row["ProcessId"] in malicious_pids or row["ParentProcessId"] in malicious_pids
        powershell_match = is_powershell(row["Image"]) or is_powershell(row["ParentImage"])
        if pid_match and powershell_match:
            return 1
        return 0

    df["label"] = df.apply(assign_label, axis=1)
    return df

if __name__ == "__main__":
    print("[*] Reading ART log...")
    malicious_pids = extract_malicious_pids(art_log_csv_path)
    print(f"[+] Found {len(malicious_pids)} malicious Process IDs")

    print("[*] Parsing Sysmon XML...")
    df = parse_sysmon_raw(sysmon_xml_path)
    print(f"[+] Parsed {len(df)} Sysmon events")

    print("[*] Labeling logs (PID + PowerShell check)...")
    df_labeled = label_logs(df, malicious_pids)

    print(f"[+] Saving to {output_csv_path}")
    df_labeled.to_csv(output_csv_path, index=False)
    print("[✓] Done.")
