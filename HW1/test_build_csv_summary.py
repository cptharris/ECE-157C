from nodes import _build_csv_summary

print(f"\n\n{'='*45} HOUSING {'='*45}\n\n")
print(_build_csv_summary("housing.csv"))
print(f"\n\n{'='*45} CUSTOM  {'='*45}\n\n")
print(_build_csv_summary("custom_dataset.csv"))
