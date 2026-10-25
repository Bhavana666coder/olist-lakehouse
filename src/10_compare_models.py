import csv

# Read the model evaluation results
file_path = "results/model_comparison.csv"

with open(file_path, "r", newline="") as file:
    reader = csv.DictReader(file)
    models = list(reader)

if not models:
    raise ValueError("No model results found in the CSV file.")

# Metrics to display
columns = [
    ("model", "Model"),
    ("roc_auc", "ROC-AUC"),
    ("pr_auc", "PR-AUC"),
    ("precision", "Precision"),
    ("recall", "Recall"),
    ("f1", "F1-score"),
    ("accuracy", "Accuracy"),
]

# Format the values
rows = []

for model in models:
    row = [
        model["model"],
        f'{float(model["roc_auc"]):.4f}',
        f'{float(model["pr_auc"]):.4f}',
        f'{float(model["precision"]):.4f}',
        f'{float(model["recall"]):.4f}',
        f'{float(model["f1"]):.4f}',
        f'{float(model["accuracy"]):.4f}',
    ]
    rows.append(row)

# Calculate column widths
headers = [column[1] for column in columns]
widths = [
    max(len(headers[i]), max(len(row[i]) for row in rows))
    for i in range(len(headers))
]

# Display the comparison table
print("\nOLIST REPEAT-PURCHASE MODEL COMPARISON")
print("=" * (sum(widths) + 3 * len(widths) + 1))

header_line = "| " + " | ".join(
    headers[i].ljust(widths[i]) for i in range(len(headers))
) + " |"

print(header_line)
print("|-" + "-|-".join("-" * width for width in widths) + "-|")

for row in rows:
    print("| " + " | ".join(
        row[i].ljust(widths[i]) for i in range(len(row))
    ) + " |")

# Select the best model based on PR-AUC
best_model = max(models, key=lambda model: float(model["pr_auc"]))

print("\n" + "=" * 60)
print("BEST MODEL SELECTION")
print("=" * 60)
print("Selection criterion: Highest PR-AUC")
print(f'Best model : {best_model["model"]}')
print(f'PR-AUC     : {float(best_model["pr_auc"]):.5f}')
print(f'ROC-AUC    : {float(best_model["roc_auc"]):.5f}')
print(f'Precision  : {float(best_model["precision"]):.5f}')
print(f'Recall     : {float(best_model["recall"]):.5f}')
print(f'F1-score   : {float(best_model["f1"]):.5f}')
print(f'Accuracy   : {float(best_model["accuracy"]):.5f}')

print("\nConclusion:")
print(
    f'{best_model["model"]} is selected because it has '
    "the highest PR-AUC among the evaluated classifiers."
)
print("Note: The best model by this metric may still have limited")
print("predictive performance.")

