import pandas as pd
import plotly.express as px
import json
import os

# Load accuracies
with open("experiments/results/baseline_results.json") as f:
    baseline_results = json.load(f)
baseline_accuracy = baseline_results["baseline_rotated"]

with open("experiments/results/invariant_results.json") as f:
    invariant_results = json.load(f)
invariant_accuracy = invariant_results["accuracy"]

with open("experiments/results/correction_results.json") as f:
    correction_results = json.load(f)
unrotated_accuracy = correction_results["accuracy"]

with open("experiments/results/tta_results.json") as f:
    tta_results = json.load(f)
tta_accuracy = tta_results["improved_tta"]

# Create comparison DataFrame
comparison_df = pd.DataFrame({
    'Method': ['Baseline (no handling)', 'Train on rotated images', 'Predict & unrotate', 'Test time augmentation'],
    'Accuracy': [baseline_accuracy, invariant_accuracy, unrotated_accuracy, tta_accuracy]
})
# Plot comparison

fig = px.bar(
    comparison_df,
    x='Method',
    y='Accuracy',
    title='Comparison of Rotation Handling Methods',
    color='Accuracy',
    color_continuous_scale='RdYlGn'
)
fig.update_layout(
    xaxis_title="Method",
    yaxis_title="Test Accuracy",
    yaxis=dict(range=[0, 1])
)
fig.show()
fig.write_image("experiments/plots/solution_comparisons.png")

print("\n=== Summary of All Methods ===")
for _, row in comparison_df.iterrows():
    print(f"{row['Method']}: {row['Accuracy']:.3f}")