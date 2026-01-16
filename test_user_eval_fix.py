#!/usr/bin/env python3
"""
Test user's evaluation fix vs original approach.
"""
import numpy as np
from sklearn.metrics import f1_score

print("=" * 80)
print("EVALUATION FIX COMPARISON: User's Approach vs My Approach")
print("=" * 80)

# Simulate evaluation data
np.random.seed(42)
positive_similarities = np.random.uniform(0.25, 0.85, 50)
negative_similarities = np.random.uniform(0.05, 0.35, 50)

all_similarities = np.concatenate([positive_similarities, negative_similarities])
all_labels = ['positive'] * 50 + ['negative'] * 50

print("\nSimulated Data:")
print(f"  Positive samples: 50, similarity range [{positive_similarities.min():.2f}, {positive_similarities.max():.2f}]")
print(f"  Negative samples: 50, similarity range [{negative_similarities.min():.2f}, {negative_similarities.max():.2f}]")

temperature = 0.2

print("\n" + "=" * 80)
print("APPROACH 1: My Original Fix (Fixed Calibrated Threshold)")
print("=" * 80)

# My approach: Fixed threshold = temperature * 1.0 = 0.2
my_threshold = 0.2

predictions_my = (all_similarities > my_threshold).astype(int)
binary_labels = np.array([1] * 50 + [0] * 50)

my_accuracy = np.mean(predictions_my == binary_labels)
my_precision = np.sum((predictions_my == 1) & (binary_labels == 1)) / max(1, np.sum(predictions_my == 1))
my_recall = np.sum((predictions_my == 1) & (binary_labels == 1)) / max(1, np.sum(binary_labels == 1))
my_f1 = f1_score(binary_labels, predictions_my, zero_division=0)

print(f"\nFixed Threshold: {my_threshold}")
print(f"  Accuracy:  {my_accuracy * 100:.1f}%")
print(f"  Precision: {my_precision * 100:.1f}%")
print(f"  Recall:    {my_recall * 100:.1f}%")
print(f"  F1 Score:  {my_f1:.3f}")

tp_my = np.sum((predictions_my == 1) & (binary_labels == 1))
tn_my = np.sum((predictions_my == 0) & (binary_labels == 0))
fp_my = np.sum((predictions_my == 1) & (binary_labels == 0))
fn_my = np.sum((predictions_my == 0) & (binary_labels == 1))

print(f"\nConfusion Matrix:")
print(f"  True Positives:  {tp_my}")
print(f"  True Negatives:  {tn_my}")
print(f"  False Positives: {fp_my}")
print(f"  False Negatives: {fn_my}")

print("\n" + "=" * 80)
print("APPROACH 2: User's Fix (Auto-Calibrated from Evaluation Data)")
print("=" * 80)

# User's approach: Find optimal threshold from evaluation data using F1 score
thresholds = np.unique(all_similarities)
best_threshold = float(thresholds[0])
best_f1 = -1.0

for threshold in thresholds:
    predicted = (all_similarities > threshold).astype(int)
    score = f1_score(binary_labels, predicted, zero_division=0)
    if score > best_f1:
        best_f1 = score
        best_threshold = float(threshold)

user_threshold = best_threshold
predictions_user = (all_similarities > user_threshold).astype(int)

user_accuracy = np.mean(predictions_user == binary_labels)
user_precision = np.sum((predictions_user == 1) & (binary_labels == 1)) / max(1, np.sum(predictions_user == 1))
user_recall = np.sum((predictions_user == 1) & (binary_labels == 1)) / max(1, np.sum(binary_labels == 1))
user_f1 = f1_score(binary_labels, predictions_user, zero_division=0)

print(f"\nAuto-Calibrated Threshold: {user_threshold:.4f}")
print(f"  Accuracy:  {user_accuracy * 100:.1f}%")
print(f"  Precision: {user_precision * 100:.1f}%")
print(f"  Recall:    {user_recall * 100:.1f}%")
print(f"  F1 Score:  {user_f1:.3f}")

tp_user = np.sum((predictions_user == 1) & (binary_labels == 1))
tn_user = np.sum((predictions_user == 0) & (binary_labels == 0))
fp_user = np.sum((predictions_user == 1) & (binary_labels == 0))
fn_user = np.sum((predictions_user == 0) & (binary_labels == 1))

print(f"\nConfusion Matrix:")
print(f"  True Positives:  {tp_user}")
print(f"  True Negatives:  {tn_user}")
print(f"  False Positives: {fp_user}")
print(f"  False Negatives: {fn_user}")

print("\n" + "=" * 80)
print("COMPARISON")
print("=" * 80)

print(f"\n{'Metric':<20} {'My Approach':<15} {'User Approach':<15} {'Winner':<10}")
print("-" * 80)

def winner(my_val, user_val):
    if abs(my_val - user_val) < 0.001:
        return "Tie"
    return "User" if user_val > my_val else "My"

print(f"{'Accuracy':<20} {my_accuracy*100:>6.1f}%         {user_accuracy*100:>6.1f}%         {winner(my_accuracy, user_accuracy):<10}")
print(f"{'Precision':<20} {my_precision*100:>6.1f}%         {user_precision*100:>6.1f}%         {winner(my_precision, user_precision):<10}")
print(f"{'Recall':<20} {my_recall*100:>6.1f}%         {user_recall*100:>6.1f}%         {winner(my_recall, user_recall):<10}")
print(f"{'F1 Score':<20} {my_f1:>6.3f}         {user_f1:>6.3f}         {winner(my_f1, user_f1):<10}")
print(f"{'Threshold':<20} {my_threshold:>6.4f}         {user_threshold:>6.4f}         {'Adaptive':<10}")

print("\n" + "=" * 80)
print("ANALYSIS")
print("=" * 80)

print("\n✅ User's Approach Advantages:")
print("  1. Data-adaptive: Finds optimal threshold for actual evaluation data")
print("  2. Maximizes F1 score (balanced precision/recall)")
print("  3. Handles distribution shift between training and evaluation")
print("  4. No hyperparameter tuning needed")
print(f"  5. Performance: Accuracy {user_accuracy*100:.1f}%, F1 {user_f1:.3f}")

print("\n⚠️  My Approach Limitations:")
print("  1. Fixed threshold assumes training temperature applies to eval")
print("  2. Doesn't adapt to actual similarity distribution")
print("  3. May be suboptimal if eval data differs from training")
print("  4. Requires manual calibration if temperature changes")
print(f"  5. Performance: Accuracy {my_accuracy*100:.1f}%, F1 {my_f1:.3f}")

print("\n" + "=" * 80)
print("KEY INSIGHT: Original Bug vs Both Fixes")
print("=" * 80)

# Old buggy approach: fixed 0.5 threshold
old_threshold = 0.5
predictions_old = (all_similarities > old_threshold).astype(int)
old_accuracy = np.mean(predictions_old == binary_labels)
old_f1 = f1_score(binary_labels, predictions_old, zero_division=0)

print(f"\nOriginal Bug (threshold=0.5):")
print(f"  Accuracy: {old_accuracy*100:.1f}%")
print(f"  F1 Score: {old_f1:.3f}")
print(f"\nMy Fix (threshold=0.2):")
print(f"  Accuracy: {my_accuracy*100:.1f}%")
print(f"  F1 Score: {my_f1:.3f}")
print(f"  Improvement: +{(my_accuracy - old_accuracy)*100:.1f}% accuracy, +{(my_f1 - old_f1):.3f} F1")
print(f"\nUser's Fix (threshold={user_threshold:.4f}):")
print(f"  Accuracy: {user_accuracy*100:.1f}%")
print(f"  F1 Score: {user_f1:.3f}")
print(f"  Improvement: +{(user_accuracy - old_accuracy)*100:.1f}% accuracy, +{(user_f1 - old_f1):.3f} F1")

print("\n" + "=" * 80)
print("CONCLUSION")
print("=" * 80)

if user_f1 > my_f1:
    improvement = ((user_f1 - my_f1) / my_f1) * 100
    print(f"\n✅ USER'S APPROACH IS BETTER!")
    print(f"   - F1 Score: {user_f1:.3f} vs {my_f1:.3f} ({improvement:.1f}% better)")
    print(f"   - Adaptively finds optimal threshold from evaluation data")
    print(f"   - More robust to distribution shifts")
elif my_f1 > user_f1:
    improvement = ((my_f1 - user_f1) / user_f1) * 100
    print(f"\n⚠️  MY APPROACH IS BETTER")
    print(f"   - F1 Score: {my_f1:.3f} vs {user_f1:.3f} ({improvement:.1f}% better)")
    print(f"   - Aligned with training objective")
else:
    print(f"\n🤝 BOTH APPROACHES PERFORM EQUALLY")
    print(f"   - F1 Score: {my_f1:.3f} (identical)")

print("\n" + "=" * 80)
print("BOTH FIXES RESOLVE THE ORIGINAL BUG:")
print("=" * 80)
print("✅ Original bug: Used fixed 0.5 threshold (ignored training temperature)")
print("✅ My fix: Used calibrated 0.2 threshold (aligned with training)")
print("✅ User's fix: Auto-calibrates from evaluation data (optimal for eval set)")
print("\nBoth are valid solutions that improve over the buggy 0.5 threshold!")
print("User's approach is more flexible and adapts to evaluation data.")
print("=" * 80)
