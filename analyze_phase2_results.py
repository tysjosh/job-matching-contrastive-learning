#!/usr/bin/env python3
"""
Comprehensive Phase 2 Fine-Tuning Analysis
"""
import json
from pathlib import Path

print("=" * 80)
print("PHASE 2 FINE-TUNING RESULTS ANALYSIS")
print("=" * 80)

# Load Phase 2 results
results_path = Path("phase2_finetuning/fine_tuning_results.json")
with open(results_path) as f:
    results = json.load(f)

# Load Phase 1 results for comparison
phase1_path = Path("phase1_pretraining/training_results.json")
with open(phase1_path) as f:
    phase1_results = json.load(f)

print("\n" + "=" * 80)
print("PHASE 2 KEY METRICS")
print("=" * 80)

print(f"\n📊 Final Results:")
print(f"  Final Accuracy: {results['final_accuracy']*100:.2f}%")
print(f"  Final Loss: {results['final_loss']:.6f}")
print(f"  Best Accuracy: {results['best_accuracy']*100:.2f}%")
print(f"  Best Loss: {results['best_loss']:.6f}")

print(f"\n⏱️  Training Time: {results['training_time']:.1f} seconds ({results['training_time']/60:.1f} minutes)")
print(f"  Total Batches: {results['total_batches']}")
print(f"  Total Samples: {results['total_samples']}")

print(f"\n🎯 Model Configuration:")
print(f"  Embedding Dim: {results['model_info']['embedding_dim']}")
print(f"  Total Parameters: {results['model_info']['total_parameters']:,}")
print(f"  Trainable Parameters: {results['model_info']['trainable_parameters']:,}")
print(f"  Frozen Parameters: {results['model_info']['frozen_parameters']:,}")
print(f"  Classification Dropout: {results['model_info']['classification_dropout']}")

print("\n" + "=" * 80)
print("EPOCH-BY-EPOCH PROGRESSION")
print("=" * 80)

print(f"\n{'Epoch':<8} {'Loss':<12} {'Accuracy':<12} {'Loss Δ':<12} {'Acc Δ':<12}")
print("-" * 80)

epoch_losses = results['epoch_losses']
epoch_accs = results['epoch_accuracies']

for i in range(len(epoch_losses)):
    epoch = i + 1
    loss = epoch_losses[i]
    acc = epoch_accs[i]
    
    if i == 0:
        loss_delta = "-"
        acc_delta = "-"
    else:
        loss_delta = f"{loss - epoch_losses[i-1]:+.6f}"
        acc_delta = f"{(acc - epoch_accs[i-1])*100:+.2f}%"
    
    print(f"{epoch:<8} {loss:<12.6f} {acc*100:<11.2f}% {loss_delta:<12} {acc_delta:<12}")

print("\n" + "=" * 80)
print("IMPROVEMENT ANALYSIS")
print("=" * 80)

initial_acc = epoch_accs[0]
final_acc = epoch_accs[-1]
best_acc = max(epoch_accs)
best_epoch = epoch_accs.index(best_acc) + 1

acc_improvement = (final_acc - initial_acc) * 100
best_improvement = (best_acc - initial_acc) * 100

initial_loss = epoch_losses[0]
final_loss = epoch_losses[-1]
loss_reduction = ((initial_loss - final_loss) / initial_loss) * 100

print(f"\nAccuracy Progress:")
print(f"  Initial (Epoch 1): {initial_acc*100:.2f}%")
print(f"  Final (Epoch 10): {final_acc*100:.2f}%")
print(f"  Best (Epoch {best_epoch}): {best_acc*100:.2f}%")
print(f"  Total Improvement: +{acc_improvement:.2f} percentage points")
print(f"  Best Improvement: +{best_improvement:.2f} percentage points")

print(f"\nLoss Progress:")
print(f"  Initial: {initial_loss:.6f}")
print(f"  Final: {final_loss:.6f}")
print(f"  Reduction: {loss_reduction:.2f}%")

print("\n" + "=" * 80)
print("COMPARISON WITH PREVIOUS PHASE 2 (54% ACCURACY)")
print("=" * 80)

previous_acc = 0.54  # From conversation history
current_acc = results['final_accuracy']
improvement = (current_acc - previous_acc) * 100

print(f"\nPrevious Phase 2 Result (with bugs):")
print(f"  Accuracy: {previous_acc*100:.1f}%")
print(f"  Issues: Fixed 0.5 threshold, unnormalized embeddings")

print(f"\nCurrent Phase 2 Result (with all 3 fixes):")
print(f"  Accuracy: {current_acc*100:.2f}%")
print(f"  Fixes Applied:")
print(f"    ✅ Model normalization (F.normalize)")
print(f"    ✅ Encoding consistency (normalize_embeddings=False)")
print(f"    ✅ Evaluation threshold (auto-calibrated)")

print(f"\n📈 Improvement: {improvement:+.2f} percentage points")

if current_acc > previous_acc:
    pct_better = ((current_acc - previous_acc) / previous_acc) * 100
    print(f"   ({pct_better:+.1f}% better than previous)")
    print("\n✅ FIXES WORKED! Accuracy improved over buggy version")
else:
    print("\n⚠️  Accuracy not improved - investigate why")

print("\n" + "=" * 80)
print("TRAINING STABILITY ANALYSIS")
print("=" * 80)

# Check last 5 epochs for convergence
last_5_accs = epoch_accs[-5:]
last_5_losses = epoch_losses[-5:]

avg_acc = sum(last_5_accs) / len(last_5_accs)
avg_loss = sum(last_5_losses) / len(last_5_losses)

acc_std = (sum((x - avg_acc) ** 2 for x in last_5_accs) / len(last_5_accs)) ** 0.5
loss_std = (sum((x - avg_loss) ** 2 for x in last_5_losses) / len(last_5_losses)) ** 0.5

print(f"\nLast 5 Epochs Stability:")
print(f"  Accuracy: {avg_acc*100:.2f}% ± {acc_std*100:.2f}%")
print(f"  Loss: {avg_loss:.6f} ± {loss_std:.6f}")
print(f"  Min Accuracy: {min(last_5_accs)*100:.2f}%")
print(f"  Max Accuracy: {max(last_5_accs)*100:.2f}%")

# Check for monotonic improvement
improving_epochs = sum(1 for i in range(1, len(epoch_accs)) if epoch_accs[i] > epoch_accs[i-1])
print(f"\nMonotonic Improvement:")
print(f"  Epochs with accuracy increase: {improving_epochs}/9")
print(f"  Convergence pattern: {'Steady' if improving_epochs >= 7 else 'Oscillating' if improving_epochs >= 4 else 'Unstable'}")

print("\n" + "=" * 80)
print("DIAGNOSTIC INSIGHTS")
print("=" * 80)

print(f"\n🔍 Key Observations:")

if current_acc < 0.60:
    print(f"\n⚠️  Accuracy still below 60% ({current_acc*100:.2f}%)")
    print("   Possible issues:")
    print("   1. Phase 1 loss (0.602) might be too high")
    print("   2. Classification head may need more capacity")
    print("   3. May need more Phase 2 epochs")
    print("   4. Learning rate might be suboptimal")
elif current_acc < 0.70:
    print(f"\n🟡 Accuracy moderate (60-70%): {current_acc*100:.2f}%")
    print("   Room for improvement:")
    print("   1. Consider training Phase 1 longer (lower loss)")
    print("   2. Try different classification head architecture")
    print("   3. Experiment with dropout rates")
else:
    print(f"\n✅ Good accuracy (>70%): {current_acc*100:.2f}%")
    print("   Fixes working well!")

# Check if Phase 1 quality is limiting Phase 2
phase1_loss = phase1_results['final_loss']
print(f"\n🔗 Phase 1 → Phase 2 Connection:")
print(f"   Phase 1 Final Loss: {phase1_loss:.6f}")
print(f"   Phase 2 Final Accuracy: {current_acc*100:.2f}%")

if phase1_loss > 0.5:
    print(f"   ⚠️  Phase 1 loss high (>{0.5:.1f})")
    print(f"      This may limit Phase 2 performance")
    print(f"      Consider retraining Phase 1 with:")
    print(f"        - More epochs (20-30)")
    print(f"        - Lower learning rate")
    print(f"        - Target loss < 0.3")

print("\n" + "=" * 80)
print("COMPARISON WITH EXPECTATIONS")
print("=" * 80)

print("\nExpected After All Fixes:")
print("  Target: >65-75% accuracy (vs previous 54%)")
print(f"  Actual: {current_acc*100:.2f}%")

if current_acc >= 0.65:
    gap = (current_acc - previous_acc) * 100
    print(f"\n✅ SUCCESS! Achieved {gap:.1f}pp improvement")
    print("   All three fixes contributed to better results")
elif current_acc > previous_acc:
    gap = (current_acc - previous_acc) * 100
    print(f"\n🟡 PARTIAL SUCCESS: {gap:.1f}pp improvement")
    print("   Fixes helped, but didn't reach target")
    print("   May need Phase 1 improvements")
else:
    print(f"\n❌ NO IMPROVEMENT over previous 54%")
    print("   Investigation needed - fixes may not be applied correctly")

print("\n" + "=" * 80)
print("RECOMMENDATIONS")
print("=" * 80)

if current_acc < 0.60:
    print("\n🔴 CRITICAL: Low accuracy - Action required")
    print("\n1. Improve Phase 1:")
    print("   - Retrain with more epochs (target loss < 0.3)")
    print("   - Consider lower learning rate (3e-5)")
    print("   - Verify all fixes are active")
    
    print("\n2. Enhance Phase 2:")
    print("   - Increase classification head capacity")
    print("   - Try different dropout rates (0.05, 0.15)")
    print("   - Train for more epochs (15-20)")
    
    print("\n3. Debug:")
    print("   - Check if evaluation threshold is being used correctly")
    print("   - Verify embeddings are normalized in inference")
    print("   - Inspect confusion matrix for error patterns")

elif current_acc < 0.70:
    print("\n🟡 MODERATE: Acceptable but improvable")
    print("\n1. Fine-tune Phase 1 for lower loss")
    print("2. Experiment with Phase 2 hyperparameters")
    print("3. Consider data augmentation quality")

else:
    print("\n✅ EXCELLENT: High accuracy achieved!")
    print("\n1. Document successful configuration")
    print("2. Run full evaluation suite")
    print("3. Test on held-out data")

print("\n" + "=" * 80)
print("NEXT STEPS")
print("=" * 80)

print(f"\nCurrent Status: Phase 2 trained with {current_acc*100:.2f}% accuracy")

if current_acc >= 0.65:
    print("\n✅ Proceed to comprehensive evaluation:")
    print("   - Calculate precision, recall, F1 scores")
    print("   - Analyze confusion matrix")
    print("   - Test on validation set")
    print("   - Generate performance visualizations")
else:
    print("\n⚠️  Consider improvements before evaluation:")
    print("   - Retrain Phase 1 for lower loss")
    print("   - Adjust Phase 2 architecture")
    print("   - Verify all fixes are correctly applied")

print("\n" + "=" * 80)
print("SUMMARY")
print("=" * 80)

print(f"\n📊 Phase 2 Final Results:")
print(f"   Accuracy: {current_acc*100:.2f}%")
print(f"   Loss: {final_loss:.6f}")
print(f"   Training: {results['training_time']/60:.1f} min")

print(f"\n📈 Improvement over previous (54%):")
print(f"   Delta: {improvement:+.2f} percentage points")

print(f"\n🔧 All 3 Fixes Applied:")
print(f"   ✅ Normalization")
print(f"   ✅ Encoding consistency")
print(f"   ✅ Evaluation threshold")

if current_acc > previous_acc:
    print(f"\n🎉 FIXES WORKED - Training improved!")
else:
    print(f"\n🔍 INVESTIGATE - Expected improvement didn't materialize")

print("=" * 80)
