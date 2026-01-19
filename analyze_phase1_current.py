#!/usr/bin/env python3
"""
Analyze Phase 1 training results and compare with previous runs.
"""
import json
from pathlib import Path

print("=" * 80)
print("PHASE 1 TRAINING ANALYSIS")
print("=" * 80)

# Load current results
results_path = Path("phase1_pretraining/training_results.json")
config_path = Path("phase1_pretraining/training_config.json")

with open(results_path) as f:
    results = json.load(f)

with open(config_path) as f:
    config = json.load(f)

print("\n" + "=" * 80)
print("CURRENT TRAINING RESULTS")
print("=" * 80)

print(f"\nFinal Loss: {results['final_loss']:.6f}")
print(
    f"Training Time: {results['training_time']:.1f} seconds ({results['training_time']/60:.1f} minutes)")
print(f"Total Epochs: {len(results['epoch_losses'])}")
print(f"Total Batches: {results['total_batches']}")
print(f"Total Samples: {results['total_samples']}")
print(f"Total Triplets: {results['metrics']['total_triplets_created']}")

print(f"\nKey Configuration:")
print(f"  Learning Rate: {config['learning_rate']}")
print(f"  Temperature: {config['temperature']}")
print(f"  Pathway Weight: {config['pathway_weight']}")
print(
    f"  Augmentation Positive Ratio: {config['augmentation_positive_ratio']}")
print(f"  Batch Size: {config['batch_size']}")
print(f"  Projection Dim: {config['projection_dim']}")

print(f"\n" + "=" * 80)
print("EPOCH-BY-EPOCH LOSS PROGRESSION")
print("=" * 80)

print(f"\n{'Epoch':<8} {'Loss':<12} {'Delta':<12} {'% Change':<12}")
print("-" * 80)

epoch_losses = results['epoch_losses']
for i, loss in enumerate(epoch_losses, 1):
    if i == 1:
        delta = "-"
        pct_change = "-"
    else:
        delta = f"{loss - epoch_losses[i-2]:+.6f}"
        pct_change = f"{((loss - epoch_losses[i-2]) / epoch_losses[i-2] * 100):+.2f}%"

    print(f"{i:<8} {loss:<12.6f} {delta:<12} {pct_change:<12}")

# Calculate convergence metrics
initial_loss = epoch_losses[0]
final_loss = epoch_losses[-1]
best_loss = min(epoch_losses)
best_epoch = epoch_losses.index(best_loss) + 1

total_improvement = ((initial_loss - final_loss) / initial_loss) * 100
best_improvement = ((initial_loss - best_loss) / initial_loss) * 100

print(f"\n" + "=" * 80)
print("CONVERGENCE METRICS")
print("=" * 80)

print(f"\nInitial Loss (Epoch 1): {initial_loss:.6f}")
print(f"Final Loss (Epoch {len(epoch_losses)}): {final_loss:.6f}")
print(f"Best Loss (Epoch {best_epoch}): {best_loss:.6f}")
print(f"\nTotal Improvement: {total_improvement:.2f}%")
print(f"Best Improvement: {best_improvement:.2f}%")

# Check for concerning patterns
print(f"\n" + "=" * 80)
print("TRAINING PATTERN ANALYSIS")
print("=" * 80)

# Check last 5 epochs for convergence
last_5_losses = epoch_losses[-5:]
last_5_avg = sum(last_5_losses) / len(last_5_losses)
last_5_std = (sum((x - last_5_avg) ** 2 for x in last_5_losses) /
              len(last_5_losses)) ** 0.5

print(f"\nLast 5 Epochs:")
print(f"  Average Loss: {last_5_avg:.6f}")
print(f"  Std Deviation: {last_5_std:.6f}")
print(f"  Min: {min(last_5_losses):.6f}")
print(f"  Max: {max(last_5_losses):.6f}")

# Check for oscillation
oscillations = 0
for i in range(1, len(epoch_losses) - 1):
    if (epoch_losses[i] > epoch_losses[i-1] and epoch_losses[i] > epoch_losses[i+1]) or \
       (epoch_losses[i] < epoch_losses[i-1] and epoch_losses[i] < epoch_losses[i+1]):
        oscillations += 1

print(f"\nOscillation Count: {oscillations} (higher = less stable)")

# Check if loss increased in final epochs
final_increasing = sum(1 for i in range(len(epoch_losses)-3, len(epoch_losses)-1)
                       if epoch_losses[i+1] > epoch_losses[i])

if final_increasing >= 2:
    print(
        f"⚠️  WARNING: Loss increased in final epochs ({final_increasing}/3)")
    print("   This suggests training may not have fully converged")

print(f"\n" + "=" * 80)
print("COMPARISON WITH EXPECTED RESULTS")
print("=" * 80)

# Based on previous discussion, expected result was ~0.062 with LR=0.000075
print("\nPrevious Best Result (with all 3 fixes):")
print("  Expected Final Loss: ~0.062")
print("  Learning Rate: 0.000075")
print("  Configuration: All normalization fixes applied")

print(f"\nCurrent Result:")
print(f"  Actual Final Loss: {final_loss:.6f}")
print(f"  Learning Rate: {config['learning_rate']}")
print(
    f"  Loss Difference: {(final_loss - 0.062):.6f} ({((final_loss - 0.062) / 0.062 * 100):+.1f}%)")

if final_loss > 0.1:
    print("\n⚠️  ISSUE: Final loss is significantly higher than expected!")
    print("\nPossible Causes:")
    print("  1. Model normalization not applied in forward pass")
    print("  2. Encoding consistency issue (normalize_embeddings settings)")
    print("  3. Learning rate may need adjustment")
    print("  4. Training needs more epochs to converge")
    print("  5. Evaluation fix changed similarity calculations")

print(f"\n" + "=" * 80)
print("RECOMMENDATIONS")
print("=" * 80)

if final_loss > 0.5:
    print("\n🔴 CRITICAL: Loss is too high (>0.5)")
    print("\nAction Items:")
    print("  1. ✅ Verify F.normalize() is in trainer.py forward pass")
    print("  2. ✅ Check embedding_cache.py normalize_embeddings=False")
    print("  3. ⚠️  Consider running more epochs (current: 10)")
    print("  4. ⚠️  Loss plateaued around 0.60-0.61, may need:")
    print("     - Lower learning rate (try 0.00005)")
    print("     - Learning rate scheduling")
    print("     - More training data")
elif final_loss > 0.2:
    print("\n🟡 MODERATE: Loss is higher than optimal (0.2-0.5)")
    print("\nAction Items:")
    print("  1. Check if all normalization fixes are applied")
    print("  2. Consider training for more epochs")
    print("  3. Verify evaluation threshold changes didn't affect training")
elif final_loss > 0.1:
    print("\n🟢 GOOD: Loss is reasonable (0.1-0.2)")
    print("  Training converged well, ready for Phase 2")
else:
    print("\n✅ EXCELLENT: Loss is very low (<0.1)")
    print("  Training converged optimally, ready for Phase 2")

print("\n" + "=" * 80)
print("NEXT STEPS")
print("=" * 80)

print("\nBased on current results:")
print(f"  Final Loss: {final_loss:.6f}")
print(f"  Best Loss: {best_loss:.6f} (Epoch {best_epoch})")

if final_loss < best_loss * 1.05:  # Within 5% of best
    print("\n✅ Training converged successfully")
    print("   Proceed to Phase 2 fine-tuning")
else:
    print(
        f"\n⚠️  Final loss is {((final_loss - best_loss) / best_loss * 100):.1f}% worse than best")
    print("   Consider:")
    print("   - Retraining from best checkpoint")
    print("   - Running additional epochs")
    print("   - Adjusting learning rate schedule")

print("=" * 80)
