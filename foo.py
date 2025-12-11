"""
QM9 SchNet - Single Batch Overfitting Test
Useful for debugging: if the model can't overfit one batch, something is wrong!
"""

import torch
from torch_geometric.datasets import QM9
from torch_geometric.loader import DataLoader
from torch_geometric.nn import SchNet
import torch.nn.functional as F

# Setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}\n")

# ============================================================================
# 1. LOAD SINGLE BATCH
# ============================================================================

print("Loading QM9 dataset...")
dataset = QM9(root="data/QM9")

# Select target property (0 = dipole moment)
TARGET_IDX = 0
dataset._data.y = dataset._data.y[:, TARGET_IDX : TARGET_IDX + 1]

# Create a small dataset (just 32 molecules)
BATCH_SIZE = 32
small_dataset = dataset[:BATCH_SIZE]

# Create loader with single batch
train_loader = DataLoader(small_dataset, batch_size=BATCH_SIZE, shuffle=False)

# Get the single batch
single_batch = next(iter(train_loader))
single_batch = single_batch.to(device)

print(f"✓ Single batch loaded:")
print(f"  Batch size: {single_batch.num_graphs}")
print(f"  Total atoms: {single_batch.num_nodes}")
print(f"  Total bonds: {single_batch.num_edges}")
print(
    f"  Target values range: [{single_batch.y.min():.3f}, {single_batch.y.max():.3f}]\n"
)

# ============================================================================
# 2. INITIALIZE SCHNET MODEL
# ============================================================================

print("Initializing SchNet model...")

model = SchNet(
    hidden_channels=128,
    num_filters=128,
    num_interactions=6,
    num_gaussians=50,
    cutoff=10.0,
).to(device)

num_params = sum(p.numel() for p in model.parameters())
print(f"Model parameters: {num_params:,}\n")

# Setup optimizer with appropriate learning rate
optimizer = torch.optim.Adam(model.parameters(), lr=0.002)

# ============================================================================
# 3. OVERFIT ON SINGLE BATCH
# ============================================================================

print("Overfitting on single batch...")
print("(If this doesn't reach near-zero loss, something is wrong!)")
print("-" * 70)
print(f"{'Iteration':<12} {'Loss':<15} {'MAE':<15}")
print("-" * 70)

model.train()
losses = []

NUM_ITERATIONS = 1000

for iteration in range(1, NUM_ITERATIONS + 1):
    # Zero gradients
    optimizer.zero_grad()

    # Forward pass on the same batch
    out = model(z=single_batch.z, pos=single_batch.pos, batch=single_batch.batch)

    # Compute loss
    loss = F.mse_loss(out, single_batch.y)

    # Backward pass
    loss.backward()

    # Clip gradients for stability
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

    optimizer.step()

    # Calculate MAE
    mae = torch.mean(torch.abs(out - single_batch.y)).item()

    losses.append(loss.item())

    # Print every 10 iterations
    if iteration % 10 == 0 or iteration == 1:
        print(f"{iteration:<12} {loss.item():<15.6f} {mae:<15.6f}")

print("-" * 70)

# ============================================================================
# 4. FINAL EVALUATION
# ============================================================================

print("\n" + "=" * 70)
print("FINAL RESULTS")
print("=" * 70)

model.eval()
with torch.no_grad():
    final_out = model(z=single_batch.z, pos=single_batch.pos, batch=single_batch.batch)
    final_loss = F.mse_loss(final_out, single_batch.y).item()
    final_mae = torch.mean(torch.abs(final_out - single_batch.y)).item()
    final_rmse = final_loss**0.5

print(f"\nFinal Metrics on Training Batch:")
print(f"  MSE:  {final_loss:.6f}")
print(f"  MAE:  {final_mae:.6f}")
print(f"  RMSE: {final_rmse:.6f}")

# Show some predictions vs actual
print(f"\n{'Actual':<12} {'Predicted':<12} {'Error':<12}")
print("-" * 40)
for i in range(min(10, len(single_batch.y))):
    actual = single_batch.y[i].item()
    predicted = final_out[i].item()
    error = abs(actual - predicted)
    print(f"{actual:<12.4f} {predicted:<12.4f} {error:<12.6f}")

# ============================================================================
# 5. DIAGNOSIS
# ============================================================================

print("\n" + "=" * 70)
print("DIAGNOSIS")
print("=" * 70)

initial_loss = losses[0]
final_loss_value = losses[-1]
reduction = (initial_loss - final_loss_value) / initial_loss * 100

print(f"\nLoss Reduction:")
print(f"  Initial loss: {initial_loss:.6f}")
print(f"  Final loss:   {final_loss_value:.6f}")
print(f"  Reduction:    {reduction:.1f}%")

if final_loss_value < 0.001:
    print("\n✅ SUCCESS! Model successfully overfitted the batch.")
    print("   Your model and training setup are working correctly!")
elif final_loss_value < 0.01:
    print("\n⚠️  PARTIAL SUCCESS. Loss is low but not near zero.")
    print("   Consider: More iterations or higher learning rate")
elif final_loss_value < 0.1:
    print("\n⚠️  CONCERNING. Loss decreased but didn't converge well.")
    print("   Possible issues:")
    print("   - Learning rate might be too low")
    print("   - Model capacity might be insufficient")
    print("   - Check if gradients are flowing properly")
else:
    print("\n❌ PROBLEM DETECTED! Model failed to overfit.")
    print("   Likely issues:")
    print("   - Bug in model forward pass")
    print("   - Incorrect loss function")
    print("   - Gradient computation problem")
    print("   - Learning rate too low")

# Check for NaN or Inf
if torch.isnan(final_out).any():
    print("\n❌ WARNING: NaN values detected in predictions!")
if torch.isinf(final_out).any():
    print("\n❌ WARNING: Inf values detected in predictions!")

print("\n" + "=" * 70)
print("Why This Test Matters:")
print("=" * 70)
print(
    """
Overfitting a single batch is a crucial sanity check:

1. If the model CAN overfit one batch:
   ✓ Model architecture is correct
   ✓ Loss function is appropriate
   ✓ Optimizer is working
   ✓ Gradients are flowing
   ✓ Data loading is correct

2. If the model CANNOT overfit one batch:
   ✗ Something is fundamentally broken
   ✗ Need to debug before training on full dataset

Expected: Loss should drop to < 0.001 within a few hundred iterations.
"""
)

print("\n✓ Overfitting test complete!")
