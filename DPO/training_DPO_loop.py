import os
import time
import torch
import matplotlib.pyplot as plt
from torch.optim import AdamW
from torch.utils.data import DataLoader, random_split
from DPO.DPO_dataset import DPOMusicDataset
from DPO.DPO_loss import dpo_loss
from DPO.extract_proba import get_batch_logprobs


if torch.cuda.is_available():
    device = torch.device("cuda")
    print("Using CUDA")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
    print("Using Apple Silicon GPU (MPS)")
else:
    device = torch.device("cpu")
    print("Using CPU")

# load the full dataset from the file maximus_dpo_preferences.pt
print("loading full DPO dataset...")
full_dataset = DPOMusicDataset("maximus_dpo_preferences.pt")

# calculate sizes for an 90% Training / 10% Validation split
dataset_size = len(full_dataset)
val_size = int(0.10 * dataset_size)
train_size = dataset_size - val_size

# randomly split the dataset
train_dataset, val_dataset = random_split(
    full_dataset,
    [train_size, val_size]
)

print(f"Dataset split: {train_size} for training, {val_size} for validation.")

# create the two DataLoaders
train_loader = DataLoader(
    train_dataset,
    batch_size=16,
    shuffle=True,
    num_workers=4,
    pin_memory=(device.type == "cuda")
)

val_loader = DataLoader(
    val_dataset,
    batch_size=16,
    shuffle=False,
    num_workers=4,
    pin_memory=(device.type == "cuda")
)

print("loading models")

# active model
active_model = load_my_maximus_phase1().to(device)
active_model.train()

# reference model
ref_model = load_my_maximus_phase1().to(device)
ref_model.eval()

# freeze the reference model weights
for param in ref_model.parameters():
    param.requires_grad = False

optimizer = AdamW(active_model.parameters(), lr=5e-6)
beta = 0.1
epochs = 3

# history 
train_loss_history = []
val_loss_history = []
train_reward_w_history = []
train_reward_l_history = []
val_reward_w_history = []
val_reward_l_history = []

# best model on validation set
best_val_loss = float("inf")
best_epoch = -1
best_model_path = None
os.makedirs("checkpoints", exist_ok=True)

# DPO training loop
print("starting DPO training...")
total_start = time.time()

for epoch in range(epochs):
    loop_start = time.time()
    train_loss = 0.0
    train_reward_w = 0.0
    train_reward_l = 0.0

    # batch_w = winner | batch_l = loser
    for batch_idx, (prompt, batch_w, batch_l) in enumerate(train_loader):

        prompt = prompt.to(device, non_blocking=True)
        batch_w = batch_w.to(device, non_blocking=True)
        batch_l = batch_l.to(device, non_blocking=True)

        optimizer.zero_grad()

        # active model pass
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=(device.type in ["cuda", "mps"])):
            logits_active_w, _ = active_model(batch_w)
            logits_active_l, _ = active_model(batch_l)

        logprobs_active_w = get_batch_logprobs(logits_active_w, batch_w)
        logprobs_active_l = get_batch_logprobs(logits_active_l, batch_l)

        # reference model pass (no gradients)
        with torch.no_grad():
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=(device.type in ["cuda", "mps"])):
                logits_ref_w, _ = ref_model(batch_w)
                logits_ref_l, _ = ref_model(batch_l)

            logprobs_ref_w = get_batch_logprobs(logits_ref_w, batch_w)
            logprobs_ref_l = get_batch_logprobs(logits_ref_l, batch_l)

        # DPO loss calculation
        loss, reward_w, reward_l = dpo_loss(
            logprobs_active_w, logprobs_ref_w,
            logprobs_active_l, logprobs_ref_l,
            beta=beta
        )

        # backward pass
        loss.backward()

        # clip gradients 
        torch.nn.utils.clip_grad_norm_(active_model.parameters(), max_norm=1.0)
        optimizer.step()

        train_loss += loss.item()
        train_reward_w += reward_w
        train_reward_l += reward_l
    avg_train_loss = train_loss / len(train_loader)
    avg_train_reward_w = train_reward_w / len(train_loader)
    avg_train_reward_l = train_reward_l / len(train_loader)

    # validation
    active_model.eval()
    val_loss = 0.0
    val_reward_w = 0.0
    val_reward_l = 0.0
    with torch.no_grad():
        for batch_idx, (prompt, batch_w, batch_l) in enumerate(val_loader):
            prompt = prompt.to(device, non_blocking=True)
            batch_w = batch_w.to(device, non_blocking=True)
            batch_l = batch_l.to(device, non_blocking=True)

            with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=(device.type in ["cuda", "mps"])):
                logits_active_w, _ = active_model(batch_w)
                logits_active_l, _ = active_model(batch_l)

            logprobs_active_w = get_batch_logprobs(logits_active_w, batch_w)
            logprobs_active_l = get_batch_logprobs(logits_active_l, batch_l)

            with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=(device.type in ["cuda", "mps"])):
                logits_ref_w, _ = ref_model(batch_w)
                logits_ref_l, _ = ref_model(batch_l)

            logprobs_ref_w = get_batch_logprobs(logits_ref_w, batch_w)
            logprobs_ref_l = get_batch_logprobs(logits_ref_l, batch_l)

            # DPO loss calculation
            loss, reward_w, reward_l = dpo_loss(
                logprobs_active_w, logprobs_ref_w,
                logprobs_active_l, logprobs_ref_l,
                beta=beta
            )
            val_loss += loss.item()
            val_reward_w += reward_w
            val_reward_l += reward_l
    avg_val_loss = val_loss / len(val_loader)
    avg_val_reward_w = val_reward_w / len(val_loader)
    avg_val_reward_l = val_reward_l / len(val_loader)


    train_loss_history.append(avg_train_loss)
    val_loss_history.append(avg_val_loss)
    train_reward_w_history.append(avg_train_reward_w)
    train_reward_l_history.append(avg_train_reward_l)
    val_reward_w_history.append(avg_val_reward_w)
    val_reward_l_history.append(avg_val_reward_l)



    print(f"Epoch {epoch+1}/{epochs}  Train Loss: {avg_train_loss:.4f}")
    print(f"Val Loss: {avg_val_loss:.4f}")
    print(f"Val Reward Winner: {avg_val_reward_w:.4f}")
    print(f"Val Reward Loser: {avg_val_reward_l:.4f}")

    # save the best model only (best  on validation dataset)
    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        best_epoch = epoch + 1
        best_model_path = f"checkpoints/maximus_dpo_best_loop_{best_epoch}.pt"
        torch.save(active_model.state_dict(), best_model_path)
        print(f"New best model saved: {best_model_path}")

    active_model.train()

total_training_time = time.time() - total_start
print(f"Total DPO training time: {total_training_time:.2f}s")

# Figures
epochs_x = list(range(1, epochs + 1))
plt.figure(figsize=(12, 8))

plt.subplot(2, 1, 1)
plt.plot(epochs_x, train_loss_history, label="Train Loss")
plt.plot(epochs_x, val_loss_history, label="Validation Loss")
if best_epoch != -1:
    plt.scatter([best_epoch], [best_val_loss], color="red", zorder=5, label="Best Model")
    plt.annotate(
        f"Best model (loop {best_epoch})\nval loss={best_val_loss:.4f}",
        (best_epoch, best_val_loss),
        textcoords="offset points",
        xytext=(10, 10)
    )
plt.title("DPO Loss Curves")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.grid(True, alpha=0.3)

plt.subplot(2, 1, 2)
plt.plot(epochs_x, train_reward_w_history, label="Train Reward Winner")
plt.plot(epochs_x, train_reward_l_history, label="Train Reward Loser")
plt.plot(epochs_x, val_reward_w_history, label="Validation Reward Winner")
plt.plot(epochs_x, val_reward_l_history, label="Validation Reward Loser")
plt.title("DPO Reward Curves")
plt.xlabel("Epoch")
plt.ylabel("Reward")
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
curve_path = "checkpoints/dpo_training_validation_curves.png"
plt.savefig(curve_path, dpi=150)
plt.show()
print(f"Curves saved to: {curve_path}")

if best_model_path is not None:
    print(f"Best model at loop {best_epoch} with val loss {best_val_loss:.4f}")
    print(f"Best model path: {best_model_path}")
