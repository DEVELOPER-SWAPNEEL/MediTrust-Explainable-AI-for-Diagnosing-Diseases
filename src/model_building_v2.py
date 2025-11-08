# ===============================================================
# 🚀 PHASE 4 v2 — Transfer Learning with EfficientNetB0
# ===============================================================
# File: src/model_building_v2.py
# Author: Swapneel Purohit
# Description: High-accuracy pneumonia classifier using pretrained EfficientNetB0
# ===============================================================

import os
import glob
import numpy as np
from termcolor import colored
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout, Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import plot_model
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau

# ===============================================================
# 🔧 CONFIG
# ===============================================================
DATA_ROOT = "../data"
PROCESSED_DIR = os.path.join(DATA_ROOT, "processed")
MODEL_DIR = "../models"
os.makedirs(MODEL_DIR, exist_ok=True)

IMG_SIZE = (224, 224, 3)
EPOCHS = 20
BATCH_SIZE = 16

# ===============================================================
# 📦 LOAD & MERGE CHUNKED DATA
# ===============================================================
def load_and_merge_npy(prefix, limit_per_part=None):
    """Merges chunked .npy parts like train_X_part0.npy, train_X_part1.npy"""
    files = sorted(glob.glob(os.path.join(PROCESSED_DIR, f"{prefix}_part*.npy")))
    if not files:
        raise FileNotFoundError(f"No .npy files found for prefix '{prefix}' in {PROCESSED_DIR}")

    arrays = []
    for f in files:
        print(colored(f"📂 Loading {os.path.basename(f)}", "blue"))
        arr = np.load(f, allow_pickle=True)
        if limit_per_part:
            arr = arr[:limit_per_part]
        arrays.append(arr)

    merged = np.concatenate(arrays, axis=0)
    print(colored(f"✅ Combined shape for {prefix}: {merged.shape}", "green"))
    return merged

# ===============================================================
# 🚀 LOAD TRAIN/VAL DATASETS
# ===============================================================
print(colored("🚀 Loading preprocessed data...", "yellow", attrs=["bold"]))

X_train = load_and_merge_npy("train_X")
y_train = load_and_merge_npy("train_y")
X_val   = load_and_merge_npy("val_X")
y_val   = load_and_merge_npy("val_y")

print(colored(f"✅ Training set: X={X_train.shape}, y={y_train.shape}", "green"))
print(colored(f"✅ Validation set: X={X_val.shape}, y={y_val.shape}", "green"))

# ===============================================================
# 🧹 DATA CLEANING & NORMALIZATION
# ===============================================================
X_train = X_train.astype("float32") / 255.0
X_val   = X_val.astype("float32") / 255.0

y_train = np.array(y_train).astype(np.float32).reshape(-1, 1)
y_val   = np.array(y_val).astype(np.float32).reshape(-1, 1)

y_train = np.clip(y_train, 0, 1)
y_val   = np.clip(y_val, 0, 1)

print(colored(f"📊 Labels — unique values: {np.unique(y_train)}", "cyan"))

# ===============================================================
# 🧠 BUILD MODEL — EfficientNetB0
# ===============================================================
print(colored("⚙️ Building EfficientNetB0 model...", "yellow", attrs=["bold"]))

def build_transfer_model(input_shape):
    base_model = EfficientNetB0(weights="imagenet", include_top=False, input_shape=input_shape)
    base_model.trainable = False  # freeze base layers for stability

    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dropout(0.4)(x)
    output = Dense(1, activation="sigmoid")(x)  # binary classification

    model = Model(inputs=base_model.input, outputs=output)
    model.compile(optimizer=Adam(learning_rate=1e-5),
                  loss="binary_crossentropy",
                  metrics=["accuracy"])
    return model

model = build_transfer_model(IMG_SIZE)
print(colored("✅ Model built successfully!", "green", attrs=["bold"]))
model.summary()

# ===============================================================
# 💾 CALLBACKS — Best Model Saver, Early Stop, LR Scheduler
# ===============================================================
checkpoint_path = os.path.join(MODEL_DIR, "best_model.keras")

callbacks = [
    ModelCheckpoint(checkpoint_path, monitor="val_accuracy", save_best_only=True, verbose=1),
    EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True, verbose=1),
    ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=2, min_lr=1e-7, verbose=1)
]

# ===============================================================
# 🚀 TRAIN MODEL
# ===============================================================
print(colored("🚀 Starting training...", "yellow", attrs=["bold"]))

history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    callbacks=callbacks,
    verbose=1
)

print(colored("🎯 Training complete!", "green", attrs=["bold"]))
print(colored(f"💾 Best model saved at: {checkpoint_path}", "cyan"))

# ===============================================================
# 🖼️ SAVE MODEL STRUCTURE & VISUALIZATION
# ===============================================================
model_json = model.to_json()
with open(os.path.join(MODEL_DIR, "efficientnet_model_structure.json"), "w") as json_file:
    json_file.write(model_json)

try:
    plot_model(model, to_file=os.path.join(MODEL_DIR, "efficientnet_architecture.png"),
               show_shapes=True, show_layer_names=True)
    print(colored("🖼️ Saved model visualization at models/efficientnet_architecture.png", "green"))
except Exception:
    print(colored("⚠️ Skipped visual model plot (install pydot & graphviz if needed)", "yellow"))

print(colored("✅ PHASE 4 v2 Completed Successfully!", "magenta", attrs=["bold"]))
