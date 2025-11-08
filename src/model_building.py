"""
PHASE 4 — MODEL BUILDING AND TRAINING (Optimized CNN)
------------------------------------------------------
✅ Supports Chunked .npy files
✅ Automatically merges data parts
✅ Uses strong augmentation, normalization, and deeper CNN
✅ Should reach >90% validation accuracy on clean datasets
"""

import os
import numpy as np
from termcolor import colored
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
)
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.utils import shuffle

# -------------------------------------------------------------------
# ⚙️ Define data paths
# -------------------------------------------------------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_ROOT = os.path.join(os.path.dirname(SCRIPT_DIR), "data")
PROCESSED_DIR = os.path.join(DATA_ROOT, "processed")
MODEL_DIR = os.path.join(DATA_ROOT, "models")
os.makedirs(MODEL_DIR, exist_ok=True)

# -------------------------------------------------------------------
# 🧩 Helper: Load and merge chunked .npy parts
# -------------------------------------------------------------------
def load_chunked_data(prefix):
    files = sorted(
        [f for f in os.listdir(PROCESSED_DIR) if f.startswith(prefix) and f.endswith(".npy")],
        key=lambda x: int(''.join(filter(str.isdigit, x)) or 0)
    )
    if not files:
        print(colored(f"❌ Error: No files found for prefix '{prefix}'", "red"))
        return None

    arrays = []
    for file in files:
        path = os.path.join(PROCESSED_DIR, file)
        print(colored(f"📦 Loading {file}", "cyan"))
        arrays.append(np.load(path, allow_pickle=True))
    combined = np.concatenate(arrays, axis=0)
    print(colored(f"✅ Combined shape for {prefix}: {combined.shape}", "green"))
    return combined

# -------------------------------------------------------------------
# 🚀 Load the datasets
# -------------------------------------------------------------------
print(colored("🚀 Loading preprocessed data...", "yellow", attrs=["bold"]))
print("=" * 80)

X_train = load_chunked_data("train_X_part")
y_train = load_chunked_data("train_y_part")
X_val = load_chunked_data("val_X_part")
y_val = load_chunked_data("val_y_part")

if X_train is None or y_train is None:
    raise FileNotFoundError("❌ Processed .npy files not found! Please run preprocessing first.")

# Normalize & Shuffle
X_train = X_train.astype("float32") / 255.0
X_val = X_val.astype("float32") / 255.0
X_train, y_train = shuffle(X_train, y_train, random_state=42)

print(colored(f"✅ Training set: X={X_train.shape}, y={y_train.shape}", "green"))
print(colored(f"✅ Validation set: X={X_val.shape}, y={y_val.shape}", "green"))

# -------------------------------------------------------------------
# 🧠 Data Augmentation
# -------------------------------------------------------------------
datagen = ImageDataGenerator(
    rotation_range=15,
    width_shift_range=0.1,
    height_shift_range=0.1,
    zoom_range=0.15,
    shear_range=0.1,
    horizontal_flip=True,
    fill_mode="nearest"
)
datagen.fit(X_train)

# -------------------------------------------------------------------
# 🧱 Optimized CNN Architecture
# -------------------------------------------------------------------
def build_model(input_shape):
    model = Sequential([
        Conv2D(32, (3,3), activation='relu', padding='same', input_shape=input_shape),
        BatchNormalization(),
        Conv2D(32, (3,3), activation='relu', padding='same'),
        BatchNormalization(),
        MaxPooling2D(2,2),
        Dropout(0.25),

        Conv2D(64, (3,3), activation='relu', padding='same'),
        BatchNormalization(),
        Conv2D(64, (3,3), activation='relu', padding='same'),
        BatchNormalization(),
        MaxPooling2D(2,2),
        Dropout(0.3),

        Conv2D(128, (3,3), activation='relu', padding='same'),
        BatchNormalization(),
        Conv2D(128, (3,3), activation='relu', padding='same'),
        BatchNormalization(),
        MaxPooling2D(2,2),
        Dropout(0.4),

        Flatten(),
        Dense(256, activation='relu'),
        BatchNormalization(),
        Dropout(0.5),
        Dense(1, activation='sigmoid')
    ])

    optimizer = Adam(learning_rate=1e-4)
    model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])
    return model

# -------------------------------------------------------------------
# 🚀 Training Setup
# -------------------------------------------------------------------
input_shape = X_train.shape[1:]
model = build_model(input_shape)
model.summary()

checkpoint_path = os.path.join(MODEL_DIR, "best_high_acc_model.keras")

callbacks = [
    EarlyStopping(monitor="val_loss", patience=7, restore_best_weights=True),
    ModelCheckpoint(checkpoint_path, monitor="val_accuracy", save_best_only=True, mode="max"),
    ReduceLROnPlateau(monitor="val_loss", factor=0.3, patience=3, min_lr=1e-6, verbose=1)
]

# -------------------------------------------------------------------
# 🏋️ Train Model with Augmentation
# -------------------------------------------------------------------
print(colored("\n🚀 Starting high-accuracy training...", "yellow", attrs=["bold"]))
history = model.fit(
    datagen.flow(X_train, y_train, batch_size=32),
    validation_data=(X_val, y_val),
    epochs=40,
    callbacks=callbacks,
    verbose=1
)

# -------------------------------------------------------------------
# 💾 Save Final Model
# -------------------------------------------------------------------
final_path = os.path.join(MODEL_DIR, "final_high_acc_model.keras")
model.save(final_path)
print(colored(f"\n✅ Model saved to: {final_path}", "green"))

print(colored("\n🎯 PHASE 4 (High Accuracy CNN) completed successfully!", "cyan", attrs=["bold"]))

