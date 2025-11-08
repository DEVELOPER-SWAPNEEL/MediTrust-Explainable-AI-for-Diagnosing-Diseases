# ===============================================================
# 🚀 PHASE 4 v3.1 — EfficientNetB0 + Grad-CAM Explainable AI
# ===============================================================
# Author: Swapneel Purohit
# Description: High-accuracy pneumonia classifier with EfficientNetB0
#              and Grad-CAM visualization for model transparency
# ===============================================================

import os
import glob
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from termcolor import colored
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import plot_model
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau

# ===============================================================
# ⚙️ CONFIGURATION
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
# 🧠 AUTO HANDLE GRAYSCALE OR RGB
# ===============================================================
def ensure_rgb(images):
    """Automatically converts grayscale (H, W, 1) → RGB (H, W, 3)"""
    if len(images.shape) == 3:  # (N, H, W)
        images = np.expand_dims(images, -1)
    if images.shape[-1] == 1:
        print(colored("🩶 Converting grayscale → RGB...", "yellow"))
        images = np.repeat(images, 3, axis=-1)
    return images

# ===============================================================
# 🚀 LOAD DATA
# ===============================================================
print(colored("🚀 Loading preprocessed data...", "yellow", attrs=["bold"]))
X_train = ensure_rgb(load_and_merge_npy("train_X"))
y_train = load_and_merge_npy("train_y")
X_val   = ensure_rgb(load_and_merge_npy("val_X"))
y_val   = load_and_merge_npy("val_y")

# ===============================================================
# 🧹 NORMALIZE & CLEAN
# ===============================================================
X_train = X_train.astype("float32") / 255.0
X_val   = X_val.astype("float32") / 255.0

y_train = np.array(y_train).astype(np.float32).reshape(-1, 1)
y_val   = np.array(y_val).astype(np.float32).reshape(-1, 1)
y_train = np.clip(y_train, 0, 1)
y_val   = np.clip(y_val, 0, 1)

print(colored(f"✅ Training set: X={X_train.shape}, y={y_train.shape}", "green"))
print(colored(f"✅ Validation set: X={X_val.shape}, y={y_val.shape}", "green"))
print(colored(f"📊 Labels — unique values: {np.unique(y_train)}", "cyan"))

# ===============================================================
# 🧠 BUILD MODEL — EfficientNetB0
# ===============================================================
print(colored("⚙️ Building EfficientNetB0 model...", "yellow", attrs=["bold"]))

# ===============================================================
# 🧠 BUILD MODEL — EfficientNetB0
# ===============================================================
print(colored("⚙️ Building EfficientNetB0 model...", "yellow", attrs=["bold"]))

# 🧩 Auto-fix for grayscale data — ensures RGB input for EfficientNet
if X_train.shape[-1] != 3:
    print(colored("⚠️ Detected grayscale input — converting to RGB (repeating channels)...", "yellow"))
    X_train = np.repeat(X_train, 3, axis=-1)
    X_val = np.repeat(X_val, 3, axis=-1)

def build_transfer_model(input_shape):
    # Force input to RGB shape
    base_model = EfficientNetB0(weights="imagenet", include_top=False, input_shape=(224, 224, 3))
    base_model.trainable = False


    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dropout(0.4)(x)
    output = Dense(1, activation="sigmoid")(x)

    model = Model(inputs=base_model.input, outputs=output)
    model.compile(optimizer=Adam(learning_rate=1e-5),
                  loss="binary_crossentropy",
                  metrics=["accuracy"])
    return model

model = build_transfer_model(IMG_SIZE)
model.summary()
print(colored("✅ Model built successfully!", "green", attrs=["bold"]))

# ===============================================================
# 💾 CALLBACKS
# ===============================================================
checkpoint_path = os.path.join(MODEL_DIR, "best_model_v3_1.keras")

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
# 🔍 GRAD-CAM — Explainability
# ===============================================================
def generate_gradcam(model, img_array, layer_name=None):
    if layer_name is None:
        layer_name = [l.name for l in model.layers if 'conv' in l.name][-1]

    grad_model = Model(inputs=model.inputs,
                       outputs=[model.get_layer(layer_name).output, model.output])

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        loss = predictions[:, 0]

    grads = tape.gradient(loss, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_outputs = conv_outputs[0]
    heatmap = tf.reduce_mean(tf.multiply(pooled_grads, conv_outputs), axis=-1)

    heatmap = np.maximum(heatmap, 0) / np.max(heatmap)
    return heatmap.numpy()

def show_gradcam(image, heatmap, alpha=0.4):
    heatmap = np.uint8(255 * heatmap)
    heatmap = np.expand_dims(heatmap, axis=-1)
    heatmap = np.repeat(heatmap, 3, axis=-1)
    heatmap = tf.image.resize(heatmap, (image.shape[0], image.shape[1])).numpy()
    plt.imshow(image)
    plt.imshow(heatmap, cmap='jet', alpha=alpha)
    plt.axis('off')
    plt.show()

# 🔬 Show Grad-CAM on few validation samples
print(colored("🩺 Generating Grad-CAM explainability...", "yellow"))
for i in range(3):
    img = X_val[i]
    input_img = np.expand_dims(img, axis=0)
    heatmap = generate_gradcam(model, input_img)
    print(colored(f"🖼️ Sample {i+1} Grad-CAM:", "cyan"))
    show_gradcam(img, heatmap)

# ===============================================================
# 🖼️ SAVE MODEL STRUCTURE
# ===============================================================
model_json = model.to_json()
with open(os.path.join(MODEL_DIR, "efficientnet_v3_1_structure.json"), "w") as f:
    f.write(model_json)

try:
    plot_model(model, to_file=os.path.join(MODEL_DIR, "efficientnet_v3_1_architecture.png"),
               show_shapes=True, show_layer_names=True)
    print(colored("🖼️ Saved model visualization!", "green"))
except Exception:
    print(colored("⚠️ Skipped model plot (install pydot & graphviz if needed)", "yellow"))

print(colored("✅ PHASE 4 v3.1 Completed Successfully!", "magenta", attrs=["bold"]))
