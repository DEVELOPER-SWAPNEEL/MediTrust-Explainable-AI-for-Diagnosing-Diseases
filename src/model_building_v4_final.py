import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
import matplotlib.pyplot as plt

# ============================================================
# CONFIG
# ============================================================
# Auto-detect data directory relative to project root
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(os.path.dirname(SCRIPT_DIR), "processed")
os.makedirs(DATA_DIR, exist_ok=True)

IMG_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 20
MODEL_PATH = "models/efficientnetb0_best.h5"

# ============================================================
# DATA LOADING UTILITY
# ============================================================
def load_data_parts(prefix):
    """Load and combine all .npy parts for a dataset prefix."""
    parts = sorted([f for f in os.listdir(DATA_DIR) if f.startswith(prefix)])
    if not parts:
        raise FileNotFoundError(f"No files found for prefix '{prefix}'")

    arrays = []
    for part in parts:
        path = os.path.join(DATA_DIR, part)
        print(f"📂 Loading {part}")
        arrays.append(np.load(path))
    combined = np.concatenate(arrays, axis=0)
    print(f"✅ Combined shape for {prefix}: {combined.shape}")
    return combined

def ensure_rgb(x):
    """Ensure data has 3 color channels (RGB)."""
    if x.ndim == 3:  # (N, H, W)
        x = np.expand_dims(x, -1)
    if x.shape[-1] == 1:
        x = np.repeat(x, 3, axis=-1)
    return x

# ============================================================
# DATA LOADING
# ============================================================
print("🚀 Loading preprocessed data...")

X_train = load_data_parts("train_X_part")
y_train = load_data_parts("train_y_part")
X_val = load_data_parts("val_X_part")
y_val = load_data_parts("val_y_part")

# Ensure RGB compatibility
X_train = ensure_rgb(X_train)
X_val = ensure_rgb(X_val)

# Normalize images
X_train = X_train / 255.0
X_val = X_val / 255.0

# Ensure labels are column vectors
y_train = y_train.reshape(-1, 1)
y_val = y_val.reshape(-1, 1)

print(f"✅ Training set: X={X_train.shape}, y={y_train.shape}")
print(f"✅ Validation set: X={X_val.shape}, y={y_val.shape}")

# ============================================================
# MODEL CREATION (EfficientNetB0)
# ============================================================
def build_transfer_model(img_size=(224, 224, 3)):
    print("⚙️ Building EfficientNetB0 model...")
    base_model = EfficientNetB0(weights="imagenet", include_top=False, input_shape=img_size)
    base_model.trainable = False  # Freeze base layers for transfer learning

    inputs = Input(shape=img_size)
    x = base_model(inputs, training=False)
    x = GlobalAveragePooling2D()(x)
    x = Dropout(0.3)(x)
    outputs = Dense(1, activation="sigmoid")(x)
    model = Model(inputs, outputs)

    model.compile(optimizer=Adam(1e-4), loss="binary_crossentropy", metrics=["accuracy"])
    print("✅ Model built successfully.")
    return model

model = build_transfer_model((*IMG_SIZE, 3))
model.summary()

# ============================================================
# TRAINING
# ============================================================
callbacks = [
    ModelCheckpoint(MODEL_PATH, save_best_only=True, monitor="val_accuracy", mode="max", verbose=1),
    EarlyStopping(patience=5, restore_best_weights=True, monitor="val_loss")
]

print("🚀 Training model...")
history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    batch_size=BATCH_SIZE,
    epochs=EPOCHS,
    callbacks=callbacks,
    verbose=1
)

print("✅ Training complete! Best model saved at:", MODEL_PATH)

# ============================================================
# EVALUATION
# ============================================================
val_loss, val_acc = model.evaluate(X_val, y_val, verbose=0)
print(f"🎯 Validation Accuracy: {val_acc:.4f}")

# ============================================================
# GRAD-CAM EXPLAINABILITY
# ============================================================
def generate_gradcam(model, image, label, layer_name='top_conv'):
    grad_model = Model([model.inputs], [model.get_layer(layer_name).output, model.output])
    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(tf.expand_dims(image, 0))
        loss = predictions[:, 0]
    grads = tape.gradient(loss, conv_outputs)
    weights = tf.reduce_mean(grads, axis=(0, 1, 2))
    cam = tf.reduce_sum(tf.multiply(weights, conv_outputs), axis=-1).numpy().squeeze()

    cam = np.maximum(cam, 0)
    cam = cam / np.max(cam)
    cam = tf.image.resize(cam[..., np.newaxis], IMG_SIZE).numpy().squeeze()
    return cam

# Visualize one Grad-CAM example
idx = np.random.randint(0, len(X_val))
test_image = X_val[idx]
true_label = y_val[idx][0]
gradcam = generate_gradcam(model, test_image, true_label)

plt.figure(figsize=(6, 3))
plt.subplot(1, 2, 1)
plt.imshow(test_image)
plt.title(f"Original (Label={true_label})")

plt.subplot(1, 2, 2)
plt.imshow(test_image)
plt.imshow(gradcam, cmap='jet', alpha=0.5)
plt.title("Grad-CAM Heatmap")
plt.tight_layout()
plt.show()
