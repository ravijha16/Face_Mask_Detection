
import tensorflow as tf
from tensorflow.keras import layers, models
from pathlib import Path

# Project settings
DATA_DIR = Path("data")
IMG_SIZE = (128, 128)
BATCH_SIZE = 32
EPOCHS = 10

print("Face Mask Detection Training Started!")

# Check dataset folders
if not (DATA_DIR / "with_mask").exists():
    raise FileNotFoundError("with_mask folder not found!")

if not (DATA_DIR / "without_mask").exists():
    raise FileNotFoundError("without_mask folder not found!")

# Load training dataset
train_data = tf.keras.utils.image_dataset_from_directory(
    DATA_DIR,
    validation_split=0.2,
    subset="training",
    seed=123,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    label_mode="binary",
    class_names=["with_mask", "without_mask"]
)

# Load validation dataset
validation_data = tf.keras.utils.image_dataset_from_directory(
    DATA_DIR,
    validation_split=0.2,
    subset="validation",
    seed=123,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    label_mode="binary",
    class_names=["with_mask", "without_mask"]
)

print("Classes:", train_data.class_names)

# Improve data loading performance
AUTOTUNE = tf.data.AUTOTUNE
train_data = train_data.prefetch(AUTOTUNE)
validation_data = validation_data.prefetch(AUTOTUNE)

# Build CNN model
model = models.Sequential([
    layers.Input(shape=(128, 128, 3)),

    layers.Rescaling(1.0 / 255),

    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.05),

    layers.Conv2D(32, (3, 3), activation="relu"),
    layers.MaxPooling2D(),

    layers.Conv2D(64, (3, 3), activation="relu"),
    layers.MaxPooling2D(),

    layers.Conv2D(128, (3, 3), activation="relu"),
    layers.MaxPooling2D(),

    layers.Flatten(),

    layers.Dense(128, activation="relu"),
    layers.Dropout(0.5),

    layers.Dense(1, activation="sigmoid")
])

# Compile model
model.compile(
    optimizer="adam",
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

model.summary()

# Save best model automatically
checkpoint = tf.keras.callbacks.ModelCheckpoint(
    "face_mask_model.keras",
    monitor="val_accuracy",
    save_best_only=True,
    verbose=1
)

early_stop = tf.keras.callbacks.EarlyStopping(
    monitor="val_loss",
    patience=3,
    restore_best_weights=True
)

# Train model
history = model.fit(
    train_data,
    validation_data=validation_data,
    epochs=EPOCHS,
    callbacks=[checkpoint, early_stop]
)

# Save final trained model
model.save("face_mask_model.keras")

print("\nTraining completed successfully!")
print("Model saved as face_mask_model.keras")
