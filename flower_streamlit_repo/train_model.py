import json
from pathlib import Path

import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

# ============================================================
# CONFIGURATION
# ============================================================

IMG_HEIGHT = 160
IMG_WIDTH = 160
IMG_SIZE = (IMG_HEIGHT, IMG_WIDTH)

BATCH_SIZE = 32
SEED = 42
VALIDATION_SPLIT = 0.20

# You can use 6 epochs for this deployment exercise.
EPOCHS = 6

MODEL_PATH = "flower_classifier.h5"
CLASS_NAMES_PATH = "class_names.json"


# ============================================================
# DOWNLOAD DATASET
# ============================================================

dataset_url = (
    "https://storage.googleapis.com/"
    "download.tensorflow.org/example_images/"
    "flower_photos.tgz"
)

archive_path = tf.keras.utils.get_file(
    fname="flower_photos.tgz",
    origin=dataset_url,
    extract=True,
)

print("Downloaded archive:", archive_path)


# ============================================================
# FIND THE CORRECT DATASET DIRECTORY
# ============================================================

expected_classes = {
    "daisy",
    "dandelion",
    "roses",
    "sunflowers",
    "tulips",
}

search_root = Path(archive_path).parent
dataset_path = None

possible_directories = [search_root] + [
    path
    for path in search_root.rglob("*")
    if path.is_dir()
]

for directory in possible_directories:
    try:
        child_folders = {
            child.name
            for child in directory.iterdir()
            if child.is_dir()
        }

        if expected_classes.issubset(child_folders):
            dataset_path = directory
            break

    except PermissionError:
        pass

if dataset_path is None:
    raise RuntimeError(
        "Could not find the flower dataset directory."
    )

print("Dataset directory:", dataset_path)


# ============================================================
# LOAD DATA
# ============================================================

train_dataset = tf.keras.utils.image_dataset_from_directory(
    dataset_path,
    validation_split=VALIDATION_SPLIT,
    subset="training",
    seed=SEED,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
)

validation_dataset = tf.keras.utils.image_dataset_from_directory(
    dataset_path,
    validation_split=VALIDATION_SPLIT,
    subset="validation",
    seed=SEED,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
)

class_names = train_dataset.class_names
num_classes = len(class_names)

print("Classes:", class_names)
print("Number of classes:", num_classes)

if num_classes != 5:
    raise ValueError(
        f"Expected 5 classes but found {num_classes}: {class_names}"
    )

with open(
    CLASS_NAMES_PATH,
    "w",
    encoding="utf-8",
) as file:
    json.dump(
        class_names,
        file,
        indent=2,
    )

print(
    f"Saved class names to {CLASS_NAMES_PATH}"
)


# ============================================================
# OPTIMIZE DATA PIPELINE
# ============================================================

AUTOTUNE = tf.data.AUTOTUNE

train_dataset = train_dataset.prefetch(
    buffer_size=AUTOTUNE
)

validation_dataset = validation_dataset.prefetch(
    buffer_size=AUTOTUNE
)


# ============================================================
# DATA AUGMENTATION
# ============================================================

data_augmentation = tf.keras.Sequential(
    [
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(0.1),
        layers.RandomZoom(0.1),
    ],
    name="data_augmentation",
)


# ============================================================
# PRETRAINED MOBILENETV2
# ============================================================

base_model = MobileNetV2(
    input_shape=(
        IMG_HEIGHT,
        IMG_WIDTH,
        3,
    ),
    include_top=False,
    weights="imagenet",
)

base_model.trainable = False


# ============================================================
# BUILD CLASSIFIER
# ============================================================

inputs = tf.keras.Input(
    shape=(
        IMG_HEIGHT,
        IMG_WIDTH,
        3,
    )
)

x = data_augmentation(inputs)

x = preprocess_input(x)

x = base_model(
    x,
    training=False,
)

x = layers.GlobalAveragePooling2D()(x)

x = layers.Dense(
    128,
    activation="relu",
)(x)

x = layers.Dropout(
    0.30
)(x)

outputs = layers.Dense(
    num_classes,
    activation="softmax",
)(x)

model = tf.keras.Model(
    inputs=inputs,
    outputs=outputs,
    name="flower_classifier",
)

print("Model output shape:", model.output_shape)

model.summary()


# ============================================================
# COMPILE
# ============================================================

model.compile(
    optimizer=tf.keras.optimizers.Adam(
        learning_rate=0.001
    ),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
)


# ============================================================
# TRAIN
# ============================================================

history = model.fit(
    train_dataset,
    validation_data=validation_dataset,
    epochs=EPOCHS,
)


# ============================================================
# EVALUATE
# ============================================================

validation_loss, validation_accuracy = model.evaluate(
    validation_dataset
)

print(
    f"Validation accuracy: {validation_accuracy:.4f}"
)

print(
    f"Validation loss: {validation_loss:.4f}"
)


# ============================================================
# SAVE MODEL
# ============================================================

model.save(
    MODEL_PATH
)

print(
    f"Saved model to {MODEL_PATH}"
)


# ============================================================
# PLOT ACCURACY
# ============================================================

epochs_range = range(
    1,
    EPOCHS + 1,
)

plt.figure(
    figsize=(8, 5)
)

plt.plot(
    epochs_range,
    history.history["accuracy"],
    label="Training Accuracy",
)

plt.plot(
    epochs_range,
    history.history["val_accuracy"],
    label="Validation Accuracy",
)

plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title("Training vs Validation Accuracy")
plt.legend()
plt.grid()
plt.show()


# ============================================================
# PLOT LOSS
# ============================================================

plt.figure(
    figsize=(8, 5)
)

plt.plot(
    epochs_range,
    history.history["loss"],
    label="Training Loss",
)

plt.plot(
    epochs_range,
    history.history["val_loss"],
    label="Validation Loss",
)

plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training vs Validation Loss")
plt.legend()
plt.grid()
plt.show()
