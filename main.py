import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers, models
from colorama import Fore
import itertools
import pandas as pd

class MeanIoU(tf.keras.metrics.Metric):
    def __init__(self, name='iou', **kwargs):
        super(MeanIoU, self).__init__(name=name, **kwargs)
        self.iou = self.add_weight(name='iou', initializer='zeros')
        self.count = self.add_weight(name='count', initializer='zeros')

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_true = tf.cast(y_true > 0.5, tf.float32)
        y_pred = tf.cast(y_pred > 0.5, tf.float32)
        intersection = tf.reduce_sum(y_true * y_pred)
        union = tf.reduce_sum(y_true) + tf.reduce_sum(y_pred) - intersection
        iou = tf.math.divide_no_nan(intersection, union)
        self.iou.assign_add(iou)
        self.count.assign_add(1.0)

    def result(self):
        return tf.math.divide_no_nan(self.iou, self.count)

    def reset_states(self):
        self.iou.assign(0.0)
        self.count.assign(0.0)


class MeanF1(tf.keras.metrics.Metric):
    def __init__(self, name='f1_score', **kwargs):
        super(MeanF1, self).__init__(name=name, **kwargs)
        self.f1 = self.add_weight(name='f1', initializer='zeros')
        self.count = self.add_weight(name='count', initializer='zeros')

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_true = tf.cast(y_true > 0.5, tf.float32)
        y_pred = tf.cast(y_pred > 0.5, tf.float32)
        intersection = tf.reduce_sum(y_true * y_pred)
        precision = tf.math.divide_no_nan(intersection, tf.reduce_sum(y_pred))
        recall = tf.math.divide_no_nan(intersection, tf.reduce_sum(y_true))
        f1 = tf.math.divide_no_nan(2 * precision * recall, precision + recall)
        self.f1.assign_add(f1)
        self.count.assign_add(1.0)

    def result(self):
        return tf.math.divide_no_nan(self.f1, self.count)

    def reset_states(self):
        self.f1.assign(0.0)
        self.count.assign(0.0)

# === Parameters ===
IMAGE_SIZE = (400,400)
BATCH_SIZE = 8
EPOCHS = 25
IMAGE_PATH = 'path/to/images'
MASK_PATH = 'path/to/masks'

# === Load Image and Mask Paths ===
image_files = sorted([os.path.join(IMAGE_PATH, f) for f in os.listdir(IMAGE_PATH)])
mask_files = sorted([os.path.join(MASK_PATH, f) for f in os.listdir(MASK_PATH)])

# === Load and Preprocess ===
def load_image(img_path, mask_path):
    img = tf.io.read_file(img_path)
    img = tf.image.decode_png(img, channels=3)
    img = tf.image.resize(img, IMAGE_SIZE)
    img = tf.cast(img, tf.float32) / 255.0

    mask = tf.io.read_file(mask_path)
    mask = tf.image.decode_png(mask, channels=1)
    mask = tf.image.resize(mask, IMAGE_SIZE, method='nearest')
    mask = tf.cast(mask > 127, tf.float32)  # Assuming binary mask
    return img, mask

def augment(img, mask):
    if tf.random.uniform(()) > 0.5:
        img = tf.image.flip_left_right(img)
        mask = tf.image.flip_left_right(mask)
    if tf.random.uniform(()) > 0.5:
        img = tf.image.flip_up_down(img)
        mask = tf.image.flip_up_down(mask)
    return img, mask


def augment_and_duplicate(img_path, mask_path, num_copies=3):
    augmented_images = []
    augmented_masks = []

    img = tf.io.read_file(img_path)
    img = tf.image.decode_png(img, channels=3)
    mask = tf.io.read_file(mask_path)
    mask = tf.image.decode_png(mask, channels=1)

    img = tf.image.resize(img, IMAGE_SIZE)
    mask = tf.image.resize(mask, IMAGE_SIZE, method='nearest')

    img = tf.cast(img, tf.float32) / 255.0
    mask = tf.cast(mask > 127, tf.float32)

    for _ in range(num_copies):
        a_img, a_mask = augment(img, mask)  # your existing OpenCV-based augment function
        augmented_images.append(a_img)
        augmented_masks.append(a_mask)

    return augmented_images, augmented_masks
def load_dataset(image_files, mask_files, augment_data=True):
    # dataset = tf.data.Dataset.from_tensor_slices((image_files, mask_files))
    # dataset = dataset.map(load_image, num_parallel_calls=tf.data.AUTOTUNE)
    # if augment_data:
    #     dataset = dataset.map(augment, num_parallel_calls=tf.data.AUTOTUNE)
    # dataset = dataset.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

    augmented_imgs = []
    augmented_masks = []
    
    for img_path, mask_path in zip(image_files[:split], mask_files[:split]):
        a_imgs, a_masks = augment_and_duplicate(img_path, mask_path, num_copies=2)
        augmented_imgs.extend(a_imgs)
        augmented_masks.extend(a_masks)
    
    dataset = tf.data.Dataset.from_tensor_slices((augmented_imgs, augmented_masks))
    dataset = dataset.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
    return dataset

# === Split Dataset ===
split = int(len(image_files) * 0.9)
train_dataset = load_dataset(image_files[:split], mask_files[:split], augment_data=True)
val_dataset = load_dataset(image_files[split:], mask_files[split:], augment_data=False)

train_size = sum(img.shape[0] for img, _ in train_dataset)
val_size = sum(img.shape[0] for img, _ in val_dataset)

print(f"{Fore.CYAN}Train dataset size:{Fore.RESET} {train_size}")
print(f"{Fore.CYAN}Val dataset size:{Fore.RESET} {val_size}")

# === Model 1: U-Net ===
def build_unet():
    inputs = layers.Input(shape=(*IMAGE_SIZE, 3))

    def conv_block(x, filters):
        x = layers.Conv2D(filters, 3, padding="same", activation="relu")(x)
        x = layers.Conv2D(filters, 3, padding="same", activation="relu")(x)
        return x

    def encoder_block(x, filters):
        c = conv_block(x, filters)
        p = layers.MaxPooling2D()(c)
        return c, p

    def decoder_block(x, skip, filters):
        x = layers.UpSampling2D()(x)
        x = layers.Concatenate()([x, skip])
        x = conv_block(x, filters)
        return x

    c1, p1 = encoder_block(inputs, 64)
    c2, p2 = encoder_block(p1, 128)
    c3, p3 = encoder_block(p2, 256)
    c4, p4 = encoder_block(p3, 512)

    b = conv_block(p4, 1024)

    d1 = decoder_block(b, c4, 512)
    d2 = decoder_block(d1, c3, 256)
    d3 = decoder_block(d2, c2, 128)
    d4 = decoder_block(d3, c1, 64)

    outputs = layers.Conv2D(1, 1, activation="sigmoid")(d4)

    return models.Model(inputs, outputs, name="U-Net")

# === Model 2: Simple CNN ===
def build_simple_cnn():
    model = models.Sequential([
        layers.Input(shape=(*IMAGE_SIZE, 3)),
        layers.Conv2D(32, 3, activation="relu", padding="same"),
        layers.MaxPooling2D(),
        layers.Conv2D(64, 3, activation="relu", padding="same"),
        layers.MaxPooling2D(),
        layers.Conv2D(128, 3, activation="relu", padding="same"),
        layers.UpSampling2D(),
        layers.Conv2D(64, 3, activation="relu", padding="same"),
        layers.UpSampling2D(),
        layers.Conv2D(1, 1, activation="sigmoid", padding="same")
    ])
    return model


def atrous_spatial_pyramid_pooling(inputs):
    dims = inputs.shape

    # 1x1 convolution
    conv_1x1 = layers.Conv2D(256, 1, padding="same", activation='relu')(inputs)

    # Dilated convolutions
    conv_3x3_r6 = layers.Conv2D(256, 3, padding="same", dilation_rate=6, activation='relu')(inputs)
    conv_3x3_r12 = layers.Conv2D(256, 3, padding="same", dilation_rate=12, activation='relu')(inputs)
    conv_3x3_r18 = layers.Conv2D(256, 3, padding="same", dilation_rate=18, activation='relu')(inputs)

    # Image-level features (global average pooling)
    image_pooling = layers.GlobalAveragePooling2D()(inputs)
    image_pooling = layers.Reshape((1, 1, dims[-1]))(image_pooling)
    image_pooling = layers.Conv2D(256, 1, padding="same", activation='relu')(image_pooling)
    image_pooling = layers.UpSampling2D(size=(dims[1], dims[2]), interpolation="bilinear")(image_pooling)

    # Concatenate all
    x = layers.Concatenate()([conv_1x1, conv_3x3_r6, conv_3x3_r12, conv_3x3_r18, image_pooling])
    x = layers.Conv2D(256, 1, padding="same", activation='relu')(x)

    return x

def build_aspp_segnet():
    inputs = layers.Input(shape=(*IMAGE_SIZE, 3))

    # Encoder
    x = layers.Conv2D(64, 3, activation='relu', padding='same')(inputs)
    x = layers.Conv2D(64, 3, activation='relu', padding='same')(x)
    x = layers.MaxPooling2D()(x)

    x = layers.Conv2D(128, 3, activation='relu', padding='same')(x)
    x = layers.Conv2D(128, 3, activation='relu', padding='same')(x)
    x = layers.MaxPooling2D()(x)

    x = layers.Conv2D(256, 3, activation='relu', padding='same')(x)
    x = layers.Conv2D(256, 3, activation='relu', padding='same')(x)

    # ASPP module
    x = atrous_spatial_pyramid_pooling(x)

    # Decoder
    x = layers.UpSampling2D(size=(2, 2))(x)
    x = layers.Conv2D(128, 3, activation='relu', padding='same')(x)
    x = layers.UpSampling2D(size=(2, 2))(x)
    x = layers.Conv2D(64, 3, activation='relu', padding='same')(x)

    outputs = layers.Conv2D(1, 1, activation='sigmoid')(x)

    return models.Model(inputs, outputs, name="ASPP-SegNet")



# === Compile and Train ===
def compile_and_train(model, name):
    model.compile(
        optimizer="adam",
        loss="binary_crossentropy",
        metrics=[MeanIoU(), MeanF1()]
    )
    print(f"\nTraining {name}...")
    history = model.fit(train_dataset, validation_data=val_dataset, epochs=EPOCHS, verbose=1)
    return model, history

def save_history_to_csv(history, model_name):
    df = pd.DataFrame(history.history)
    filename = f"{model_name.replace(' ', '_').lower()}_history.csv"
    df.to_csv(filename, index=False)
    print(f"Saved history to {filename}")

unet_model = build_unet()
cnn_model = build_simple_cnn()
aspp_model = build_aspp_segnet()

unet_model, unet_hist = compile_and_train(unet_model, "U-Net")
save_history_to_csv(unet_hist, "U-Net")

cnn_model, cnn_hist = compile_and_train(cnn_model, "Simple CNN")
save_history_to_csv(cnn_hist, "Simple CNN")

aspp_model, aspp_hist = compile_and_train(aspp_model, "ASPP SegNet")
save_history_to_csv(aspp_hist, "ASPP SegNet")

# === Visualization ===
def show_predictions(dataset, models_dict, num=3):
    for img_batch, mask_batch in dataset.take(1):
        preds = {name: model.predict(img_batch) for name, model in models_dict.items()}

        for i in range(num):
            plt.figure(figsize=(12, 4))
            plt.subplot(1, 5, 1)
            plt.imshow(img_batch[i])
            plt.title("Input")
            plt.axis("off")

            plt.subplot(1, 5, 2)
            plt.imshow(mask_batch[i].numpy().squeeze(), cmap="gray")
            plt.title("Ground Truth")
            plt.axis("off")

            plt.subplot(1, 5, 3)
            plt.imshow(preds["U-Net"][i].squeeze() > 0.5, cmap="gray")
            plt.title("U-Net")
            plt.axis("off")

            plt.subplot(1, 5, 4)
            plt.imshow(preds["Simple CNN"][i].squeeze() > 0.5, cmap="gray")
            plt.title("Simple CNN")
            plt.axis("off")

            plt.subplot(1, 5, 5)
            plt.imshow(preds["ASPP SegNet"][i].squeeze() > 0.5, cmap="gray")
            plt.title("ASPP SegNet")
            plt.axis("off")

            plt.tight_layout()
            plt.show()



def plot_metrics(histories, metric_name, title_prefix):
    # Plot training metrics
    plt.figure(figsize=(10, 5))
    for label, df in histories.items():
        if metric_name in df.columns:
            plt.plot(df[metric_name], label=label)
    plt.title(f"{title_prefix} - Training")
    plt.xlabel("Epoch")
    plt.ylabel(metric_name)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # Plot validation metrics
    plt.figure(figsize=(10, 5))
    for label, df in histories.items():
        val_key = f"val_{metric_name}"
        if val_key in df.columns:
            plt.plot(df[val_key], label=label)
    plt.title(f"{title_prefix} - Validation")
    plt.xlabel("Epoch")
    plt.ylabel(val_key)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


# Load histories directly as DataFrames
unet_df = pd.read_csv("u-net_history.csv")
cnn_df = pd.read_csv("simple_cnn_history.csv")
aspp_df = pd.read_csv("aspp_segnet_history.csv")


plot_metrics(
    {
        "U-Net": unet_df,
        "Simple CNN": cnn_df,
        "ASPP SegNet": aspp_df
    },
    metric_name="f1_score",
    title_prefix="F1 Score"
)

plot_metrics(
    {
        "U-Net": unet_df,
        "Simple CNN": cnn_df,
        "ASPP SegNet": aspp_df
    },
    metric_name="iou",
    title_prefix="IoU"
)


print(f"unet best val_f1_score {unet_df['val_f1_score'].max()}")
print(f"simple cnn best val_f1_score {cnn_df['val_f1_score'].max()}")
print(f"aspp best val_f1_score {aspp_df['val_f1_score'].max()}")

print(f"unet best val_iou {unet_df['val_iou'].max()}")
print(f"simple cnn best val_iou {cnn_df['val_iou'].max()}")
print(f"aspp best val_iou {aspp_df['val_iou'].max()}")

print(f"unet best val_loss {unet_df['val_loss'].min()}")
print(f"simple cnn best val_loss {cnn_df['val_loss'].min()}")
print(f"aspp best val_loss {aspp_df['val_loss'].min()}")
