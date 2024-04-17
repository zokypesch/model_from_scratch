import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import os
import matplotlib.pyplot as plt
import glob


image_size = (180, 180)
batch_size = 128

PATH = '/Users/maulana/python/semantic-search/fine_tune/images/source_images'
train_dir = os.path.join(PATH, 'PetImages')
validation_dir = os.path.join(PATH, 'validation')

def check_corrupt():
    img_paths = glob.glob(os.path.join(train_dir,'*/*.*')) # assuming you point to the directory containing the label folders.
    bad_paths = []

    for image_path in img_paths:
        try:
            img_bytes = tf.io.read_file(image_path)
            decoded_img = tf.io.decode_image(img_bytes)
        except tf.errors.InvalidArgumentError as e:
            print(f"Found bad path {image_path}...{e}")
            bad_paths.append(image_path)

        print(f"{image_path}: OK")

    print("BAD PATHS:")
    for bad_path in bad_paths:
        print(f"{bad_path}")
        os.unlink(bad_path)



check_corrupt()


# train_ds, val_ds = tf.keras.utils.image_dataset_from_directory(train_dir, shuffle=True, batch_size=batch_size,image_size=image_size)               
# validation_dataset = tf.keras.utils.image_dataset_from_directory(validation_dir,shuffle=True,batch_size=BATCH_SIZE,image_size=IMG_SIZE)                
# class_names = train_ds.class_names

train_ds, val_ds = tf.keras.utils.image_dataset_from_directory(
    train_dir, # direktori gambar yang akan diproses
    validation_split=0.2, #brp jumlah data validasi 0.2 * jumlah data
    subset="both", # jika both maka akan mengembalikan langsung validasi dan hasil trainninya
    seed=1500, # adalah jumlah angka random yang digunakan agar mendapatkan nilai trainning yang konsisten
    image_size=image_size, # resize ukuran gam bar yang akan diproses
    batch_size=batch_size, # batch jumlah brp banyak yang akan diulang
)

print('Number of training batches: %d' % tf.data.experimental.cardinality(train_ds).numpy())
print('Number of validation batches: %d' % tf.data.experimental.cardinality(val_ds).numpy())

# AUTOTUNE = tf.data.AUTOTUNE
# train_dataset = train_ds.prefetch(buffer_size=AUTOTUNE)
# validation_dataset = val_ds.prefetch(buffer_size=AUTOTUNE)

data_augmentation = keras.Sequential(
    [
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(0.1),
    ]
)
 
# plt.figure(figsize=(10, 10))
# for images, _ in train_ds.take(1):
#     for i in range(9):
#         augmented_images = data_augmentation(images)
#         ax = plt.subplot(3, 3, i + 1)
#         plt.imshow(augmented_images[0].numpy().astype("uint8"))
#         plt.axis("off")

# augmented_train_ds = train_ds.map(
#     lambda x, y: (data_augmentation(x, training=True), y))

train_ds = train_ds.map(
    lambda img, label: (data_augmentation(img), label),
    num_parallel_calls=tf.data.AUTOTUNE,
)
train_ds = train_ds.prefetch(tf.data.AUTOTUNE)
val_ds = val_ds.prefetch(tf.data.AUTOTUNE)

def make_model(input_shape, num_classes):
    inputs = keras.Input(shape=input_shape)

    x = layers.Rescaling(1.0 / 255)(inputs)
    x = layers.Conv2D(128, 3, strides=2, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)

    previous_block_activation = x 

    for size in [256, 512, 728]:
        x = layers.Activation("relu")(x)
        x = layers.SeparableConv2D(size, 3, padding="same")(x)
        x = layers.BatchNormalization()(x)

        x = layers.Activation("relu")(x)
        x = layers.SeparableConv2D(size, 3, padding="same")(x)
        x = layers.BatchNormalization()(x)

        x = layers.MaxPooling2D(3, strides=2, padding="same")(x)

        residual = layers.Conv2D(size, 1, strides=2, padding="same")(
            previous_block_activation
        )
        x = layers.add([x, residual])  
        previous_block_activation = x  # Set aside next residual

    x = layers.SeparableConv2D(1024, 3, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)

    x = layers.GlobalAveragePooling2D()(x)
    if num_classes == 2:
        activation = "sigmoid"
        units = 1
    else:
        activation = "softmax"
        units = num_classes

    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(units, activation=activation)(x)
    return keras.Model(inputs, outputs)

model = make_model(input_shape=image_size + (3,), num_classes=2)
keras.utils.plot_model(model, show_shapes=True)

epochs = 25

callbacks = [
    keras.callbacks.ModelCheckpoint("save_at_{epoch}.keras"),
]
model.compile(
    optimizer=keras.optimizers.Adam(1e-3),
    loss="binary_crossentropy",
    metrics=["accuracy"],
)
model.fit(
    train_ds,
    epochs=epochs,
    callbacks=callbacks,
    validation_data=val_ds,
)

