import os
import json
import tensorflow as tf
from tensorflow.keras import layers, models
import kagglehub

# 1. Download do Dataset
print("Baixando o dataset...")
path = kagglehub.dataset_download("ajithdari/papaya-leaf-disease-dataset")
print("Caminho base do dataset:", path)

# 2. Definição correta do diretório de treino
# Baseado na sua estrutura, vamos apontar diretamente para a pasta 'new' 
# (ou se preferir usar a 'augmented', basta trocar o nome abaixo)
data_dir = os.path.join(path, "new")

if not os.path.exists(data_dir):
    data_dir = path  # Fallback caso a estrutura mude

print("Diretório de imagens utilizado:", data_dir)

# 3. Configurações de Treinamento
BATCH_SIZE = 32
IMG_HEIGHT = 224
IMG_WIDTH = 224
EPOCHS = 10

print("Carregando o conjunto de dados...")

# Criando dataset de treino (80%)
train_ds = tf.keras.utils.image_dataset_from_directory(
    data_dir,
    validation_split=0.2,
    subset="training",
    seed=123,
    image_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=BATCH_SIZE
)

# Criando dataset de validação (20%)
val_ds = tf.keras.utils.image_dataset_from_directory(
    data_dir,
    validation_split=0.2,
    subset="validation",
    seed=123,
    image_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=BATCH_SIZE
)

# Recupera automaticamente os nomes exatos das pastas (classes)
class_names = train_ds.class_names
print("Classes encontradas e mapeadas:", class_names)
num_classes = len(class_names)

# Otimização de I/O
AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.cache().prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

# 4. Data Augmentation
data_augmentation = tf.keras.Sequential([
    layers.RandomFlip("horizontal_rotation"),
    layers.RandomRotation(0.2),
    layers.RandomZoom(0.2),
])

# 5. Construção do Modelo (Transfer Learning com MobileNetV2)
print("Construindo o modelo...")
base_model = tf.keras.applications.MobileNetV2(
    input_shape=(IMG_HEIGHT, IMG_WIDTH, 3),
    include_top=False,
    weights='imagenet'
)

base_model.trainable = False

inputs = tf.keras.Input(shape=(IMG_HEIGHT, IMG_WIDTH, 3))
x = data_augmentation(inputs)
x = tf.keras.applications.mobilenet_v2.preprocess_input(x)
x = base_model(x, training=False)
x = layers.GlobalAveragePooling2D()(x)
x = layers.Dropout(0.2)(x)
outputs = layers.Dense(num_classes, activation='softmax')(x)

model = models.Model(inputs, outputs)

# 6. Compilação
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss=tf.keras.losses.SparseCategoricalCrossentropy(),
    metrics=['accuracy']
)

model.summary()

# 7. Treinamento
print("Iniciando o treinamento...")
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS
)

# 8. Salvando o modelo (.keras) e o mapeamento oficial de classes (classes.json)
model.save("modelo_pragas_mamao.keras")

with open("classes.json", "w") as f:
    json.dump(class_names, f)

print("Sucesso! Modelo salvo como 'modelo_pragas_mamao.keras' e classes salvas em 'classes.json'.")