import os
from PIL import Image
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import to_categorical
import tensorflow as tf

# Defina o caminho para a pasta contendo as imagens
image_folder = r'D:\FIAP\2024\1_Semestre\AI ENGENEERING, COGNITIVE AND SEMANTIC COMPUTATION & IOT\CP\KenAI\Reconhecimento_pessoa\images\Larissa'

# Obtenha uma lista de todos os arquivos na pasta
image_files = [f for f in os.listdir(image_folder) if os.path.isfile(os.path.join(image_folder, f))]


# Função para carregar e preprocessar as imagens
def load_and_preprocess_images(image_folder, image_files, target_size=(32, 32)):
    images = []
    for image_file in image_files:
        image_path = os.path.join(image_folder, image_file)
        image = Image.open(image_path).convert('RGB')  # Abre a imagem e converte para RGB
        image = image.resize(target_size)  # Redimensiona a imagem para o tamanho desejado
        image = np.array(image)  # Converte a imagem para um array NumPy
        images.append(image)
    images = np.array(images).astype('float32') / 255.0  # Normaliza os valores dos pixels para [0, 1]
    return images

x_train = load_and_preprocess_images(image_folder, image_files)
labels = list(range(len(x_train)))
num_classes = len(set(labels))
y_train = to_categorical(labels, num_classes=num_classes) 

# y_train = to_categorical(labels, num_classes=359)  
# y_train = to_categorical(labels, num_classes=233)  

# unique_labels = set(labels)
# classes = len(unique_labels)
# print("Número de classes:", classes)

modelo = Sequential()

modelo.add(Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)))
modelo.add(BatchNormalization())
modelo.add(MaxPooling2D(pool_size=(2, 2)))

modelo.add(Conv2D(64, (3, 3), activation='relu'))
modelo.add(BatchNormalization())
modelo.add(MaxPooling2D(pool_size=(2, 2)))

modelo.add(Conv2D(128, (3, 3), activation='relu'))
modelo.add(BatchNormalization())
modelo.add(MaxPooling2D(pool_size=(2, 2)))

modelo.add(Flatten())
modelo.add(Dense(128, activation='relu'))
modelo.add(Dropout(0.5))
modelo.add(Dense(num_classes, activation='softmax'))  # Usar num_classes aqui

modelo.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

modelo.summary()

# early_stopping = EarlyStopping(monitor='val_loss',
#                                patience=10,
#                                verbose=1,
#                                restore_best_weights=True)

# reduce_lr = ReduceLROnPlateau(monitor='val_loss',
#                               factor=0.2,
#                               patience=5,
#                               min_lr=0.001,
#                               verbose=1)

datagen = ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)


# history = modelo.fit(datagen.flow(x_train, y_train, batch_size=32),
#                      epochs=20,
#                      validation_data=(x_train, y_train),
#                      callbacks=[early_stopping, reduce_lr])

history = modelo.fit(datagen.flow(x_train, y_train, batch_size=32),
                     epochs=50,
                     validation_data=(x_train, y_train))

plt.plot(history.history['accuracy'], label='accuracy')
plt.plot(history.history['val_accuracy'], label='val_accuracy')
plt.xlabel('Épocas')
plt.ylabel('Acurácia')
plt.legend()
plt.show()

_, accuracy = modelo.evaluate(x_train, y_train)
print('Acurácia:', accuracy)

modelo.save(r'D:\FIAP\2024\1_Semestre\AI ENGENEERING, COGNITIVE AND SEMANTIC COMPUTATION & IOT\CP\KenAI\Reconhecimento_pessoa\modelo.h5')
model_save_path = r'D:\FIAP\2024\1_Semestre\AI ENGENEERING, COGNITIVE AND SEMANTIC COMPUTATION & IOT\CP\KenAI\Reconhecimento_pessoa\modelo.h5'
best_model = tf.keras.models.load_model(model_save_path)


