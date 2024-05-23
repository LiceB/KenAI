import os
from PIL import Image
import matplotlib.pyplot as plt

# Defina o caminho para a pasta contendo as imagens
image_folder = r'D:\FIAP\2024\1_Semestre\AI ENGENEERING, COGNITIVE AND SEMANTIC COMPUTATION & IOT\CP\KenAI\Reconhecimento_pessoa\images\Larissa'

# Obtenha uma lista de todos os arquivos na pasta
image_files = [f for f in os.listdir(image_folder) if os.path.isfile(os.path.join(image_folder, f))]

# Filtre os arquivos para incluir apenas imagens (aqui considerando jpg e png)
image_files = [f for f in image_files if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

# Verifique se há imagens na pasta
if not image_files:
    print("Nenhuma imagem encontrada na pasta.")
else:
    # Exiba cada imagem
    for image_file in image_files:
        image_path = os.path.join(image_folder, image_file)
        image = Image.open(image_path)

        # Mostrar a imagem
        plt.imshow(image)
        plt.title(image_file)
        plt.axis('off')  # Ocultar os eixos
        plt.show()
