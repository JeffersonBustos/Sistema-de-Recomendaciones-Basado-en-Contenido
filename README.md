# Sistema-de-Recomendaciones-Basado-en-Contenido
from google.colab import drive
drive.mount('/content/drive')

# Ruta de direcionamiento 
    dir_ = "/content/drive/MyDrive/Colab Notebooks/Sistema basado en contenido"

# Limpieza de datos
      import re
      
      def limpiar_texto(texto):
          # Eliminar emoticones
          texto = re.sub(r"[^\w\sáéíóúüñÁÉÍÓÚÜÑ]", "", texto)
      
          # Eliminar @username y hashtags
          texto = re.sub(r"(@\w+|#\w+)", "", texto)
      
          # Eliminar punto, punto y coma y comas
          texto = re.sub(r"[.;,]", "", texto)
      
          return texto
      
  # Abrir el archivo .tsv y leer su contenido
      with open('contenido1.tsv', 'r') as archivo:
          contenido = archivo.read()
  
  # Eliminar caracteres no deseados
      contenido_limpio = limpiar_texto(contenido)
  
  # Escribir el contenido limpio en un nuevo archivo
      with open('archivo_limpo.tsv', 'w') as archivo_limpo:
          archivo_limpo.write(contenido_limpio)
      
# transformación de los datos
      import csv
      
      def tsv_to_csv(tsv_file, csv_file):
          with open(tsv_file, 'r') as tsvfile:
              tsv_reader = csv.reader(tsvfile, delimiter='\t')
              with open(csv_file, 'w', newline='') as csvfile:
                  csv_writer = csv.writer(csvfile, delimiter=',')
                  for row in tsv_reader:
                      csv_writer.writerow(row)
      
          # Ejemplo de uso
          tsv_file = 'archivo_limpo.tsv'
      csv_file = 'archivo_limpo.csv'
      tsv_to_csv(tsv_file, csv_file)
      
# visualización de datos
    import pandas as pd
    data = pd.read_csv('archivo_limpo.tsv', delimiter='\t')
    data.head()

   # Crear un diccionario para asignar IDs únicos a los usuarios
    user_ids = {user: i+1 for i, user in enumerate(data['username'].unique())}

  # Asignar IDs de usuario a los datos
    data['username'] = data['username'].map(user_ids)
    data.head()

  # importacion de  librerias
    import numpy as np
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import mean_squared_error
    import torch
    import torch.nn as nn
    import torch.optim as optim
# creación de archivos de entrenamiento

  # Realizar la partición estratificada basada en las calificaciones
    train_df, test_df = train_test_split(data, test_size=0.2, stratify=data['score'])
    
  # Verificar si hay usuarios en ambos conjuntos
    common_users = set(train_df['username']) & set(test_df['username'])
    num_common_users = len(common_users)
    num_unique_users = len(set(data['username']))

  # Calcular porcentaje de usuarios que están en ambos conjuntos
    percentage_common_users = (num_common_users / num_unique_users) * 100

  # Verificar si todos los productos están en ambos conjuntos
    common_products = set(train_df['site']) & set(test_df['site'])
    num_common_products = len(common_products)
    num_unique_products = len(set(data['site']))

  # Cantidad de usuairos y reseñas en cada conjunto
    num_users_train = len(train_df['username'])
    num_users_test = len(test_df['username'])
    
  # Calcular porcentaje de productos que están en ambos conjuntos
    percentage_common_products = (num_common_products / num_unique_products) * 100

  # Imprimir resultados
    print(f"Número de usuarios en ambos conjuntos: {num_common_users}")
    print(f"Número de usuarios únicos: {num_unique_users}")
    print(f"Porcentaje de usuarios en ambos conjuntos: {percentage_common_users}%")
    print(f"Número de productos en ambos conjuntos: {num_common_products}")
    print(f"Número de productos únicos: {num_unique_products}")
    print(f"Porcentaje de productos en ambos conjuntos: {percentage_common_products}%")
    print(f"Cantidad de usuarios en el conjunto de entrenamiento: {num_users_train}")
    print(f"Cantidad de usuarios en el conjunto de prueba: {num_users_test}")

    train_df.to_csv(dir_+"trainbc.csv",sep=',',index=False)
    test_df.to_csv(dir_+"testbc.csv",sep=',',index=False)
  
  # Creación de matriz de trabajo
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity

  # Crear una matriz de características TF-IDF basada en las reseñas del producto
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(train_df['text'])

  # Calcular la matriz de similitud del coseno entre los productos
    similarity_matrix = cosine_similarity(tfidf_matrix)

  # Función para obtener las recomendaciones para un producto dado
      def get_recommendations(site, num_recommendations):
          # Obtener el índice del producto en la matriz de similitud
          product_index = data[data['site'] == site].index[0]

  # Obtener las puntuaciones de similitud del producto actual con todos los demás productos
    product_scores = similarity_matrix[product_index]

  # Obtener los índices de los productos más similares (excluyendo el producto actual)
    similar_indices = product_scores.argsort()[::-1][1:num_recommendations + 1]

  # Obtener los nombres de los productos más similares
    similar_products = data.loc[similar_indices, 'site']

    return similar_products
  # Ejemplo de uso: obtener 3 recomendaciones para el producto 'Producto1'
    recommendations = get_recommendations('Parque Fundadores', 5)
    print(recommendations)
