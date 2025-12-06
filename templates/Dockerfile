# Usamos una imagen base de Python
FROM python:3.8-slim

# Establecemos el directorio de trabajo
WORKDIR /app

# Copiamos los archivos del proyecto al contenedor
COPY . /app

# Instalamos las dependencias
RUN pip install --no-cache-dir -r requirements.txt

# Exponemos el puerto que utilizará Flask
EXPOSE 5000

# Comando para ejecutar la app
CMD ["python", "app.py"]