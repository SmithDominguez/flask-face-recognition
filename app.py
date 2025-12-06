import gradio as gr
import cv2
import numpy as np
import urllib.request
import face_recognition
import base64

# ---------- CONFIGURACIÓN ----------
URLS_CELEBRIDADES = {
    "Messi": "https://upload.wikimedia.org/wikipedia/commons/thumb/b/b4/Lionel-Messi-Argentina-2022-FIFA-World-Cup_%28cropped%29.jpg/250px-Lionel-Messi-Argentina-2022-FIFA-World-Cup_%28cropped%29.jpg",
    "Shakira": "https://upload.wikimedia.org/wikipedia/commons/thumb/0/0b/2023-11-16_Gala_de_los_Latin_Grammy%2C_03_%28cropped%2901.jpg/330px-2023-11-16_Gala_de_los_Latin_Grammy%2C_03_%28cropped%2901.jpg"
}
URL_IMAGEN_PRUEBA = "https://upload.wikimedia.org/wikipedia/commons/thumb/2/26/Leo_messi_barce_2005.jpg/250px-Leo_messi_barce_2005.jpg"
TOLERANCIA = 0.6
HEADERS = {'User-Agent': 'Mozilla/5.0'}
# ------------------------------------

def descargar_imagen(url):
    """Descarga una imagen desde URL usando cabeceras de navegador."""
    req = urllib.request.Request(url, headers=HEADERS)
    with urllib.request.urlopen(req) as response:
        data = response.read()
    return data

def cargar_celebridades():
    nombres = []
    embeddings = []
    for nombre, url in URLS_CELEBRIDADES.items():
        try:
            print(f"[INFO] Descargando {nombre}...")
            data = descargar_imagen(url)
            img_array = np.asarray(bytearray(data), dtype=np.uint8)
            imagen = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
            rgb = cv2.cvtColor(imagen, cv2.COLOR_BGR2RGB)
            caras = face_recognition.face_encodings(rgb)
            if len(caras) == 0:
                print(f"[AVISO] No se detectó cara en {nombre}")
                continue
            nombres.append(nombre)
            embeddings.append(caras[0])
            print(f"[OK] {nombre} cargado correctamente.")
        except Exception as e:
            print(f"[ERROR] No se pudo procesar {nombre}: {e}")
    return nombres, embeddings

def procesar_imagen(file):
    print("[INFO] Cargando dataset de la nube...")
    nombres_conocidos, embeddings_conocidos = cargar_celebridades()
    if not nombres_conocidos:
        return "No se cargaron celebridades correctamente.", None

    # Obtener imagen del POST
    img_array = np.frombuffer(file.read(), np.uint8)
    imagen_prueba = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
    rgb_prueba = cv2.cvtColor(imagen_prueba, cv2.COLOR_BGR2RGB)

    ubicaciones_caras = face_recognition.face_locations(rgb_prueba)
    embeddings_prueba = face_recognition.face_encodings(rgb_prueba, ubicaciones_caras)

    for (top, right, bottom, left), embedding in zip(ubicaciones_caras, embeddings_prueba):
        coincidencias = face_recognition.compare_faces(embeddings_conocidos, embedding, tolerance=TOLERANCIA)
        distancias = face_recognition.face_distance(embeddings_conocidos, embedding)
        nombre_detectado = "Desconocido"
        if True in coincidencias:
            indice = np.argmin(distancias)
            nombre_detectado = nombres_conocidos[indice]

        cv2.rectangle(imagen_prueba, (left, top), (right, bottom), (0, 255, 0), 2)
        cv2.putText(imagen_prueba, nombre_detectado, (left, top - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    # Convertir imagen procesada a formato base64
    _, img_encoded = cv2.imencode('.jpg', imagen_prueba)
    img_base64 = base64.b64encode(img_encoded).decode('utf-8')

    return "Imagen procesada correctamente", img_base64

def run_gradio_interface():
    interface = gr.Interface(
        fn=procesar_imagen,
        inputs=gr.inputs.Image(type="file"),
        outputs=[gr.outputs.Textbox(), gr.outputs.Image(type="auto")],
        live=True
    )
    interface.launch()

if __name__ == "__main__":
    run_gradio_interface()
