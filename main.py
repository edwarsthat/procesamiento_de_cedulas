from pathlib import Path
import re
from typing import List, Optional, Dict, Any
import shutil
from pyzbar.pyzbar import decode
import cv2
import numpy as np
import pytesseract
import fitz  
import zxingcpp

### PRE PROCESAMIENTO ###
def pdf_to_images(pdf_path, dpi, page_numbers):

    pdf_path = Path(pdf_path).resolve()

    if not pdf_path.exists():
        raise FileNotFoundError(f"No existe el archivo PDF: {pdf_path}")

    if pdf_path.suffix.lower() != ".pdf":
        raise ValueError(f"El archivo no es un PDF válido: {pdf_path}")

    images: List[np.ndarray] = []
    with fitz.open(pdf_path) as doc:
        total_pages = doc.page_count
        pages = page_numbers if page_numbers is not None else range(total_pages)

        for page_index in pages:
            if page_index < 0 or page_index >= total_pages:
                raise IndexError(f"Página fuera de rango: {page_index}")

            page = doc.load_page(page_index)
            pix = page.get_pixmap(dpi=dpi)

            img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)

            if pix.n == 4:
                img = cv2.cvtColor(img, cv2.COLOR_RGBA2BGR)
            elif pix.n == 3:
                img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            elif pix.n == 1:
                img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
            else:
                img = img[:, :, :3]

            images.append(img)

    return images

def obtener_imagen_para_barcode(pdf_path: str, dpi: int = 300) -> np.ndarray:
    """
    Devuelve una imagen (np.ndarray BGR) adecuada para detectar PDF417:
    - Si el PDF tiene 2+ páginas: usa la segunda página completa
    - Si el PDF tiene 1 página: recorta la mitad inferior
    """

    # 1) Cargar TODAS las páginas (necesitamos saber cuántas hay)
    images = pdf_to_images(pdf_path, dpi=dpi, page_numbers=None)

    if not images:
        raise ValueError("No se pudieron extraer imágenes del PDF")

    # 2) Elegir imagen base
    if len(images) >= 2:
        img = images[1]  # segunda página (índice 1)
    else:
        img = images[0]  # única página

        # 3) Recortar mitad inferior
        h, w = img.shape[:2]
        img = img[int(h * 0.5):h, 0:w]

    return img

def preprocess_for_ocr(
    image: np.ndarray,
    clahe_clip: float = 0.6,
    clahe_tile: int = 16,
    blur_kind: str = "gaussian",   # "none" | "gaussian" | "median"
    blur_ksize: int = 3,
    blur_sigma: float = 0.5,
    threshold_kind: str = "otsu",  # "none" | "otsu" | "adaptive"
    adaptive_block: int = 31,
    adaptive_C: int = 10,
) -> np.ndarray:
    if image is None or not isinstance(image, np.ndarray):
        raise ValueError("Imagen inválida para preprocesamiento")

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    clahe = cv2.createCLAHE(
        clipLimit=float(clahe_clip),
        tileGridSize=(int(clahe_tile), int(clahe_tile))
    )
    proc = clahe.apply(gray)

    # Blur (opcional)
    if blur_kind == "gaussian":
        k = int(blur_ksize)
        if k % 2 == 0:
            k += 1
        proc = cv2.GaussianBlur(proc, (k, k), float(blur_sigma))
    elif blur_kind == "median":
        k = int(blur_ksize)
        if k % 2 == 0:
            k += 1
        proc = cv2.medianBlur(proc, k)
    elif blur_kind == "none":
        pass
    else:
        raise ValueError(f"blur_kind inválido: {blur_kind}")

    # Threshold (opcional)
    if threshold_kind == "otsu":
        _, proc = cv2.threshold(proc, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    elif threshold_kind == "adaptive":
        b = int(adaptive_block)
        if b % 2 == 0:
            b += 1
        proc = cv2.adaptiveThreshold(
            proc, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            b,
            int(adaptive_C)
        )
    elif threshold_kind == "none":
        pass
    else:
        raise ValueError(f"threshold_kind inválido: {threshold_kind}")

    return proc

def show_resized(window_name, image, max_width=1200, max_height=800):
    h, w = image.shape[:2]
    scale = min(max_width / w, max_height / h, 1.0)

    resized = cv2.resize(
        image,
        (int(w * scale), int(h * scale)),
        interpolation=cv2.INTER_AREA
    )

    cv2.imshow(window_name, resized)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def crop_mrz_last_quarter(img: np.ndarray, ratio: float = 0.65) -> np.ndarray:
    """
    Recorta el último cuarto (o porcentaje configurable) de la imagen.
    ratio=0.75 -> último 25%
    ratio=0.65 -> último 35%
    """
    h, w = img.shape[:2]
    y0 = int(h * ratio)
    return img[y0:h, 0:w]

def ocr_mrz(img):
    config = (
        "--oem 3 --psm 6 "
        "-c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789<"
    )
    return pytesseract.image_to_string(img, lang="eng", config=config)


### dETECTAR PDF417 ###
def _to_cv(img):
    # PIL -> np
    if not isinstance(img, np.ndarray):
        img = np.array(img)

    # si viene RGB (PIL), pásalo a BGR para OpenCV
    if img.ndim == 3 and img.shape[2] == 3:
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    return img

def leer_pdf417_zxing(img):
    if not isinstance(img, np.ndarray):
        img = np.array(img)

    if img.dtype != np.uint8:
        img = img.astype(np.uint8)

    if img.ndim == 3:
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    else:
        img_rgb = img

    results = zxingcpp.read_barcodes(img_rgb)

    print("ZXing encontró:", len(results))
    for r in results:
        print("Formato:", r.format.name, "len:", len(r.text))

    for r in results:
        if r.format.name == "PDF417" and r.text:
            return r.text

    return None

def limpiar_pdf417(raw: str) -> str:
    s = raw
    s = re.sub(r"<[^>]*>", "", s)
    s = re.sub(r"[\x00-\x1F\x7F]", "|", s)
    s = re.sub(r"[\s|]+", "|", s).strip("|").strip()
    return s

def extraer_datos_cedula_pdf417(raw: str) -> dict:
    print("=" * 80)
    print("RAW COMPLETO (primeros 500 chars):")
    print(repr(raw[:500]))
    print("=" * 80)
    
    # Reemplazar <NUL> y caracteres nulos reales por pipe
    s = raw.replace("<NUL>", "|").replace("\x00", "|")
    
    # Limpiar pipes múltiples
    s = re.sub(r"\|+", "|", s)
    
    print("STRING LIMPIO (primeros 300 chars):")
    print(repr(s[:300]))
    print("=" * 80)
    
    # Dividir por pipes
    parts = [p.strip() for p in s.split("|") if p.strip()]
    print(f"PARTES EXTRAÍDAS: {parts[:20]}")
    print("=" * 80)
    
    # 1) Cédula: primer número de 8-10 dígitos
    cedula = next((p for p in parts if re.fullmatch(r"\d{8,10}", p)), None)
    print(f"Cédula: {cedula}")
    
    # 2) Buscar palabras que sean SOLO letras mayúsculas (apellidos y nombres)
    letras = [p for p in parts if re.fullmatch(r"[A-ZÁÉÍÓÚÑ]+", p) and len(p) >= 3]
    print(f"Palabras de solo letras: {letras}")
    
    apellidos = None
    nombres = None
    
    if len(letras) >= 3:
        # Típicamente: APELLIDO1, APELLIDO2, NOMBRE(S)
        apellidos = f"{letras[0]} {letras[1]}"
        nombres = " ".join(letras[2:])
    elif len(letras) == 2:
        apellidos = letras[0]
        nombres = letras[1]
    elif len(letras) == 1:
        apellidos = letras[0]
    
    print(f"Apellidos: {apellidos}")
    print(f"Nombres: {nombres}")
    
    # 3) Sexo y fecha de nacimiento
    sexo = None
    fecha_nacimiento = None
    
    # Buscar en todo el string el patrón 0M/1F seguido de fecha
    sexo_fecha_match = re.search(r"[01]([MF])(\d{8})", raw)
    if sexo_fecha_match:
        sexo = sexo_fecha_match.group(1)
        fecha_nacimiento = sexo_fecha_match.group(2)
    
    print(f"Sexo: {sexo}, Fecha: {fecha_nacimiento}")
    
    # 4) RH
    rh = None
    rh_match = re.search(r"(AB|A|B|O)[+-]", raw)
    if rh_match:
        rh = rh_match.group(0)
    
    print(f"RH: {rh}")
    print("=" * 80)
    
    return {
        "cedula": cedula,
        "apellidos": apellidos,
        "nombres": nombres,
        "fecha_nacimiento": fecha_nacimiento,
        "sexo": sexo,
        "rh": rh,
    }

### VALIDANDO MRZ #####
def get_mrz_candidate_lines(text: str) -> list[str]:
    lines = []
    for ln in text.splitlines():
        ln = ln.strip()
        if len(ln) >= 20 and "<" in ln:
            lines.append(ln)
    return lines

def validar_numero_caracteres(mrz_line) -> str:
    numero_caracteres = len(mrz_line)
    if numero_caracteres == 30:
        return "Correcto"
    else:
        return "Incorrecto"

def fix_common_mrz_errors(text: str) -> str:
    t = text

    # Correcciones típicas OCR
    t = t.replace("SSS", "<<<")
    t = t.replace("SS", "<<")
    t = t.replace("S<S", "<<<")
    t = t.replace("K<", "<<")
    t = t.replace("E<C", "I<C")
    t = t.replace("EC", "I<")

    return t
#Validando primera linea
def validar_mrz_tipo_documento(mrz_line) -> str:
    tipo_documento = mrz_line[0]
    if tipo_documento == "I":
        return "Identificacion"
    elif tipo_documento == "P":
        return "Pasaporte"
    elif tipo_documento == "A":
        return "Especial"
    else:
        return "desconocido"

def validar_mrz_pais(mrz_line) -> str:
    pais = mrz_line[2:5]
    return pais

def obtener_numero_identidad(mrz_line) -> str:
    substring = mrz_line[5:]
    match = re.search(r'[<kK]', substring)
    return substring[:match.start()] if match else substring

#Validando segunda linea
def obtener_fecha_nacimiento(mrz_line) -> str:
    substring = mrz_line[0:6]
    checksum_real = int(mrz_line[6]) if len(mrz_line) > 6 else None
    
    anno = substring[0:2]
    mes = substring[2:4]
    dia = substring[4:6]
    
    # Calcular checksum
    pesos = [7, 3, 1]
    suma = 0
    
    for i, char in enumerate(substring):
        if char.isdigit():
            valor = int(char)
        elif char.isalpha():
            valor = ord(char.upper()) - ord('A') + 10
        elif char == '<':
            valor = 0
        else:
            valor = 0
        
        peso = pesos[i % 3]
        suma += valor * peso
    
    checksum_calculado = suma % 10
    if checksum_real is not None:
        valido = checksum_calculado == checksum_real
    else:
        valido = None
    
    if valido:
        return f"{dia}-{mes}-{anno}"
    else:
        return "Invalido"

def obtener_fecha_expiracion(mrz_line) -> str:
    substring = mrz_line[8:14]
    checksum_real = int(mrz_line[14]) if len(mrz_line) > 14 else None
    
    anno = substring[0:2]
    mes = substring[2:4]
    dia = substring[4:6]
    
    # Calcular checksum
    pesos = [7, 3, 1]
    suma = 0
    
    for i, char in enumerate(substring):
        if char.isdigit():
            valor = int(char)
        elif char.isalpha():
            valor = ord(char.upper()) - ord('A') + 10
        elif char == '<':
            valor = 0
        else:
            valor = 0
        
        peso = pesos[i % 3]
        suma += valor * peso
    
    checksum_calculado = suma % 10
    if checksum_real is not None:
        valido = checksum_calculado == checksum_real
    else:
        valido = None
    
    if valido:
        return f"{dia}-{mes}-{anno}"
    else:
        return "Invalido"

def obtener_nacionalidad(mrz_line) -> str:
    substring = mrz_line[15:18]
    return substring

def obtener_numero_identidad2(mrz_line) -> str:
    substring = mrz_line[19:]
    match = re.search(r'[<kK]', substring)
    return substring[:match.start()] if match else substring

#validar tecera linea
def obtener_nombre_apellido(mrz_line):

    data = mrz_line.split("<<")

    apellido = data[0] if len(data) > 0 else ""
    nombre = data[1] if len(data) > 1 else ""

    apellido = apellido.replace("<", " ").strip()
    nombre = nombre.replace("<", " ").strip()

    return apellido, nombre


###Obtener QR###
def leer_qr_code(img):
    """
    Lee códigos QR de la imagen usando ZXing.
    """
    if not isinstance(img, np.ndarray):
        img = np.array(img)

    if img.dtype != np.uint8:
        img = img.astype(np.uint8)

    if img.ndim == 3:
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    else:
        img_rgb = img

    results = zxingcpp.read_barcodes(img_rgb)

    print("ZXing encontró:", len(results))
    for r in results:
        print(f"  - Formato: {r.format.name}, len: {len(r.text)}")

    # Buscar QR Code
    for r in results:
        if r.format.name == "QRCode" and r.text:
            print(f"✓ QR Code encontrado con {len(r.text)} caracteres")
            return r.text

    return None

def extraer_datos_qr(raw: str) -> dict:
    """
    Extrae datos del QR. El formato puede variar, pero generalmente
    contiene datos separados por caracteres especiales o en JSON.
    """
    print("=" * 80)
    print("QR RAW (primeros 500 chars):")
    print(repr(raw[:500]))
    print("=" * 80)
    
    # Intentar parsear como JSON primero
    try:
        import json
        data = json.loads(raw)
        return {
            "cedula": data.get("numeroDocumento") or data.get("cedula"),
            "apellidos": data.get("apellidos"),
            "nombres": data.get("nombres"),
            "fecha_nacimiento": data.get("fechaNacimiento"),
            "sexo": data.get("sexo"),
            "formato": "JSON"
        }
    except:
        pass
    
    # Si no es JSON, intentar parsear como texto delimitado
    # Similar al PDF417
    s = raw.replace("\x00", "|").replace("<NUL>", "|")
    s = re.sub(r"\|+", "|", s)
    parts = [p.strip() for p in s.split("|") if p.strip()]
    
    print(f"QR PARTS: {parts[:15]}")
    
    cedula = next((p for p in parts if re.fullmatch(r"\d{8,10}", p)), None)
    letras = [p for p in parts if re.fullmatch(r"[A-ZÁÉÍÓÚÑ]+", p) and len(p) >= 3]
    
    apellidos = None
    nombres = None
    
    if len(letras) >= 2:
        apellidos = letras[0]
        nombres = " ".join(letras[1:])
    
    sexo = None
    fecha_nacimiento = None
    sexo_fecha_match = re.search(r"[01]?([MF])(\d{8})", raw)
    if sexo_fecha_match:
        sexo = sexo_fecha_match.group(1)
        fecha_nacimiento = sexo_fecha_match.group(2)
    
    return {
        "cedula": cedula,
        "apellidos": apellidos,
        "nombres": nombres,
        "fecha_nacimiento": fecha_nacimiento,
        "sexo": sexo,
        "formato": "DELIMITADO"
    }
    
### MAIN ###
def main():
    tesseract_path = shutil.which("tesseract")

    if not tesseract_path:
        raise RuntimeError(
            "❌ Tesseract no está instalado o no está en el PATH.\n"
            "Instálalo con: sudo apt install tesseract-ocr"
        )

    pytesseract.pytesseract.tesseract_cmd = tesseract_path
    pdf_path = "./cedulas/10.pdf"

    # 1) PDF -> imagen (la original)
    images = obtener_imagen_para_barcode(pdf_path, dpi=300)
    original = images

    show_resized("Processed", original)

    datos_finales = {}
    metodo_usado = None
    
    # ============================================================
    # ESTRATEGIA 1: Intentar leer PDF417 (MÁS COMÚN EN CÉDULAS)
    # ============================================================
    print("\n" + "="*80)
    print("ESTRATEGIA 1: INTENTANDO LEER PDF417...")
    print("="*80)
    
    pdf417_data = leer_pdf417_zxing(original)
    
    if pdf417_data:
        print("✓ PDF417 detectado")
        info_pdf417 = extraer_datos_cedula_pdf417(pdf417_data)
        
        if info_pdf417.get("cedula") and info_pdf417.get("apellidos") and info_pdf417.get("nombres"):
            print("✓ Datos completos extraídos del PDF417")
            datos_finales = {
                "metodo": "PDF417",
                "cedula": info_pdf417["cedula"],
                "apellidos": info_pdf417["apellidos"],
                "nombres": info_pdf417["nombres"],
                "fecha_nacimiento": info_pdf417["fecha_nacimiento"],
                "sexo": info_pdf417["sexo"],
                "rh": info_pdf417.get("rh"),
            }
            metodo_usado = "PDF417"
        else:
            print("⚠ PDF417 detectado pero datos incompletos")
    else:
        print("✗ No se detectó código PDF417")

    # ============================================================
    # ESTRATEGIA 2: Intentar leer QR CODE
    # ============================================================
    if not metodo_usado:
        print("\n" + "="*80)
        print("ESTRATEGIA 2: INTENTANDO LEER QR CODE...")
        print("="*80)
        
        qr_data = leer_qr_code(original)
        
        if qr_data:
            print("✓ QR Code detectado")
            info_qr = extraer_datos_qr(qr_data)
            
            if info_qr.get("cedula") and (info_qr.get("apellidos") or info_qr.get("nombres")):
                print("✓ Datos extraídos del QR Code")
                datos_finales = {
                    "metodo": "QR",
                    "cedula": info_qr["cedula"],
                    "apellidos": info_qr["apellidos"],
                    "nombres": info_qr["nombres"],
                    "fecha_nacimiento": info_qr.get("fecha_nacimiento"),
                    "sexo": info_qr.get("sexo"),
                }
                metodo_usado = "QR"
            else:
                print("⚠ QR detectado pero datos incompletos")
        else:
            print("✗ No se detectó código QR")

    # ============================================================
    # ESTRATEGIA 3: FALLBACK FINAL - OCR del MRZ
    # ============================================================
    if not metodo_usado:
        print("\n" + "="*80)
        print("ESTRATEGIA 3 (FALLBACK): USANDO OCR DEL MRZ...")
        print("="*80)
        
        try:
            processed = preprocess_for_ocr(
                original,
                clahe_clip=1.2,
                clahe_tile=8,
                blur_kind="gaussian",
                blur_ksize=5,
                blur_sigma=0.5,
                threshold_kind="adaptive",
                adaptive_block=16,
                adaptive_C=3,
            )

            # processed = crop_mrz_last_quarter(processed)
            raw_text = ocr_mrz(processed)
            mrz_lines = get_mrz_candidate_lines(raw_text)

            if len(mrz_lines) < 3:
                raise ValueError("No se encontraron 3 líneas MRZ")

            mrz_lines[0] = fix_common_mrz_errors(mrz_lines[0])
            mrz_lines[1] = fix_common_mrz_errors(mrz_lines[1])
            mrz_lines[2] = fix_common_mrz_errors(mrz_lines[2])

            apellido, nombre = obtener_nombre_apellido(mrz_lines[2])
            
            datos_finales = {
                "metodo": "MRZ-OCR",
                "tipo_documento": validar_mrz_tipo_documento(mrz_lines[0]),
                "pais": validar_mrz_pais(mrz_lines[0]),
                "cedula": obtener_numero_identidad(mrz_lines[0]),
                "fecha_nacimiento": obtener_fecha_nacimiento(mrz_lines[1]),
                "sexo": mrz_lines[1][7],
                "fecha_expiracion": obtener_fecha_expiracion(mrz_lines[1]),
                "nacionalidad": obtener_nacionalidad(mrz_lines[1]),
                "apellidos": apellido,
                "nombres": nombre,
            }
            metodo_usado = "MRZ-OCR"
            print("✓ Datos extraídos del MRZ correctamente")
            
        except Exception as e:
            print(f"✗ Error al procesar MRZ: {e}")
            datos_finales = {"metodo": "ERROR", "error": str(e)}

    # ============================================================
    # RESULTADO FINAL
    # ============================================================
    print("\n" + "="*80)
    print(f"✓ MÉTODO EXITOSO: {metodo_usado}")
    print("="*80)
    print("\nDATOS EXTRAÍDOS:")
    for key, value in datos_finales.items():
        print(f"  {key}: {value}")
    print("="*80)
    
    return datos_finales


if __name__ == "__main__":
    main()