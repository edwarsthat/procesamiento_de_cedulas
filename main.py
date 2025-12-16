from pathlib import Path
import re
from typing import List, Optional, Dict, Any

import cv2
import numpy as np
import pytesseract
import fitz  

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

def save_image(image, filename):
    output_path = Path(__file__).parent / filename
    cv2.imwrite(str(output_path), image)
    print(f"📸 Imagen guardada en: {output_path}")

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

def run_grid_search(image: np.ndarray) -> list[Path]:
    root = Path(__file__).parent
    out_dir = root / "grid_outputs"
    out_dir.mkdir(exist_ok=True)

    clahe_clips = [0.7, 1.2, 1.8]  # Aumenta el contraste (más definición de borde)
    clahe_tiles = [8, 16]         # 8 es mejor para texto pequeño
    blur_kinds = ["none", "gaussian"] # Quitamos "median" y probamos "none" para máxima nitidez
    blur_ksizes = [3]
    threshold_kinds = ["otsu", "adaptive"]
    adaptive_blocks = [21, 31]    # Bloques más pequeños para documentos con muchas líneas
    adaptive_cs = [5, 15]

    # clahe_clips = [0.4, 0.7]
    # clahe_tiles = [16]
    # blur_kinds = ["none", "median"]
    # blur_ksizes = [3]
    # threshold_kinds = ["otsu", "adaptive"]
    # adaptive_blocks = [31]
    # adaptive_cs = [10]
    saved = []
    idx = 0

    for clip in clahe_clips:
        for tile in clahe_tiles:
            for bkind in blur_kinds:
                for bks in blur_ksizes:
                    for tkind in threshold_kinds:
                        # Para adaptive, probamos block/C. Si no, ignoramos esos parámetros.
                        blocks = adaptive_blocks if tkind == "adaptive" else [31]
                        cs = adaptive_cs if tkind == "adaptive" else [10]

                        for ab in blocks:
                            for ac in cs:
                                idx += 1
                                processed = preprocess_for_ocr(
                                    image,
                                    clahe_clip=clip,
                                    clahe_tile=tile,
                                    blur_kind=bkind,
                                    blur_ksize=bks,
                                    blur_sigma=0.5,
                                    threshold_kind=tkind,
                                    adaptive_block=ab,
                                    adaptive_C=ac,
                                )

                                name = (
                                    f"{idx:03d}_clip{clip}_tile{tile}_"
                                    f"blur{bkind}{bks}_thr{tkind}_ab{ab}_c{ac}.png"
                                )
                                out_path = out_dir / name
                                cv2.imwrite(str(out_path), processed)
                                saved.append(out_path)

    print(f"✅ Guardadas {len(saved)} imágenes en: {out_dir}")
    return saved


MRZ_RE = re.compile(
    r"(I[A-Z0-9<]{20,}\n[A-Z0-9<]{20,}\n[A-Z0-9<]{10,})",
    re.MULTILINE
)
def extract_mrz_block(ocr_text: str) -> str | None:
    # Normaliza
    t = ocr_text.upper().replace(" ", "")
    t = t.replace("«", "<").replace("‹", "<").replace("❮", "<")
    t = re.sub(r"[^\nA-Z0-9<]", "", t)  # deja solo letras/números/< y saltos

    m = MRZ_RE.search(t)
    return m.group(1) if m else None

def parse_mrz_basic(mrz_block: str) -> dict:
    lines = [ln.strip() for ln in mrz_block.splitlines() if ln.strip()]
    if len(lines) < 3:
        raise ValueError("MRZ incompleta")

    l1, l2, l3 = lines[0], lines[1], lines[2]

    # Nombres (línea 3 suele ser APELLIDO<APELLIDO<<NOMBRE<NOMBRE<<<)
    parts = l3.split("<<", 1)
    surname_raw = parts[0]
    name_raw = parts[1] if len(parts) > 1 else ""

    surnames = " ".join([p for p in surname_raw.split("<") if p])
    names = " ".join([p for p in name_raw.split("<") if p])

    # Línea 2: suele traer fecha nac (YYMMDD), sexo (M/F) y fecha exp (YYMMDD)
    # Ej: 0509064F3311031COL....
    birth = None
    sex = None
    expiry = None

    m2 = re.search(r"(\d{6})([MF])(\d{6})", l2)
    if m2:
        birth = m2.group(1)   # YYMMDD
        sex = m2.group(2)     # M/F
        expiry = m2.group(3)  # YYMMDD

    return {
        "mrz_lines": lines[:3],
        "names": names,
        "surnames": surnames,
        "birth_YYMMDD": birth,
        "sex": sex,
        "expiry_YYMMDD": expiry,
    }

def score_ocr_text(text: str) -> int:
    score = 0

    mrz = extract_mrz_block(text)
    if mrz:
        score += 5
        score += min(mrz.count("<"), 30) // 5  # bonus por estructura MRZ

    blood = extract_blood_type(text)
    if blood:
        score += 2

    t = text.upper()
    if "REPUBLICA" in t or "REPÚBLICA" in t:
        score += 1
    if "COLOMBIA" in t:
        score += 1

    # penaliza ruido típico OCR
    score -= t.count("?")

    return score


BLOOD_RE = re.compile(r"\b(AB|A|B|O)\s*([+-])\b", re.IGNORECASE)

def extract_blood_type(ocr_text: str) -> Optional[str]:
    t = ocr_text.upper()

    # Normalizaciones típicas OCR: 0+ -> O+
    t = re.sub(r"\b0\s*([+-])\b", r"O\1", t)

    m = BLOOD_RE.search(t)
    if not m:
        return None
    return f"{m.group(1)}{m.group(2)}"


def ocr_text_from_image(img: np.ndarray) -> str:
    """
    OCR genérico para documento completo.
    Para MRZ, conviene 1) recortar esa zona y 2) otro PSM,
    pero por ahora lo dejamos simple.
    """
    config = "--oem 3 --psm 6"
    return pytesseract.image_to_string(img, lang="eng", config=config)


def pick_best_variant(saved_paths: List[Path]) -> Dict[str, Any]:
    best = {
        "path": None,
        "score": -10**9,
        "text": "",
        "mrz": None,
        "blood": None,
        "mrz_data": None,
    }

    for p in saved_paths:
        img = cv2.imread(str(p))
        if img is None:
            continue

        text = ocr_text_from_image(img)
        s = score_ocr_text(text)

        mrz_block = extract_mrz_block(text)
        blood = extract_blood_type(text)

        if s > best["score"]:
            best.update({
                "path": p,
                "score": s,
                "text": text,
                "mrz": mrz_block,
                "blood": blood
            })

    if best["mrz"]:
        try:
            best["mrz_data"] = parse_mrz_basic(best["mrz"])
        except Exception:
            best["mrz_data"] = None

    return best

def main():
    # Ajusta esta ruta si tu instalación quedó en otro lugar
    TESSERACT_EXE = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

    if not Path(TESSERACT_EXE).exists():
        raise FileNotFoundError(f"No encuentro tesseract.exe en: {TESSERACT_EXE}")

    pytesseract.pytesseract.tesseract_cmd = TESSERACT_EXE

    pdf_path = "4.pdf"

    # 1) PDF -> imagen (la original)
    images = pdf_to_images(pdf_path, dpi=300, page_numbers=[0])
    original = images[0]

    # 2) Grid search: crea muchas variantes y las guarda en /grid_outputs
    saved_paths = run_grid_search(original)

    # 3) OCR + scoring: escoger la mejor variante automáticamente
    best = pick_best_variant(saved_paths)

    if not best["path"]:
        print("❌ No se pudo evaluar ninguna imagen.")
        return

    print("\n🏁 Mejor candidata:")
    print(f" - Archivo: {best['path'].name}")
    print(f" - Score:   {best['score']}")
    print(f" - Sangre:  {best['blood']}")
    print(f" - MRZ:     {'✅' if best['mrz'] else '❌'}")

    # 4) Parsing final (MRZ)
    if best["mrz_data"]:
        print("\n📌 MRZ parseada:")
        for k, v in best["mrz_data"].items():
            print(f" - {k}: {v}")

    # 5) Mostrar la mejor (escalada para verla completa)
    best_img = cv2.imread(str(best["path"]))
    show_resized("Mejor variante", best_img)


if __name__ == "__main__":
    main()