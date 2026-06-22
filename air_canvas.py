import os
import time
from collections import deque
from typing import Deque, List, Optional, Tuple

import cv2
import numpy as np

import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

KORSATKICH_BARMOK_UCHI = 8
BOSH_BARMOQ_UCHI = 4

MIN_QOL_ANIQLASH_ISHONCHI = 0.5
MIN_QOL_MAVJUDLIGI_ISHONCHI = 0.5
MIN_KUZATISH_ISHONCHI = 0.5

RANGLAR = [
    (0, 255, 0),
    (0, 0, 255),
    (255, 0, 0),
    (0, 255, 255),
    (255, 128, 0),
    (255, 0, 255),
]
RANG_NOMLARI = ["Yashil", "Qizil", "Kok", "Sariq", "Apelsin", "Pushti"]
OCHIRGICH_QALINLIGI = 30

TEKISLASH_DERAZASI = 5
MAX_NUQTA_MASOFA = 50

ASBOBLAR_BALANDLIGI = 80
ASBOBLAR_ORQA_FON = (30, 30, 30)
TUGMA_BALANDLIGI = 50
TUGMA_KENGLIGI = 55
TUGMA_ORALIGI = 6
CHIMDIK_CHEGARASI_PX = 40

FONT = cv2.FONT_HERSHEY_SIMPLEX
FONT_OLCHAMI = 0.55
FONT_QALINLIGI = 1
MATN_RANGI = (255, 255, 255)

KURSOR_RADIUSI = 6

TUGMA_RANG_1 = ord("1")
TUGMA_RANG_2 = ord("2")
TUGMA_RANG_3 = ord("3")
TUGMA_RANG_4 = ord("4")
TUGMA_RANG_5 = ord("5")
TUGMA_RANG_6 = ord("6")
TUGMA_QALIN_OSHIR = ord("=")
TUGMA_QALIN_KAMAYTIR = ord("-")
TUGMA_OCHIRGICH = ord("e")
TUGMA_QAYTARISH = ord("z")
TUGMA_TOZALA = ord("c")
TUGMA_SAQLA = ord("s")
TUGMA_YOZIB_OL = ord("r")
TUGMA_CHIQISH_Q = ord("q")
TUGMA_CHIQISH_ESC = 27

MODEL_NOMZODLARI = [
    "hand_landmarker.task",
    os.path.join("models", "hand_landmarker.task"),
]

MODEL_YUKLASH_URLI = (
    "https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
    "hand_landmarker/float16/latest/hand_landmarker.task"
)


def model_topish() -> str:
    skript_papka = os.path.dirname(os.path.abspath(__file__))
    for yol in MODEL_NOMZODLARI + [
        os.path.join(skript_papka, MODEL_NOMZODLARI[0]),
        os.path.join(skript_papka, MODEL_NOMZODLARI[1]),
    ]:
        if os.path.isfile(yol):
            return os.path.abspath(yol)
    raise FileNotFoundError(
        "HandLandmarker model fayli topilmadi.\n"
        f"Yuklab oling:\n  {MODEL_YUKLASH_URLI}\n"
        "va skript yoniga yoki models/ papkasiga joylashtiring."
    )


class ChizishHolati:
    def __init__(self) -> None:
        self.chiziqlar: List[List[Tuple[int, int]]] = []
        self.joriy_chiziq: List[Tuple[int, int]] = []
        self.chizish_davom: bool = False
        self._tarix: Deque[Tuple[float, float]] = deque(maxlen=TEKISLASH_DERAZASI)
        self.rang_index: int = 0
        self.qalinlik: int = 4
        self.ochirgich: bool = False
        self.tozalash_kerak: bool = False

    def chiziqni_boshlash(self) -> None:
        if self.joriy_chiziq:
            self.chiziqlar.append(self.joriy_chiziq)
            self.joriy_chiziq = []
        self.chizish_davom = True

    def chiziqni_tugatish(self) -> None:
        if self.joriy_chiziq:
            self.chiziqlar.append(self.joriy_chiziq)
            self.joriy_chiziq = []
        self.chizish_davom = False
        self._tarix.clear()

    def nuqta_qoshish(self, x: int, y: int) -> None:
        self.joriy_chiziq.append((x, y))

    def tekislash(self, xom_x: float, xom_y: float) -> Tuple[int, int]:
        if self._tarix:
            ort_x = sum(p[0] for p in self._tarix) / len(self._tarix)
            ort_y = sum(p[1] for p in self._tarix) / len(self._tarix)
        else:
            ort_x, ort_y = xom_x, xom_y
        dx = xom_x - ort_x
        dy = xom_y - ort_y
        masofa = np.sqrt(dx * dx + dy * dy)
        if masofa > MAX_NUQTA_MASOFA and self._tarix:
            ishlat_x, ishlat_y = ort_x, ort_y
        else:
            ishlat_x, ishlat_y = xom_x, xom_y
        self._tarix.append((ishlat_x, ishlat_y))
        oxirgi_x = sum(p[0] for p in self._tarix) / len(self._tarix)
        oxirgi_y = sum(p[1] for p in self._tarix) / len(self._tarix)
        return int(round(oxirgi_x)), int(round(oxirgi_y))

    def tozala(self) -> None:
        self.chiziqlar.clear()
        self.joriy_chiziq.clear()
        self.chizish_davom = False
        self._tarix.clear()

    def oxirgini_qaytar(self) -> None:
        if self.chiziqlar:
            self.chiziqlar.pop()

    def joriy_rang(self) -> Tuple[int, int, int]:
        if self.ochirgich:
            return (0, 0, 0)
        return RANGLAR[self.rang_index]

    def joriy_qalinlik(self) -> int:
        if self.ochirgich:
            return OCHIRGICH_QALINLIGI
        return self.qalinlik


class AsboblarPaneli:
    def __init__(self) -> None:
        self.tugmalar: List[dict] = []
        self.ustida_turgan: Optional[int] = None

    def qurish(self, kenglik: int) -> None:
        self.tugmalar.clear()
        x = TUGMA_ORALIGI
        for i, (rang, nom) in enumerate(zip(RANGLAR, RANG_NOMLARI)):
            self.tugmalar.append({
                "turi": "rang",
                "index": i,
                "hudud": (x, 10, x + TUGMA_KENGLIGI, 10 + TUGMA_BALANDLIGI),
                "rang": rang,
                "nom": nom[0],
            })
            x += TUGMA_KENGLIGI + TUGMA_ORALIGI
        x += TUGMA_ORALIGI
        for belgi in [
            ("ochirgich", "O", (180, 180, 180)),
            ("qaytarish", "Q", (180, 180, 180)),
            ("tozala", "T", (180, 180, 180)),
        ]:
            self.tugmalar.append({
                "turi": belgi[0],
                "hudud": (x, 10, x + TUGMA_KENGLIGI, 10 + TUGMA_BALANDLIGI),
                "rang": belgi[2],
                "nom": belgi[1],
            })
            x += TUGMA_KENGLIGI + TUGMA_ORALIGI

    def urilgan_tekshir(self, px: int, py: int) -> Optional[int]:
        for i, tugma in enumerate(self.tugmalar):
            x1, y1, x2, y2 = tugma["hudud"]
            if x1 <= px <= x2 and y1 <= py <= y2:
                return i
        return None


def asboblar_panelini_chiz(kadr: np.ndarray, panel: AsboblarPaneli, holat: ChizishHolati) -> None:
    h, w, _ = kadr.shape
    qoplama = kadr.copy()
    cv2.rectangle(qoplama, (0, 0), (w, ASBOBLAR_BALANDLIGI), ASBOBLAR_ORQA_FON, -1)
    cv2.addWeighted(qoplama, 0.85, kadr, 0.15, 0, kadr)

    for i, tugma in enumerate(panel.tugmalar):
        x1, y1, x2, y2 = tugma["hudud"]
        rang = tugma["rang"]
        faol = False
        if tugma["turi"] == "rang":
            faol = holat.rang_index == tugma["index"] and not holat.ochirgich
        elif tugma["turi"] == "ochirgich":
            faol = holat.ochirgich
        if faol:
            cv2.rectangle(kadr, (x1 - 2, y1 - 2), (x2 + 2, y2 + 2), (255, 255, 255), 2)
        if panel.ustida_turgan == i:
            cv2.rectangle(kadr, (x1 - 2, y1 - 2), (x2 + 2, y2 + 2), (200, 200, 200), 1)
        if tugma["turi"] == "rang":
            cv2.rectangle(kadr, (x1 + 4, y1 + 4), (x2 - 4, y2 - 4), rang, -1)
        else:
            cv2.rectangle(kadr, (x1 + 4, y1 + 4), (x2 - 4, y2 - 4), rang, -1)
            (tw, th), _ = cv2.getTextSize(tugma["nom"], FONT, 0.6, 2)
            tx = (x1 + x2 - tw) // 2
            ty = (y1 + y2 + th) // 2
            cv2.putText(kadr, tugma["nom"], (tx, ty), FONT, 0.6, (0, 0, 0), 2)

    joriy_rang = holat.joriy_rang()
    cv2.rectangle(kadr, (w - 50, 15), (w - 10, 55), joriy_rang, -1)
    cv2.putText(kadr, f"Qalin: {holat.qalinlik}", (w - 140, 40), FONT, 0.5, MATN_RANGI, 1)


def kadrni_qayta_ishla(
    kadr: np.ndarray,
    belgilagich: vision.HandLandmarker,
    holat: ChizishHolati,
    panel: AsboblarPaneli,
    vaqt_belgisi_ms: int,
) -> Tuple[Optional[Tuple[int, int]], Optional[Tuple[int, int]]]:
    rgb = cv2.cvtColor(kadr, cv2.COLOR_BGR2RGB)
    mp_tasvir = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    natija = belgilagich.detect_for_video(mp_tasvir, vaqt_belgisi_ms)

    korsatkich_holati: Optional[Tuple[int, int]] = None
    bosh_barmoq_holati: Optional[Tuple[int, int]] = None

    if natija.hand_landmarks:
        belgilar = natija.hand_landmarks[0]
        h, w, _ = kadr.shape

        kb = belgilar[KORSATKICH_BARMOK_UCHI]
        bb = belgilar[BOSH_BARMOQ_UCHI]
        kx, ky = kb.x * w, kb.y * h
        bx, by = bb.x * w, bb.y * h
        korsatkich_holati = (int(round(kx)), int(round(ky)))
        bosh_barmoq_holati = (int(round(bx)), int(round(by)))
        chimdik_px = np.sqrt((kx - bx) ** 2 + (ky - by) ** 2)
        chimdik_bor = chimdik_px < CHIMDIK_CHEGARASI_PX
        asboblar_ichida = ky < ASBOBLAR_BALANDLIGI

        if asboblar_ichida:
            panel.ustida_turgan = panel.urilgan_tekshir(int(round(kx)), int(round(ky)))
            if chimdik_bor:
                if holat.chizish_davom:
                    holat.chiziqni_tugatish()
                tugma_index = panel.ustida_turgan
                if tugma_index is not None:
                    tugma = panel.tugmalar[tugma_index]
                    if tugma["turi"] == "rang":
                        holat.rang_index = tugma["index"]
                        holat.ochirgich = False
                    elif tugma["turi"] == "ochirgich":
                        holat.ochirgich = not holat.ochirgich
                    elif tugma["turi"] == "qaytarish":
                        holat.oxirgini_qaytar()
                    elif tugma["turi"] == "tozala":
                        holat.tozala()
                        holat.tozalash_kerak = True
            return (korsatkich_holati, bosh_barmoq_holati)

        panel.ustida_turgan = None

        if chimdik_bor:
            if holat.chizish_davom:
                holat.chiziqni_tugatish()
            return (korsatkich_holati, bosh_barmoq_holati)

        tx, ty = holat.tekislash(kx, ky)
        if not holat.chizish_davom:
            holat.chiziqni_boshlash()
        holat.nuqta_qoshish(tx, ty)
    else:
        if holat.chizish_davom:
            holat.chiziqni_tugatish()
        panel.ustida_turgan = None

    return (korsatkich_holati, bosh_barmoq_holati)


def kanvasni_chiz(kanvas: np.ndarray, holat: ChizishHolati) -> None:
    for chiziq in holat.chiziqlar:
        _chiziqni_chiz(kanvas, chiziq, holat)
    _chiziqni_chiz(kanvas, holat.joriy_chiziq, holat)


def _chiziqni_chiz(kanvas: np.ndarray, chiziq: List[Tuple[int, int]], holat: ChizishHolati) -> None:
    if len(chiziq) < 2:
        return
    rang = holat.joriy_rang()
    qalin = holat.joriy_qalinlik()
    for i in range(1, len(chiziq)):
        cv2.line(kanvas, chiziq[i - 1], chiziq[i], rang, qalin, cv2.LINE_AA)


def interfeysni_chiz(
    kadr: np.ndarray,
    fps: float,
    korsatkich_holati: Optional[Tuple[int, int]],
    bosh_barmoq_holati: Optional[Tuple[int, int]],
    yozib_olish: bool,
    holat: ChizishHolati,
) -> None:
    h, w, _ = kadr.shape

    holat_matni = "OCHIRGICH" if holat.ochirgich else "CHIZISH"
    holat_rangi = (0, 100, 255) if holat.ochirgich else (0, 255, 0)
    belgi = holat_matni if korsatkich_holati else "QOL YOQ"
    rang = holat_rangi if korsatkich_holati else (0, 0, 255)
    (sw, sh), _ = cv2.getTextSize(belgi, FONT, 0.72, FONT_QALINLIGI + 1)
    sx, sy = w - sw - 15, ASBOBLAR_BALANDLIGI + 30
    cv2.putText(kadr, belgi, (sx, sy), FONT, 0.72, rang, FONT_QALINLIGI + 1, cv2.LINE_AA)

    if yozib_olish:
        yozuv_belgi = "● YOZUV"
        (rw, rh), _ = cv2.getTextSize(yozuv_belgi, FONT, 0.7, 2)
        rx, ry = 10, ASBOBLAR_BALANDLIGI + 30
        cv2.putText(kadr, yozuv_belgi, (rx, ry), FONT, 0.7, (0, 0, 255), 2, cv2.LINE_AA)

    korsatma = "[1-6]Rang  [+/-]Qalin  [E]Ochir  [Z]Qaytar  [C]Tozala  [S]Saqla  [R]Yozuv  [Q]Chiq"
    (iw, ih), _ = cv2.getTextSize(korsatma, FONT, FONT_OLCHAMI, FONT_QALINLIGI)
    ix, iy = (w - iw) // 2, h - 12
    bar = kadr.copy()
    cv2.rectangle(bar, (ix - 6, iy - ih - 6), (ix + iw + 6, iy + 6), (0, 0, 0), -1)
    cv2.addWeighted(bar, 0.5, kadr, 0.5, 0, kadr)
    cv2.putText(kadr, korsatma, (ix, iy), FONT, FONT_OLCHAMI, MATN_RANGI, FONT_QALINLIGI, cv2.LINE_AA)

    if korsatkich_holati:
        cx, cy = korsatkich_holati
        cv2.circle(kadr, (cx, cy), KURSOR_RADIUSI, (0, 0, 255), 2, cv2.LINE_AA)
        qoplama = kadr.copy()
        cv2.circle(qoplama, (cx, cy), KURSOR_RADIUSI - 2, (0, 0, 255), -1)
        cv2.addWeighted(qoplama, 0.35, kadr, 0.65, 0, kadr)

    if bosh_barmoq_holati and korsatkich_holati:
        bx, by = bosh_barmoq_holati
        cv2.circle(kadr, (bx, by), 4, (255, 100, 0), -1)
        cv2.line(kadr, korsatkich_holati, bosh_barmoq_holati, (200, 200, 200), 1, cv2.LINE_AA)


def asosiy() -> None:
    try:
        model_yoli = model_topish()
    except FileNotFoundError as exc:
        print(f"XATO: {exc}")
        return
    print(f"Model: {model_yoli}")

    asosiy_tanlov = python.BaseOptions(model_asset_path=model_yoli)
    tanlov = vision.HandLandmarkerOptions(
        base_options=asosiy_tanlov,
        running_mode=vision.RunningMode.VIDEO,
        num_hands=1,
        min_hand_detection_confidence=MIN_QOL_ANIQLASH_ISHONCHI,
        min_hand_presence_confidence=MIN_QOL_MAVJUDLIGI_ISHONCHI,
        min_tracking_confidence=MIN_KUZATISH_ISHONCHI,
    )
    belgilagich = vision.HandLandmarker.create_from_options(tanlov)

    kamera = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    if not kamera.isOpened():
        print("XATO: Kamerani ochib bolmadi.")
        belgilagich.close()
        return

    kamera.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    kamera.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    time.sleep(0.2)

    ret, test = kamera.read()
    if not ret:
        print("XATO: Birinchi kadrni okib bolmadi.")
        kamera.release()
        belgilagich.close()
        return
    by, bx = test.shape[:2]

    kanvas = np.zeros((by, bx, 3), dtype=np.uint8)
    holat = ChizishHolati()
    panel = AsboblarPaneli()
    panel.qurish(bx)

    fps = 0.0
    kadr_soni = 0
    fps_taymer = time.perf_counter()

    birlashtirilgan = np.empty_like(test)

    yozib_olish = False
    video_yozuvchi: Optional[cv2.VideoWriter] = None

    print(f"Air Canvas — {bx}x{by}")
    print("Rangni ozgartirish: [1-6] tugmalari yoki barmoq bilan panelga bosish (chimdik).")

    try:
        while True:
            ret, kadr = kamera.read()
            if not ret:
                print("OGOHLANTIRISH: Kadr tushib ketdi.")
                continue

            kadr = cv2.flip(kadr, 1)
            vaqt_belgisi_ms = int(time.perf_counter() * 1000)

            korsatkich_holati, bosh_barmoq_holati = kadrni_qayta_ishla(
                kadr, belgilagich, holat, panel, vaqt_belgisi_ms
            )

            if holat.tozalash_kerak:
                kanvas.fill(0)
                holat.tozalash_kerak = False

            kanvasni_chiz(kanvas, holat)

            kulrang = cv2.cvtColor(kanvas, cv2.COLOR_BGR2GRAY)
            _, niqob = cv2.threshold(kulrang, 1, 255, cv2.THRESH_BINARY)
            niqob_teskari = cv2.bitwise_not(niqob)
            fon = cv2.bitwise_and(kadr, kadr, mask=niqob_teskari)
            rasm = cv2.bitwise_and(kanvas, kanvas, mask=niqob)
            cv2.add(fon, rasm, dst=birlashtirilgan)

            asboblar_panelini_chiz(birlashtirilgan, panel, holat)

            kadr_soni += 1
            if kadr_soni >= 30:
                hozir = time.perf_counter()
                fps = 30.0 / (hozir - fps_taymer)
                kadr_soni = 0
                fps_taymer = hozir

            fps_belgi = f"FPS: {fps:.1f}"
            cv2.putText(
                birlashtirilgan, fps_belgi, (10, ASBOBLAR_BALANDLIGI + 55),
                FONT, 0.5, MATN_RANGI, 1,
            )

            interfeysni_chiz(
                birlashtirilgan, fps, korsatkich_holati,
                bosh_barmoq_holati, yozib_olish, holat,
            )

            if yozib_olish and video_yozuvchi is not None:
                video_yozuvchi.write(birlashtirilgan)

            cv2.imshow("Air Canvas", birlashtirilgan)

            tugma = cv2.waitKey(1) & 0xFF

            for k, ri in [
                (TUGMA_RANG_1, 0), (TUGMA_RANG_2, 1), (TUGMA_RANG_3, 2),
                (TUGMA_RANG_4, 3), (TUGMA_RANG_5, 4), (TUGMA_RANG_6, 5),
            ]:
                if tugma == k:
                    holat.rang_index = ri
                    holat.ochirgich = False
                    break
            else:
                if tugma == TUGMA_QALIN_OSHIR:
                    holat.qalinlik = min(holat.qalinlik + 1, 30)
                elif tugma == TUGMA_QALIN_KAMAYTIR:
                    holat.qalinlik = max(holat.qalinlik - 1, 1)
                elif tugma == TUGMA_OCHIRGICH:
                    holat.ochirgich = not holat.ochirgich
                elif tugma == TUGMA_QAYTARISH:
                    holat.oxirgini_qaytar()
                elif tugma == TUGMA_TOZALA:
                    kanvas.fill(0)
                    holat.tozala()
                    print("Kanvas tozalandi.")
                elif tugma == TUGMA_SAQLA:
                    yol = f"air_canvas_{int(time.time())}.png"
                    cv2.imwrite(yol, birlashtirilgan)
                    print(f"Screenshot saqlandi -> {yol}")
                elif tugma == TUGMA_YOZIB_OL:
                    if not yozib_olish:
                        fourcc = cv2.VideoWriter_fourcc(*"XVID")
                        yozuv_yoli = f"air_canvas_yozuv_{int(time.time())}.avi"
                        video_yozuvchi = cv2.VideoWriter(
                            yozuv_yoli, fourcc, 20.0, (bx, by)
                        )
                        yozib_olish = True
                        print(f"Yozuv boshlandi -> {yozuv_yoli}")
                    else:
                        yozib_olish = False
                        if video_yozuvchi is not None:
                            video_yozuvchi.release()
                            video_yozuvchi = None
                        print("Yozuv tugatildi.")
                elif tugma in (TUGMA_CHIQISH_Q, TUGMA_CHIQISH_ESC):
                    print("Chiqilmoqda.")
                    break

    except KeyboardInterrupt:
        print("\nToxtatildi.")
    except Exception as exc:
        print(f"XATO: {exc}")
    finally:
        kamera.release()
        if video_yozuvchi is not None:
            video_yozuvchi.release()
        belgilagich.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    asosiy()
