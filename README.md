# 🎨 Air Canvas — AI Qo‘l Harakati Bilan Chizish

Air Canvas — bu sun'iy intellekt va kompyuter ko‘rish texnologiyalari yordamida yaratilgan virtual chizish dasturi. Dastur webcam orqali qo‘l harakatlarini kuzatadi va foydalanuvchiga havoda chizish imkonini beradi.

Ko‘rsatkich barmoq virtual qalam sifatida ishlaydi, bosh barmoq va ko‘rsatkich barmoq pinch gesture orqali esa menyu va funksiyalar boshqariladi.

---

## ✨ Asosiy imkoniyatlar

### 🖐️ Real-time Hand Tracking

* MediaPipe HandLandmarker (Tasks API)
* Real vaqt rejimida qo‘lni aniqlash
* Ko‘rsatkich barmoq orqali chizish
* Thumb + Index pinch gesture orqali boshqaruv

### 🎨 Air Drawing

* Havoda chizish imkoniyati
* Silliq chiziqlar (Smooth Drawing)
* 5-frame Moving Average Filter
* Noise va sakrashlarni filtrlash

### 🌈 Ranglar

Dasturda 6 xil rang mavjud:

* 🟢 Yashil
* 🔴 Qizil
* 🔵 Ko‘k
* 🟡 Sariq
* 🟠 Apelsin
* 🩷 Pushti

### 🧽 Chizish vositalari

* Eraser (O‘chirg‘ich)
* Undo (Oxirgi chizmani bekor qilish)
* Clear Canvas (Hammasini tozalash)
* Brush Thickness (Qalam qalinligini o‘zgartirish)

### 📸 Qo‘shimcha funksiyalar

* Screenshot saqlash (PNG)
* Video yozib olish (AVI)
* FPS hisoblagichi
* Status indikator

---

## ⌨️ Klaviatura boshqaruvi

| Tugma | Funksiya                           |
| ----- | ---------------------------------- |
| 1 – 6 | Rang tanlash                       |
| + / = | Qalam qalinligini oshirish         |
| -     | Qalam qalinligini kamaytirish      |
| E     | Eraser yoqish/o‘chirish            |
| Z     | Undo                               |
| C     | Canvasni tozalash                  |
| S     | Screenshot saqlash                 |
| R     | Video yozishni boshlash/to‘xtatish |
| Q     | Dasturdan chiqish                  |
| ESC   | Dasturdan chiqish                  |

---

## 🧠 Gesture boshqaruv

Toolbar bilan ishlash uchun:

1. Ko‘rsatkich barmoqni kerakli tugma ustiga olib boring.
2. Bosh barmoq va ko‘rsatkich barmoqni bir-biriga yaqinlashtiring.
3. Pinch gesture bajarilganda tugma faollashadi.

---

## 📦 O‘rnatish

### 1. Repository'ni yuklab olish

```bash
git clone https://github.com/OrginalCoder/air-canvas.git
cd air-canvas
```

### 2. Kerakli kutubxonalarni o‘rnatish

```bash
pip install opencv-python mediapipe numpy
```

---

## 🚀 Ishga tushirish

```bash
python air_canvas.py
```

---

## 📁 Loyiha tuzilishi

```text
air_canvas.py
hand_landmarker.task

outputs/
├── air_canvas_*.png
└── air_canvas_rec_*.avi
```

> `hand_landmarker.task` fayli repository ichida mavjud va alohida yuklab olish talab qilinmaydi.

---

## ⚙️ Texnik ma'lumotlar

* Python 3.14+
* OpenCV
* MediaPipe Tasks API
* NumPy
* RunningMode.VIDEO
* Real-time frame processing
* Smooth Drawing algoritmi
* Gesture-Based UI

---

## 💡 Qo‘llanish sohalari

* Computer Vision o‘rganish
* AI loyihalar
* Gesture Control tizimlari
* Interaktiv taqdimotlar
* Virtual Whiteboard

---

## 👨‍💻 Dasturchi

**Muallif:** Isroil Mahfiraliyev

* Telegram: @Org_Coder
* GitHub: https://github.com/OrginalCoder

Ushbu loyiha Python, OpenCV, MediaPipe Tasks API va NumPy texnologiyalari yordamida ishlab chiqilgan.

Agar loyiha sizga foydali bo‘lgan bo‘lsa, repositoryga ⭐ Star berishni unutmang.

---

## 🤝 Hissa qo‘shish

Takliflar, xatoliklar yoki yangi funksiyalar bo‘yicha Pull Request va Issue'lar mamnuniyat bilan qabul qilinadi.

---

## 📜 Litsenziya

MIT License — erkin foydalanish, o‘zgartirish va tarqatish mumkin.
# air-canvas
