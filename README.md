# 🫀 ECG Platform

A self-learning fullstack project that simulates and visualizes real-time ECG (Electrocardiogram) signals.  
Built with **HTML, CSS, JavaScript**, and **Flask** (Python) for backend processing and data simulation.

> 🔗 **Live Demo**: [ecg-platform-54mj.onrender.com](https://ecg-platform-54mj.onrender.com)

---

## 📌 Table of Contents

- [Features](#-features)
- [Tech Stack](#-tech-stack)
- [Preview](#-preview)
- [Project Structure](#-project-structure)
- [Getting Started](#-getting-started)
- [Learning Objectives](#-learning-objectives)
- [Future Improvements](#-future-improvements)
- [About Me](#-about-me)
- [License](#-license)

---

## ✨ Features

- 📈 Real-time ECG waveform simulation on browser
- 🧠 Backend logic handled by **Flask**
- 📦 API endpoint to serve ECG data dynamically
- ⏯ Interactive controls (start/stop)
- 📱 Responsive UI
- 🧪 No external frontend libraries used

---

## 🛠 Tech Stack

| Layer      | Technology               |
|------------|--------------------------|
| Frontend   | HTML5, CSS3, JavaScript  |
| Backend    | Python + Flask           |
| Hosting    | Render                   |

---

## 📷 Preview

> *(Insert image here if available)*

Live site 👉 [ecg-platform-54mj.onrender.com](https://ecg-platform-54mj.onrender.com)

---

## 📁 Project Structure

ecg-platform/
├── static/
│ ├── style.css # Frontend styling
│ └── script.js # JS logic to render ECG
├── templates/
│ └── index.html # Main HTML with Jinja2
├── app.py # Flask app + API endpoint
├── ecg_data.py # (Optional) ECG signal generator
└── requirements.txt # Python dependencies

---

## 🚀 Getting Started

### 1. Clone the repo

```bash
git clone https://github.com/TranDucLuong2201/ecg-platform.git
cd ecg-platform
```
# 2. Set up Python environment
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```
# 3. Run the Flask server
```bash
python app.py
```
