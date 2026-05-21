# 🚗 Vehicle Detection & Tracking System

A real-time vehicle detection, tracking, and counting system built with Python and OpenCV. It detects vehicles from a video feed, classifies them by type, tracks their movement, estimates speed, and counts them across virtual lines.

---

## 📸 Features

- ✅ Real-time vehicle detection using MOG2 Background Subtraction
- ✅ Multi-object tracking with IoU matching
- ✅ Vehicle classification — Car, Motorbike, Truck/Bus
- ✅ Speed estimation in km/h
- ✅ Direction detection — Up→Down / Down→Up
- ✅ Movement trail visualization
- ✅ Live stats panel on screen

---

## 🛠️ Technologies Used

- Python 3.x
- OpenCV (cv2)
- NumPy

---

## 📁 Project Structure

```
Car-tracker/
│
├── vehicle_tracker.py     # Main program
├── traffic.mp4            # Sample input video
└── README.md              # Project documentation
```

---

## ⚙️ Installation

**1. Clone the repository**
```bash
git clone https://github.com/sazzad101737/Car-tracker.git
cd Car-tracker
```

**2. Install required libraries**
```bash
pip install opencv-python numpy
```

---

## ▶️ How to Run

```bash
python vehicle_tracker.py
```

> Make sure `traffic.mp4` is in the same folder as `vehicle_tracker.py`

Press **Q** to quit the program.

---

## 🔍 How It Works

| Step | Description |
|------|-------------|
| 1 | Video frames are read and resized |
| 2 | MOG2 background subtractor detects moving objects |
| 3 | Contours are filtered by area to remove noise |
| 4 | IoU matching links detections to existing tracks |
| 5 | Vehicles are classified by bounding box size |
| 6 | Speed is calculated using time between Line A and Line B |
| 7 | Final count is displayed when a track disappears |

---

## 📊 Output

- Bounding box around each detected vehicle
- Vehicle ID, type, and speed shown on screen
- Two counting lines (Line A and Line B)
- Live panel showing total count by type and direction

---

## 🙋 Author

**Sazzad**  
GitHub: [@sazzad101737](https://github.com/sazzad101737)
