
# 🛡️ Face Mask Detection System

An intelligent **Face Mask Detection** web application built using **Django**, **OpenCV (Haar Cascade)**, and **VGG-19**.  
This project was developed as part of an initiative to improve face recognition systems in public settings where masks are mandatory due to COVID-19 social restrictions.

## 🧠 Overview

This system detects faces in an image and classifies them as either **with mask** or **without mask** using a hybrid approach:
- **Face Detection**: Haar Cascade Classifier
- **Mask Classification**: Deep Learning using VGG-19 (pretrained on a face mask dataset)

> ⚠️ Useful during health crises where identifying masked individuals becomes critical in enforcing protocols and public safety.

---

## 🔍 Technologies Used

| Component         | Description                                                                 |
|------------------|-----------------------------------------------------------------------------|
| 💻 Framework      | [Django](https://www.djangoproject.com/) for building the web application |
| 📷 Detection      | [OpenCV Haar Cascade](https://github.com/opencv/opencv/blob/master/data/haarcascades/haarcascade_frontalface_default.xml) |
| 🧠 Deep Learning  | [VGG-19 CNN](https://www.kaggle.com/code/greynolan/face-mask-detection-vgg19/notebook) pretrained model |
| 📊 Visualization  | Matplotlib and Base64 image embedding                                      |

---

## 📁 Project Structure

```
face-mask-detection-main/
├── home/
│   ├── views.py
│   ├── templates/
│   └── urls.py
├── media/                  # Saved user image uploads and results
├── static/                 # Static files (CSS, JS, etc.)
├── Face Mask Dataset/      # Training/validation/test images
├── VGG19-Face Mask Detection.h5  # Trained model file
├── haarcascade_frontalface_default.xml
├── manage.py
├── requirements.txt
└── README.md
```

---

## 📂 Dataset & Model

- **Dataset Source**: [Face Mask Detection Dataset - Kaggle](https://www.kaggle.com/code/greynolan/face-mask-detection-vgg19/notebook)
- **Model Download**: [VGG-19 Trained Model Output](https://www.kaggle.com/code/greynolan/face-mask-detection-vgg19/output)
- **Face Detector**: [Haarcascade frontalface XML](https://github.com/opencv/opencv/blob/master/data/haarcascades/haarcascade_frontalface_default.xml)

> Dataset structure:
```
Face Mask Dataset/
├── Train/
├── Validation/
└── Test/
```

---

## 💡 How It Works

1. **User uploads an image**
2. **Haar Cascade** detects faces in the image
3. Each face is cropped and resized to 128x128
4. **VGG-19 model** classifies each cropped face:
    - With Mask 😷
    - Without Mask 😐
5. Final image is returned with detection boxes and predictions

---

## 🧪 Try It Locally

### ✅ Requirements

Install the dependencies from `requirements.txt`:

```bash
pip install -r requirements.txt
```

Ensure you place the following files in the correct directories:
- `VGG19-Face Mask Detection.h5` in project root
- `Face Mask Dataset` inside the root directory
- `haarcascade_frontalface_default.xml` inside root or specify the path in `views.py`

### 🚀 Run the App

```bash
python manage.py runserver
```

Open browser at: `http://127.0.0.1:8000/face_mask_detection/`

---

## 🧱 Core Concepts

### 📌 What is Haar Cascade?

> A machine learning-based approach for object detection proposed by Viola and Jones (2001).  
It works by extracting **simple features** from an image and applying **cascade classifiers** to detect faces with high speed and accuracy.

### 📌 What is Deep Learning?

> A subfield of machine learning using **Artificial Neural Networks (ANN)** that mimics the human brain’s structure.  
Deep learning enables systems to **learn from massive data** and recognize complex patterns effectively.

### 📌 What is VGG-19?

> A deep CNN architecture with **19 weight layers** (mostly convolutional) known for its simplicity and effectiveness.  
VGG-19 uses **3x3 convolution filters** and is excellent at learning features from visual data such as images of masked vs. unmasked faces.

---

## 🧾 License

This project is intended for academic and research purposes.  
For commercial or production use, please consult the original dataset and model licenses.

---

## 🙋‍♂️ Author

**Gigih Pambuko**  
🌐 [LinkedIn](https://linkedin.com/in/gigihko) | 💻 [GitHub](https://github.com/gigihko)

---
