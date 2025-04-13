# 🦴 Osteoporosis Detection in Spine using Deep Learning Methods

This project aims to enhance early diagnosis of **Osteoporosis** using deep learning techniques applied to spine DXA scan images. It implements and compares multiple models including **CNN**, **VGG16**, **ResNet**, and a **hybrid ensemble model** combining these with a **Random Forest (RF)** classifier. The best-performing model is deployed using a **FastAPI backend** and a simple **HTML/CSS frontend** as a web application.

---

## 📄 Project Highlights

- Implements **CNN, VGG16, ResNet**, and **Random Forest** classifiers.
- Combines models using **Ensemble Learning (Stacking)** to improve prediction accuracy.
- Achieves up to **97% accuracy** using CNN; ensemble model reaches **93%**.
- Trained on **DXA spine scan images**.
- Deployed as a **web application** using FastAPI for backend and HTML/CSS for frontend.
- Predicts whether a patient has osteoporosis based on spine images.

---

## 📁 Files Included

- `Research Paper Osteoporosis.pdf` – Full documentation and system methodology.
- `Final Report` – Report of how we approached the problem and how it is designed.
- `osteoporosis model.h5` – a basic model of CNN.
- `Models (if needed contact me)` – Trained models (CNN, ResNet, VGG16, Ensembled model) is not included as it is more than 25 mb.
- `README.md` – This documentation file.
- `app.py` - This is the frontend of our project.

---

## 🛠️ Tools & Technologies

- **Python 3.10**
- **TensorFlow**, **Keras**, **PyTorch**
- **Scikit-learn**
- **FastAPI**
- **HTML/CSS**
- **DXA (Dual-energy X-ray Absorptiometry) spine scan dataset**

---

## 📊 Model Architecture

### Base Models:
- **CNN** – Extracts spatial features from spine images.
- **VGG16** – Pre-trained deep CNN fine-tuned on the dataset.
- **ResNet + Random Forest** – Combines deep feature extraction with ensemble classification.
- ** Stacking method used model** - A hybrid model deployed to get better accuracy and keeping in mind the future enhancement.

### Ensemble Approach:
- **Stacking** – Combines all base model outputs to improve robustness.
- **Meta-Model** – Logistic Regression used as the final decision layer.

---

## 🧪 Performance Evaluation

| Model                  | Accuracy |
|-----------------------|----------|
| CNN                   | 97%      |
| VGG16                 | 90%      |
| ResNet + RF Classifier| 87%      |
| **Stacked Ensemble**  | **93%**  |

- Evaluation metrics: Accuracy, Precision, Recall, F1-score
- Validated with cross-validation and real-world test set

---

## 🚀 Deployment

The final model is deployed using **FastAPI**, and accessed through a browser interface where users can:

1. Upload a spine image.
2. Get real-time prediction.
3. Access FAQs and treatment recommendations.

---

## 💡 Future Enhancements

- Integration with Clinical Decision Support Systems (CDSS)
- Support for **multimodal imaging** (CT, DXA, Ultrasound)
- Explainable AI (using Grad-CAM, SHAP)
- Real-time edge device deployment
- Tracking progression via RNNs or LSTMs
- Expansion to other skeletal disorders (scoliosis, fractures)

---

## 👥 Authors

- **Dr. K. R. Shylaja** – Professor, AIML Dept.
- **Shreyas D**, **Nithin Suresh**, **Roopitha G Nayak**, **Shrinidhi S Hegde**  
  Department of Information Science, Dr. Ambedkar Institute of Technology

---

## 📜 License

This project is for academic and educational purposes only. Contact the authors for permission to use in clinical settings.

---

## 📬 Contact

Feel free to reach out for project code, datasets and collabrations:

**Shreyas D**  
[LinkedIn](www.linkedin.com/in/shreyas-d-9668a422a)  
📧 shreyasappu952003@gmail.com

