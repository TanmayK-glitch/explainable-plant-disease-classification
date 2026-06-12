# Explainable Plant Disease Classification

EfficientNet-B0 plant disease classification with prediction, evaluation,
confusion-matrix generation, and Grad-CAM visualization.

## Setup

Python 3.10 or newer is recommended.

```bash
cd /path/to/explainable-plant-disease-classification
python -m venv venv
source venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r app/requirements.txt
```

The dataset must use this directory structure:

```text
data/
└── PlantVillage/
    ├── Pepper__bell___Bacterial_spot/
    ├── Pepper__bell___healthy/
    ├── Potato___Early_blight/
    └── ...
```

The trained checkpoint is expected at:

```text
models/best_model.pth
```

## Commands

Train the model:

```bash
python src/train.py
```

Predict one image:

```bash
python src/predict.py path/to/leaf_image.jpg
```

Evaluate the best checkpoint and save the confusion matrix:

```bash
python src/evaluate.py
```

The confusion matrix is saved to `outputs/confusion_matrix.png`.

Generate five correct, five healthy, and five disease Grad-CAM examples:

```bash
python src/generate_gradcam_examples.py
```

The images and CSV manifest are saved under `outputs/gradcam/`.
