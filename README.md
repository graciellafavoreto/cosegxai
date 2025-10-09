# Context vs. Segmentation: Insights from Saliency Maps in Pulmonary Disease Classification

This repository contains the implementation and evaluation code for our research on the counterintuitive relationship between classification performance and model explainability in chest X-ray analysis.

## Abstract

We investigate while lung segmentation reduces classification performance, it significantly improves model explainability. Using the COVID-19 Radiography Database with 21,165 chest X-ray images across four diagnostic classes, we trained ResNet50, DenseNet201, and VGG19 models on both contextual (complete) and ROI-based (lung-segmented) images. Models trained on complete images achieved superior classification performance (92.80% vs 88.08% accuracy), but ROI-based models produced more interpretable saliency maps with better correlation metrics.

## Key Findings

- **Performance**: Complete images yield better classification accuracy
- **Explainability Advantage**: Segmented images produce more focused, clinically interpretable saliency maps
- **Saliency Evaluation**: Comprehensive evaluation using Insertion Correlation (IC), Deletion Correlation (DC), and Sparsity metrics
- **Clinical Implications**: Trade-off between accuracy and interpretability has important implications for medical AI deployment

## Dataset

We use the **COVID-19 Radiography Database** containing:
- **Total Images**: 21,165 chest X-rays (299x299 pixels, resized to 224x224)
- **Classes**: COVID-19 (3,616), Normal (10,192), Lung Opacity (6,012), Viral Pneumonia (1,345)
- **Split**: 70% training, 15% validation, 15% testing
- **Evaluation Subset**: 951 stratified samples (30% per class) for explainability analysis

## Methodology

### Model Training
- **Architectures**: ResNet50, DenseNet201, VGG19
- **Training Paradigms**: Contextual (complete images) vs ROI-based (lung-segmented)
- **Hyperparameters**: SGD optimizer (lr=1e-3, momentum=0.9), dropout=0.3, 20 epochs

### Explainability Evaluation
- **Saliency Method**: Grad-CAM
- **Quantitative Metrics**: 
  - Insertion Correlation (IC)
  - Deletion Correlation (DC) 
  - Sparsity
- **Perturbation Method**: Progressive Gaussian blur (σ=8) in 4 steps
- **Statistical Analysis**: Pearson correlation across stratified test samples

## Results

| Model | Training Type | Accuracy | Precision | Recall | F1-Score | AUC |
|-------|---------------|----------|-----------|--------|----------|-----|
| DenseNet201 | Contextual | **92.80%** | **93.23%** | **92.80%** | **92.73%** | **98.62%** |
| DenseNet201 | ROI-based | 88.08% | 88.96% | 88.08% | 87.79% | 97.90% |
| ResNet50 | Contextual | 90.81% | 91.45% | 90.81% | 90.71% | 98.30% |
| ResNet50 | ROI-based | 85.75% | 87.31% | 85.75% | 85.25% | 97.17% |



**Explainability Results**: ROI-based models consistently achieved better IC, DC, and Sparsity scores, indicating more focused and interpretable attention patterns despite lower classification performance.

