# Multi-Modal-Skin-Lesion-Classification-with-EfficientNetB1
A high-resolution (384×384) multi-modal deep learning framework for skin lesion classification on the HAM10000 dataset, combining dermoscopic images and clinical metadata using EfficientNetB1 and a two-stage fine-tuning strategy to improve minority-class performance.
In this project, we developed a high-resolution multi-modal deep learning model for skin lesion classification using the HAM10000 dataset, a well-known benchmark in dermatological image analysis. The dataset contains 10,015 dermoscopic images belonging to 7 diagnostic categories (akiec, bcc, bkl, df, mel, nv, vasc) along with clinically relevant metadata such as age, sex, and lesion localization.
Dataset link:
👉 https://www.kaggle.com/datasets/kmader/skin-cancer-mnist-ham10000

Data Preparation

We began by extracting the dataset from Google Drive and loading the metadata file (HAM10000_metadata.csv). Missing age values were filled with the mean age, categorical variables (sex and localization) were one-hot encoded, and the age feature was standardized using StandardScaler. Image paths were reconstructed from the two image folders, and labels were encoded using a sorted class mapping to ensure consistency. The dataset was split using a stratified 80/20 train–validation split to preserve class distribution. Due to strong class imbalance, class weights were computed and applied during training.

High-Resolution Image Processing

Unlike standard approaches that use 224×224 images, we increased the input resolution to 384×384, allowing the model to capture finer-grained visual patterns such as lesion borders, color variations, and texture irregularities. To fit this resolution within GPU memory constraints, the batch size was reduced to 16. A tf.data pipeline was used for efficient loading, resizing, batching, and prefetching.

Multi-Modal Architecture

The model follows a dual-branch (multi-modal) architecture:

Image branch:
A pretrained EfficientNetB1 backbone (ImageNet weights, no classification head) processes the dermoscopic images. Input images are passed through EfficientNet’s preprocessing function and a lightweight data augmentation pipeline (horizontal flip and small rotations). Global Average Pooling is applied to extract compact visual features.

Metadata branch:
Clinical metadata (≈17 features) is processed through a small MLP with a dense layer (32 units, ReLU).

The outputs of both branches are concatenated and passed directly to a final softmax classifier with 7 outputs.

Two-Stage Training Strategy

Training was performed in two stages:

Head-only training:
The EfficientNet backbone was frozen, and only the classification layers were trained using Adam optimizer (learning rate = 1e-3) with Early Stopping.

Fine-tuning:
The top 40 layers of EfficientNetB1 were unfrozen, and the model was fine-tuned with a much smaller learning rate (1e-5). This allowed the network to adapt high-level visual features to the medical domain while avoiding overfitting.

Final Results

The final model achieved strong and well-balanced performance on the validation set:

Validation Accuracy: 72.59%

Weighted F1-score: ≈ 0.75

Macro F1-score: ≈ 0.66

Notably, the model showed substantial improvements on minority classes, which are typically challenging in HAM10000:

akiec: F1 ≈ 0.68

bcc: F1 ≈ 0.77

vasc: F1 ≈ 0.71

mel: Recall ≈ 0.74

These results highlight the effectiveness of combining high-resolution images with clinical metadata and using a careful fine-tuning strategy. The approach demonstrates a practical and scalable framework for medical image classification tasks where both visual and non-visual patient information are available
