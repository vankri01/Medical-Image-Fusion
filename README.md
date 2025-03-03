# Medical-Image-Fusion
This project aims to enhance diagnostic capabilities by effectively combining MRI and CT 
images of the brain. By fusing these complementary imaging modalities, we can provide a 
comprehensive view that captures both soft and hard tissue details, aiding in improved medical 
diagnosis. Our study focuses on two distinct image fusion algorithms—Principal Component 
Analysis (PCA) and Convolutional Neural Network (CNN)-based fusion—to evaluate their 
efficacy in producing high-quality fused images. 

## Features

- Implementation of PCA-based and CNN-based image fusion.

- Utilizes Discrete Wavelet Transform (DWT) for image decomposition.

- Performance evaluation using PSNR and MSE.

- Pre-processing techniques including noise reduction and contrast enhancement.

- Comparison of fusion outputs with qualitative and quantitative metrics.

## Dataset

- Source: Harvard Medical School Brain MRI-CT dataset.

## Formats: TIFF images.

## Number of Pairs: 8 MRI-CT image pairs.

## Technologies Used

- Programming Language: Python

- Libraries: OpenCV, NumPy, Scikit-Image, PyTorch (for CNN model)

## Methodology

1. Data Collection: MRI & CT image acquisition.

2. Preprocessing:

3. Noise reduction (Non-Local Means Filter)

4. Contrast enhancement (CLAHE)

5. Edge sharpening (Unsharp Masking)

6.Fusion Techniques:

- PCA-based Fusion: Extracts principal components from both images and merges them.

- CNN-based Fusion (VGG19): Deep learning model trained on multimodal images.
- DWT-based Fusion: Decomposes images into frequency components and merges them.
7. Evaluation Metrics:

- 📊 PSNR: Measures the quality of the fused image.

- 📊 MSE: Determines error between fused and original images.
DWT-based Fusion: Decomposes images into frequency components and merges them.
## 📈 Results
| DATASET   |  PCA(PSNR dB) |  CNN (PSNR dB) |  PCA (MSE) |  CNN (MSE) |
|-----------|---------------|----------------|------------|------------|
|Image 1    | 27.65         | 27.91          | 111.70     | 105.13     |
|Image 2    | 27.65         | 27.91          | 111.70     | 105.13     |
|Image 3    | 27.65         | 27.90          | 111.70     | 105.13     |
|Image 4    | 27.65         | 27.91          | 111.70     | 105.13     |




### Observations: CNN-based fusion outperforms PCA-based fusion in preserving image details and reducing error.
## Installation & Usage
# Clone the repository
```
git clone https://github.com/yourusername/yourrepository.git

# Install dependencies
pip install opencv-python numpy scikit-image torch torchvision

# Run the script
python fusion.py
```
## 📸 Screenshots

### Preprocessed Images
![Preprocessed Image](https://raw.githubusercontent.com/vankri01/Medical-Image-Fusion/refs/heads/main/ct_final%20(1).png)  ![Preprocessed Image](https://raw.githubusercontent.com/vankri01/Medical-Image-Fusion/refs/heads/main/mri_final.png)
#### CT &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; MRI
### Fusion Output
![Fusion Output](https://raw.githubusercontent.com/vankri01/Medical-Image-Fusion/refs/heads/main/fused_image.png)

## Contributors
- [Deepti Kushwaha](https://github.com/Deeptikushwaha)
- [Chitranshi Joshi](https://github.com/chitranshi-j)



## 📜 License

This project is licensed under the **MIT License**.

[![MIT License](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)


