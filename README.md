# 🤓 WDT-MD: Wavelet Diffusion Transformers for Microaneurysm Detection in Fundus Images

This code is a **pytorch** implementation of our paper "**WDT-MD: Wavelet Diffusion Transformers for Microaneurysm Detection in Fundus Images**".

<img width="5912" height="1680" alt="d1" src="https://github.com/user-attachments/assets/f8845132-acbc-456e-bf67-be77ea995208" />

<div align="center">
  Figure 1: Overview of our proposed WDT-MD method.
</div>

## 🧑🏻‍🏫 Background

**Diabetic Retinopathy** (**DR**) is a serious complication affecting individuals with diabetes and can result in severe vision loss if not treated promptly. In the initial stages of DR, retinal capillaries are damaged due to hyperglycemia, which weakens the capillary walls and leads to **Microaneurysms** (**MAs**). MAs are small outpouchings in the lumen of the retinal vessels, typically measuring 15-60 $\mu m$ in diameter. Identification of MAs allows for timely recognition of DR, thus providing an opportunity for early intervention in patients. To analyze them, fundus images are widely used where small red dots are an indication of MAs. Nevertheless, MAs are tiny and inconspicuous with variations in brightness, contrast, and shape, making it difficult for physicians to detect them. Therefore, automated MA detection methods with high accuracy in fundus images are of great significance.

<img width="2011" height="941" alt="data" src="https://github.com/user-attachments/assets/9b521b12-31c6-43e5-8537-0991e921f889" />

<div align="center">
  Figure 2: An illustration of MAs in fundus images.
</div>

## 😖 Current Challenges

1. The inherent risk of learning "``identity mapping``" still persists in existing frameworks based on diffusion models. "Identity mapping" refers to the behavior of directly copying the input as output, whether normal or abnormal. This contradicts the foundational assumption that anomalies induce significant reconstruction deviations, ultimately causing false negatives.

2. The inability to distinguish MAs from other anomalies leads to ``high false positives``. Existing methods lacking pixel-level supervision signals tend to treat all reconstruction errors as homogeneous indicators of abnormality, disregarding the unique morphological and contextual signatures of the target anomalies. Consequently, confounding factors such as imaging artifacts or coexisting lesions can be indiscriminately flagged as MA candidates, undermining clinical utility.

3. The ``suboptimal reconstruction quality`` of normal features hampers the performance of AD. In retinal imaging, incomplete restoration of vascular patterns may introduce spurious reconstruction errors, masking true MA lesions or misclassifying normal variations as anomalies. 

## 🌟 Primary Contributions

To address these challenges, we propose a **W**avelet **D**iffusion **T**ransformer framework for **M**A **D**etection (**WDT-MD**). This is a supervised image-conditioned wavelet-domain model based on  Diffusion Transformers (DiTs). Our contributions can be summarized as follows:

1. In order to mitigate "identity mapping", we propose a ``noise-encoded image conditioning`` mechanism for diffusion-based MA detection. By perturbing the image condition with random intensities during training, the model is driven to capture the normal pattern.

2. To alleviate the issue of high false positives, we introduce pixel-level supervision signals in the training process through ``pseudo-normal pattern synthesis``. Specifically, we obtain the pseudo-normal labels align with the spatial distribution of real fundus images using inpainting techniques. This enables the model to distinguish MAs from other anomalies, thereby improving the detection performance.

3. To improve the reconstruction quality of normal features, we propose a ``wavelet diffusion Transformer`` architecture, which combines the global modelling capability of DiTs with the multi-scale analysis advantage of wavelet decomposition to better understand the overall structure and detailed information of fundus images.

4. Comprehensive experiments on the IDRiD and e-ophtha MA datasets demonstrate exceptional performance of our WDT-MD, holding significant promise for improving early DR screening.

## 🚀 Pre-Trained Models

<img width="4181" height="1888" alt="d2" src="https://github.com/user-attachments/assets/98e4bad8-7ccc-4165-a673-5708d59b65c2" />

<div align="center">
  Figure 3: The training process of our WDT-MD.
</div>

## 📦 Datasets

Two publicly available datasets, namely [IDRiD](https://ieee-dataport.org/openaccess/indian-diabetic-retinopathy-image-dataset-idrid) and [e-ophtha MA](https://www.adcis.net/en/third-party/e-ophtha), are adopted for extensive evaluation.

**The IDRiD dataset**, a benchmark resource for diabetic retinopathy analysis, was adapted for our study. For MA detection, we curated a subset of 249 samples, including 199 training cases, 24 validation cases, and 26 test cases. Specifically, the training set contains 134 normal images and 65 abnormal images. Contrast Limited Adaptive Histogram Equalization (CLAHE) was applied with 8 $\times$ 8 tile grids and a 2.0 clip limit to enhance contrast. Considering  the computational overhead, we implemented dimension standardization through bilinear downsampling to 300 $\times$ 200 pixels. 

**The e-ophtha MA dataset** consists of 381 cases divided into 304 training, 38 validation, and 39 test samples. Specifically, the training set contains 188 normal images and 116 abnormal images. The preprocessing pipeline maintained strict consistency with IDRiD: (1) CLAHE (8 $\times$ 8 tile grids, 2.0 clip limit); (2) downsampling to 300 $\times$ 200 pixels.

<img width="1341" height="1371" alt="dataset" src="https://github.com/user-attachments/assets/701ce580-033a-458a-836b-dd525a628473" />

<div align="center">
  Figure 4: Raw and preprocessed data from the IDRiD and e-ophtha MA datasets (partial).
</div>

## 🌵 Dependencies

```
pip install -r requirements.txt
```

## 🍳 Training

```
python wdt_train.py
```

## 🚅 Inference

```
python wdt_eval.py
```
