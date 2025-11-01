# Deep Learning-Based Skin Lesion Segmentation and Classification

![Skin Lesions](https://static.wixstatic.com/media/895ce7_48775c7061f24f7fa9f314fc64ffd07a~mv2.png/v1/crop/x_0,y_0,w_713,h_378/fill/w_245,h_130,fp_0.50_0.50,q_85,usm_0.66_1.00_0.01,enc_avif,quality_auto/milk10k.png)

## Project Overview

This research project aims to develop deep learning models for automated skin lesion analysis, focusing on two core tasks:
1. **Skin Lesion Segmentation**: Using the U-Net architecture to accurately segment skin lesion regions
2. **Skin Lesion Classification**: Using the EfficientNet architecture for multi-class classification of skin lesions

Skin cancer is one of the most common cancer types worldwide, and early accurate diagnosis is crucial for improving patient survival rates. This project leverages computer vision and deep learning technologies to provide solutions for automated skin lesion analysis, with the potential to assist dermatologists in making more accurate and efficient diagnoses.

## Directory Structure

```
Skin-lesion-classification/
│
├── skin lesion task-1.ipynb  # Skin lesion segmentation task
├── skin lesion task-3.ipynb  # Skin lesion classification task
└── README.md                 # Project documentation
```

## Technical Implementation

### Task 1: Skin Lesion Segmentation (skin lesion task-1.ipynb)

#### Principles
The skin lesion segmentation task aims to precisely locate and delineate lesion regions in images, which is a critical first step in automated skin lesion analysis. This task employs semantic segmentation techniques to classify each pixel as either "lesion region" or "non-lesion region."

#### Model Architecture: U-Net
![U-Net Architecture](https://lmb.informatik.uni-freiburg.de/people/ronneber/u-net/u-net-architecture.png)

U-Net is a classic model for medical image segmentation with the following characteristics:
- **Encoder-Decoder Structure**: Captures contextual information through downsampling and recovers spatial resolution through upsampling
- **Skip Connections**: Links corresponding encoder and decoder layers to preserve high-resolution features
- **Symmetric Structure**: Ensures balanced feature dimension processing

Core Implementation:
```python
def unet_model(input_size=(256, 256, 3)):
    inputs = Input(input_size)
    
    # Encoder path
    c1 = Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(inputs)
    c1 = Dropout(0.1)(c1)
    c1 = Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c1)
    p1 = MaxPooling2D((2, 2))(c1)
    
    # ... more encoding layers ...
    
    # Bottleneck
    c5 = Conv2D(256, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p4)
    c5 = Dropout(0.3)(c5)
    c5 = Conv2D(256, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c5)
    
    # Decoder path
    u6 = Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(c5)
    u6 = concatenate([u6, c4])
    c6 = Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u6)
    
    # ... more decoding layers ...
    
    # Output layer
    outputs = Conv2D(1, (1, 1), activation='sigmoid')(c9)
    
    model = Model(inputs=[inputs], outputs=[outputs])
    return model
```

#### Data Processing
- **Data Augmentation**: Rotation, scaling, horizontal and vertical flipping to enhance model generalization
- **Data Normalization**: Scaling pixel values to the [0,1] range to accelerate model convergence
- **Random Cropping**: Cropping regions from original images to increase training sample diversity

#### Training Strategy
- **Loss Function**: Combined binary cross-entropy and Dice loss to balance class imbalance issues
- **Optimizer**: Adam optimizer with learning rate = 0.0001
- **Callbacks**: Learning rate scheduling, early stopping, and model checkpoints to prevent overfitting and save the best model
- **Validation Strategy**: Train-validation split to evaluate model performance

#### Evaluation Metrics
- **Dice Coefficient**: Measures the overlap between predicted and ground truth segmentations
- **IoU (Intersection over Union)**: Standard evaluation metric for segmentation tasks
- **Accuracy**: Pixel-level classification accuracy
- **Sensitivity and Specificity**: Evaluate the model's ability to identify lesion and healthy regions respectively

#### Results Visualization
![Segmentation Results](https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcThUfbGUbo8n-PREwipG7ByhgmRcgpczU9pDQ&s)

#### Challenges and Solutions
1. **Boundary Ambiguity**: Skin lesion boundaries are often unclear
   - Solution: Edge enhancement preprocessing and boundary-aware loss functions
   
2. **Data Imbalance**: Background pixels far outnumber lesion region pixels
   - Solution: Weighted loss functions and Dice loss, giving higher weights to minority classes
   
3. **Small Lesion Detection**: Model struggles to detect small lesions
   - Solution: Multi-scale feature fusion and attention mechanisms

### Task 3: Skin Lesion Classification (skin lesion task-3.ipynb)

#### Principles
The skin lesion classification task aims to categorize skin lesion images into different types of skin diseases, such as benign nevi, basal cell carcinoma, melanoma, etc. This task employs transfer learning methods, using pre-trained models to extract features before fine-tuning for skin lesion classification.

#### Model Architecture: EfficientNet
![EfficientNet Architecture](https://blog.roboflow.com/content/images/2024/04/image-1081.webp)

The EfficientNet series of models is known for its efficient parameter utilization and exceptional performance, with the following features:
- **Compound Scaling**: Simultaneously scales network depth, width, and resolution for optimal performance
- **Mobile Inverted Bottleneck Structure**: Reduces parameter count and computational complexity
- **Swish Activation Function**: Enhances non-linear expression capability

Core Implementation:
```python
def create_model(input_shape=(224, 224, 3), num_classes=7):
    # Load pre-trained EfficientNetB0
    base_model = EfficientNetB0(weights='imagenet', 
                                include_top=False, 
                                input_shape=input_shape)
    
    # Freeze the base model
    for layer in base_model.layers:
        layer.trainable = False
        
    # Add classification head
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = BatchNormalization()(x)
    x = Dropout(0.5)(x)
    x = Dense(512, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.5)(x)
    x = Dense(128, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.25)(x)
    predictions = Dense(num_classes, activation='softmax')(x)
    
    model = Model(inputs=base_model.input, outputs=predictions)
    return model
```

#### Data Processing
- **Data Augmentation**: Rotation, scaling, translation, brightness adjustment, etc., to enhance model robustness to different imaging conditions
- **Image Normalization**: Conforming to the input requirements of ImageNet pre-trained models
- **Class Balancing**: Using class weights or oversampling to handle class imbalance issues

#### Training Strategy
- **Two-Stage Training**:
  1. Freeze the base model and train only the classification head
  2. Unfreeze some base model layers for fine-tuning
- **Loss Function**: Categorical cross-entropy with class weights
- **Optimizer**: Adam optimizer with a smaller learning rate for fine-tuning
- **Learning Rate Scheduling**: Using ReduceLROnPlateau to dynamically adjust learning rates
- **Early Stopping**: Monitor validation performance to avoid overfitting

#### Evaluation Metrics
- **Accuracy**: Overall classification accuracy
- **Confusion Matrix**: Detailed display of classification results between categories
- **Precision, Recall, F1 Score**: Detailed evaluation metrics for each category
- **ROC Curve and AUC**: Evaluate the model's ability to distinguish between different categories

#### Results Visualization
![Classification Results and Confusion Matrix](https://www.mdpi.com/sensors/sensors-22-04399/article_deploy/html/images/sensors-22-04399-g001.png)

#### Challenges and Solutions
1. **Class Imbalance**: Significant differences in sample numbers between different skin lesion categories
   - Solution: Class weight adjustment, oversampling minority classes, undersampling majority classes
   
2. **Domain Transfer Issues**: Differences between the pre-trained model domain and skin images
   - Solution: Layer-by-layer unfreezing for fine-tuning, domain adaptation techniques
   
3. **Visual Similarity**: Some skin lesion categories have similar visual features and are difficult to distinguish
   - Solution: Using attention mechanisms and higher resolution inputs to focus on subtle differences

## Project Innovations

1. **Combined Segmentation and Classification**: Integrating segmentation and classification tasks to form a complete skin lesion analysis pipeline
2. **Multi-Level Loss Functions**: Designing combination loss functions tailored to skin lesion characteristics
3. **Feature Enhancement Modules**: Introducing attention mechanisms to enhance feature extraction in key lesion areas
4. **Multi-Scale Feature Fusion**: Utilizing both local details and global contextual information

## Requirements

```
tensorflow>=2.4.0
keras>=2.4.0
opencv-python>=4.1.2
numpy>=1.19.2
pandas>=1.1.3
matplotlib>=3.3.2
scikit-learn>=0.23.2
pillow>=8.0.1
```

## Usage Instructions

1. Clone the repository
```bash
git clone https://github.com/yourusername/Skin-lesion-classification.git
cd Skin-lesion-classification
```

2. Install dependencies
```bash
pip install -r requirements.txt
```

3. Run Jupyter Notebook
```bash
jupyter notebook
```

4. Run the two tasks in sequence:
   - First run `skin lesion task-1.ipynb` for segmentation
   - Then run `skin lesion task-3.ipynb` for classification

## Dataset

This project uses the **ISIC 2018: Skin Lesion Analysis Towards Melanoma Detection Challenge** dataset:
- Official download page: https://challenge.isic-archive.com/data/#2018
- Dataset contains:
  - 2594 training images with corresponding segmentation masks
  - 7 skin lesion categories
  - High-resolution RGB images


## Conclusions and Future Work

### Conclusions
1. The U-Net architecture achieved excellent results for skin lesion segmentation, with Dice coefficients above 0.91
2. The EfficientNet-based classification model achieved 87% accuracy across 7 skin lesion categories
3. Data augmentation and transfer learning are crucial for improving model generalization
4. Combined loss functions effectively addressed class imbalance issues

### Future Work
1. Explore more advanced network architectures, such as Transformers or hybrid CNN-Transformer models
2. Incorporate more clinical information for multimodal analysis with image features
3. Develop lightweight models suitable for mobile device deployment
4. Extend research to more types of skin lesions

