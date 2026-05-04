# Automatic Surgical Phase Recognition using Video Transformer Models

This project focuses on automatic classification of surgical phases in laparoscopic cholecystectomy videos using Video Transformer models. The main goal is to recognize the current phase of a surgery from video frames, which can support surgical workflow analysis, education, and future decision-support systems in the operating room.

## Project Overview

Surgical phase recognition is a computer vision task where a model predicts which phase of an operation is currently being performed. In this project, I worked with laparoscopic gallbladder removal videos and trained a deep learning model to classify each video segment into one of seven surgical phases.

The project uses a Video Swin Transformer model because surgical phase recognition depends not only on individual frames, but also on temporal information across multiple frames. Instead of classifying one image at a time, the model receives short video sequences and learns motion and context over time.

## Datasets

The project uses two surgical video datasets:

### Cholec80

Cholec80 is used for pretraining the model. It contains laparoscopic cholecystectomy videos annotated with surgical phase labels.

### CholecTrack20

CholecTrack20 is used for fine-tuning and evaluation. It contains videos, extracted frames, JSON annotations, and metadata files.

The model predicts seven surgical phases:

1. Preparation
2. Calot Triangle Dissection
3. Clipping and Cutting
4. Gallbladder Dissection
5. Gallbladder Packaging
6. Cleaning and Coagulation
7. Gallbladder Retraction

## Methodology

### Data Preparation

The video frames were processed into frame-level labels using the annotation files. After that, the frames were grouped into fixed-length temporal sequences.

The main sequence settings were:

- Sequence length: 16 frames
- Stride: 8 frames

This means that each training example contains 16 consecutive frames. A stride of 8 allows overlap between sequences, which gives the model more training samples and helps preserve temporal continuity.

### Model

The main model used in this project is:

- Video Swin Transformer
- Pretrained on Kinetics-400
- Adapted for 7 surgical phase classes

Video Swin Transformer was chosen because it is designed for video understanding tasks. It can learn both spatial features from frames and temporal relationships between frames.

### Transfer Learning

Transfer learning was used because training a video transformer from scratch would require a very large amount of data and computational resources.

The model was first initialized with pretrained weights, then adapted to the surgical phase recognition task. The general idea was:

1. Use pretrained video features from a large video dataset.
2. Replace the classification head with a new layer for 7 surgical phases.
3. Fine-tune the model on surgical video data.

## Training

The training pipeline included:

- PyTorch
- Video Swin Transformer
- AdamW optimizer
- Cross-entropy loss / weighted loss
- Data augmentation
- Transfer learning
- GPU acceleration with CUDA

AdamW was used because it is commonly used for transformer-based architectures and includes weight decay, which helps with regularization.

Data augmentation was used to reduce overfitting and improve generalization. Examples of augmentations include resizing, cropping, horizontal flipping, and light color jitter.

## Evaluation Metrics

The model was evaluated using:

### Frame Accuracy

Frame accuracy measures the percentage of correctly classified frames or sequences.

It is easy to understand, but it can be misleading when the dataset is imbalanced.

### F1 Score

F1 score combines precision and recall. It is especially useful in this project because some surgical phases appear much more often than others.

Because of class imbalance, F1 score is more informative than accuracy alone.

## Results

The model achieved strong performance on the surgical phase recognition task.

Example evaluation results:

- Frame Accuracy: around 94.88%
- Macro F1 Score: around 0.90

These results show that the model was able to learn meaningful temporal and visual patterns from surgical videos.

## Challenges

### Class Imbalance

Some surgical phases appear much more frequently than others. This can cause the model to perform better on common phases and worse on rare phases.

To address this, class weights and F1 score evaluation were used.

### Disk Bottleneck

One practical challenge was slow data loading. Since the model uses many image frames, the disk was often at high usage, which slowed down training and reduced GPU utilization.

Possible solutions included:

- Increasing DataLoader workers
- Using batch size experiments
- Caching images in RAM
- Optimizing frame loading

### Temporal Context

Surgical phases are not always obvious from a single frame. Some phases look visually similar, so temporal context is important. This is why the model uses sequences instead of individual frames.

## Technologies Used

- Python
- PyTorch
- TorchVision
- Video Swin Transformer
- CUDA
- NumPy
- Pandas
- Scikit-learn
- OpenCV
- Cholec80 dataset
- CholecTrack20 dataset
