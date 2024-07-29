
# Face Generation with GANs using CelebA Dataset

## Introduction to GANs
Generative Adversarial Networks (GANs) are a class of artificial intelligence algorithms used in unsupervised machine learning, implemented by a system of two neural networks contesting with each other in a zero-sum game framework. This innovative approach can generate photographs that look at least superficially authentic to human observers, featuring many realistic characteristics. Initially proposed as a form of generative model for unsupervised learning, GANs have also found utility in semi-supervised learning, fully supervised learning, and reinforcement learning.

## Purpose of the Project
The objective of this project is to develop a GAN that can generate realistic human faces using the CelebA dataset. This provides a practical exploration into the complex dynamics of GANs and their applications in realistic image generation. The project serves as a foundational step towards understanding and leveraging the power of neural networks in creating lifelike images.

## Ongoing Development
This project is actively under development with aims to improve the performance and stability of the GAN models. Future updates may include more sophisticated GAN architectures, enhanced preprocessing techniques, and optimizations for better resource utilization. Contributions and suggestions are welcome to help advance this project further.

## Project Setup Guide

### Step-by-Step Instructions

#### 1. Downloading the CelebA Dataset
- Download the CelebA dataset from the following link: [CelebA Dataset](http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html).
- Extract the downloaded zip file and place it in the `data` folder at the root of the project directory.

#### 2. Create a Conda Environment
- To isolate and manage the project's dependencies, create a new Conda environment by running the following command:
  ```bash
  conda create -n gan_project python=3.8
  ```
- Activate the environment using:
  ```bash
  conda activate gan_project
  ```

#### 3. Install Required Packages
- Ensure `requirements.txt` contains all necessary packages, then install them using:
  ```bash
  pip install -r requirements.txt
  ```

#### 4. Install TensorFlow
- This project requires TensorFlow. Ensure your machine has the proper CUDA and cuDNN installations to support TensorFlow with GPU acceleration. Follow the detailed instructions here: [Install TensorFlow](https://www.tensorflow.org/install/pip).

#### 5. Preprocess Image Data
- Use the Jupyter notebook `image_processing.ipynb` to preprocess the image data from CelebA. This notebook will guide you through resizing and normalizing images.

#### 6. Create Datasets
- Run `dataset.py` to organize the processed images into training, validation, and test datasets:
  ```bash
  python dataset.py
  ```

#### 7. Train the GAN
- Execute `train.py` to start training the GAN:
  ```bash
  python train.py
  ```

#### 8. Test the Model
- After training, generate new images by running `test.py`:
  ```bash
  python test.py
  ```

## Conclusion
This project offers a dynamic and collaborative environment for exploring and enhancing GANs for image generation, which is under development. 
