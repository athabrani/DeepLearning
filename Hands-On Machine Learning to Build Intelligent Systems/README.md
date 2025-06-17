# Hands-On Machine Learning: A Chapter-by-Chapter Guide

This repository contains chapter-by-chapter code reproductions and theoretical explanations for Aurélien Géron's book, "Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow, 2nd Edition." It's designed to help students and practitioners deepen their understanding by actively engaging with the material.

## Part I: The Fundamentals of Machine Learning

* **[Chapter 1: The Machine Learning Landscape](#chapter-1-the-machine-learning-landscape)**: Introduces the core concepts of Machine Learning. We explore what ML is, the types of systems (supervised, unsupervised, reinforcement), the main challenges (e.g., overfitting, underfitting), and the general workflow of an ML project.

* **[Chapter 2: End-to-End Machine Learning Project](#chapter-2-end-to-end-machine-learning-project)**: A practical, hands-on chapter where we build a complete project from scratch using the California Housing dataset. It covers data fetching, exploratory data analysis, data preparation pipelines, model selection (Linear Regression, Decision Trees, Random Forests), and model fine-tuning with Grid Search.

* **[Chapter 3: Classification](#chapter-3-classification)**: This chapter focuses on classification tasks using the MNIST dataset as a case study. We cover key performance metrics like accuracy, confusion matrix, precision, recall, F1-score, and the ROC curve. We also explore different types of classification: binary, multiclass, multilabel, and multioutput.

* **[Chapter 4: Training Models](#chapter-4-training-models)**: We dive into the mechanics of how models are trained. This chapter covers Linear and Polynomial Regression, the Normal Equation, and an in-depth look at Gradient Descent algorithms (Batch, Stochastic, Mini-batch). It also introduces regularization techniques like Ridge, Lasso, and Elastic Net, and concludes with Logistic and Softmax Regression for classification.

* **[Chapter 5: Support Vector Machines (SVMs)](#chapter-5-support-vector-machines-svms)**: This chapter explains the core concepts of Support Vector Machines. Key topics include large margin classification, handling non-linear data with the kernel trick (polynomial, Gaussian RBF), and extending SVMs to regression tasks.

* **[Chapter 6: Decision Trees](#chapter-6-decision-trees)**: We explore one of the most intuitive models: Decision Trees. This chapter covers how to train, visualize, and make predictions. It discusses the CART algorithm, concepts like Gini impurity and entropy, regularization hyperparameters, and the model's inherent instability.

* **[Chapter 7: Ensemble Learning and Random Forests](#chapter-7-ensemble-learning-and-random-forests)**: This chapter introduces the power of combining multiple models. We explore ensemble techniques such as voting classifiers, bagging, pasting, and boosting (AdaBoost, Gradient Boosting). A significant focus is placed on Random Forests, one of the most powerful and popular ML algorithms.

* **[Chapter 8: Dimensionality Reduction](#chapter-8-dimensionality-reduction)**: We tackle the "curse of dimensionality." This chapter explains the main approaches to reducing the number of features in a dataset: projection and Manifold Learning, with a focus on implementing PCA, Kernel PCA, and LLE.

* **[Chapter 9: Unsupervised Learning Techniques](#chapter-9-unsupervised-learning-techniques)**: This chapter shifts the focus to learning from unlabeled data. We explore clustering algorithms like K-Means and DBSCAN, discuss anomaly detection, and delve into density estimation using Gaussian Mixture Models (GMMs).

## Part II: Neural Networks and Deep Learning

* **[Chapter 10: Introduction to Artificial Neural Networks with Keras](#chapter-10-introduction-to-artificial-neural-networks-with-keras)**: We transition from classical ML to Deep Learning. This chapter traces the history from biological to artificial neurons, introduces Perceptrons and Multilayer Perceptrons (MLPs), and provides a comprehensive guide to building models using Keras's Sequential, Functional, and Subclassing APIs.

* **[Chapter 11: Training Deep Neural Networks](#chapter-11-training-deep-neural-networks)**: This chapter addresses the key challenges of training deep networks. We cover unstable gradients (vanishing/exploding), advanced optimization algorithms (Momentum, Adam), weight initialization (Glorot, He), non-saturating activation functions (ReLU, ELU), Batch Normalization, and transfer learning.

* **[Chapter 12: Custom Models and Training with TensorFlow](#chapter-12-custom-models-and-training-with-tensorflow)**: For maximum flexibility, we dive into TensorFlow's lower-level API. This chapter teaches you how to create custom loss functions, metrics, layers, and models. It culminates in writing a custom training loop from scratch and explains how TensorFlow uses AutoGraph to optimize Python code.

* **[Chapter 13: Loading and Preprocessing Data with TensorFlow](#chapter-13-loading-and-preprocessing-data-with-tensorflow)**: Efficient data handling is crucial for large-scale projects. This chapter focuses on building high-performance data input pipelines using the `tf.data` API. We also cover the efficient TFRecord binary format and the Keras preprocessing layers for building end-to-end models.

* **[Chapter 14: Deep Computer Vision Using Convolutional Neural Networks (CNNs)](#chapter-14-deep-computer-vision-using-convolutional-neural-networks-cnns)**: This chapter explores the premier architecture for image tasks: CNNs. We examine the core building blocks (convolutional and pooling layers) and review influential architectures like LeNet-5, AlexNet, GoogLeNet, and ResNet. We apply these to tasks like object detection and semantic segmentation.

* **[Chapter 15: Processing Sequences Using RNNs and CNNs](#chapter-15-processing-sequences-using-rnns-and-cnns)**: We turn to sequential data like time series and text. This chapter introduces Recurrent Neural Networks (RNNs) and their more powerful variants, LSTM and GRU cells, which address short-term memory limitations. We also see how 1D CNNs like WaveNet can effectively process sequences.

* **[Chapter 16: Natural Language Processing with RNNs and Attention](#chapter-16-natural-language-processing-with-rnns-and-attention)**: This chapter applies deep learning to NLP. We build models for text generation and sentiment analysis. The core focus is on Encoder-Decoder architectures for machine translation, enhanced by the game-changing concept of attention mechanisms, which leads to the powerful Transformer architecture.

* **[Chapter 17: Representation Learning and Generative Learning Using Autoencoders and GANs](#chapter-17-representation-learning-and-generative-learning-using-autoencoders-and-gans)**: We explore two powerful unsupervised techniques. Autoencoders (including denoising, sparse, and variational variants) are used for feature extraction. Generative Adversarial Networks (GANs) are introduced for their incredible ability to generate new, realistic data samples.

* **[Chapter 18: Reinforcement Learning](#chapter-18-reinforcement-learning)**: This chapter introduces the exciting field of RL, where an agent learns to make decisions by interacting with an environment. We cover core concepts like policies and rewards, and implement algorithms like Policy Gradients and Deep Q-Networks (DQN). The chapter concludes with an introduction to the powerful TF-Agents library.

* **[Chapter 19: Training and Deploying TensorFlow Models at Scale](#chapter-19-training-and-deploying-tensorflow-models-at-scale)**: The final chapter bridges the gap between training a model and using it in the real world. We cover deploying models using TF Serving and Google Cloud AI Platform, and how to scale training across multiple GPUs and servers using TensorFlow's Distribution Strategies API.
