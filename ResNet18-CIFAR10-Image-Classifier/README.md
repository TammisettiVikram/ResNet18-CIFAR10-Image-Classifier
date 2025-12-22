---
title: CIFAR-10 Dual Backend Classifier
emoji: 🖼️
colorFrom: blue
colorTo: green
sdk: gradio
sdk_version: "4.44.1"
app_file: app.py
pinned: false
---

# CIFAR-10 Dual Backend Image Classifier

[![Hugging Face Spaces](https://img.shields.io/badge/🤗%20Hugging%20Face-Spaces-blue)](https://huggingface.co/spaces/TAMMISETTIVIKRAM/ResNet18-CIFAR10-Image-Classifier)
[![Docker](https://img.shields.io/badge/Docker-Ready-blue)](https://hub.docker.com)
[![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.104.1-green)](https://fastapi.tiangolo.com/)
[![Gradio](https://img.shields.io/badge/Gradio-4.44.1-orange)](https://gradio.app/)

A production-ready, full-stack machine learning application that classifies images into CIFAR-10 categories using dual deep learning backends. This project demonstrates advanced ML engineering skills through a scalable FastAPI backend, interactive Gradio frontend, and containerized deployment.

## 🚀 Live Demo

Experience the classifier in action: [Hugging Face Space](https://huggingface.co/spaces/TAMMISETTIVIKRAM/ResNet18-CIFAR10-Image-Classifier)

## ✨ Key Features

- **Dual ML Backends**: PyTorch ResNet18 and TensorFlow CNN implementations
- **Top-5 Predictions**: Premium user experience with confidence scores and visual bars
- **RESTful API**: Async FastAPI with automatic OpenAPI documentation
- **Interactive UI**: Modern Gradio interface with drag-and-drop uploads
- **Database Integration**: SQLite with SQLAlchemy for prediction logging and analytics
- **Containerized**: Docker-ready for seamless deployment
- **Cross-Platform**: Works on CPU with optimized inference

## 🛠️ Tech Stack

**Backend:**
- FastAPI (async web framework)
- SQLAlchemy (ORM with async support)
- PyTorch & TorchVision (ResNet18 fine-tuning)
- TensorFlow/Keras (CNN architecture)
- Uvicorn (ASGI server)

**Frontend:**
- Gradio (interactive ML demos)
- Vanilla JavaScript + HTML/CSS (custom web interface)
- Responsive design with dark/light themes

**DevOps:**
- Docker & Docker Compose
- Hugging Face Spaces (serverless deployment)
- Python packaging with requirements.txt

## 📊 Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Gradio UI     │    │   FastAPI       │    │   SQLite DB     │
│   (HuggingFace) │◄──►│   Backend       │◄──►│   Analytics     │
└─────────────────┘    └─────────────────┘    └─────────────────┘
                              │
                              ▼
                       ┌─────────────────┐
                       │   ML Models     │
                       │ PyTorch │ TF    │
                       └─────────────────┘
```

## 🎯 What Makes This Special

This project showcases enterprise-level ML engineering:
- **Model Serving**: Efficient inference with pre-loaded models
- **API Design**: RESTful endpoints with proper error handling
- **Data Pipeline**: Image preprocessing and postprocessing
- **User Experience**: Progressive enhancement from basic to premium features
- **Scalability**: Async operations and containerization for production

Built with modern Python practices, this classifier represents a complete ML solution from model training to user-facing application.