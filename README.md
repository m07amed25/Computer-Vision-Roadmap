# 🧠 Computer Vision Plan (12 Weeks)

Welcome to your journey to becoming a **Computer Vision Master using PyTorch**! This README outlines a structured weekly learning path, complete with learning objectives, detailed learning topics, resource links, coding projects, and key concepts to take you from solid ML basics to state-of-the-art GenAI computer vision capabilities. Let’s gooo 💻📸🚀

---

## 🗖️ Weekly Plan Overview

### **Week 1–2: CV + PyTorch Foundations**

**✅ Objectives:**

* Understand image formats: RGB, grayscale, channels, pixel ranges
* Learn tensors: creation, shape, slicing, broadcasting
* Get comfortable with PyTorch core modules
* Learn the full training loop: forward, loss, backward, optimizer step
* Visualize data, loss curves, and predictions

**📚 Resources:**

* [🔥 YouTube Course: Intro to CV with PyTorch](https://www.youtube.com/playlist?list=PLS84ypkqWiQ8-TL0AmTRynkzK0v-d4C5m)
* PyTorch Docs (Tensors & Autograd): [https://pytorch.org/tutorials/beginner/blitz/autograd\_tutorial.html](https://pytorch.org/tutorials/beginner/blitz/autograd_tutorial.html)
* CS231n Lecture 1 & 2

**👨‍💻 What to Learn:**

* Tensors: `torch.tensor()`, `requires_grad`, `.backward()`
* Layers: `nn.Linear`, `nn.ReLU`, `nn.Conv2d`, `nn.Softmax`
* Modules: `nn.Module`, `forward()` method
* Optimizers: `SGD`, `Adam`, learning rate tuning
* Losses: `CrossEntropyLoss`, `MSELoss`
* Data loading: `DataLoader`, `TensorDataset`

**👨‍💻 Mini Project:**

* Classify MNIST using a 2-layer CNN
* Visualize sample predictions
* Plot training/validation accuracy and loss

**🧠 Concepts to Master:**

* Difference between weights and activations
* How gradients flow
* Underfitting vs. overfitting

---

### **Week 3–4: CNN Architectures + Transfer Learning**

**✅ Objectives:**

* Explore CNN building blocks in detail
* Learn standard architectures from scratch and pre-trained
* Apply transfer learning using torchvision

**📚 Resources:**

* [Stanford CS231n: CNN Architectures](https://cs231n.github.io/convolutional-networks/)
* torchvision.models: [https://pytorch.org/vision/stable/models.html](https://pytorch.org/vision/stable/models.html)
* [ResNet Paper (He et al.)](https://arxiv.org/abs/1512.03385)

**👨‍💻 What to Learn:**

* Conv layers: `nn.Conv2d`, `nn.MaxPool2d`
* BatchNorm, Dropout, AdaptiveAvgPool2d
* Flattening layers, FC layers
* Transfer learning strategies: freeze base, fine-tune head
* `model.eval()` vs `model.train()` modes

**👨‍💻 Mini Project:**

* Load ResNet18 from torchvision
* Fine-tune on TrashNet dataset
* Experiment with freezing different layers

**🧠 Concepts to Master:**

* Depth vs. width in CNNs
* Role of residual connections
* Transfer learning vs. training from scratch

---

### **Week 5: Data Augmentation & Pipelines**

**✅ Objectives:**

* Learn data preprocessing best practices
* Boost generalization with strong augmentations
* Write custom PyTorch `Dataset` classes

**📚 Resources:**

* [Albumentations Docs](https://albumentations.ai/docs/)
* [Data Loading Tutorial](https://pytorch.org/tutorials/beginner/data_loading_tutorial.html)

**👨‍💻 What to Learn:**

* Transformations: `Resize`, `ToTensor`, `Normalize`, `RandomCrop`, `RandomHorizontalFlip`
* Composing transforms: `transforms.Compose()`
* Creating `__getitem__`, `__len__`
* Handling class imbalance with `WeightedRandomSampler`

**👨‍💻 Mini Project:**

* Create custom dataset class for TrashNet
* Use Albumentations to apply augmentations
* Visualize batch with `matplotlib`

**🧠 Concepts to Master:**

* Difference between train/test transforms
* Why augmentation prevents overfitting
* Data pipeline performance tips

---

### **Week 6–7: Object Detection (YOLO Time!)**

**✅ Objectives:**

* Learn bounding box regression and metrics
* Train your own YOLOv5 model

**📚 Resources:**

* [YOLOv5 Official Repo](https://github.com/ultralytics/yolov5)
* [Roboflow Annotation Tool](https://roboflow.com/)
* [FastObjectDetection with YOLOv8](https://docs.ultralytics.com/)

**👨‍💻 What to Learn:**

* YOLO architecture: backbone, neck, head
* Anchors, strides, NMS
* Labeling formats (COCO, Pascal VOC)
* Training script config (epochs, batch size, img size)

**👨‍💻 Mini Project:**

* Annotate 100 images with Roboflow
* Export in YOLO format
* Train YOLOv5s on TrashNet
* Inference with webcam

**🧠 Concepts to Master:**

* IoU, precision, recall, mAP\@.5
* Multi-class object detection
* Real-time inference tricks

---

### **Week 8: Semantic & Instance Segmentation**

**✅ Objectives:**

* Learn how to assign a class to each pixel
* Use segmentation models for pixel-wise predictions

**📚 Resources:**

* [UNet Paper](https://arxiv.org/abs/1505.04597)
* [Segmentation Models PyTorch](https://github.com/qubvel/segmentation_models.pytorch)

**👨‍💻 What to Learn:**

* `nn.ConvTranspose2d`, skip connections
* Loss functions: Dice Loss, BCEWithLogitsLoss
* Segmentation masks encoding/decoding
* Creating mask datasets from polygons

**👨‍💻 Mini Project:**

* Train UNet on semantic segmentation dataset
* Overlay masks on image with OpenCV

**🧠 Concepts to Master:**

* Binary vs. multiclass segmentation
* Handling overlapping masks
* Use of sigmoid vs softmax

---

### **Week 9: Explainability & Debugging**

**✅ Objectives:**

* Interpret CNN decisions
* Use tools like Captum and Grad-CAM

**📚 Resources:**

* [Captum Library](https://captum.ai/)
* [Grad-CAM GitHub](https://github.com/jacobgil/pytorch-grad-cam)

**👨‍💻 What to Learn:**

* Saliency maps, occlusion sensitivity, Integrated Gradients
* Visualizing layer activations
* Why explainability matters in real-world AI

**👨‍💻 Mini Project:**

* Apply Grad-CAM on TrashNet CNN model
* Compare heatmaps between true/false predictions

**🧠 Concepts to Master:**

* ReLU effect on saliency
* Visualization of internal layers

---

### **Week 10: Optimization & Edge Deployment**

**✅ Objectives:**

* Optimize models for edge deployment
* Export models to ONNX, TorchScript

**📚 Resources:**

* [ONNX Export Tutorial](https://pytorch.org/tutorials/advanced/super_resolution_with_onnxruntime.html)
* [Quantization Guide](https://pytorch.org/docs/stable/quantization.html)

**👨‍💻 What to Learn:**

* `torch.jit.script` vs `torch.jit.trace`
* Static vs. dynamic quantization
* Export to ONNX and run with OpenCV DNN

**👨‍💻 Mini Project:**

* Quantize YOLO model
* Export to ONNX, run live webcam inference

**🧠 Concepts to Master:**

* Accuracy drop after quantization
* Speed-up metrics
* Edge use-cases

---

### **Week 11: Self-Supervised & Foundation Models**

**✅ Objectives:**

* Explore vision-language models and zero-shot inference
* Learn about foundation models like CLIP and SAM

**📚 Resources:**

* [CLIP GitHub](https://github.com/openai/CLIP)
* [Segment Anything GitHub](https://github.com/facebookresearch/segment-anything)
* Hugging Face Model Hub

**👨‍💻 What to Learn:**

* CLIP: image and text embeddings
* SAM: promptable segmentation
* Contrastive learning: SimCLR, BYOL (concept only)

**👨‍💻 Mini Project:**

* Use CLIP for zero-shot TrashNet classification
* Use SAM to segment objects and match descriptions

**🧠 Concepts to Master:**

* Prompt engineering for vision tasks
* Foundation models training principles

---

### **Week 12: Capstone Project 🚀**

**✅ Final Challenge:**

* Combine what you've learned into a real-world project

**Project Ideas:**

* Smart Waste Sorting Assistant (YOLO + UNet + Streamlit)
* Real-time Sign Language Recognition
* Drone Surveillance using YOLO + ViT

**Deliverables:**

* GitHub Repo
* Streamlit or Gradio demo (optional)
* Edge deployment script (optional)

---

## 🛠️ Tools Checklist (Use Throughout):

* PyTorch, Torchvision
* OpenCV, Albumentations
* Matplotlib, Seaborn
* Gradio, Streamlit
* ONNX, TorchScript
* Roboflow, Label Studio
* Docker (optional), Jetson Nano (optional)

---

## 🎓 Final Words

Stick to this plan, finish your mini-projects, post them online (LinkedIn, GitHub), and you'll be miles ahead. Keep going, you're literally training your brain like a neural net. 💪🔥
