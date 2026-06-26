# watch-ai-detector
Real-time watch detection using MobileNetV2 and OpenCV
A real-time AI system that detects wristwatches live via webcam using Transfer Learning with MobileNetV2 and OpenCV. It shows WATCH DETECTED or NO WATCH with a confidence score on screen.
About the Project
This is a binary image classification project. Instead of training from scratch, it uses MobileNetV2 pretrained on ImageNet as a feature extractor. Only the top layers are trained on a custom dataset. The result is a fast, accurate detector that runs on any CPU without a GPU.
Features
Real-time webcam detection using OpenCV. Transfer Learning with MobileNetV2. Live confidence score on screen. Data Augmentation for better accuracy. Runs on CPU — no GPU needed.
Model Architecture
Input image of 128x128 is passed through frozen MobileNetV2, then GlobalAveragePooling2D, then Dense layer of 64 neurons with ReLU, then Dropout of 0.3, and finally Dense of 1 neuron with Sigmoid giving Watch or No Watch output. Optimizer is Adam, Loss is Binary Crossentropy, trained for 25 epochs.
Dataset
Images downloaded from Google Images. Two classes: Watch and No Watch. Organized in train/watch and train/no_watch folders. Recommended 200 plus images per class for good accuracy.
Data Augmentation
Rescaling to 1/255, rotation up to 30 degrees, zoom of 0.3, horizontal flip, brightness range of 0.4 to 1.6, shear of 0.2, and 20 percent validation split were applied to improve generalization.
Installation
Clone the repo and run pip install tensorflow opencv-python numpy.
Usage
Run python model.py to train and save the model as watch_detector.h5. Then run python detect.py to start webcam detection. Press Q to quit. Green label means watch detected, red means no watch.
Results
Training accuracy around 95 percent, validation accuracy around 90 percent, runs in real time on CPU, model size around 14 MB.
Future Improvements
Multi-class watch brand detection, YOLO bounding box, Flask web app deployment, TensorFlow Lite for mobile, and training graphs.
Author
Saad Ullah Khan — github.com/saad-ullah-khan14
If this helped you, please give it a Star!
