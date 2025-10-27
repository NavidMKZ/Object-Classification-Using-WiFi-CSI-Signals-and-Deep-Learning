# Object-Classification-Using-WiFi-CSI-Signals-and-Deep-Learning

This project implements a deep learning model for classifying objects such as humans, cats, and other creatures using WiFi Channel State Information (CSI) signals. The model is trained on the **Animal Crossing WiFi CSI Dataset** and utilizes an LSTM-based architecture to process sequential signal data. The model achieves promising results in classifying different objects from wireless signals.

## Dataset

The dataset used in this project is the **Animal Crossing WiFi CSI Dataset**. It contains WiFi CSI signals labeled according to the object/creature present, such as humans, cats, and other animals. The dataset captures continuous WiFi signals and represents them as time-series data across multiple subcarriers.

- **Dataset Link**: [Animal Crossing WiFi CSI Dataset](https://github.com/IBM/Animal-Crossing-WiFi-CSI)  
- **Access**: The dataset is downloaded as `TRAIN.parquet` and `TEST.parquet` files for training and evaluation.  

---

## Model Components

### 1. Data Preprocessing
- The continuous CSI signals are reshaped into a 3D tensor `(num_samples, TIME_FRAMES, N_SUBCARRIERS)` suitable for LSTM input.
- Labels are encoded as integers corresponding to each object class.

### 2. LSTM Model
- An LSTM-based neural network is used to capture temporal dependencies in WiFi signals.
- The network consists of multiple LSTM layers followed by a fully connected layer to output class probabilities.

### 3. Training
- The model is trained using cross-entropy loss and optimized with the Adam optimizer.
- GPU acceleration is utilized for faster training on large datasets.

---

## Requirements

- Python 3.x  
- PyTorch  
- NumPy  
- pandas  
- scikit-learn  
- Matplotlib (for visualization)  

---

## Results

- The LSTM model successfully classifies objects from WiFi CSI signals.  
- Performance metrics include accuracy, F1 score, and confusion matrix.  
- Example results show high accuracy in differentiating humans, cats, and other creatures using WiFi data.

---

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for more details.
