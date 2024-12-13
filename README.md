## **Animal Health Prediction Project**

This repository contains a machine learning-based application designed to predict whether an animal's health condition is dangerous or not, based on provided symptoms. The project includes model training, feature encoding, and a web-based user interface built using Streamlit.

---

### **Project Overview**
The project leverages a **Random Forest Classifier** to analyze symptoms and predict the health condition of animals. This system provides a user-friendly interface for entering symptoms and retrieving predictions about whether the condition is dangerous.

---

### **Files in This Repository**
1. **`animalpredictionjup.ipynb`**:  
   A Jupyter Notebook containing the complete workflow for data preprocessing, model training, and evaluation.

2. **`app.py`**:  
   A Streamlit application that serves as a user-friendly interface for making health predictions. *(The app was created using Spyder)*

3. **`data.csv`**:  
   The dataset containing symptoms and their corresponding labels (dangerous or not).

4. **`random_forest.joblib`**:  
   A serialized Random Forest model trained to predict animal health conditions.

5. **`onehot_encoder.pkl`**:  
   A pre-trained OneHotEncoder used for encoding categorical features in the dataset.

---

### **How to Run the Project**

#### **Prerequisites**
Ensure you have **Python 3.x** installed on your system. The following libraries are required for this project:

- `streamlit`
- `scikit-learn`
- `pandas`
- `numpy`
- `joblib`

To ensure consistency between the **Jupyter Notebook** and **Anaconda terminal**, it's important that both environments use the same versions of the libraries.

#### **Steps to Set Up and Run the Application**
1. **Create and Activate the Environment** (Optional but recommended)  
   It's best to create a virtual environment for the project to ensure consistent versions across both the Jupyter Notebook and Anaconda terminal.
   ```bash
   conda create --name animalhealth python=3.x
   conda activate animalhealth
   ```

2. **Install the Required Libraries**  
   After activating your environment, install the necessary dependencies:
   ```bash
   pip install streamlit scikit-learn pandas numpy joblib
   ```

3. **Run the Streamlit Application**  
   Once everything is set up, you can run the Streamlit application:
   ```bash
   streamlit run app.py
   ```

---

### **Project Workflow**
1. **Data Preprocessing**:  
   - Cleaned and transformed the dataset using techniques such as **OneHotEncoding** for categorical features.

2. **Model Training**:  
   - Trained a **Random Forest Classifier** to predict whether the health condition is dangerous or not, based on symptoms.

3. **Deployment**:  
   - Integrated the trained model into a Streamlit app for real-time user interaction.

---

### **Features**
- Predict if an animal's health condition is dangerous based on symptoms.
- Interactive web-based user interface for input and output.
- Reusable **OneHotEncoder** and **Random Forest Classifier**.
---

### **Technologies Used**
- **Programming Language**: Python
- **Libraries**:
  - **Streamlit**: For building the interactive web application.
  - **scikit-learn**: For machine learning and preprocessing.
  - **pandas**: For data manipulation.
  - **numpy**: For numerical operations.
  - **joblib**: For saving and loading the model and encoder.

---

### **Future Enhancements**
- Incorporate more features into the dataset to improve prediction accuracy.
- Expand the dataset to cover a wider range of symptoms and conditions.
- Add visualization features to the Streamlit application for better insights.
---

### **Contributors**
- **Alla Pavani**  
  Machine Learning Enthusiast  

Feel free to reach out with feedback or suggestions!
