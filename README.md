DigitSense – AI-Powered Handwritten Digit Recognition

Introduction

DigitSense is a machine learning-powered web application that recognizes handwritten digits. Built using Random Forest, Streamlit, and Docker, this project enables users to input digits and receive real-time predictions. The model is deployed on AWS EC2 for cloud-based accessibility.

Features

✅ Handwritten Digit Recognition using a trained Random Forest model✅ Interactive Web Interface powered by Streamlit✅ Containerized Deployment with Docker✅ Cloud-Hosted on AWS EC2 for seamless access✅ Fast and Efficient Predictions with optimized preprocessing

Tech Stack

Machine Learning: Random Forest, Scikit-learn

Web Framework: Streamlit

Backend Processing: Python, OpenCV, NumPy, Pandas

Containerization: Docker

Cloud Deployment: AWS EC2 (Ubuntu Server)

Installation & Usage

1. Clone the Repository

git clone https://github.com/your-username/DigitSense.git
cd DigitSense

2. Install Dependencies

pip install -r requirements.txt

3. Run the Streamlit App

streamlit run app.py

Docker Deployment

1. Build Docker Image

docker build -t digitsense .

2. Run Docker Container

docker run -p 8501:8501 digitsense

The app will be accessible at http://localhost:8501.

Deploying on AWS EC2

1. Set Up EC2 Instance

Choose Ubuntu Server

Ensure port 8501 is open in security settings

2. Install Docker on EC2

sudo apt update && sudo apt install docker.io -y

3. Pull and Run the Docker Image

docker pull your-dockerhub-username/digitsense
docker run -d -p 8501:8501 your-dockerhub-username/digitsense

4. Access the Application

Open http://<EC2-Public-IP>:8501 in a browser.

Example Usage

1️⃣ Upload or draw a handwritten digit2️⃣ Click "Predict" to analyze the input3️⃣ View the predicted digit with confidence score

