# PAN---card-tampering-detection
This repository contains a small production-ready Flask web app for PAN card tamper detection. The app accepts uploaded PAN card images and runs a tamper-detection model and returns a prediction (tampered / genuine), a confidence score, and an optional visualization showing regions of concern. 
🚀 Project Setup & Run Guide

Follow the steps below to set up and run the application.

🔧 Prerequisites

Anaconda
 (or Miniconda)

Python (version specified in requirements.txt)

⚡ Steps to Run the Application
1. Clone or Copy the Project

Download or clone this repository to your local machine.

2. Navigate to Project Folder

Open Command Prompt (or Anaconda Prompt) and change directory to where the app.py file is located:

cd path\to\your\project

3. Create a Virtual Environment

Create a new conda environment:

conda create --name myenv python=3.9


Replace myenv with any name you like.
You can also change the Python version depending on your needs.

4. Activate the Environment
conda activate myenv

5. Install Dependencies

Install all required packages from requirements.txt:

python -m pip install -r requirements.txt

6. Run the Application

Start the app using:

python app.py


You will see a local URL (for example:
http://127.0.0.1:5000/)

Copy and paste it into your browser to open the application.

7. Test with Sample Data

Use the sample_data folder (included in the project) to test the application with example images.

✅ Notes

If you face issues with conda, you can also use Python’s built-in venv:

python -m venv venv
venv\Scripts\activate   # On Windows
source venv/bin/activate # On macOS/Linux


Make sure you are inside the correct environment before running the app.

✨ That’s it! You’re ready to run the project.
