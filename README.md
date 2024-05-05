# Appendicitis-prediction
The objective of this project is to build a robust machine learning model that can predict the likelihood of appendicitis based on basic parameters and a set of clinical features using machine learning algorithms.


Standard Operating Procedure
Purpose:
To establish guideline to run the project files and a guide to run user interface.

Requirements:
Jupyter Notebook
Python 3.11.4
Anaconda navigator
Visual studio

Required libraries (for both jupyter notebook and visual studio):
Use this command: pip install library_name
flask
numpy
pandas
sklearn
seaborn
matplotlib

Step 1: To run Pre-processing file.
This file consists the code for preprocessing steps and code 
Open Jupyter notebook. Find the preprocessing.ipynb file. Make sure the app_data.xlsx dataset is present in same folder. And then click on run and press run all cells to run all cells at once. Otherwise run each cell by clicking the play button ▷ at top.

Step 2: To run EDA file.
This file consists the code for detailed exploratory data analysis of data
Open Jupyter notebook. Find the EDA.ipynb file. And then click on run and press run all cells to run all cells at once. Otherwise run each cell by clicking the play button ▷ at top.


Step 3: To run Models file (With oversampling).
This file consists the code for different classification algorithms that were used.
Open Jupyter notebook. Find the models.ipynb file. And then click on run and press run all cells to run all cells at once. Otherwise run each cell by clicking the play button ▷ at top.

Step 4: To run Models file (Without oversampling).
This file consists the code for different classification algorithms that were used for data without oversampling.
Open Jupyter notebook. Find the models-1.ipynb file. And then click on run and press run all cells to run all cells at once. Otherwise run each cell by clicking the play button ▷ at top.

For user interface
Make sure the app.py, static and templates folder is in a directory. predict.html file should be present inside templates folder. Similarly bc.jpg should be present inside static folder. 
 
Open app.py in visual studio. Make sure all the necessary libraries and extensions are there. Next run and debug the file.
 
From the terminal copy this link and paste it in any web browser.
Enter the values in the UI page and click on predict button. To clear the page click on clear button.
