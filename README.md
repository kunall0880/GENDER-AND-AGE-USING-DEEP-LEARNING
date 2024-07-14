# GENDER-AND-AGE-USING-DEEP-LEARNING
# LINK TO DATASET -(https://drive.google.com/file/d/1VdSma4nctoU4aG_tCV2UHotbcygpXzVU/view?usp=sharing)

-------------------------------------------------------------------------------------------------

IF YOU WANT TO RUN IN LOCAL 
Prerequisites
Python: Ensure you have Python installed. Preferably Python 3.7 or later.
Virtual Environment: It's a good practice to use virtual environments to manage dependencies.
Step 1: Set Up a Virtual Environment
Open a terminal and create a virtual environment:

bash
Copy code
python -m venv age-gender-env
Activate the virtual environment:

On Windows:
bash
Copy code
age-gender-env\Scripts\activate

On macOS/Linux:
bash
Copy code
source age-gender-env/bin/activate

Step 2: Install Required Libraries
Install the necessary libraries using pip:

bash
Copy code
pip install tensorflow keras numpy matplotlib seaborn pandas tqdm pillow
Step 3: Set Up the Project Directory
Create a project directory and navigate into it:

Additional Notes
Ensure you replace the img_to_test path with the path to your test image.
If you encounter any issues with library versions or compatibility, consider creating a requirements.txt file with the exact versions of the libraries used, and install them using pip install -r requirements.txt.
