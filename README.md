# GENDER-AND-AGE-USING-DEEP-LEARNING

## Link to Dataset
[Download Dataset](https://drive.google.com/file/d/1VdSma4nctoU4aG_tCV2UHotbcygpXzVU/view?usp=sharing)

---

## If You Want to Run Locally

### Prerequisites

- **Python**: Ensure you have Python installed. Preferably Python 3.7 or later.
- **Virtual Environment**: It's a good practice to use virtual environments to manage dependencies.

### Step 1: Set Up a Virtual Environment

Open a terminal and create a virtual environment:

\`\`\`bash
python -m venv age-gender-env
\`\`\`

Activate the virtual environment:

**On Windows:**
\`\`\`bash
age-gender-env\Scripts\activate
\`\`\`

**On macOS/Linux:**
\`\`\`bash
source age-gender-env/bin/activate
\`\`\`

### Step 2: Install Required Libraries

Install the necessary libraries using pip:

\`\`\`bash
pip install tensorflow keras numpy matplotlib seaborn pandas tqdm pillow
\`\`\`

### Step 3: Set Up the Project Directory

Create a project directory and navigate into it:

\`\`\`bash
mkdir gender-age-detection
cd gender-age-detection
\`\`\`

### Additional Notes

- Ensure you replace the \`img_to_test\` path with the path to your test image.
- If you encounter any issues with library versions or compatibility, consider creating a \`requirements.txt\` file with the exact versions of the libraries used, and install them using:

\`\`\`bash
pip install -r requirements.txt
\`\`\`

### Example \`requirements.txt\`

Here's an example of what your \`requirements.txt\` file might look like:

\`\`\`text
tensorflow==2.5.0
keras==2.4.3
numpy==1.19.5
matplotlib==3.3.4
seaborn==0.11.1
pandas==1.2.4
tqdm==4.61.0
Pillow==8.2.0
\`\`\`

### Running the Project

To run the project, ensure all the dependencies are installed and the dataset is downloaded. Replace the \`img_to_test\` path with the path to your test image in your script, and execute your Python script.

\`\`\`bash
python your_script.py
\`\`\`

Replace \`your_script.py\` with the actual name of your script.

---

This README file provides a clear, step-by-step guide for setting up and running your gender and age detection project using deep learning on a local machine.
