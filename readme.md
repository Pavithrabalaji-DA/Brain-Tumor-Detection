# Brain Tumor Detection Web App

This is a web application built using Flask for detecting brain tumors from MRI images.

## Features

- **Upload MRI Images**: Users can upload MRI images of the brain.
- **Tumor Detection**: The uploaded images are processed using a machine learning model to detect the presence of a brain tumor.
- **Display Results**: The application displays the prediction results to the user.

## Installation

1. Clone the repository:

    ```bash
    git clone https://github.com/Pavithrabalaji-DA/brain-tumor-detection-app.git
    ```

2. Navigate to the project directory:

    ```bash
    cd brain-tumor-detection-app
    ```

3. Install and create a virtual environment:

    ```bash
    pip install virtualenv
    virtualenv env
    .\env\Scripts\activate.ps1
    ```

4. Install the dependencies:

    ```bash
    pip install -r requirements.txt
    ```

## Usage

1. Start the Flask server:

    ```bash
    python app.py
    ```

2. Open your web browser and navigate to [http://127.0.0.1:5000](http://127.0.0.1:5000).

3. Upload an MRI image of the brain using the provided form.

4. Wait for the application to process the image and display the prediction results.

## File Structure

- `app.py`: Main Flask application file.
- `model/`: Directory containing the trained machine learning model.
- `static/`: Directory for static files (e.g., CSS, JavaScript, images).
- `templates/`: Directory for HTML templates.

## Dependencies

- Flask: Web framework for Python.
- Keras: Deep learning library.
- TensorFlow: Machine learning library.

## Screenshots

### Home Page
![Home Page](![Screenshot (245)](https://github.com/user-attachments/assets/c16af5d7-cf8c-4ebc-917d-b83dcc6a4db2)
)

### Upload Page
![Upload Page](![Screenshot (246)](https://github.com/user-attachments/assets/507149f4-14d6-4e2f-8f31-903893163b90)
)

### Results Page
![Results Page](![Screenshot (247)](https://github.com/user-attachments/assets/0d1f2561-d867-45bd-a23f-05e4b7f4cb01)
)

Replace `path/to/your-screenshot` with the actual paths to the screenshots in your repository.

## Contributing

Contributions are welcome! Please create a pull request with your changes.

## Author

Developed by [Pavithra](https://github.com/Pavithrabalaji-DA).  
Find more of my work on my GitHub profile: [https://github.com/Pavithrabalaji-DA](https://github.com/Pavithrabalaji-DA).
