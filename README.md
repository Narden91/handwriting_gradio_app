# Handwriting Recognition Gradio App

A Gradio web application for recognizing and extracting text from handwritten images.

## Features

- Upload images containing handwritten text
- Extract and digitize text from handwritten content
- Simple and intuitive user interface

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/handwriting_gradio_app.git
cd handwriting_gradio_app

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Usage

```bash
python app.py
```

The application will be available at http://127.0.0.1:7860

## Project Structure

```
handwriting_gradio_app/
├── app/
│   ├── components/
│   ├── utils/
│   └── main.py
├── assets/
├── models/
├── data/
├── tests/
├── app.py
└── requirements.txt
```

## License

[MIT](LICENSE)
