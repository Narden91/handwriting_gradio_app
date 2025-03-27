from setuptools import setup, find_packages

setup(
    name="handwriting_gradio_app",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "gradio>=4.0.0",
        "numpy>=1.24.0",
        "pandas>=2.0.0",
        "pillow>=9.0.0",
        "torch>=2.0.0",
        "transformers>=4.28.0",
    ],
    author="Emanuele Nardone",
    author_email="eminik2006@gmail.com",
    description="A Gradio app for handwriting recognition",
    keywords="gradio, handwriting, recognition, machine learning",
    python_requires=">=3.8",
)
