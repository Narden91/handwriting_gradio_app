import gradio as gr
from app.main import create_interface

# Create the Gradio interface
demo = create_interface()

if __name__ == "__main__":
    # Launch the app locally when script is run directly
    demo.launch(server_name="0.0.0.0", server_port=7860)
else:
    # For Hugging Face Spaces deployment
    demo.launch(share=False)