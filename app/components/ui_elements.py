import gradio as gr

def create_header(title, description=None):
    """Create a header component for the app"""
    return gr.Markdown(f"# {title}\n{description if description else ''}")

def create_footer():
    """Create a footer component for the app"""
    return gr.Markdown("Developed with Gradio")
