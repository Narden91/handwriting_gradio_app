import gradio as gr

def create_header(title, description=None):
    """Create a header component for the app"""
    header_md = f"# {title}\n"
    if description:
        header_md += f"{description}\n"
    header_md += "---"
    return gr.Markdown(header_md)

def create_footer():
    """Create a footer component for the app"""
    footer_md = """
    ---
    ### About This App
    
    This application analyzes handwriting based on samples provided through the drawing canvas 
    and additional user information. Please note that the actual handwriting recognition and 
    analysis functionality is currently a placeholder.
    
    Developed with [Gradio](https://gradio.app/) | © 2025
    """
    return gr.Markdown(footer_md)

def create_info_box(message, type="info"):
    """Create an info box with custom styling"""
    color_map = {
        "info": "blue",
        "warning": "orange",
        "error": "red",
        "success": "green"
    }
    color = color_map.get(type, "blue")
    
    html = f"""
    <div style="padding: 10px; border-radius: 8px; background-color: {color}15; border: 1px solid {color}50;">
        <p style="margin: 0; color: {color}E0;">{message}</p>
    </div>
    """
    return gr.HTML(html)