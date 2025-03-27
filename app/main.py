import gradio as gr
from app.components.ui_elements import create_header, create_footer
from app.utils.helpers import preprocess_image, postprocess_result

def create_interface():
    """
    Create and return the Gradio interface for the handwriting recognition app
    with drawing canvas and additional user fields
    """
    with gr.Blocks(title="Handwriting Recognition App") as demo:
        # Header
        create_header(
            "Handwriting Recognition App", 
            "Write on the canvas and fill in your information to analyze your handwriting"
        )
        
        with gr.Row():
            # Left column for drawing and results
            with gr.Column(scale=2):
                # For newer Gradio versions, use gr.Paint() instead of gr.Image with tool parameter
                try:
                    # Try to use Paint component (available in newer versions)
                    canvas_input = gr.Paint(
                        label="Write here",
                        height=400,
                        width=600
                    )
                except (AttributeError, TypeError):
                    # Fallback to basic Image component with minimal parameters
                    canvas_input = gr.Image(
                        label="Write here",
                        height=400,
                        width=600
                    )
                
                # Display recognized text
                text_output = gr.Textbox(label="Recognized Text", lines=3)
                
                # Buttons
                with gr.Row():
                    clear_btn = gr.Button("Clear Canvas")
                    submit_btn = gr.Button("Recognize Text")
            
            # Right column for user information
            with gr.Column(scale=1):
                name_input = gr.Textbox(label="Name")
                surname_input = gr.Textbox(label="Surname")
                age_input = gr.Number(label="Age")
                
                work_type_input = gr.Dropdown(
                    label="Type of Work",
                    choices=["Intellectual", "Manual", "Mixed", "Other"]
                )
                
                degree_input = gr.Dropdown(
                    label="Degree of Instruction",
                    choices=[
                        "Elementary School",
                        "Middle School",
                        "High School",
                        "Bachelor's Degree",
                        "Master's Degree",
                        "Doctorate",
                        "Other"
                    ]
                )
                
                sex_input = gr.Dropdown(
                    label="Sex",
                    choices=["Male", "Female", "Other", "Prefer not to say"]
                )
                
                analysis_output = gr.Textbox(
                    label="Handwriting Analysis", 
                    lines=8
                )
        
        # Footer
        create_footer()
        
        # Define functions
        def recognize_handwriting(image):
            """Process the handwritten image and extract text"""
            if image is None:
                return "Please draw something on the canvas."
            
            # Preprocess the image
            processed_image = preprocess_image(image)
            
            # TODO: Implement actual handwriting recognition model
            # For now, return a placeholder
            result = "Sample recognized text. Replace with actual model output."
            
            # Postprocess the result
            processed_result = postprocess_result(result)
            
            return processed_result
        
        def analyze_handwriting(image, name, surname, age, work_type, degree, sex):
            """Analyze handwriting based on user input and drawn text"""
            text = recognize_handwriting(image)
            
            # Create a user profile string
            user_profile = f"Analysis for {name} {surname}\n"
            
            if age:
                user_profile += f"Age: {age}\n"
            
            user_profile += f"Work Type: {work_type}\n"
            user_profile += f"Education: {degree}\n"
            user_profile += f"Sex: {sex}\n\n"
            
            # TODO: Implement actual handwriting analysis
            # This is a placeholder for the handwriting analysis
            analysis = "Handwriting analysis will appear here once implemented."
            
            return text, user_profile + analysis
        
        def clear_canvas():
            """Clear the drawing canvas"""
            return None
        
        # Set up event handlers
        clear_btn.click(fn=clear_canvas, inputs=None, outputs=canvas_input)
        
        submit_btn.click(
            fn=analyze_handwriting,
            inputs=[
                canvas_input,
                name_input,
                surname_input,
                age_input,
                work_type_input,
                degree_input,
                sex_input
            ],
            outputs=[text_output, analysis_output]
        )
        
    return demo