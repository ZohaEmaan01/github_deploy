import gradio as gr
import pickle
import numpy as np

# Load the model
with open('iris_model.pkl', 'rb') as file:
    model = pickle.load(file)

# Define the prediction function
def predict(sepal_length, sepal_width, petal_length, petal_width):
    input_data = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
    prediction = model.predict(input_data)
    return prediction[0]

# Create the Gradio interface
iface = gr.Interface(
    fn=predict,
    inputs=[
        gr.inputs.Number(label="Sepal Length"),
        gr.inputs.Number(label="Sepal Width"),
        gr.inputs.Number(label="Petal Length"),
        gr.inputs.Number(label="Petal Width")
    ],
    outputs="text"
)

# Launch the interface
iface.launch()
