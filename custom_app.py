import gradio as gr
import ollama_gradio

gr.load(
    name='gpt-4-turbo',
    src=ollama_gradio.registry,
    title='Ollama-Gradio Integration',
    description="Chat with llama3.1-8b model.",
    examples=["Explain quantum gravity to a 5-year old.", "How many R are there in the word Strawberry?"]
).launch()