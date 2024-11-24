import gradio as gr
import ollama_gradio

gr.load(
    name='llama3.1:8b',
    src=ollama_gradio.registry,
).launch()