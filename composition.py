import gradio as gr
import ollama_gradio

with gr.Blocks() as demo:
    with gr.Tab("LLaMA3.1-8B"):
        gr.load('llama3.1:8b', src=ollama_gradio.registry)
    with gr.Tab("GPT-3.5-turbo"):
        gr.load('gpt-3.5-turbo', src=ollama_gradio.registry)

demo.launch()