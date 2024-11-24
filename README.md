# `ollama-gradio`

is a Python package that makes it very easy for developers to create machine learning apps that are powered by Ollama models.

# Installation

You can install `ollama-gradio` directly using pip:

```bash
pip install ollama-gradio
```

That's it! 

# Basic Usage

Make sure you have Ollama installed and running locally. Then in a Python file, write:

```python
import gradio as gr
import ollama_gradio

gr.load(
    name='llama2',  # or any other Ollama model
    src=ollama_gradio.registry,
).launch()
```

Run the Python file, and you should see a Gradio Interface connected to your local Ollama model!

![ChatInterface](chatinterface.png)

# Customization 

You can customize the Gradio UI by setting your own title, description, examples, or any other arguments supported by `gr.ChatInterface`. For example:

```python
import gradio as gr
import ollama_gradio

gr.load(
    name='llama2',
    src=ollama_gradio.registry,
    title='Ollama-Gradio Integration',
    description="Chat with local LLMs using Ollama",
    examples=["Explain quantum gravity to a 5-year old.", "How many R are there in the word Strawberry?"]
).launch()
```

# Composition

You can use multiple Ollama models within larger Gradio Web UIs, e.g.

```python
import gradio as gr
import ollama_gradio

with gr.Blocks() as demo:
    with gr.Tab("llama2"):
        gr.load('llama2', src=ollama_gradio.registry)
    with gr.Tab("mistral"):
        gr.load('mistral', src=ollama_gradio.registry)

demo.launch()
```

# Features

- Chat interface with streaming responses
- Support for multimodal inputs (text and images)
- Compatible with all Ollama models
- Easy integration with larger Gradio applications

# Supported Models

All models available through Ollama are compatible with this integration. You can use any model that you've pulled to your local Ollama installation.

To see available models or pull new ones, refer to the [Ollama documentation](https://github.com/ollama/ollama).