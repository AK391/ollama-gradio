import os
import gradio as gr
from typing import Callable
import base64
import ollama
import httpx
from ollama import AsyncClient

__version__ = "0.0.3"


async def get_fn(model_name: str, preprocess: Callable, postprocess: Callable):
    async def fn(message, history):
        try:
            inputs = preprocess(message, history)
            client = AsyncClient()
            
            async for chunk in await client.chat(
                model=model_name,
                messages=inputs["messages"],
                stream=True,
            ):
                if 'message' in chunk and 'content' in chunk['message']:
                    delta = chunk['message']['content']
                    yield postprocess(delta)
                    
        except httpx.ConnectError:
            error_msg = (
                "Could not connect to Ollama server. "
                "Please make sure Ollama is running and accessible at http://localhost:11434. "
                "You can start it by running 'ollama serve' in your terminal."
            )
            yield postprocess(error_msg)
        except Exception as e:
            error_msg = f"An error occurred: {str(e)}"
            yield postprocess(error_msg)

    return fn


def get_image_base64(url: str, ext: str):
    with open(url, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read()).decode('utf-8')
    return "data:image/" + ext + ";base64," + encoded_string


def handle_user_msg(message: str):
    if type(message) is str:
        return message
    elif type(message) is dict:
        if message["files"] is not None and len(message["files"]) > 0:
            ext = os.path.splitext(message["files"][-1])[1].strip(".")
            if ext.lower() in ["png", "jpg", "jpeg", "gif", "pdf"]:
                encoded_str = get_image_base64(message["files"][-1], ext)
            else:
                raise NotImplementedError(f"Not supported file type {ext}")
            content = [
                    {"type": "text", "text": message["text"]},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": encoded_str,
                        }
                    },
                ]
        else:
            content = message["text"]
        return content
    else:
        raise NotImplementedError


def get_interface_args(pipeline):
    if pipeline == "chat":
        inputs = None
        outputs = None

        def preprocess(message, history):
            messages = []
            files = None
            for user_msg, assistant_msg in history:
                if assistant_msg is not None:
                    messages.append({"role": "user", "content": handle_user_msg(user_msg)})
                    messages.append({"role": "assistant", "content": assistant_msg})
                else:
                    files = user_msg
            if type(message) is str and files is not None:
                message = {"text":message, "files":files}
            elif type(message) is dict and files is not None:
                if message["files"] is None or len(message["files"]) == 0:
                    message["files"] = files
            messages.append({"role": "user", "content": handle_user_msg(message)})
            return {"messages": messages}

        postprocess = lambda x: x
    else:
        # Add other pipeline types when they will be needed
        raise ValueError(f"Unsupported pipeline type: {pipeline}")
    return inputs, outputs, preprocess, postprocess


def get_pipeline(model_name):
    # Determine the pipeline type based on the model name
    # For simplicity, assuming all models are chat models at the moment
    return "chat"


def registry(name: str, token: str | None = None, **kwargs):
    """
    Create a Gradio Interface for an Ollama model.
    This function matches the signature expected by gr.load().

    Parameters:
        - name (str): The name of the Ollama model to run locally
        - token (str | None): Unused parameter, required for gr.load() compatibility 
                            since Ollama runs models locally
        - **kwargs: Additional keyword arguments passed to the interface
    """
    # If name includes the src prefix (e.g., "ollama/model-name"), strip it
    if "/" in name:
        _, name = name.split("/", 1)

    pipeline = get_pipeline(name)
    inputs, outputs, preprocess, postprocess = get_interface_args(pipeline)
    fn = await get_fn(name, preprocess, postprocess)

    if pipeline == "chat":
        interface = gr.ChatInterface(fn=fn, multimodal=True, **kwargs)
    else:
        interface = gr.Interface(fn=fn, inputs=inputs, outputs=outputs, **kwargs)

    return interface
