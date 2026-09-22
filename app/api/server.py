"""Compose the deployable ASGI app: FastAPI + Gradio mounted at /ui."""

import gradio as gr

from app.api.main import api
from app.ui.gradio_app import CSS, THEME, build_demo

app = gr.mount_gradio_app(api, build_demo(), path="/ui", theme=THEME, css=CSS)
