import base64
from pathlib import Path

import vertexai
from vertexai.generative_models import GenerativeModel, Part


class VertexAIClient:
    """Client pour Vertex AI Gemini Vision."""

    def __init__(
        self,
        project_id: str,
        location: str = "us-central1",
        model_name: str = "gemini-2.0-flash-001",
    ):
        self.project_id = project_id
        self.location = location
        self.model_name = model_name

        vertexai.init(project=project_id, location=location)
        self.model = GenerativeModel(model_name)

    def extract_from_images(
        self,
        images: list[bytes],
        prompt: str,
    ) -> str:
        """Extrait des informations depuis une liste d'images."""
        parts = []

        for img_bytes in images:
            parts.append(Part.from_data(img_bytes, mime_type="image/png"))

        parts.append(prompt)

        response = self.model.generate_content(parts)
        return response.text
