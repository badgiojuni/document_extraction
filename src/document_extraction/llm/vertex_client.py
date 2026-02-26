"""Client Vertex AI pour l'extraction structurée."""

import json
import logging
import os
from abc import ABC, abstractmethod
from typing import Any, Optional

logger = logging.getLogger(__name__)


class LLMClientError(Exception):
    """Erreur du client LLM."""

    pass


class BaseLLMClient(ABC):
    """Interface de base pour les clients LLM."""

    @abstractmethod
    def generate(self, prompt: str, **kwargs) -> str:
        """
        Génère une réponse à partir d'un prompt.

        Args:
            prompt: Le prompt à envoyer au modèle
            **kwargs: Arguments additionnels

        Returns:
            La réponse générée
        """
        pass

    @abstractmethod
    def generate_json(self, prompt: str, **kwargs) -> dict[str, Any]:
        """
        Génère une réponse JSON structurée.

        Args:
            prompt: Le prompt à envoyer au modèle
            **kwargs: Arguments additionnels

        Returns:
            La réponse parsée en dictionnaire
        """
        pass

    @abstractmethod
    def is_available(self) -> bool:
        """Vérifie si le client est disponible."""
        pass


class VertexAIClient(BaseLLMClient):
    """Client pour Vertex AI (Google Cloud)."""

    def __init__(
        self,
        project_id: str,
        location: str = "europe-west1",
        model_name: str = "gemini-1.5-flash",
        credentials_path: Optional[str] = None,
    ):
        """
        Initialise le client Vertex AI.

        Args:
            project_id: ID du projet Google Cloud
            location: Région GCP (europe-west1, us-central1, etc.)
            model_name: Nom du modèle (gemini-1.5-flash, gemini-1.5-pro)
            credentials_path: Chemin vers le fichier de credentials JSON
        """
        self.project_id = project_id
        self.location = location
        self.model_name = model_name
        self._model = None
        self._initialized = False

        # Configuration des credentials
        if credentials_path and os.path.exists(credentials_path):
            os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = credentials_path

        logger.info(
            f"VertexAIClient configuré (project={project_id}, "
            f"location={location}, model={model_name})"
        )

    def _initialize(self) -> None:
        """Initialise le client Vertex AI (lazy loading)."""
        if self._initialized:
            return

        try:
            import vertexai
            from vertexai.generative_models import GenerativeModel

            vertexai.init(project=self.project_id, location=self.location)
            self._model = GenerativeModel(self.model_name)
            self._initialized = True
            logger.info("Vertex AI initialisé avec succès")

        except ImportError:
            raise LLMClientError(
                "google-cloud-aiplatform n'est pas installé. "
                "Installez-le avec: pip install google-cloud-aiplatform"
            )
        except Exception as e:
            raise LLMClientError(f"Erreur d'initialisation Vertex AI: {e}")

    def is_available(self) -> bool:
        """Vérifie si Vertex AI est disponible."""
        try:
            self._initialize()
            return True
        except LLMClientError:
            return False

    def generate(
        self,
        prompt: str,
        temperature: float = 0.1,
        max_tokens: int = 4096,
        **kwargs,
    ) -> str:
        """
        Génère une réponse avec Vertex AI.

        Args:
            prompt: Le prompt à envoyer
            temperature: Température de génération (0-1)
            max_tokens: Nombre maximum de tokens en sortie

        Returns:
            La réponse générée
        """
        self._initialize()

        try:
            from vertexai.generative_models import GenerationConfig

            config = GenerationConfig(
                temperature=temperature,
                max_output_tokens=max_tokens,
            )

            response = self._model.generate_content(
                prompt,
                generation_config=config,
            )

            return response.text

        except Exception as e:
            raise LLMClientError(f"Erreur lors de la génération: {e}")

    def generate_json(
        self,
        prompt: str,
        temperature: float = 0.0,
        **kwargs,
    ) -> dict[str, Any]:
        """
        Génère une réponse JSON structurée.

        Args:
            prompt: Le prompt demandant une sortie JSON
            temperature: Température (0 recommandé pour JSON)

        Returns:
            La réponse parsée en dictionnaire
        """
        response_text = self.generate(prompt, temperature=temperature, **kwargs)

        # Nettoyage de la réponse
        response_text = response_text.strip()

        # Supprimer les marqueurs de code markdown si présents
        if response_text.startswith("```json"):
            response_text = response_text[7:]
        elif response_text.startswith("```"):
            response_text = response_text[3:]

        if response_text.endswith("```"):
            response_text = response_text[:-3]

        response_text = response_text.strip()

        try:
            return json.loads(response_text)
        except json.JSONDecodeError as e:
            logger.error(f"Erreur de parsing JSON: {e}")
            logger.debug(f"Réponse brute: {response_text[:500]}")
            raise LLMClientError(f"La réponse n'est pas un JSON valide: {e}")
