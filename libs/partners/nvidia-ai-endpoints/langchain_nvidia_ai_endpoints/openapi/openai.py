
from __future__ import annotations

import logging
import os
import sys
from typing import (
    Any,
    Dict,
    List,
    Optional,
    Tuple,
)

import openai
from openai._exceptions import OpenAIError
import tiktoken
from langchain_core.messages import BaseMessage

from langchain_core.pydantic_v1 import BaseModel, Field, root_validator
from langchain_core.utils import get_from_dict_or_env

from langchain_nvidia_ai_endpoints.openapi.openapi import OpenAPI
from langchain_nvidia_ai_endpoints.openapi._openapi_client import (
    SyncClientMixin,
    AsyncClientMixin,
    SyncHttpxClientMixin,
    AsyncHttpxClientMixin,
)

logger = logging.getLogger(__name__)


class OpenAIMixin:
    @classmethod
    def pull_env_dict(cls, **kwargs: dict):
        """
        This automatically infers the following arguments from their corresponding environment variables if they are not provided:
        - `api_key` from `OPENAI_API_KEY`
        - `organization` from `OPENAI_ORG_ID`
        """
        kws = {k.replace("openai_", ""): v for k,v in kwargs.items()}
        kws = {k.replace("nvidia_", ""): v for k,v in kwargs.items()}
        kws["api_key"] = (
            kws.get("api_key")
            or os.environ.get("OPENAI_API_KEY")
        )

        if not kws.get("api_key"):
            raise OpenAIError(
                "The api_key client option must be set either by passing api_key"
                " to the client or by setting the OPENAI_API_KEY environment variable"
            )
        
        kws["organization"] = (
            kws.get("organization")
            or os.environ.get("OPENAI_ORG_ID")
            or os.environ.get("OPENAI_ORGANIZATION")
        )
        
        if "api_base" in kws:
            kws["base_url"] = kws.pop("api_base")
        
        kws["base_url"] = (
            kws.get("base_url")
            or os.environ.get("OPENAI_BASE_URL")
            or os.environ.get("OPENAI_API_BASE")
            or "https://api.openai.com/v1"
        )

        kws["proxy"] = (
            kws.get("proxy")
            or os.environ.get("OPENAI_PROXY")
        )

        return kws


class SyncOpenAI(OpenAIMixin, SyncClientMixin, SyncHttpxClientMixin):
    pass


class AsyncOpenAI(OpenAIMixin, AsyncClientMixin, AsyncHttpxClientMixin):
    pass


class OpenAI(OpenAIMixin, OpenAPI):
    
    api_key: Optional[str] = Field(default=None, alias="openai_api_key")
    """Automatically inferred from env var `OPENAI_API_KEY` if not provided."""
    base_url: Optional[str] = Field(default=None, alias="openai_api_base")
    """Base URL path for API requests, leave blank if not using a proxy or service 
        emulator."""
    organization: Optional[str] = Field(default=None, alias="openai_organization")
    """Automatically inferred from env var `OPENAI_ORG_ID` if not provided."""
    # to support explicit proxy for OpenAI
    proxy: Optional[str] = Field(default=None, alias="openai_organization")

    @property
    def _llm_type(self) -> str:
        """Return type of chat model."""
        return "openai"

    @property
    def lc_secrets(self) -> Dict[str, str]:
        return {"api_key": "OPENAI_API_KEY"}

    @classmethod
    def get_lc_namespace(cls) -> List[str]:
        """Get the namespace of the langchain object."""
        return ["langchain", "llms", "openai"]

    @classmethod
    def get_client_classes(cls):
        return SyncOpenAI, AsyncOpenAI

    @staticmethod
    def modelname_to_contextsize(modelname: str) -> int:
        """Calculate the maximum number of tokens possible to generate for a model.

        Args:
            modelname: The modelname we want to know the context size for.

        Returns:
            The maximum context size

        Example:
            .. code-block:: python

                max_tokens = openai.modelname_to_contextsize("gpt-3.5-turbo-instruct")
        """
        model_token_mapping = {
            "gpt-4": 8192,
            "gpt-4-0314": 8192,
            "gpt-4-0613": 8192,
            "gpt-4-32k": 32768,
            "gpt-4-32k-0314": 32768,
            "gpt-4-32k-0613": 32768,
            "gpt-3.5-turbo": 4096,
            "gpt-3.5-turbo-0301": 4096,
            "gpt-3.5-turbo-0613": 4096,
            "gpt-3.5-turbo-16k": 16385,
            "gpt-3.5-turbo-16k-0613": 16385,
            "gpt-3.5-turbo-instruct": 4096,
            "text-ada-001": 2049,
            "ada": 2049,
            "text-babbage-001": 2040,
            "babbage": 2049,
            "text-curie-001": 2049,
            "curie": 2049,
            "davinci": 2049,
            "text-davinci-003": 4097,
            "text-davinci-002": 4097,
            "code-davinci-002": 8001,
            "code-davinci-001": 8001,
            "code-cushman-002": 2048,
            "code-cushman-001": 2048,
        }

        # handling finetuned models
        if "ft-" in modelname:
            modelname = modelname.split(":")[0]

        context_size = model_token_mapping.get(modelname, None)

        if context_size is None:
            raise ValueError(
                f"Unknown model: {modelname}. Please provide a valid OpenAI model name."
                "Known models are: " + ", ".join(model_token_mapping.keys())
            )

        return context_size