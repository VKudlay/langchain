
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
import tiktoken
from langchain_core.messages import BaseMessage

from langchain_core.pydantic_v1 import Field, root_validator
from langchain_core.utils import get_from_dict_or_env

from langchain_nvidia_ai_endpoints.openapi.openapi import OpenAPI
from langchain_nvidia_ai_endpoints.openapi._openapi_client import (
    SyncClientMixin,
    AsyncClientMixin,
    SyncHttpxClientMixin,
    AsyncHttpxClientMixin,
)

logger = logging.getLogger(__name__)


class NVIDIAMixin:
    @classmethod
    def pull_env_dict(cls, **kwargs: dict):
        """
        This automatically infers the following arguments from their corresponding environment variables if they are not provided:
        - `api_key` from `OPENAI_API_KEY`
        - `organization` from `OPENAI_ORG_ID`
        """
        kws = {k.replace("nvidia_", ""): v for k,v in kwargs.items()}
        kws = {k.replace("openai_", ""): v for k,v in kwargs.items()}
        kws["api_key"] = (
            kws.get("api_key")
            or os.environ.get("NVIDIA_API_KEY")
        )

        if not kws.get("api_key"):
            raise Exception(
                "The api_key client option must be set either by passing api_key"
                " to the client or by setting the OPENAI_API_KEY environment variable"
            )
        
        if "api_base" in kws:
            kws["base_url"] = kws.pop("api_base")
        
        kws["base_url"] = (
            kws.get("base_url")
            or os.environ.get("NVIDIA_BASE_URL")
            or os.environ.get("NVIDIA_API_BASE")
            or "https://integrate.api.nvidia.com/v1"
        )

        kws["proxy"] = (
            kws.get("base_url")
            or os.environ.get("NVIDIA_PROXY")
        )

        return kws


class SyncNVIDIA(NVIDIAMixin, SyncClientMixin, SyncHttpxClientMixin):
    pass


class AsyncNVIDIA(NVIDIAMixin, AsyncClientMixin, AsyncHttpxClientMixin):
    pass


class OpenNVIDIA(NVIDIAMixin, OpenAPI):
    
    api_key: Optional[str] = Field(default=None, alias="nvidia_api_key")
    """Automatically inferred from env var `NVIDIA_API_KEY` if not provided."""
    base_url: Optional[str] = Field(default=None, alias="nvidia_api_base")
    """Base URL path for API requests, leave blank if not using a proxy or service 
        emulator."""
    organization: Optional[str] = Field(default=None, alias="nvidia_organization")
    """Automatically inferred from env var `OPENAI_ORG_ID` if not provided."""
    # to support explicit proxy for OpenAI
    proxy: Optional[str] = Field(default=None, alias="nvidia_proxy")

    @property
    def _llm_type(self) -> str:
        """Return type of chat model."""
        return "nvidia"

    @property
    def lc_secrets(self) -> Dict[str, str]:
        return {"api_key": "NVIDIA_API_KEY"}

    @classmethod
    def get_lc_namespace(cls) -> List[str]:
        """Get the namespace of the langchain object."""
        return ["langchain", "llms", "nvidia"]

    @classmethod
    def get_client_classes(cls):
        return SyncNVIDIA, AsyncNVIDIA


    @staticmethod
    def modelname_to_contextsize(modelname: str) -> int:
        """Calculate the maximum number of tokens possible to generate for a model.

        Args:
            modelname: The modelname we want to know the context size for.

        Returns:
            The maximum context size
        """
        return 4096
