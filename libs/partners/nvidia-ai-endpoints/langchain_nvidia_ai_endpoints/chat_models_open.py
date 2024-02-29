
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

from langchain_nvidia_ai_endpoints.chat_openapi import ChatOpenAPI, _convert_message_to_dict

logger = logging.getLogger(__name__)


class ChatOpenNVIDIA(ChatOpenAPI):
    
    nvidia_api_key: Optional[str] = Field(default=None, alias="api_key")
    """Automatically inferred from env var `OPENAI_API_KEY` if not provided."""
    nvidia_api_base: Optional[str] = Field(default=None, alias="base_url")
    """Base URL path for API requests, leave blank if not using a proxy or service 
        emulator."""
    nvidia_organization: Optional[str] = Field(default=None, alias="organization")
    """Automatically inferred from env var `OPENAI_ORG_ID` if not provided."""
    # to support explicit proxy for OpenAI
    nvidia_proxy: Optional[str] = None

    @property
    def _llm_type(self) -> str:
        """Return type of chat model."""
        return "nvidia-chat"

    @property
    def lc_secrets(self) -> Dict[str, str]:
        return {"nvidia_api_key": "NVIDIA_API_KEY"}

    @classmethod
    def get_lc_namespace(cls) -> List[str]:
        """Get the namespace of the langchain object."""
        return ["langchain", "chat_models", "nvidia"]

    @property
    def lc_attributes(self) -> Dict[str, Any]:
        attributes: Dict[str, Any] = {}

        if self.openai_organization:
            attributes["nvidia_organization"] = self.nvidia_organization

        if self.openai_api_base:
            attributes["nvidia_api_base"] = self.nvidia_api_base

        if self.openai_proxy:
            attributes["nvidia_proxy"] = self.nvidia_proxy

        return attributes
    
    @root_validator()
    def validate_environment(cls, values: Dict) -> Dict:
        """Validate that api key and python package exists in environment."""
        if values["n"] < 1:
            raise ValueError("n must be at least 1.")
        if values["n"] > 1 and values["streaming"]:
            raise ValueError("n must be 1 when streaming.")

        values["nvidia_api_key"] = get_from_dict_or_env(
            values, "nvidia_api_key", "NVIDIA_API_KEY"
        )
        # Check OPENAI_ORGANIZATION for backwards compatibility.
        values["nvidia_organization"] = (
            values["nvidia_organization"]
            or os.getenv("NVIDIA_ORG_ID")
            or os.getenv("NVIDIA_ORGANIZATION")
        )
        values["nvidia_api_base"] = values["nvidia_api_base"] or os.getenv(
            "NVIDIA_API_BASE"
        )
        values["nvidia_proxy"] = get_from_dict_or_env(
            values,
            "nvidia_proxy",
            "NVIDIA_PROXY",
            default="",
        )

        client_params = {
            "api_key": values["nvidia_api_key"],
            "organization": values["nvidia_organization"],
            "base_url": values["nvidia_api_base"],
            "timeout": values["request_timeout"],
            "max_retries": values["max_retries"],
            "default_headers": values["default_headers"],
            "default_query": values["default_query"],
            "http_client": values["http_client"],
        }

        if not values.get("client"):
            values["client"] = openai.OpenAI(**client_params).chat.completions
        if not values.get("async_client"):
            values["async_client"] = openai.AsyncOpenAI(**client_params).chat.completions
        return values

    def _get_encoding_model(self) -> Tuple[str, tiktoken.Encoding]:
        if self.tiktoken_model_name is not None:
            model = self.tiktoken_model_name
        else:
            model = self.model_name
            if model == "gpt-3.5-turbo":
                # gpt-3.5-turbo may change over time.
                # Returning num tokens assuming gpt-3.5-turbo-0301.
                model = "gpt-3.5-turbo-0301"
            elif model == "gpt-4":
                # gpt-4 may change over time.
                # Returning num tokens assuming gpt-4-0314.
                model = "gpt-4-0314"
        # Returns the number of tokens used by a list of messages.
        try:
            encoding = tiktoken.encoding_for_model(model)
        except KeyError:
            logger.warning("Warning: model not found. Using cl100k_base encoding.")
            model = "cl100k_base"
            encoding = tiktoken.get_encoding(model)
        return model, encoding

    def get_num_tokens_from_messages(self, messages: List[BaseMessage]) -> int:
        """Calculate num tokens for gpt-3.5-turbo and gpt-4 with tiktoken package.

        Official documentation: https://github.com/openai/openai-cookbook/blob/
        main/examples/How_to_format_inputs_to_ChatGPT_models.ipynb"""
        model, encoding = self._get_encoding_model()
        if sys.version_info[1] <= 7:
            return super().get_num_tokens_from_messages(messages)
        # if model.startswith("gpt-3.5-turbo-0301"):
        #     # every message follows <im_start>{role/name}\n{content}<im_end>\n
        #     tokens_per_message = 4
        #     # if there's a name, the role is omitted
        #     tokens_per_name = -1
        # elif model.startswith("gpt-3.5-turbo") or model.startswith("gpt-4"):
        #     tokens_per_message = 3
        #     tokens_per_name = 1
        else:
            raise NotImplementedError(
                f"get_num_tokens_from_messages() is not presently implemented for model {model}."
                " See https://platform.openai.com/docs/guides/text-generation/managing-tokens"
                " for information on how messages are converted to tokens."
            )
        num_tokens = 0
        messages_dict = [_convert_message_to_dict(m) for m in messages]
        for message in messages_dict:
            num_tokens += tokens_per_message
            for key, value in message.items():
                # Cast str(value) in case the message value is not a string
                # This occurs with function messages
                num_tokens += len(encoding.encode(str(value)))
                if key == "name":
                    num_tokens += tokens_per_name
        # every reply is primed with <im_start>assistant
        num_tokens += 3
        return num_tokens