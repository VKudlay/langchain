
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

from langchain_nvidia_ai_endpoints.openapi.chat_openapi import ChatOpenAPI, _convert_message_to_dict
from langchain_nvidia_ai_endpoints.openapi._openapi_client import (
    SyncClientMixin,
    AsyncClientMixin,
    SyncHttpxClientMixin,
    AsyncHttpxClientMixin,
)

logger = logging.getLogger(__name__)

from langchain_nvidia_ai_endpoints.openapi.openai import OpenAIMixin, SyncOpenAI, AsyncOpenAI


class ChatOpenAI(OpenAIMixin, ChatOpenAPI):
    
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
        return "openai-chat"

    @property
    def lc_secrets(self) -> Dict[str, str]:
        return {"api_key": "OPENAI_API_KEY"}

    @classmethod
    def get_lc_namespace(cls) -> List[str]:
        """Get the namespace of the langchain object."""
        return ["langchain", "chat_models", "openai"]

    @classmethod
    def get_client_classes(cls):
        return SyncOpenAI, AsyncOpenAI

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
        if sys.version_info[1] <= 7:
            return super().get_num_tokens_from_messages(messages)
        model, encoding = self._get_encoding_model()
        if model.startswith("gpt-3.5-turbo-0301"):
            # every message follows <im_start>{role/name}\n{content}<im_end>\n
            tokens_per_message = 4
            # if there's a name, the role is omitted
            tokens_per_name = -1
        elif model.startswith("gpt-3.5-turbo") or model.startswith("gpt-4"):
            tokens_per_message = 3
            tokens_per_name = 1
        else:
            raise NotImplementedError(
                f"get_num_tokens_from_messages() is not presently implemented "
                f"for model {model}. See "
                "https://platform.openai.com/docs/guides/text-generation/managing-tokens"
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