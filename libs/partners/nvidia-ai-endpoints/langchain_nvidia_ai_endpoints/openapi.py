from __future__ import annotations

import os
from typing import Any, Callable, Union, Mapping
from typing_extensions import Self, override

import httpx

from openai import OpenAI as SyncOpenAI, AsyncOpenAI
from openai import resources
from openai._types import NOT_GIVEN, Omit, Timeout

from openai._base_client import (
    SyncAPIClient,
    AsyncAPIClient,
    DEFAULT_MAX_RETRIES,
)

from langchain_core.pydantic_v1 import Field, root_validator, SecretStr
from langchain_openai import OpenAI, ChatOpenAI, OpenAIEmbeddings, OpenAIEmbeddings
from langchain_core.utils import convert_to_secret_str

class ClientMixin:
    def __init__(
        self,
        *,
        api_key: str | None = None,
        organization: str | None = None,
        base_url: str | httpx.URL | None = None,
        timeout: Union[float, Timeout, None, NotGiven] = NOT_GIVEN,
        max_retries: int = DEFAULT_MAX_RETRIES,
        default_headers: Mapping[str, str] | None = None,
        default_query: Mapping[str, object] | None = None,
        http_client: httpx.Client | None = None,
        _strict_response_validation: bool = False,
        use_base_as_endpoint: bool = False,
        **kwargs: Any,
    ) -> None:
        """Construct a new synchronous openai client instance."""
        if api_key is None:
            api_key = os.environ.get("NVIDIA_API_KEY")
        if api_key is None:
            raise Exception(
                "The api_key client option must be set either by passing api_key"
                " to the client or by setting the NVIDIA_API_KEY environment variable"
            )
        self.api_key = api_key

        if base_url is None:
            base_url = os.environ.get("NVIDIA_BASE_URL")
        if base_url is None:
            base_url = f"https://api.nvidia.com/v1"
        
        self.use_base_as_endpoint = use_base_as_endpoint

        super().__init__(
            api_key = api_key,
            organization = organization,
            base_url = base_url,
            timeout = timeout,
            max_retries = max_retries,
            default_headers = default_headers,
            default_query = default_query,
            http_client = http_client,
            _strict_response_validation = _strict_response_validation,
        )

    def default_headers(self) -> dict[str, str | Omit]:
        return {**super().default_headers, **self._custom_headers}

    def _prepare_url(self, url: str) -> httpx.URL:
        """
        Merge a URL argument together with any 'base_url' on the client,
        to create the URL used for the outgoing request.
        """
        merge_url = httpx.URL(url)
        if self.use_base_as_endpoint:
            merge_url = httpx.URL(str(self.base_url).rstrip("/"))
        elif merge_url.is_relative_url:
            merge_raw_path = self.base_url.raw_path + merge_url.raw_path.lstrip(b"/")
            merge_url = self.base_url.copy_with(raw_path=merge_raw_path)
        return merge_url


class SyncNVIDIA(ClientMixin, SyncOpenAI):
    pass


class AsyncNVIDIA(ClientMixin, AsyncOpenAI):
    pass


class NVIDIAMixin:
    
    partner_name = "NVIDIA"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **self.__class__.get_from_env(kwargs))

    @property
    def nvidia_api_key(self) -> SecretStr:
        return self.api_key

    @property
    def nvidia_organization(self) -> str:
        return self.organization

    @property
    def nvidia_api_base(self) -> str:
        return self.base_url
    
    @property
    def openai_api_key(self) -> SecretStr:
        return self.api_key

    @property
    def openai_organization(self) -> str:
        return self.organization

    @property
    def openai_api_base(self) -> str:
        return self.base_url

    @classmethod
    def get_from_env(cls, values: dict) -> dict:
        env_key = os.getenv(f"{cls.partner_name}_API_KEY")
        env_url1 = os.getenv(f"{cls.partner_name}_API_BASE")
        env_url2 = os.getenv(f"{cls.partner_name}_BASE_URL")
        env_org1 = os.getenv(f"{cls.partner_name}_ORG_ID")
        env_org2 = os.getenv(f"{cls.partner_name}_ORGANIZATION")
        env_prox = os.getenv(f"{cls.partner_name}_PROXY")

        def get_partner_keys(values, bases, prefixes=["nvidia_", "", "openai_"]) -> None:
            for base in bases: 
                for pref in prefixes:
                    v = values.pop(f"{pref}{base}", None)
                    if v: 
                        return v

        values["api_key"] = get_partner_keys(values, ["api_key"]) or env_key
        values["base_url"] = get_partner_keys(values, ["api_base", "base_url"]) or env_url1 or env_url2
        values["base_url"] = values["base_url"] or "https://integrate.api.nvidia.com/v1"
        values["organization"] = get_partner_keys(values, ["organization", "org_id"]) or env_org1 or env_org2
        return values
    
    @staticmethod
    def _get_resource(client: httpx._client.BaseClient):
        return client.completion
    
    @classmethod
    def _set_client(cls, values: dict) -> dict:

        client_params = {
            "api_key": values["api_key"],
            "organization": values["organization"],
            "base_url": values["base_url"],
            "timeout": values["request_timeout"],
            "max_retries": values["max_retries"],
            "default_headers": values["default_headers"],
            "default_query": values["default_query"],
            "http_client": values["http_client"],
        }

        if not values.get("client"):
            values["client"] = cls._get_resource(SyncNVIDIA(**client_params))
        if not values.get("async_client"):
            values["async_client"] = cls._get_resource(AsyncNVIDIA(**client_params))

        return values
    
    @property
    def lc_secrets(self) -> Dict[str, str]:
        return {"api_key": "NVIDIA_API_KEY"}

    @property
    def lc_attributes(self) -> Dict[str, Any]:
        attributes: Dict[str, Any] = {}
        if self.base_url:
            attributes["base_url"] = self.base_url

        if self.organization:
            attributes["organization"] = self.organization

        return attributes


class OpenNVIDIA(NVIDIAMixin, OpenAI):

    @staticmethod
    def _get_resource(client: httpx._client.BaseClient):
        return client.completions

    @root_validator()
    def validate_environment(cls, values: Dict) -> Dict:
        """Validate that api key and python package exists in environment."""
        if values["n"] < 1:
            raise ValueError("n must be at least 1.")
        if values["streaming"] and values["n"] > 1:
            raise ValueError("Cannot stream results when n > 1.")
        if values["streaming"] and values["best_of"] > 1:
            raise ValueError("Cannot stream results when best_of > 1.")

        values = cls.get_from_env(values)
        values = cls._set_client(values)
        values['api_key'] = convert_to_secret_str(values['api_key'])
        return values


class ChatOpenNVIDIA(NVIDIAMixin, ChatOpenAI):
    
    @staticmethod
    def _get_resource(client: httpx._client.BaseClient):
        return client.chat.completions

    @root_validator()
    def validate_environment(cls, values: Dict) -> Dict:
        """Validate that api key and python package exists in environment."""
        if values["n"] < 1:
            raise ValueError("n must be at least 1.")
        if values["streaming"] and values["n"] > 1:
            raise ValueError("Cannot stream results when n > 1.")

        values = cls.get_from_env(values)
        values = cls._set_client(values)
        values['api_key'] = convert_to_secret_str(values['api_key'])
        return values


class OpenNVIDIAEmbeddings(NVIDIAMixin, OpenAIEmbeddings):

    @staticmethod
    def _get_resource(client: httpx._client.BaseClient):
        return client.embeddings

    @property
    def _invocation_params(self) -> Dict[str, Any]:
        if is_openai_v1():
            default_args: Dict = {"model": self.model, **self.model_kwargs}
        else:
            default_args = {
                "model": self.model,
                "request_timeout": self.request_timeout,
                "headers": self.headers,
                "api_key": self.api_key,
                "organization": self.organization,
                "api_base": self.api_base,
                "api_type": self.api_type,
                "api_version": self.api_version,
                **self.model_kwargs,
            }
        return default_args

    @root_validator()
    def validate_environment(cls, values: Dict) -> Dict:
        """Validate that api key and python package exists in environment."""
        values = cls.get_from_env(values)
        values = cls._set_client(values)
        values['api_key'] = convert_to_secret_str(values['api_key'])
        return values


class output_parsers:

    ## TODO: Should this be in a try-catch?
    from langchain.output_parsers.openai_tools import (
        JsonOutputKeyToolsParser,
        JsonOutputToolsParser,
        PydanticToolsParser,
    )

    __all__ = ["PydanticToolsParser", "JsonOutputToolsParser", "JsonOutputKeyToolsParser"]
