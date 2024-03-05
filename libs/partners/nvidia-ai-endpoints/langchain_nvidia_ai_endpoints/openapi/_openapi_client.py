# File generated from our OpenAPI spec by Stainless.

from __future__ import annotations

import os
from typing import Any, Union, Mapping
from typing_extensions import Self, override

import httpx

from openai import resources, _exceptions
from openai._qs import Querystring
from openai._types import (
    NOT_GIVEN,
    Omit,
    Timeout,
    NotGiven,
    Transport,
    ProxiesTypes,
    RequestOptions,
)
from openai._utils import (
    is_given,
    is_mapping,
    get_async_library,
)
from openai._version import __version__
from openai._streaming import Stream as Stream, AsyncStream as AsyncStream
from openai._exceptions import OpenAIError, APIStatusError
from openai._base_client import (
    DEFAULT_MAX_RETRIES,
    SyncAPIClient,
    AsyncAPIClient,
)


__all__ = [
    "Timeout",
    "Transport",
    "ProxiesTypes",
    "RequestOptions",
    "resources",
    "OpenAI",
    "AsyncOpenAI",
    "Client",
    "AsyncClient",
]


class OpenAIWithRawResponse:
    def __init__(self, client: httpx.Client) -> None:
        self.completions = resources.CompletionsWithRawResponse(client.completions)
        self.chat = resources.ChatWithRawResponse(client.chat)
        self.embeddings = resources.EmbeddingsWithRawResponse(client.embeddings)
        self.files = resources.FilesWithRawResponse(client.files)
        self.images = resources.ImagesWithRawResponse(client.images)
        self.audio = resources.AudioWithRawResponse(client.audio)
        self.moderations = resources.ModerationsWithRawResponse(client.moderations)
        self.models = resources.ModelsWithRawResponse(client.models)
        self.fine_tuning = resources.FineTuningWithRawResponse(client.fine_tuning)
        self.beta = resources.BetaWithRawResponse(client.beta)


class AsyncOpenAIWithRawResponse:
    def __init__(self, client: httpx.AsyncClient) -> None:
        self.completions = resources.AsyncCompletionsWithRawResponse(client.completions)
        self.chat = resources.AsyncChatWithRawResponse(client.chat)
        self.embeddings = resources.AsyncEmbeddingsWithRawResponse(client.embeddings)
        self.files = resources.AsyncFilesWithRawResponse(client.files)
        self.images = resources.AsyncImagesWithRawResponse(client.images)
        self.audio = resources.AsyncAudioWithRawResponse(client.audio)
        self.moderations = resources.AsyncModerationsWithRawResponse(client.moderations)
        self.models = resources.AsyncModelsWithRawResponse(client.models)
        self.fine_tuning = resources.AsyncFineTuningWithRawResponse(client.fine_tuning)
        self.beta = resources.AsyncBetaWithRawResponse(client.beta)


class OpenAIWithStreamedResponse:
    def __init__(self, client: httpx.Client) -> None:
        self.completions = resources.CompletionsWithStreamingResponse(client.completions)
        self.chat = resources.ChatWithStreamingResponse(client.chat)
        self.embeddings = resources.EmbeddingsWithStreamingResponse(client.embeddings)
        self.files = resources.FilesWithStreamingResponse(client.files)
        self.images = resources.ImagesWithStreamingResponse(client.images)
        self.audio = resources.AudioWithStreamingResponse(client.audio)
        self.moderations = resources.ModerationsWithStreamingResponse(client.moderations)
        self.models = resources.ModelsWithStreamingResponse(client.models)
        self.fine_tuning = resources.FineTuningWithStreamingResponse(client.fine_tuning)
        self.beta = resources.BetaWithStreamingResponse(client.beta)


class AsyncOpenAIWithStreamedResponse:
    def __init__(self, client: httpx.AsyncClient) -> None:
        self.completions = resources.AsyncCompletionsWithStreamingResponse(client.completions)
        self.chat = resources.AsyncChatWithStreamingResponse(client.chat)
        self.embeddings = resources.AsyncEmbeddingsWithStreamingResponse(client.embeddings)
        self.files = resources.AsyncFilesWithStreamingResponse(client.files)
        self.images = resources.AsyncImagesWithStreamingResponse(client.images)
        self.audio = resources.AsyncAudioWithStreamingResponse(client.audio)
        self.moderations = resources.AsyncModerationsWithStreamingResponse(client.moderations)
        self.models = resources.AsyncModelsWithStreamingResponse(client.models)
        self.fine_tuning = resources.AsyncFineTuningWithStreamingResponse(client.fine_tuning)
        self.beta = resources.AsyncBetaWithStreamingResponse(client.beta)


class SyncClientMixin:

    completions: resources.Completions
    chat: resources.Chat
    embeddings: resources.Embeddings
    files: resources.Files
    images: resources.Images
    audio: resources.Audio
    moderations: resources.Moderations
    models: resources.Models
    fine_tuning: resources.FineTuning
    beta: resources.Beta
    with_raw_response: OpenAIWithRawResponse
    with_streaming_response: OpenAIWithStreamedResponse

    def _set_resources(self):
        self._default_stream_cls = Stream
        self.completions = resources.Completions(self)
        self.chat = resources.Chat(self)
        self.embeddings = resources.Embeddings(self)
        self.files = resources.Files(self)
        self.images = resources.Images(self)
        self.audio = resources.Audio(self)
        self.moderations = resources.Moderations(self)
        self.models = resources.Models(self)
        self.fine_tuning = resources.FineTuning(self)
        self.beta = resources.Beta(self)
        self.with_raw_response = OpenAIWithRawResponse(self)
        self.with_streaming_response = OpenAIWithStreamedResponse(self)

class AsyncClientMixin:

    completions: resources.AsyncCompletions
    chat: resources.AsyncChat
    embeddings: resources.AsyncEmbeddings
    files: resources.AsyncFiles
    images: resources.AsyncImages
    audio: resources.AsyncAudio
    moderations: resources.AsyncModerations
    models: resources.AsyncModels
    fine_tuning: resources.AsyncFineTuning
    beta: resources.AsyncBeta
    with_raw_response: AsyncOpenAIWithRawResponse
    with_streaming_response: AsyncOpenAIWithStreamedResponse

    def _set_resources(self):
        self._default_stream_cls = AsyncStream
        self.completions = resources.AsyncCompletions(self)
        self.chat = resources.AsyncChat(self)
        self.embeddings = resources.AsyncEmbeddings(self)
        self.files = resources.AsyncFiles(self)
        self.images = resources.AsyncImages(self)
        self.audio = resources.AsyncAudio(self)
        self.moderations = resources.AsyncModerations(self)
        self.models = resources.AsyncModels(self)
        self.fine_tuning = resources.AsyncFineTuning(self)
        self.beta = resources.AsyncBeta(self)
        self.with_raw_response = AsyncOpenAIWithRawResponse(self)
        self.with_streaming_response = AsyncOpenAIWithStreamedResponse(self)


class DefaultEnvMixin:
    @classmethod
    def pull_env_dict(cls, **kwargs: dict):
        kwargs = {k.replace("openai_", ""): v for k,v in kwargs.items()}
        kwargs = {k.replace("nvidia_", ""): v for k,v in kwargs.items()}
        if "api_base" in kwargs:
            kwargs["base_url"] = kwargs.pop("api_base")
        return kwargs

class HttpxClientMixin(DefaultEnvMixin):

    APIClient: type[SyncAPIClient | AsyncAPIClient] | None = None
    # client options
    # api_key: str
    # organization: str | None

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
        http_client: httpx._client.BaseClient | None = None,
        use_base_as_endpoint: bool = False,
        _strict_response_validation: bool = False,
        **kwargs: Any,
    ) -> None:
        kws = {
            "api_key": api_key,
            "organization": organization,
            "base_url": base_url,
            "timeout": timeout,
            "max_retries": max_retries,
            "default_headers": default_headers,
            "default_query": default_query,
            "http_client": http_client,
            "_strict_response_validation": _strict_response_validation,
        }

        kws = self.__class__.pull_env_dict(**kws)
        self.api_key = kws.get("api_key")
        self.organization = kws.get("organization")
        self.use_base_as_endpoint = use_base_as_endpoint
        # print(self.__class__.__name__, kws)

        self.__class__.APIClient.__init__(
            self,
            version=__version__,
            base_url=kws['base_url'],
            max_retries=max_retries,
            timeout=timeout,
            http_client=http_client,
            custom_headers=default_headers,
            custom_query=default_query,
            _strict_response_validation=_strict_response_validation,
        )

        self._set_resources()

    @property
    @override
    def default_headers(self) -> dict[str, str | Omit]:
        return {
            **super().default_headers,
            **self._custom_headers,
        }

    @classmethod
    def pull_env_dict(cls, **kwargs: dict):
        kwargs = {k.replace("openai_", ""): v for k,v in kwargs.items()}
        kwargs = {k.replace("nvidia_", ""): v for k,v in kwargs.items()}
        if "api_base" in kwargs:
            kwargs["base_url"] = kwargs.pop("api_base")
        return kwargs

    def _set_default_headers(self, kws) -> None:
        if self.api_key is None:
            raise Exception(
                "The api_key client option must be set either by passing api_key to"
                " the client or by setting the NVIDIA_API_KEY environment variable"
            )
        if not kws.get("base_url"):
            kws["base_url"] = "https://api.openai.com/v1"

    @property
    @override
    def qs(self) -> Querystring:
        return Querystring(array_format="comma")

    @property
    @override
    def auth_headers(self) -> dict[str, str]:
        api_key = self.api_key
        return {"Authorization": f"Bearer {api_key}"}

    @property
    @override
    def default_headers(self) -> dict[str, str | Omit]:
        return {
            **super().default_headers,
            **self._custom_headers,
        }

    def copy(
        self,
        *,
        api_key: str | None = None,
        organization: str | None = None,
        base_url: str | httpx.URL | None = None,
        timeout: float | Timeout | None | NotGiven = NOT_GIVEN,
        http_client: httpx._client.BaseClient | None = None,
        max_retries: int | NotGiven = NOT_GIVEN,
        default_headers: Mapping[str, str] | None = None,
        set_default_headers: Mapping[str, str] | None = None,
        default_query: Mapping[str, object] | None = None,
        set_default_query: Mapping[str, object] | None = None,
        _extra_kwargs: Mapping[str, Any] = {},
    ) -> Self:
        """
        Create a new client instance re-using the same options given to the current client with optional overriding.
        """
        if default_headers is not None and set_default_headers is not None:
            raise ValueError("The `default_headers` and `set_default_headers` arguments are mutually exclusive")

        if default_query is not None and set_default_query is not None:
            raise ValueError("The `default_query` and `set_default_query` arguments are mutually exclusive")

        headers = self._custom_headers
        if default_headers is not None:
            headers = {**headers, **default_headers}
        elif set_default_headers is not None:
            headers = set_default_headers

        params = self._custom_query
        if default_query is not None:
            params = {**params, **default_query}
        elif set_default_query is not None:
            params = set_default_query

        http_client = http_client or self._client
        return self.__class__(
            api_key=api_key or self.api_key,
            organization=organization or self.organization,
            base_url=base_url or self.base_url,
            timeout=self.timeout if isinstance(timeout, NotGiven) else timeout,
            http_client=http_client,
            max_retries=max_retries if is_given(max_retries) else self.max_retries,
            default_headers=headers,
            default_query=params,
            **_extra_kwargs,
        )

    # Alias for `copy` for nicer inline usage, e.g.
    # client.with_options(timeout=10).foo.create(...)
    with_options = copy

    @override
    def _make_status_error(
        self,
        err_msg: str,
        *,
        body: object,
        response: httpx.Response,
    ) -> APIStatusError:
        data = body.get("error", body) if is_mapping(body) else body
        if response.status_code == 400:
            return _exceptions.BadRequestError(err_msg, response=response, body=data)

        if response.status_code == 401:
            return _exceptions.AuthenticationError(err_msg, response=response, body=data)

        if response.status_code == 403:
            return _exceptions.PermissionDeniedError(err_msg, response=response, body=data)

        if response.status_code == 404:
            return _exceptions.NotFoundError(err_msg, response=response, body=data)

        if response.status_code == 409:
            return _exceptions.ConflictError(err_msg, response=response, body=data)

        if response.status_code == 422:
            return _exceptions.UnprocessableEntityError(err_msg, response=response, body=data)

        if response.status_code == 429:
            return _exceptions.RateLimitError(err_msg, response=response, body=data)

        if response.status_code >= 500:
            return _exceptions.InternalServerError(err_msg, response=response, body=data)
        return APIStatusError(err_msg, response=response, body=data)

    def _prepare_url(self, url: str) -> httpx.URL:
        """
        Merge a URL argument together with any 'base_url' on the client,
        to create the URL used for the outgoing request.
        """
        # Copied from httpx's `_merge_url` method.
        merge_url = httpx.URL(url)
        if self.use_base_as_endpoint:
            merge_url = httpx.URL(str(self.base_url).rstrip("/"))
        elif merge_url.is_relative_url:
            merge_raw_path = self.base_url.raw_path + merge_url.raw_path.lstrip(b"/")
            merge_url = self.base_url.copy_with(raw_path=merge_raw_path)
        print(merge_url)
        return merge_url


#################################################################
## Mixin Combinations


class SyncHttpxClientMixin(HttpxClientMixin, SyncAPIClient):
    APIClient: type[SyncAPIClient | AsyncAPIClient] | None = SyncAPIClient


class AsyncHttpxClientMixin(HttpxClientMixin, AsyncAPIClient):
    APIClient: type[SyncAPIClient | AsyncAPIClient] | None = AsyncAPIClient
