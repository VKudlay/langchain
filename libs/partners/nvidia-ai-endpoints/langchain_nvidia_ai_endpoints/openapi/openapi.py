from __future__ import annotations

import logging
import os
import sys
from typing import (
    AbstractSet,
    Any,
    AsyncIterator,
    Collection,
    Dict,
    Iterator,
    List,
    Literal,
    Mapping,
    Optional,
    Set,
    Tuple,
    Union,
)

import openai
import tiktoken
from httpx._client import BaseClient
from langchain_core.callbacks import (
    AsyncCallbackManagerForLLMRun,
    CallbackManagerForLLMRun,
)
from langchain_core.language_models.llms import BaseLLM
from langchain_core.outputs import Generation, GenerationChunk, LLMResult
from langchain_core.pydantic_v1 import Field, SecretStr, root_validator
from langchain_core.utils import (
    convert_to_secret_str,
    get_from_dict_or_env,
    get_pydantic_field_names,
)
from langchain_core.utils.utils import build_extra_kwargs

from langchain_nvidia_ai_endpoints.openapi._openapi_client import DefaultEnvMixin

logger = logging.getLogger(__name__)


def _update_token_usage(
    keys: Set[str], response: Dict[str, Any], token_usage: Dict[str, Any]
) -> None:
    """Update token usage."""
    _keys_to_use = keys.intersection(response["usage"])
    for _key in _keys_to_use:
        if _key not in token_usage:
            token_usage[_key] = response["usage"][_key]
        else:
            token_usage[_key] += response["usage"][_key]


def _stream_response_to_generation_chunk(
    stream_response: Dict[str, Any],
) -> GenerationChunk:
    """Convert a stream response to a generation chunk."""
    if not stream_response["choices"]:
        return GenerationChunk(text="")
    return GenerationChunk(
        text=stream_response["choices"][0]["text"],
        generation_info=dict(
            finish_reason=stream_response["choices"][0].get("finish_reason", None),
            logprobs=stream_response["choices"][0].get("logprobs", None),
        ),
    )


class OpenAPI(DefaultEnvMixin, BaseLLM):
    """OpenAI large language models.

    To use, you should have the ``openai`` python package installed, and the
    environment variable ``OPENAI_API_KEY`` set with your API key.

    Any parameters that are valid to be passed to the openai.create call can be passed
    in, even if not explicitly saved on this class.

    Example:
        .. code-block:: python

            from langchain_community.llms import OpenAI
            openai = OpenAI(model_name="gpt-3.5-turbo-instruct")
    """

    client: Any = Field(default=None, exclude=True)  #: :meta private:
    async_client: Any = Field(default=None, exclude=True)  #: :meta private:
    model_name: str = Field(default="gpt-3.5-turbo-instruct", alias="model")
    temperature: float = 0.7
    max_tokens: int = 256
    top_p: float = 1
    frequency_penalty: float = 0
    presence_penalty: float = 0
    n: int = 1
    best_of: int = 1
    model_kwargs: Dict[str, Any] = Field(default_factory=dict)
    api_key: Optional[SecretStr] = Field(default=None)
    base_url: Optional[str] = Field(default=None)
    organization: Optional[str] = Field(default=None)
    proxy: Optional[str] = None
    batch_size: int = 20
    request_timeout: Union[float, Tuple[float, float], Any, None] = Field(
        default=None, alias="timeout"
    )
    logit_bias: Optional[Dict[str, float]] = Field(default_factory=dict)
    max_retries: int = 2
    streaming: bool = False
    allowed_special: Union[Literal["all"], AbstractSet[str]] = set()
    disallowed_special: Union[Literal["all"], Collection[str]] = "all"
    tiktoken_model_name: Optional[str] = None
    default_headers: Union[Mapping[str, str], None] = None
    default_query: Union[Mapping[str, object], None] = None
    http_client: Union[Any, None] = None
    use_base_as_endpoint: bool = Field(False)

    class Config:
        """Configuration for this pydantic object."""
        allow_population_by_field_name = True

    ## Properties/meta

    @property
    def _llm_type(self) -> str:
        """Return type of llm."""
        return "openapi"

    @property
    def lc_secrets(self) -> Dict[str, str]:
        return {"api_key": "API_KEY"}

    @property
    def lc_attributes(self) -> Dict[str, Any]:
        attributes: Dict[str, Any] = {}
        if self.organization:
            attributes["organization"] = self.organization
        if self.base_url:
            attributes["base_url"] = self.base_url
        if self.proxy:
            attributes["proxy"] = self.proxy
        return attributes

    @property
    def _default_params(self) -> Dict[str, Any]:
        """Get the default parameters for calling OpenAI API."""
        normal_params: Dict[str, Any] = {
            "temperature": self.temperature,
            "top_p": self.top_p,
            "frequency_penalty": self.frequency_penalty,
            "presence_penalty": self.presence_penalty,
            "n": self.n,
            "logit_bias": self.logit_bias,
        }

        if self.max_tokens is not None:
            normal_params["max_tokens"] = self.max_tokens

        # Azure gpt-35-turbo doesn't support best_of
        # don't specify best_of if it is 1
        if self.best_of > 1:
            normal_params["best_of"] = self.best_of

        return {**normal_params, **self.model_kwargs}

    @property
    def available_models(self) -> list:
        """Map the available models that can be invoked."""
        return [m.id for m in self.client._client.models.list().data]

    @property
    def _identifying_params(self) -> Mapping[str, Any]:
        """Get the identifying parameters."""
        return {**{"model_name": self.model_name}, **self._default_params}

    @classmethod
    def get_lc_namespace(cls) -> List[str]:
        """Get the namespace of the langchain object."""
        return ["langchain", "llms", "openai"]

    @classmethod
    def is_lc_serializable(cls) -> bool:
        """Return whether this model can be serialized by Langchain."""
        return True

    @property
    def _invocation_params(self) -> Dict[str, Any]:
        return {**{"model": self.model_name}, **self._default_params}

    @classmethod
    def is_lc_serializable(cls) -> bool:
        """Return whether this model can be serialized by Langchain."""
        return True

    @classmethod
    def get_client_classes(cls):
        return openai.OpenAI, openai.AsyncOpenAI
    
    @classmethod
    def get_client_model_list(cls, client: Optional[BaseClient]=None, **kwargs: Any) -> list:
        """Map the available models that can be invoked. Callable from class"""
        new_kws = cls.pull_env_dict(**kwargs)
        if client is None:
            client = cls.get_client_classes()[0]
        return client(**new_kws).models.list().data

    @classmethod
    def get_available_models(cls, **kwargs: Any) -> list:
        """Map the available models that can be invoked. Callable from class"""
        return [m.id for m in cls.get_client_model_list(**kwargs)]

    def get_model_details(self, model_name: Optional[str] = None) -> dict:
        """Get more meta-details about a model retrieved by a given name"""
        model_name = model_name or self.model_name or None
        known_fns = self.client._client.models.list().data
        fn_spec = [f for f in known_fns if f.id == model_name or not model_name][0]
        return fn_spec

    def get_model_clients(self) -> list:
        """Map the available models that can be invoked."""
        return {model_name: self.bind(model=model_name) for model_name in self.available_models}

    ## Validators

    @root_validator(pre=True)
    def build_extra(cls, values: Dict[str, Any]) -> Dict[str, Any]:
        """Build extra kwargs from additional params that were passed in."""
        all_required_field_names = get_pydantic_field_names(cls)
        extra = values.get("model_kwargs", {})
        values["model_kwargs"] = build_extra_kwargs(
            extra, values, all_required_field_names
        )
        return values

    @root_validator()
    def validate_environment(cls, values: Dict) -> Dict:
        """Validate that api key and python package exists in environment."""
        if values["n"] < 1:
            raise ValueError("n must be at least 1.")
        if values["streaming"] and values["n"] > 1:
            raise ValueError("Cannot stream results when n > 1.")
        if values["streaming"] and values["best_of"] > 1:
            raise ValueError("Cannot stream results when best_of > 1.")

        values = cls.pull_env_dict(**values)

        client_params = {
            "api_key": values["api_key"],
            "organization": values["organization"],
            "base_url": values["base_url"],
            "timeout": values["request_timeout"],
            "max_retries": values["max_retries"],
            "default_headers": values["default_headers"],
            "default_query": values["default_query"],
            "http_client": values["http_client"],
            "use_base_as_endpoint": values["use_base_as_endpoint"],
        }

        SClient, AClient = cls.get_client_classes()
        if not values.get("client"):
            values["client"] = SClient(**client_params).completions
        if not values.get("async_client"):
            values["async_client"] = AClient(**client_params).completions

        return values

    ## Standard forward-pass methods

    def _stream(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> Iterator[GenerationChunk]:
        params = {**self._invocation_params, **kwargs, "stream": True}
        self.get_sub_prompts(params, [prompt], stop)  # this mutates params
        for stream_resp in self.client.create(prompt=prompt, **params):
            if not isinstance(stream_resp, dict):
                stream_resp = stream_resp.model_dump()
            chunk = _stream_response_to_generation_chunk(stream_resp)

            if run_manager:
                run_manager.on_llm_new_token(
                    chunk.text,
                    chunk=chunk,
                    verbose=self.verbose,
                    logprobs=(
                        chunk.generation_info["logprobs"]
                        if chunk.generation_info
                        else None
                    ),
                )
            yield chunk

    async def _astream(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[AsyncCallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> AsyncIterator[GenerationChunk]:
        params = {**self._invocation_params, **kwargs, "stream": True}
        self.get_sub_prompts(params, [prompt], stop)  # this mutates params
        async for stream_resp in await self.async_client.create(
            prompt=prompt, **params
        ):
            if not isinstance(stream_resp, dict):
                stream_resp = stream_resp.model_dump()
            chunk = _stream_response_to_generation_chunk(stream_resp)

            if run_manager:
                await run_manager.on_llm_new_token(
                    chunk.text,
                    chunk=chunk,
                    verbose=self.verbose,
                    logprobs=(
                        chunk.generation_info["logprobs"]
                        if chunk.generation_info
                        else None
                    ),
                )
            yield chunk

    def _generate(
        self,
        prompts: List[str],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> LLMResult:
        """Call out to OpenAI's endpoint with k unique prompts.

        Args:
            prompts: The prompts to pass into the model.
            stop: Optional list of stop words to use when generating.

        Returns:
            The full LLM output.

        Example:
            .. code-block:: python

                response = openai.generate(["Tell me a joke."])
        """
        # TODO: write a unit test for this
        params = self._invocation_params
        params = {**params, **kwargs}
        sub_prompts = self.get_sub_prompts(params, prompts, stop)
        choices = []
        token_usage: Dict[str, int] = {}
        # Get the token usage from the response.
        # Includes prompt, completion, and total tokens used.
        _keys = {"completion_tokens", "prompt_tokens", "total_tokens"}
        system_fingerprint: Optional[str] = None
        for _prompts in sub_prompts:
            if self.streaming:
                if len(_prompts) > 1:
                    raise ValueError("Cannot stream results with multiple prompts.")

                generation: Optional[GenerationChunk] = None
                for chunk in self._stream(_prompts[0], stop, run_manager, **kwargs):
                    if generation is None:
                        generation = chunk
                    else:
                        generation += chunk
                assert generation is not None
                choices.append(
                    {
                        "text": generation.text,
                        "finish_reason": (
                            generation.generation_info.get("finish_reason")
                            if generation.generation_info
                            else None
                        ),
                        "logprobs": (
                            generation.generation_info.get("logprobs")
                            if generation.generation_info
                            else None
                        ),
                    }
                )
            else:
                print(params)
                response = self.client.create(prompt=_prompts, **params)
                if not isinstance(response, dict):
                    # V1 client returns the response in an PyDantic object instead of
                    # dict. For the transition period, we deep convert it to dict.
                    response = response.model_dump()

                choices.extend(response["choices"])
                _update_token_usage(_keys, response, token_usage)
                if not system_fingerprint:
                    system_fingerprint = response.get("system_fingerprint")
        return self.create_llm_result(
            choices,
            prompts,
            params,
            token_usage,
            system_fingerprint=system_fingerprint,
        )

    async def _agenerate(
        self,
        prompts: List[str],
        stop: Optional[List[str]] = None,
        run_manager: Optional[AsyncCallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> LLMResult:
        """Call out to OpenAI's endpoint async with k unique prompts."""
        params = self._invocation_params
        params = {**params, **kwargs}
        sub_prompts = self.get_sub_prompts(params, prompts, stop)
        choices = []
        token_usage: Dict[str, int] = {}
        # Get the token usage from the response.
        # Includes prompt, completion, and total tokens used.
        _keys = {"completion_tokens", "prompt_tokens", "total_tokens"}
        system_fingerprint: Optional[str] = None
        for _prompts in sub_prompts:
            if self.streaming:
                if len(_prompts) > 1:
                    raise ValueError("Cannot stream results with multiple prompts.")

                generation: Optional[GenerationChunk] = None
                async for chunk in self._astream(
                    _prompts[0], stop, run_manager, **kwargs
                ):
                    if generation is None:
                        generation = chunk
                    else:
                        generation += chunk
                assert generation is not None
                choices.append(
                    {
                        "text": generation.text,
                        "finish_reason": (
                            generation.generation_info.get("finish_reason")
                            if generation.generation_info
                            else None
                        ),
                        "logprobs": (
                            generation.generation_info.get("logprobs")
                            if generation.generation_info
                            else None
                        ),
                    }
                )
            else:
                response = await self.async_client.create(prompt=_prompts, **params)
                if not isinstance(response, dict):
                    response = response.model_dump()
                choices.extend(response["choices"])
                _update_token_usage(_keys, response, token_usage)
        return self.create_llm_result(
            choices,
            prompts,
            params,
            token_usage,
            system_fingerprint=system_fingerprint,
        )
    
    ## Helper Methods

    def get_sub_prompts(
        self,
        params: Dict[str, Any],
        prompts: List[str],
        stop: Optional[List[str]] = None,
    ) -> List[List[str]]:
        """Get the sub prompts for llm call."""
        if stop is not None:
            if "stop" in params:
                raise ValueError("`stop` found in both the input and default params.")
            params["stop"] = stop
        if params["max_tokens"] == -1:
            if len(prompts) != 1:
                raise ValueError(
                    "max_tokens set to -1 not supported for multiple inputs."
                )
            params["max_tokens"] = self.max_tokens_for_prompt(prompts[0])
        sub_prompts = [
            prompts[i : i + self.batch_size]
            for i in range(0, len(prompts), self.batch_size)
        ]
        return sub_prompts

    def create_llm_result(
        self,
        choices: Any,
        prompts: List[str],
        params: Dict[str, Any],
        token_usage: Dict[str, int],
        *,
        system_fingerprint: Optional[str] = None,
    ) -> LLMResult:
        """Create the LLMResult from the choices and prompts."""
        generations = []
        n = params.get("n", self.n)
        for i, _ in enumerate(prompts):
            sub_choices = choices[i * n : (i + 1) * n]
            generations.append(
                [
                    Generation(
                        text=choice["text"],
                        generation_info=dict(
                            finish_reason=choice.get("finish_reason"),
                            logprobs=choice.get("logprobs"),
                        ),
                    )
                    for choice in sub_choices
                ]
            )
        llm_output = {"token_usage": token_usage, "model_name": self.model_name}
        if system_fingerprint:
            llm_output["system_fingerprint"] = system_fingerprint
        return LLMResult(generations=generations, llm_output=llm_output)

    def get_token_ids(self, text: str) -> List[int]:
        """Get the token IDs using the tiktoken package."""
        # tiktoken NOT supported for Python < 3.8
        if sys.version_info[1] < 8:
            return super().get_num_tokens(text)

        model_name = self.tiktoken_model_name or self.model_name
        try:
            enc = tiktoken.encoding_for_model(model_name)
        except KeyError:
            enc = tiktoken.get_encoding("cl100k_base")

        return enc.encode(
            text,
            allowed_special=self.allowed_special,
            disallowed_special=self.disallowed_special,
        )

    @property
    def max_context_size(self) -> int:
        """Get max context size for this model."""
        return self.modelname_to_contextsize(self.model_name)


    def max_tokens_for_prompt(self, prompt: str) -> int:
        """Calculate the maximum number of tokens possible to generate for a prompt.

        Args:
            prompt: The prompt to pass into the model.

        Returns:
            The maximum number of tokens to generate for a prompt.

        Example:
            .. code-block:: python

                max_tokens = openai.max_token_for_prompt("Tell me a joke.")
        """
        num_tokens = self.get_num_tokens(prompt)
        return self.max_context_size - num_tokens

    @staticmethod
    def modelname_to_contextsize(modelname: str) -> int:
        return 2040
