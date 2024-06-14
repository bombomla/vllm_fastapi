import argparse
import asyncio
import json
import os
import time
from http import HTTPStatus
from typing import AsyncGenerator, Dict, List, Optional, Tuple, Union

from logging_test import logger

import fastapi
import uvicorn
from fastapi import Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.encoders import jsonable_encoder
from fastapi.responses import JSONResponse, StreamingResponse, Response
from packaging import version
from pydantic import BaseModel, Field

from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.engine.async_llm_engine import AsyncLLMEngine
from vllm.entrypoints.openai.protocol import (
    CompletionResponse, CompletionResponseChoice,
    CompletionResponseStreamChoice, CompletionStreamResponse,
    ChatCompletionResponse,
    ChatCompletionResponseChoice, ChatCompletionResponseStreamChoice,
    ChatCompletionStreamResponse, ChatMessage, DeltaMessage, ErrorResponse,
    CompletionLogProbs, ModelCard, ModelList, ModelPermission, UsageInfo)
from vllm.logger import init_logger
from vllm.outputs import RequestOutput
from vllm.sampling_params import SamplingParams
from vllm.transformers_utils.tokenizer import get_tokenizer
from vllm.utils import random_uuid

try:
    import fastchat
    from fastchat.conversation import Conversation, SeparatorStyle
    from fastchat.model.model_adapter import get_conversation_template
    _fastchat_available = True
except ImportError:
    _fastchat_available = False

TIMEOUT_KEEP_ALIVE = 5  # seconds

#API_KEY = os.environ.get("API_KEY", None)

#logger = init_logger(__name__)
served_model = None
app = fastapi.FastAPI()
engine = None


class ChatCompletionRequest(BaseModel):
    model: str
    #api_key: str
    messages: Union[str, List[Dict[str, str]]]
    temperature: Optional[float] = 0.7
    top_p: Optional[float] = 1.0
    n: Optional[int] = 1
    max_tokens: Optional[int] = None
    stop: Optional[Union[str, List[str]]] = Field(default_factory=list)
    stream: Optional[bool] = False
    presence_penalty: Optional[float] = 0.0
    frequency_penalty: Optional[float] = 0.0
    logit_bias: Optional[Dict[str, float]] = None
    user: Optional[str] = None
    best_of: Optional[int] = None
    top_k: Optional[int] = -1
    ignore_eos: Optional[bool] = False
    use_beam_search: Optional[bool] = False
    stop_token_ids: Optional[List[int]] = Field(default_factory=list)
    skip_special_tokens: Optional[bool] = True
    spaces_between_special_tokens: Optional[bool] = True


class CompletionRequest(BaseModel):
    model: str
   #api_key: str
    prompt: Union[List[int], List[List[int]], str, List[str]]
    suffix: Optional[str] = None
    max_tokens: Optional[int] = 16
    temperature: Optional[float] = 1.0
    top_p: Optional[float] = 1.0
    n: Optional[int] = 1
    stream: Optional[bool] = False
    logprobs: Optional[int] = None
    echo: Optional[bool] = False
    stop: Optional[Union[str, List[str]]] = Field(default_factory=list)
    presence_penalty: Optional[float] = 0.0
    frequency_penalty: Optional[float] = 0.0
    best_of: Optional[int] = None
    logit_bias: Optional[Dict[str, float]] = None
    user: Optional[str] = None
    # Additional parameters supported by vLLM
    top_k: Optional[int] = -1
    ignore_eos: Optional[bool] = False
    use_beam_search: Optional[bool] = False
    stop_token_ids: Optional[List[int]] = Field(default_factory=list)
    skip_special_tokens: Optional[bool] = True
    spaces_between_special_tokens: Optional[bool] = True


def create_error_response(status_code: HTTPStatus,
                          message: str) -> JSONResponse:
    #return JSONResponse(ErrorResponse(message=message,
    #                                  type="invalid_request_error").dict(),
    #                    status_code=status_code.value)
    return JSONResponse(status_code=status_code.value, content={"detail":message})


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request, exc):  # pylint: disable=unused-argument
    return create_error_response(HTTPStatus.BAD_REQUEST, str(exc))


async def check_model(request) -> Optional[JSONResponse]:
    if request.model == served_model:
        return
    ret = create_error_response(
        HTTPStatus.NOT_FOUND,
        f"The model `{request.model}` does not exist, the model name is {served_model}.",
    )
    return ret


async def get_gen_prompt(request) -> str:
    if not _fastchat_available:
        raise ModuleNotFoundError(
            "fastchat is not installed. Please install fastchat to use "
            "the chat completion and conversation APIs: `$ pip install fschat`"
        )
    if version.parse(fastchat.__version__) < version.parse("0.2.23"):
        raise ImportError(
            f"fastchat version is low. Current version: {fastchat.__version__} "
            "Please upgrade fastchat to use: `$ pip install -U fschat`")
    #conv = get_conversation_template(request.model)
    if model_name is not None:
        conv = get_conversation_template(model_name)
    else:
        conv = get_conversation_template(served_model)
    conv = Conversation(
        name=conv.name,
        system_template=conv.system_template,
        system_message=conv.system_message,
        roles=conv.roles,
        messages=list(conv.messages),  # prevent in-place modification
        offset=conv.offset,
        sep_style=SeparatorStyle(conv.sep_style),
        sep=conv.sep,
        sep2=conv.sep2,
        stop_str=conv.stop_str,
        stop_token_ids=conv.stop_token_ids,
    )

    if isinstance(request.messages, str):
        prompt = request.messages
    else:
        for message in request.messages:
            msg_role = message["role"]
            if msg_role == "system":
                conv.system_message = message["content"]
            elif msg_role == "user":
                conv.append_message(conv.roles[0], message["content"])
                
            elif msg_role == "assistant":
                conv.append_message(conv.roles[1], message["content"])
            else:
                raise ValueError(f"Unknown role: {msg_role}")

        # Add a blank message for the assistant.
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()
    

    return prompt


async def check_length(
    request: Union[ChatCompletionRequest, CompletionRequest],
    prompt: Optional[str] = None,
    prompt_ids: Optional[List[int]] = None
) -> Tuple[List[int], Optional[JSONResponse]]:
    assert (not (prompt is None and prompt_ids is None)
            and not (prompt is not None and prompt_ids is not None)
            ), "Either prompt or prompt_ids should be provided."
    if prompt_ids is not None:
        input_ids = prompt_ids
    else:
        input_ids = tokenizer(prompt).input_ids
    token_num = len(input_ids)

    if request.max_tokens is None:
        request.max_tokens = max_model_len - token_num
    if token_num + request.max_tokens > max_model_len:
        return input_ids, create_error_response(
            HTTPStatus.BAD_REQUEST,
            f"This model's maximum context length is {max_model_len} tokens. "
            f"However, you requested {request.max_tokens + token_num} tokens "
            f"({token_num} in the messages, "
            f"{request.max_tokens} in the completion). "
            f"Please reduce the length of the messages or completion.",
        )
    else:
        return input_ids, None


@app.get("/health")
async def health() -> Response:
    """Health check."""
    return Response(status_code=200)


@app.get("/models")
async def show_available_models():
    """Show available models. Right now we only have one model."""
    model_cards = [
        ModelCard(id=served_model,
                  root=served_model,
                  permission=[ModelPermission()])
    ]
    return ModelList(data=model_cards)


def create_logprobs(token_ids: List[int],
                    id_logprobs: List[Dict[int, float]],
                    initial_text_offset: int = 0) -> CompletionLogProbs:
    """Create OpenAI-style logprobs."""
    CompletionLogProbs = CompletionLogProbs()
    last_token_len = 0
    for token_id, id_logprob in zip(token_ids, id_logprobs):
        token = tokenizer.convert_ids_to_tokens(token_id)
        CompletionLogProbs.tokens.append(token)
        CompletionLogProbs.token_logprobs.append(id_logprob[token_id])
        if len(CompletionLogProbs.text_offset) == 0:
            CompletionLogProbs.text_offset.append(initial_text_offset)
        else:
            CompletionLogProbs.text_offset.append(CompletionLogProbs.text_offset[-1] +
                                        last_token_len)
        last_token_len = len(token)

        CompletionLogProbs.top_logprobs.append({
            tokenizer.convert_ids_to_tokens(i): p
            for i, p in id_logprob.items()
        })
    return CompletionLogProbs


@app.post("/v1/chat/completions")
async def create_chat_completion(request: ChatCompletionRequest,
                                 raw_request: Request):
    """Completion API similar to OpenAI's API.
    See  https://platform.openai.com/docs/api-reference/chat/create
    for the API specification. This API mimics the OpenAI ChatCompletion API.
    NOTE: Currently we do not support the following features:
        - function_call (Users should implement this by themselves)
        - logit_bias (to be supported by vLLM engine)
    """
    # != API_KEY or API_KEY is None:
    #    return create_error_response(HTTPStatus.UNAUTHORIZED,
    #                                 "Invalid API key")

    #logger.info(f"Received chat completion request: {request}")
    print(f"Received chat completion request: {request}")

    error_check_ret = await check_model(request)
    if error_check_ret is not None:
        return error_check_ret

    if request.logit_bias is not None and len(request.logit_bias) > 0:
        # TODO: support logit_bias in vLLM engine.
        return create_error_response(HTTPStatus.BAD_REQUEST,
                                     "logit_bias is not currently supported")

    prompt = await get_gen_prompt(request)
    token_ids, error_check_ret = await check_length(request, prompt=prompt)
    if error_check_ret is not None:
        return error_check_ret

    model_name = request.model
    request_id = f"cmpl-{random_uuid()}"
    created_time = int(time.monotonic())
    try:
        spaces_between_special_tokens = request.spaces_between_special_tokens
        sampling_params = SamplingParams(
            n=request.n,
            presence_penalty=request.presence_penalty,
            frequency_penalty=request.frequency_penalty,
            temperature=request.temperature,
            top_p=request.top_p,
            stop=request.stop,
            stop_token_ids=request.stop_token_ids,
            max_tokens=request.max_tokens,
            best_of=request.best_of,
            top_k=request.top_k,
            ignore_eos=request.ignore_eos,
            use_beam_search=request.use_beam_search,
            skip_special_tokens=request.skip_special_tokens,
            spaces_between_special_tokens=spaces_between_special_tokens,
        )
    except ValueError as e:
        return create_error_response(HTTPStatus.BAD_REQUEST, str(e))

    start_time = time.time()
    log_dict = dict()
    result_generator = engine.generate(prompt, sampling_params, request_id)

    def create_stream_response_json(
        index: int,
        text: str,
        finish_reason: Optional[str] = None,
        usage: Optional[UsageInfo] = None,
    ) -> str:
        choice_data = ChatCompletionResponseStreamChoice(
            index=index,
            delta=DeltaMessage(content=text),
            finish_reason=finish_reason,
        )
        response = ChatCompletionStreamResponse(
            id=request_id,
            created=created_time,
            model=model_name,
            choices=[choice_data],
        )
        if usage is not None:
            response.usage = usage
        # exclude unset to leave details out of each sse
        #response_json = response.json(exclude_unset=True, ensure_ascii=False)
        response_json = json.dumps(response.model_dump(exclude_unset=True),ensure_ascii=False)

        return response_json

    async def completion_stream_generator() -> AsyncGenerator[str, None]:
        # First chunk with role
        for i in range(request.n):
            choice_data = ChatCompletionResponseStreamChoice(
                index=i,
                delta=DeltaMessage(role="assistant"),
                finish_reason=None,
            )
            chunk = ChatCompletionStreamResponse(id=request_id,
                                                 choices=[choice_data],
                                                 model=model_name)
            #data = chunk.json(exclude_unset=True, ensure_ascii=False)
            data = json.dumps(chunk.model_dump(exclude_unset=True),ensure_ascii=False)
            yield f"data: {data}\n\n"

        previous_texts = [""] * request.n
        stream_txt = []
        previous_num_tokens = [0] * request.n
        async for res in result_generator:
            res: RequestOutput
            for output in res.outputs:
                i = output.index
                delta_text = output.text[len(previous_texts[i]):]
                previous_texts[i] = output.text
                completion_tokens = len(output.token_ids)
                previous_num_tokens[i] = completion_tokens
                response_json = create_stream_response_json(
                    index=i,
                    text=delta_text,
                )
                stream_txt.append(delta_text)
                yield f"data: {response_json}\n\n"
                if output.finish_reason is not None:
                    prompt_tokens = len(res.prompt_token_ids)
                    final_usage = UsageInfo(
                        prompt_tokens=prompt_tokens,
                        completion_tokens=completion_tokens,
                        total_tokens=prompt_tokens + completion_tokens,
                    )
                    response_json = create_stream_response_json(
                        index=i,
                        text="",
                        finish_reason=output.finish_reason,
                        usage=final_usage,
                    )
                    
                    yield f"data: {response_json}\n\n"
        time_string = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(start_time))
        end_time = time.time()
        total_time = str(end_time - start_time)
        log_dict["requestTime"] = time_string
        log_dict["prompt"] = prompt
        log_dict["response"] = "".join(stream_txt)
        log_dict["total_time"] = total_time
        log_dict_str = jsonable_encoder(log_dict)
        logger.info(log_dict_str)

        #print("".join(stream_txt))
        yield "data: [DONE]\n\n"

    # Streaming response
    if request.stream:
        return StreamingResponse(completion_stream_generator(),
                                 media_type="text/event-stream")

    # Non-streaming response
    final_res: RequestOutput = None
    async for res in result_generator:
        if await raw_request.is_disconnected():
            # Abort the request if the client disconnects.
            await engine.abort(request_id)
            return create_error_response(HTTPStatus.BAD_REQUEST,
                                         "Client disconnected")
        final_res = res
    assert final_res is not None
    choices = []
    for output in final_res.outputs:
        choice_data = ChatCompletionResponseChoice(
            index=output.index,
            message=ChatMessage(role="assistant", content=output.text),
            finish_reason=output.finish_reason,
        )
        choices.append(choice_data)

    num_prompt_tokens = len(final_res.prompt_token_ids)
    num_generated_tokens = sum(
        len(output.token_ids) for output in final_res.outputs)
    usage = UsageInfo(
        prompt_tokens=num_prompt_tokens,
        completion_tokens=num_generated_tokens,
        total_tokens=num_prompt_tokens + num_generated_tokens,
    )
    response = ChatCompletionResponse(
        id=request_id,
        created=created_time,
        model=model_name,
        choices=choices,
        usage=usage,
    )

    if request.stream:
        # When user requests streaming but we don't stream, we still need to
        # return a streaming response with a single event.
        response_json = response.json(ensure_ascii=False)

        async def fake_stream_generator() -> AsyncGenerator[str, None]:
            yield f"data: {response_json}\n\n"
            yield "data: [DONE]\n\n"

        return StreamingResponse(fake_stream_generator(),
                                 media_type="text/event-stream")
    time_string = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(start_time))
    end_time = time.time()
    total_time = str(end_time - start_time)
    log_dict["requestTime"] = time_string
    log_dict["prompt"] = prompt
    log_dict["response"] = jsonable_encoder(choices)
    log_dict["total_time"] = total_time
    log_dict_str = jsonable_encoder(log_dict)
    logger.info(log_dict_str)
    return response

'''
@app.post("/completions")
async def create_completion(request: CompletionRequest, raw_request: Request):
    """Completion API similar to OpenAI's API.
    See https://platform.openai.com/docs/api-reference/completions/create
    for the API specification. This API mimics the OpenAI Completion API.
    NOTE: Currently we do not support the following features:
        - echo (since the vLLM engine does not currently support
          getting the logprobs of prompt tokens)
        - suffix (the language models we currently support do not support
          suffix)
        - logit_bias (to be supported by vLLM engine)
    """
    #if request.api_key != API_KEY or API_KEY is None:
    #    return create_error_response(HTTPStatus.UNAUTHORIZED,
    #                                 "Invalid API key")

    logger.info(f"Received completion request: {request}")

    error_check_ret = await check_model(request)
    if error_check_ret is not None:
        return error_check_ret

    if request.echo:
        # We do not support echo since the vLLM engine does not
        # currently support getting the logprobs of prompt tokens.
        return create_error_response(HTTPStatus.BAD_REQUEST,
                                     "echo is not currently supported")

    if request.suffix is not None:
        # The language models we currently support do not support suffix.
        return create_error_response(HTTPStatus.BAD_REQUEST,
                                     "suffix is not currently supported")

    if request.logit_bias is not None and len(request.logit_bias) > 0:
        # TODO: support logit_bias in vLLM engine.
        return create_error_response(HTTPStatus.BAD_REQUEST,
                                     "logit_bias is not currently supported")

    model_name = request.model
    request_id = f"cmpl-{random_uuid()}"

    use_token_ids = False
    if isinstance(request.prompt, list):
        if len(request.prompt) == 0:
            return create_error_response(HTTPStatus.BAD_REQUEST,
                                         "please provide at least one prompt")
        first_element = request.prompt[0]
        if isinstance(first_element, int):
            use_token_ids = True
            prompt = request.prompt
        elif isinstance(first_element, (str, list)):
            # TODO: handles multiple prompt case in list[list[int]]
            if len(request.prompt) > 1:
                return create_error_response(
                    HTTPStatus.BAD_REQUEST,
                    "multiple prompts in a batch is not currently supported")
            use_token_ids = not isinstance(first_element, str)
            prompt = request.prompt[0]
    else:
        prompt = request.prompt

    if use_token_ids:
        _, error_check_ret = await check_length(request, prompt_ids=prompt)
    else:
        token_ids, error_check_ret = await check_length(request, prompt=prompt)
    if error_check_ret is not None:
        return error_check_ret

    created_time = int(time.monotonic())
    try:
        spaces_between_special_tokens = request.spaces_between_special_tokens
        sampling_params = SamplingParams(
            n=request.n,
            best_of=request.best_of,
            presence_penalty=request.presence_penalty,
            frequency_penalty=request.frequency_penalty,
            temperature=request.temperature,
            top_p=request.top_p,
            top_k=request.top_k,
            stop=request.stop,
            stop_token_ids=request.stop_token_ids,
            ignore_eos=request.ignore_eos,
            max_tokens=request.max_tokens,
            logprobs=request.logprobs,
            use_beam_search=request.use_beam_search,
            skip_special_tokens=request.skip_special_tokens,
            spaces_between_special_tokens=spaces_between_special_tokens,
        )
    except ValueError as e:
        return create_error_response(HTTPStatus.BAD_REQUEST, str(e))

    if use_token_ids:
        result_generator = engine.generate(None,
                                           sampling_params,
                                           request_id,
                                           prompt_token_ids=prompt)
    else:
        result_generator = engine.generate(prompt, sampling_params, request_id,
                                           token_ids)

    # Similar to the OpenAI API, when n != best_of, we do not stream the
    # results. In addition, we do not stream the results when use beam search.
    stream = (request.stream
              and (request.best_of is None or request.n == request.best_of)
              and not request.use_beam_search)

    def create_stream_response_json(
        index: int,
        text: str,
        logprobs: Optional[LogProbs] = None,
        finish_reason: Optional[str] = None,
        usage: Optional[UsageInfo] = None,
    ) -> str:
        choice_data = CompletionResponseStreamChoice(
            index=index,
            text=text,
            logprobs=logprobs,
            finish_reason=finish_reason,
        )
        response = CompletionStreamResponse(
            id=request_id,
            created=created_time,
            model=model_name,
            choices=[choice_data],
        )
        if usage is not None:
            response.usage = usage
        #response_json = response.json(exclude_unset=True, ensure_ascii=False)
        response_json = json.dumps(response.exclude_unset(exclude_unset=True),ensure_ascii=False)

        return response_json

    async def completion_stream_generator() -> AsyncGenerator[str, None]:
        previous_texts = [""] * request.n
        previous_num_tokens = [0] * request.n
        async for res in result_generator:
            res: RequestOutput
            for output in res.outputs:
                i = output.index
                delta_text = output.text[len(previous_texts[i]):]
                if request.logprobs is not None:
                    logprobs = create_logprobs(
                        output.token_ids[previous_num_tokens[i]:],
                        output.logprobs[previous_num_tokens[i]:],
                        len(previous_texts[i]))
                else:
                    logprobs = None
                previous_texts[i] = output.text
                previous_num_tokens[i] = len(output.token_ids)
                response_json = create_stream_response_json(
                    index=i,
                    text=delta_text,
                    logprobs=logprobs,
                )
                yield f"data: {response_json}\n\n"
                if output.finish_reason is not None:
                    logprobs = (LogProbs()
                                if request.logprobs is not None else None)
                    prompt_tokens = len(res.prompt_token_ids)
                    completion_tokens = len(output.token_ids)
                    final_usage = UsageInfo(
                        prompt_tokens=prompt_tokens,
                        completion_tokens=completion_tokens,
                        total_tokens=prompt_tokens + completion_tokens,
                    )
                    response_json = create_stream_response_json(
                        index=i,
                        text="",
                        logprobs=logprobs,
                        finish_reason=output.finish_reason,
                        usage=final_usage,
                    )
                    yield f"data: {response_json}\n\n"
        yield "data: [DONE]\n\n"

    # Streaming response
    if stream:
        return StreamingResponse(completion_stream_generator(),
                                 media_type="text/event-stream")

    # Non-streaming response
    final_res: RequestOutput = None
    async for res in result_generator:
        if await raw_request.is_disconnected():
            # Abort the request if the client disconnects.
            await engine.abort(request_id)
            return create_error_response(HTTPStatus.BAD_REQUEST,
                                         "Client disconnected")
        final_res = res
    assert final_res is not None
    choices = []
    for output in final_res.outputs:
        if request.logprobs is not None:
            logprobs = create_logprobs(output.token_ids, output.logprobs)
        else:
            logprobs = None
        choice_data = CompletionResponseChoice(
            index=output.index,
            text=output.text,
            logprobs=logprobs,
            finish_reason=output.finish_reason,
        )
        choices.append(choice_data)

    num_prompt_tokens = len(final_res.prompt_token_ids)
    num_generated_tokens = sum(
        len(output.token_ids) for output in final_res.outputs)
    usage = UsageInfo(
        prompt_tokens=num_prompt_tokens,
        completion_tokens=num_generated_tokens,
        total_tokens=num_prompt_tokens + num_generated_tokens,
    )
    response = CompletionResponse(
        id=request_id,
        created=created_time,
        model=model_name,
        choices=choices,
        usage=usage,
    )

    if request.stream:
        # When user requests streaming but we don't stream, we still need to
        # return a streaming response with a single event.
        response_json = response.json(ensure_ascii=False)

        async def fake_stream_generator() -> AsyncGenerator[str, None]:
            yield f"data: {response_json}\n\n"
            yield "data: [DONE]\n\n"

        return StreamingResponse(fake_stream_generator(),
                                 media_type="text/event-stream")

    return response
'''

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="vLLM OpenAI-Compatible RESTful API server.")
    parser.add_argument("--host", type=str, default=None, help="host name")
    parser.add_argument("--port", type=int, default=8000, help="port number")
    parser.add_argument("--allow-credentials",
                        action="store_true",
                        help="allow credentials")
    parser.add_argument("--allowed-origins",
                        type=json.loads,
                        default=["*"],
                        help="allowed origins")
    parser.add_argument("--allowed-methods",
                        type=json.loads,
                        default=["*"],
                        help="allowed methods")
    parser.add_argument("--allowed-headers",
                        type=json.loads,
                        default=["*"],
                        help="allowed headers")
    parser.add_argument("--served-model-name",
                        type=str,
                        default=None,
                        help="The model name used in the API. If not "
                        "specified, the model name will be the same as "
                        "the huggingface name.")

    parser.add_argument("--model", type=str, default="facebook/opt-125m",
                        help="the model ID")

    parser.add_argument("--gpu-memory-utilization", type=float, default=0.9,
                        help="gpu memory use")

    parser.add_argument("--tensor-parallel-size", type=int, default=1,
                        help="tensor-parallel-size")
    parser.add_argument("--input-len",
                        type=int,
                        default=None,
                        help="Input prompt length for each request")
    parser.add_argument("--output-len",
                        type=int,
                        default=None,
                        help="Output length for each request. Overrides the "
                        "output length from the dataset.")
    parser.add_argument("--tokenizer", type=str, default=None)
    parser.add_argument('--quantization',
                        '-q',
                        choices=['awq', 'gptq', 'squeezellm', None],
                        default=None)
    parser.add_argument("--use-beam-search", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--hf-max-batch-size",
                        type=int,
                        default=None,
                        help="Maximum batch size for HF backend.")
    parser.add_argument('--trust-remote-code',
                        action='store_true',
                        help='trust remote code from huggingface')
    parser.add_argument(
        '--max-model-len',
        type=int,
        default=None,
        help='Maximum length of a sequence (including prompt and output). '
        'If None, will be derived from the model.')
    parser.add_argument(
        '--dtype',
        type=str,
        default='auto',
        choices=['auto', 'half', 'float16', 'bfloat16', 'float', 'float32'],
        help='data type for model weights and activations. '
        'The "auto" option will use FP16 precision '
        'for FP32 and FP16 models, and BF16 precision '
        'for BF16 models.')
    parser.add_argument("--enforce-eager",
                        action="store_true",
                        help="enforce eager execution")
    parser.add_argument(
        "--kv-cache-dtype",
        type=str,
        choices=["auto", "fp8_e5m2"],
        default="auto",
        help=
        'Data type for kv cache storage. If "auto", will use model data type.')
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        choices=["cuda"],
        help='device type for vLLM execution, supporting CUDA only currently.')
    #parser = AsyncEngineArgs.add_cli_args(parser)
    args = parser.parse_args()


    app.add_middleware(
        CORSMiddleware,
        allow_origins=args.allowed_origins,
        allow_credentials=args.allow_credentials,
        allow_methods=args.allowed_methods,
        allow_headers=args.allowed_headers,
    )


    if args.served_model_name is not None:
        served_model = args.served_model_name
    else:
        served_model = args.model

    if 'qwen' in args.served_model_name.lower() or 'qwen' in args.model.lower():
        model_name = 'qwen'
    else:
        model_name = None

    new_llm_args = {
        "model":args.model,
        "served_model_name": args.served_model_name,
        "gpu_memory_utilization": args.gpu_memory_utilization,
        "tensor_parallel_size": args.tensor_parallel_size,
        "dtype": args.dtype,
        "tensor_parallel_size": args.tensor_parallel_size,
        "max_model_len": args.max_model_len,
        "trust_remote_code": args.trust_remote_code
    }

    engine_args = AsyncEngineArgs(**new_llm_args)
    engine = AsyncLLMEngine.from_engine_args(engine_args)
    engine_model_config = asyncio.run(engine.get_model_config())
    max_model_len = engine_model_config.max_model_len

    # A separate tokenizer to map token IDs to strings.
    tokenizer = get_tokenizer(engine_args.tokenizer,
                              tokenizer_mode=engine_args.tokenizer_mode,
                              trust_remote_code=engine_args.trust_remote_code)

    uvicorn.run(app,
                host=args.host,
                port=args.port,
                log_level="info",
                timeout_keep_alive=TIMEOUT_KEEP_ALIVE)