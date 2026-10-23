# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""CPU coverage of the frontend boundaries for vllm-project/vllm#58830."""

import logging
from argparse import Namespace
from contextlib import nullcontext
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from vllm.entrypoints.openai.api_server import build_app
from vllm.entrypoints.openai.chat_completion.batch_serving import OpenAIServingChatBatch
from vllm.entrypoints.openai.chat_completion.serving import OpenAIServingChat
from vllm.entrypoints.openai.completion.serving import OpenAIServingCompletion
from vllm.entrypoints.openai.responses.serving import OpenAIServingResponses
from vllm.entrypoints.pooling.classify.serving import ServingClassification
from vllm.entrypoints.pooling.embed.serving import ServingEmbedding
from vllm.entrypoints.pooling.pooling.serving import ServingPooling
from vllm.entrypoints.pooling.scoring.serving import ServingScores
from vllm.entrypoints.serve.render.serving import OpenAIServingRender
from vllm.entrypoints.serve.tokenize.serving import OpenAIServingTokenization

FIELDS = ["mm_processor_kwargs", "media_io_kwargs"]
VALUES = [{"patch_size": 1}, {"video": {"num_frames": 1_000_000}}]
OVERRIDE_VARIANTS: list[dict[str, Any]] = [
    {},
    {f: None for f in FIELDS},
    {f: {} for f in FIELDS},
    dict(zip(FIELDS, VALUES)),
]
CHAT = {"messages": [{"role": "user", "content": "hello"}]}
# The two completion protocols accept these as extra fields; preprocess_cmpl
# forwards mm_processor_kwargs even though it is not a declared protocol field.
ENDPOINTS: list[tuple[str, dict[str, Any]]] = [
    ("/v1/chat/completions", CHAT),
    ("/v1/chat/completions/batch", {"messages": [CHAT["messages"]]}),
    ("/v1/completions", {"prompt": "hello"}),
    ("/v1/responses", {"input": "hello"}),
    ("/v1/chat/completions/render", CHAT),
    ("/v1/completions/render", {"prompt": "hello"}),
    ("/tokenize", CHAT),
    ("/tokenize", {"prompt": "hello"}),
    ("/v1/embeddings", {"input": "hello"}),
    ("/v1/embeddings", CHAT),
    ("/classify", {"input": "hello"}),
    ("/classify", CHAT),
    ("/pooling", {"input": "hello"}),
    ("/pooling", CHAT),
    ("/pooling", {"data": "hello", "task": "plugin"}),
    ("/score", {"queries": "q", "documents": ["d"]}),
    ("/score", {"queries": "q", "items": ["d"]}),
    ("/score", {"data_1": "q", "data_2": "d"}),
    ("/score", {"text_1": "q", "text_2": "d"}),
    ("/v1/score", {"queries": "q", "documents": ["d"]}),
    ("/rerank", {"query": "q", "documents": ["d"]}),
    ("/v1/rerank", {"query": "q", "documents": ["d"]}),
    ("/v2/rerank", {"query": "q", "documents": ["d"]}),
    ("/invocations", CHAT),
    ("/invocations", {"prompt": "hello"}),
]


class ReachedProcessor(Exception):
    """Stop before media loading or inference in a CPU-only boundary test."""


def make_render(trusted=False):
    from tests.entrypoints.openai.chat_completion.test_chat_error import (
        MockModelConfig,
    )

    renderer = MagicMock()
    renderer.render_chat_async = AsyncMock(side_effect=ReachedProcessor)
    renderer.render_cmpl_async = AsyncMock(side_effect=ReachedProcessor)
    renderer.track_mm_cache_items.side_effect = lambda: nullcontext([])
    model_config = MockModelConfig()
    render = OpenAIServingRender(
        model_config=model_config,
        renderer=renderer,
        model_registry=MagicMock(),
        request_logger=None,
        chat_template=None,
        chat_template_content_format="auto",
    )
    # Setting this independently also lets the regression run on the base tree.
    render.trust_request_mm_kwargs = trusted
    render._check_model = AsyncMock(return_value=None)
    return render


def make_serving(cls, render):
    serving = cls.__new__(cls)
    serving.openai_serving_render = render
    serving.renderer = render.renderer
    serving.model_config = render.model_config
    serving.engine_client = MagicMock(errored=False)
    serving.models = MagicMock()
    serving.has_kv_connector = False
    serving._check_model = AsyncMock(return_value=None)
    serving._maybe_get_adapters = MagicMock(return_value=None)
    serving.chat_template = None
    serving.chat_template_content_format = "auto"
    serving.default_chat_template_kwargs = {}
    serving.trust_request_chat_template = False
    serving.reasoning_parser_cls = None
    serving.parser = None
    serving.use_harmony = False
    serving.enable_store = False
    serving._validate_create_responses_input = MagicMock(return_value=None)
    serving.trust_request_mm_kwargs = render.trust_request_mm_kwargs
    serving.io_processor = MagicMock()
    serving.io_processor.create_pooling_params.return_value = None
    serving.get_io_processor = MagicMock(return_value=serving.io_processor)
    serving._validate_request = MagicMock()
    serving._preprocessing_async = AsyncMock(side_effect=ReachedProcessor)
    serving.enable_flash_late_interaction = False
    return serving


def make_app(trusted=False, harmony=False, flash=False, invocation_task=None):
    args = Namespace(
        disable_fastapi_docs=True,
        enable_offline_docs=False,
        root_path="",
        allowed_origins=["*"],
        allow_credentials=False,
        allowed_methods=["*"],
        allowed_headers=["*"],
        api_key=None,
        enable_request_id_headers=False,
        middleware=[],
        log_error_stack=True,
    )
    model_config = MagicMock()
    model_config.get_pooling_task.return_value = invocation_task
    app = build_app(
        args,
        supported_tasks=(invocation_task or "generate",),
        model_config=model_config if invocation_task else None,
    )

    from vllm.entrypoints.pooling.classify.api_router import router as classify
    from vllm.entrypoints.pooling.embed.api_router import router as embed
    from vllm.entrypoints.pooling.pooling.api_router import router as pooling
    from vllm.entrypoints.pooling.scoring.api_router import router as scoring

    for router in (classify, embed, pooling, scoring):
        app.include_router(router)
    render = make_render(trusted)
    render.use_harmony = harmony
    app.state.openai_serving_render = render
    for name, cls in [
        ("openai_serving_chat", OpenAIServingChat),
        ("openai_serving_chat_batch", OpenAIServingChatBatch),
        ("openai_serving_completion", OpenAIServingCompletion),
        ("openai_serving_responses", OpenAIServingResponses),
        ("openai_serving_tokenization", OpenAIServingTokenization),
        ("serving_embedding", ServingEmbedding),
        ("serving_classification", ServingClassification),
        ("serving_pooling", ServingPooling),
        ("serving_scores", ServingScores),
    ]:
        setattr(app.state, name, make_serving(cls, render))
    app.state.openai_serving_responses.use_harmony = harmony
    app.state.serving_scores.enable_flash_late_interaction = flash
    return app


@pytest.mark.parametrize(("path", "payload"), ENDPOINTS)
@pytest.mark.parametrize(("field", "value"), zip(FIELDS, VALUES))
def test_http_rejects_before_processing(path, payload, field, value, caplog):
    app = make_app()
    logger = logging.getLogger("vllm.entrypoints.openai.server_utils")
    logger.addHandler(caplog.handler)
    try:
        # Default raise_server_exceptions=True catches any escaped ASGI error.
        response = TestClient(app).post(
            path, json={"model": "test", **payload, field: value}
        )
    finally:
        logger.removeHandler(caplog.handler)
    assert response.status_code == 400, response.text
    error = response.json()["error"]
    assert error["code"] == 400
    assert error["param"] == field
    assert f"Per-request {field} are disabled" in error["message"]
    assert "--trust-request-mm-kwargs" in error["message"]
    records = [r for r in caplog.records if r.name == logger.name]
    assert len(records) == 1
    assert records[0].levelno == logging.WARNING
    assert records[0].exc_info is None
    render = app.state.openai_serving_render.renderer
    render.render_chat_async.assert_not_called()
    render.render_cmpl_async.assert_not_called()
    app.state.serving_scores._preprocessing_async.assert_not_called()


@pytest.mark.parametrize("kind", ["chat", "batch", "responses"])
@pytest.mark.parametrize(("field", "value"), zip(FIELDS, VALUES))
def test_harmony_also_rejects(kind, field, value):
    path, payload = ENDPOINTS[{"chat": 0, "batch": 1, "responses": 3}[kind]]
    response = TestClient(make_app(harmony=True)).post(
        path, json={"model": "test", **payload, field: value}
    )
    assert response.status_code == 400, response.text
    assert response.json()["error"]["param"] == field


@pytest.mark.parametrize(("field", "value"), zip(FIELDS, VALUES))
def test_flash_late_interaction_rejects_before_preprocessing(field, value):
    app = make_app(flash=True)
    response = TestClient(app).post(
        "/score",
        json={"model": "test", "queries": "q", "documents": ["d"], field: value},
    )
    assert response.status_code == 400, response.text
    app.state.serving_scores._preprocessing_async.assert_not_called()


@pytest.mark.parametrize(("path", "payload"), ENDPOINTS)
@pytest.mark.parametrize(
    "kwargs",
    [{}, {f: None for f in FIELDS}, {f: {} for f in FIELDS}, dict(zip(FIELDS, VALUES))],
)
def test_accepted_requests_reach_unchanged_pipeline(path, payload, kwargs):
    trusted = bool(kwargs.get("mm_processor_kwargs"))
    app = make_app(trusted=trusted)
    if path == "/v1/responses":
        serving = app.state.openai_serving_responses
        serving.msg_store = {}
        serving.parser = None
        serving.chat_template_kwargs = {}
    with pytest.raises(ReachedProcessor):
        TestClient(app).post(path, json={"model": "test", **payload, **kwargs})


@pytest.mark.asyncio
@pytest.mark.parametrize("chat", [False, True])
@pytest.mark.parametrize(
    "kwargs",
    [{}, {f: None for f in FIELDS}, {f: {} for f in FIELDS}, dict(zip(FIELDS, VALUES))],
)
async def test_preprocessing_preserves_kwargs(chat, kwargs):
    from vllm.entrypoints.openai.chat_completion.protocol import ChatCompletionRequest
    from vllm.entrypoints.openai.completion.protocol import CompletionRequest

    render = make_render(trusted=True)
    before = dict(kwargs)
    request = (
        ChatCompletionRequest(**CHAT, **kwargs)
        if chat
        else CompletionRequest(prompt="hello", **kwargs)
    )
    with pytest.raises(ReachedProcessor):
        if chat:
            await render.preprocess_chat(request, CHAT["messages"], None, "auto", {})
        else:
            await render.preprocess_completion(request, "hello", None)
    method = (
        render.renderer.render_chat_async if chat else render.renderer.render_cmpl_async
    )
    extras = method.call_args.kwargs["prompt_extras"]
    assert extras == (
        {"mm_processor_kwargs": kwargs["mm_processor_kwargs"]}
        if kwargs.get("mm_processor_kwargs") is not None
        else {}
    )
    if chat:
        params = method.call_args.args[1]
        expected = (
            {"video": {"num_frames": 1_000_000}}
            if kwargs.get("media_io_kwargs")
            else None
        )
        assert params.media_io_kwargs == expected
    assert kwargs == before


@pytest.mark.asyncio
@pytest.mark.parametrize("index", [0, 8, 15, 20])
@pytest.mark.parametrize(("field", "value"), zip(FIELDS, VALUES))
async def test_run_batch_returns_400(index, field, value):
    from vllm.entrypoints.openai.run_batch import (
        BatchProgressTracker,
        BatchRequestInput,
        run_request,
    )

    app = make_app()
    path, payload = ENDPOINTS[index]
    handler = {
        0: app.state.openai_serving_chat.create_chat_completion,
        8: app.state.serving_embedding,
        15: app.state.serving_scores,
        20: app.state.serving_scores,
    }[index]
    request = BatchRequestInput(
        custom_id="test",
        method="POST",
        url=path,
        body={"model": "test", **payload, field: value},
    )
    result = await run_request(handler, request, BatchProgressTracker())
    assert result.response.status_code == 400
    assert result.error.error.param == field


def test_trust_flag_is_frontend_only(tmp_path):
    from dataclasses import asdict, fields

    from vllm.config import CompilationConfig, MultiModalConfig, VllmConfig
    from vllm.engine.arg_utils import EngineArgs
    from vllm.entrypoints.openai.cli_args import make_arg_parser
    from vllm.utils.argparse_utils import FlexibleArgumentParser

    parser = make_arg_parser(FlexibleArgumentParser())
    default = parser.parse_args(["--model", str(tmp_path)])
    trusted = parser.parse_args(["--model", str(tmp_path), "--trust-request-mm-kwargs"])
    assert asdict(EngineArgs.from_cli_args(default)) == asdict(
        EngineArgs.from_cli_args(trusted)
    )
    for cls in (EngineArgs, MultiModalConfig, CompilationConfig, VllmConfig):
        assert "trust_request_mm_kwargs" not in {field.name for field in fields(cls)}


@pytest.mark.asyncio
@pytest.mark.parametrize(("field", "value"), zip(FIELDS, VALUES))
async def test_responses_next_turn_rejects(field, value):
    from vllm.entrypoints.openai.responses.protocol import ResponsesRequest
    from vllm.exceptions import VLLMValidationError

    serving = make_serving(OpenAIServingResponses, make_render())
    serving.chat_template_kwargs = {}
    request = ResponsesRequest(input="hello", **{field: value})
    with pytest.raises(VLLMValidationError, match=f"Per-request {field} are disabled"):
        await serving._render_next_turn(request, [], None, None, None, "auto")


@pytest.mark.parametrize(
    ("mm", "media", "parameter"),
    [
        ({"use_audio_in_video": False}, None, "mm_processor_kwargs"),
        (None, {"video": {}}, "media_io_kwargs"),
        ({"size": None}, {"video": {}}, "mm_processor_kwargs"),
    ],
)
def test_nonempty_means_top_level_dict(mm, media, parameter):
    from vllm.entrypoints.utils import validate_request_mm_kwargs
    from vllm.exceptions import VLLMValidationError

    with pytest.raises(VLLMValidationError) as error:
        validate_request_mm_kwargs(
            mm_processor_kwargs=mm, media_io_kwargs=media, trust_request_mm_kwargs=False
        )
    assert error.value.parameter == parameter


@pytest.mark.asyncio
@pytest.mark.parametrize("trusted", [False, True])
async def test_frontend_initializers_propagate_trust(trusted):
    from starlette.datastructures import State

    from vllm.entrypoints.openai.api_server import init_app_state, init_render_app_state
    from vllm.entrypoints.openai.cli_args import make_arg_parser
    from vllm.entrypoints.pooling.factories import init_pooling_state
    from vllm.utils.argparse_utils import FlexibleArgumentParser

    args = make_arg_parser(FlexibleArgumentParser()).parse_args([])
    args.trust_request_mm_kwargs = trusted
    engine = MagicMock()
    engine.vllm_config.lora_config = None
    engine.model_config.get_pooling_task.return_value = "embed"
    models = MagicMock()
    models.init_static_loras = AsyncMock()
    with (
        patch(
            "vllm.entrypoints.openai.api_server.OpenAIServingModels",
            return_value=models,
        ),
        patch("vllm.entrypoints.serve.tokenize.serving.OpenAIServingTokenization"),
        patch(
            "vllm.sm70_profiles.acceleration.collect_worker_reports", new=AsyncMock()
        ),
        patch("vllm.entrypoints.openai.api_server.OpenAIServingRender") as render,
    ):
        await init_app_state(engine, State(), args, supported_tasks=())
        assert render.call_args.kwargs["trust_request_mm_kwargs"] is trusted
    with (
        patch("vllm.renderers.renderer_from_config"),
        patch("vllm.entrypoints.serve.render.serving.OpenAIServingRender") as render,
    ):
        await init_render_app_state(engine.vllm_config, State(), args)
        assert render.call_args.kwargs["trust_request_mm_kwargs"] is trusted
    with (
        patch(
            "vllm.entrypoints.pooling.factories.enable_scoring_api", return_value=False
        ),
        patch("vllm.entrypoints.pooling.pooling.serving.ServingPooling"),
        patch("vllm.entrypoints.pooling.embed.serving.ServingEmbedding") as embed,
    ):
        state = State()
        state.openai_serving_models = models
        init_pooling_state(engine, state, args, None, ("embed",))
        config = embed.call_args.kwargs["chat_template_config"]
        assert config.trust_request_mm_kwargs is trusted


@pytest.mark.parametrize("trusted", [False, True])
def test_serving_constructors_use_trust(trusted):
    from vllm.entrypoints.chat_utils import ChatTemplateConfig
    from vllm.entrypoints.pooling.base.serving import PoolingServingBase

    render = make_render()
    kwargs = dict(
        model_config=render.model_config,
        renderer=render.renderer,
        model_registry=MagicMock(),
        request_logger=None,
        chat_template=None,
        chat_template_content_format="auto",
        trust_request_mm_kwargs=trusted,
    )
    assert OpenAIServingRender(**kwargs).trust_request_mm_kwargs is trusted
    serving = ServingEmbedding.__new__(ServingEmbedding)
    models = MagicMock(model_config=render.model_config)
    PoolingServingBase.__init__(
        serving,
        MagicMock(),
        models,
        request_logger=None,
        chat_template_config=ChatTemplateConfig(trust_request_mm_kwargs=trusted),
    )
    assert serving.trust_request_mm_kwargs is trusted


@pytest.mark.parametrize(("field", "value"), zip(FIELDS, VALUES))
def test_cohere_ignores_unsupported_overrides(field, value):
    app = make_app()
    with pytest.raises(ReachedProcessor):
        TestClient(app).post(
            "/v2/embed", json={"model": "test", "texts": ["hello"], field: value}
        )
    ctx = app.state.serving_embedding._preprocessing_async.call_args.args[1]
    assert not hasattr(ctx.request, field)


@pytest.mark.parametrize("index", [8, 10, 12, 14, 15, 20])
@pytest.mark.parametrize(("field", "value"), zip(FIELDS, VALUES))
def test_sagemaker_pooling_rejects(index, field, value):
    app = make_app(invocation_task="classify" if index == 10 else "embed")
    _, payload = ENDPOINTS[index]
    response = TestClient(app).post(
        "/invocations", json={"model": "test", **payload, field: value}
    )
    assert response.status_code == 400, response.text
    assert response.json()["error"]["param"] == field


def make_demo(monkeypatch, trusted):
    from vllm.entrypoints import api_server as demo

    async def generate(prompt, params, request_id):
        yield SimpleNamespace(prompt="hello", outputs=[SimpleNamespace(text="ok")])

    engine = MagicMock()
    engine.generate.side_effect = generate
    monkeypatch.setattr(demo, "engine", engine)
    monkeypatch.setattr(
        demo.app.state, "args", Namespace(log_error_stack=True), raising=False
    )
    monkeypatch.setattr(
        demo.app.state, "trust_request_mm_kwargs", trusted, raising=False
    )
    return TestClient(demo.app), engine


@pytest.mark.parametrize("encoder_decoder", [False, True])
@pytest.mark.parametrize(("field", "value"), zip(FIELDS, VALUES))
def test_demo_raw_prompt_rejects(monkeypatch, encoder_decoder, field, value):
    client, engine = make_demo(monkeypatch, False)
    prompt = {"prompt": "hello", field: value}
    if encoder_decoder:
        prompt = {"encoder_prompt": prompt, "decoder_prompt": "hello"}
    response = client.post("/generate", json={"prompt": prompt})
    assert response.status_code == 400, response.text
    assert response.json()["error"]["param"] == field
    assert "--trust-request-mm-kwargs" in response.json()["error"]["message"]
    engine.generate.assert_not_called()


@pytest.mark.parametrize("encoder_decoder", [False, True])
@pytest.mark.parametrize(
    ("trusted", "kwargs"),
    [
        (trusted, kwargs)
        for trusted in (False, True)
        for kwargs in OVERRIDE_VARIANTS
        if trusted or not any(kwargs.values())
    ],
)
def test_demo_preserves_prompt_overrides(monkeypatch, trusted, kwargs, encoder_decoder):
    client, engine = make_demo(monkeypatch, trusted)
    prompt = {"prompt": "hello", **kwargs}
    if encoder_decoder:
        prompt = {"encoder_prompt": prompt, "decoder_prompt": "hello"}
    response = client.post("/generate", json={"prompt": prompt})
    assert response.status_code == 200
    assert response.json() == {"text": ["hellook"]}
    assert engine.generate.call_args.args[0] == prompt


def test_demo_cli_exposes_trust_flag():
    import subprocess
    import sys

    result = subprocess.run(
        [sys.executable, "-m", "vllm.entrypoints.api_server", "--help"],
        capture_output=True,
        text=True,
        check=True,
    )
    assert "--trust-request-mm-kwargs" in result.stdout
    assert "--no-trust-request-mm-kwargs" in result.stdout


@pytest.mark.parametrize("trusted", [False, True])
def test_demo_keeps_engine_errors_unchanged(monkeypatch, trusted):
    from vllm.exceptions import VLLMValidationError

    client, engine = make_demo(monkeypatch, trusted)
    engine.generate.side_effect = VLLMValidationError("existing engine failure")
    with pytest.raises(VLLMValidationError, match="existing engine failure"):
        client.post("/generate", json={"prompt": {"prompt": "hello"}})
