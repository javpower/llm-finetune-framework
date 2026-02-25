"""
API服务模块
提供OpenAI兼容的API接口
"""

import os
import time
import json
from typing import List, Optional, Dict, Any, AsyncGenerator
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import StreamingResponse, JSONResponse
from pydantic import BaseModel, Field
import uvicorn
import logging

logger = logging.getLogger(__name__)


# ==================== 数据模型 ====================


class ChatMessage(BaseModel):
    """聊天消息"""

    role: str
    content: str
    name: Optional[str] = None


class ChatCompletionRequest(BaseModel):
    """聊天完成请求"""

    model: str = "default"
    messages: List[ChatMessage]
    temperature: Optional[float] = 0.7
    top_p: Optional[float] = 0.9
    top_k: Optional[int] = 50
    n: Optional[int] = 1
    max_tokens: Optional[int] = 512
    stop: Optional[List[str]] = None
    stream: Optional[bool] = False
    presence_penalty: Optional[float] = 0.0
    frequency_penalty: Optional[float] = 0.0
    user: Optional[str] = None


class ChatCompletionChoice(BaseModel):
    """聊天完成选项"""

    index: int
    message: ChatMessage
    finish_reason: Optional[str] = None


class ChatCompletionResponse(BaseModel):
    """聊天完成响应"""

    id: str
    object: str = "chat.completion"
    created: int
    model: str
    choices: List[ChatCompletionChoice]
    usage: Dict[str, int]


class ChatCompletionStreamChoice(BaseModel):
    """流式聊天完成选项"""

    index: int
    delta: Dict[str, str]
    finish_reason: Optional[str] = None


class ChatCompletionStreamResponse(BaseModel):
    """流式聊天完成响应"""

    id: str
    object: str = "chat.completion.chunk"
    created: int
    model: str
    choices: List[ChatCompletionStreamChoice]


class ModelInfo(BaseModel):
    """模型信息"""

    id: str
    object: str = "model"
    created: int
    owned_by: str = "custom"


class ModelListResponse(BaseModel):
    """模型列表响应"""

    object: str = "list"
    data: List[ModelInfo]


# ==================== 推理管理器 ====================


class InferenceManager:
    """
    推理管理器
    管理模型加载和推理
    """

    def __init__(self):
        self.engine = None
        self.model_path = None
        self.engine_type = None

    def initialize(self, model_path: str, engine_type: str = "transformers", **kwargs):
        """
        初始化推理引擎

        Args:
            model_path: 模型路径
            engine_type: 引擎类型
            **kwargs: 其他参数
        """
        from ..inference.engine import InferenceEngineFactory

        self.model_path = model_path
        self.engine_type = engine_type

        logger.info(f"初始化推理引擎: {engine_type}")
        logger.info(f"模型路径: {model_path}")

        self.engine = InferenceEngineFactory.create_engine(
            engine_type, model_path, **kwargs
        )

        # 预加载模型
        self.engine.load_model()

        logger.info("推理引擎初始化完成")

    def generate(self, messages: List[Dict[str, str]], **kwargs) -> str:
        """
        生成回复

        Args:
            messages: 消息列表
            **kwargs: 生成参数

        Returns:
            生成的回复
        """
        if self.engine is None:
            raise RuntimeError("推理引擎未初始化")

        # 提取系统提示词
        system_prompt = ""
        chat_history = []

        for msg in messages[:-1]:
            if msg["role"] == "system":
                system_prompt = msg["content"]
            elif msg["role"] == "user":
                chat_history.append([msg["content"], ""])
            elif msg["role"] == "assistant" and chat_history:
                chat_history[-1][1] = msg["content"]

        # 最后一条用户消息
        query = messages[-1]["content"] if messages else ""

        # 生成回复
        return self.engine.generate(
            query,
            history=chat_history if chat_history else None,
            system_prompt=system_prompt,
            **kwargs,
        )

    def stream_generate(
        self, messages: List[Dict[str, str]], **kwargs
    ) -> AsyncGenerator[str, None]:
        """
        流式生成回复

        Args:
            messages: 消息列表
            **kwargs: 生成参数

        Yields:
            生成的token
        """
        if self.engine is None:
            raise RuntimeError("推理引擎未初始化")

        # 提取系统提示词
        system_prompt = ""
        chat_history = []

        for msg in messages[:-1]:
            if msg["role"] == "system":
                system_prompt = msg["content"]
            elif msg["role"] == "user":
                chat_history.append([msg["content"], ""])
            elif msg["role"] == "assistant" and chat_history:
                chat_history[-1][1] = msg["content"]

        # 最后一条用户消息
        query = messages[-1]["content"] if messages else ""

        # 流式生成
        for token in self.engine.stream_generate(
            query,
            history=chat_history if chat_history else None,
            system_prompt=system_prompt,
            **kwargs,
        ):
            yield token


# 全局推理管理器实例
inference_manager = InferenceManager()


# ==================== FastAPI应用 ====================


@asynccontextmanager
async def lifespan(app: FastAPI):
    """应用生命周期管理"""
    # 启动时执行
    logger.info("API服务启动")
    yield
    # 关闭时执行
    logger.info("API服务关闭")


app = FastAPI(
    title="LLM微调框架 API",
    description="OpenAI兼容的大语言模型API服务",
    version="1.0.0",
    lifespan=lifespan,
)


@app.get("/health")
async def health_check():
    """健康检查"""
    return {
        "status": "healthy",
        "model_loaded": inference_manager.engine is not None,
        "model_path": inference_manager.model_path,
    }


@app.get("/v1/models", response_model=ModelListResponse)
async def list_models():
    """获取可用模型列表"""
    models = []

    if inference_manager.model_path:
        models.append(
            ModelInfo(
                id=os.path.basename(inference_manager.model_path),
                created=int(time.time()),
            )
        )

    return ModelListResponse(data=models)


@app.post("/v1/chat/completions")
async def chat_completion(request: ChatCompletionRequest):
    """
    聊天完成接口
    兼容OpenAI API格式
    """
    if inference_manager.engine is None:
        raise HTTPException(status_code=503, detail="模型未加载")

    try:
        # 转换消息格式
        messages = [
            {"role": msg.role, "content": msg.content} for msg in request.messages
        ]

        # 生成参数
        gen_kwargs = {
            "temperature": request.temperature,
            "top_p": request.top_p,
            "max_new_tokens": request.max_tokens,
        }

        if request.stream:
            # 流式响应
            async def generate_stream():
                completion_id = f"chatcmpl-{int(time.time())}"
                created = int(time.time())
                model = (
                    os.path.basename(inference_manager.model_path)
                    if inference_manager.model_path
                    else "default"
                )

                # 发送开始标记
                start_data = {
                    "id": completion_id,
                    "object": "chat.completion.chunk",
                    "created": created,
                    "model": model,
                    "choices": [{"index": 0, "delta": {"role": "assistant"}, "finish_reason": None}]
                }
                yield f"data: {json.dumps(start_data)}\n\n"

                # 流式生成内容
                full_response = ""
                async for token in inference_manager.stream_generate(
                    messages, **gen_kwargs
                ):
                    full_response += token
                    chunk_data = {
                        "id": completion_id,
                        "object": "chat.completion.chunk",
                        "created": created,
                        "model": model,
                        "choices": [{"index": 0, "delta": {"content": token}, "finish_reason": None}]
                    }
                    yield f"data: {json.dumps(chunk_data)}\n\n"

                # 发送结束标记
                end_data = {
                    "id": completion_id,
                    "object": "chat.completion.chunk",
                    "created": created,
                    "model": model,
                    "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]
                }
                yield f"data: {json.dumps(end_data)}\n\n"

                yield "data: [DONE]\n\n"

            return StreamingResponse(generate_stream(), media_type="text/event-stream")

        else:
            # 非流式响应
            response_text = inference_manager.generate(messages, **gen_kwargs)

            completion_id = f"chatcmpl-{int(time.time())}"
            created = int(time.time())
            model = (
                os.path.basename(inference_manager.model_path)
                if inference_manager.model_path
                else "default"
            )

            # 估算token数（简化计算）
            prompt_tokens = sum(len(msg.content) // 4 for msg in request.messages)
            completion_tokens = len(response_text) // 4

            return ChatCompletionResponse(
                id=completion_id,
                created=created,
                model=model,
                choices=[
                    ChatCompletionChoice(
                        index=0,
                        message=ChatMessage(role="assistant", content=response_text),
                        finish_reason="stop",
                    )
                ],
                usage={
                    "prompt_tokens": prompt_tokens,
                    "completion_tokens": completion_tokens,
                    "total_tokens": prompt_tokens + completion_tokens,
                },
            )

    except Exception as e:
        logger.error(f"生成失败: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/v1/completions")
async def text_completion(request: Request):
    """
    文本完成接口（简化版）
    """
    body = await request.json()
    prompt = body.get("prompt", "")

    # 转换为聊天格式
    messages = [{"role": "user", "content": prompt}]

    # 复用聊天完成逻辑
    chat_request = ChatCompletionRequest(
        model=body.get("model", "default"),
        messages=[ChatMessage(role="user", content=prompt)],
        temperature=body.get("temperature", 0.7),
        top_p=body.get("top_p", 0.9),
        max_tokens=body.get("max_tokens", 512),
        stream=body.get("stream", False),
    )

    return await chat_completion(chat_request)


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """全局异常处理"""
    logger.error(f"请求处理失败: {exc}")
    return JSONResponse(
        status_code=500,
        content={"error": {"message": str(exc), "type": "internal_error"}},
    )


# ==================== 服务启动函数 ====================


def start_server(
    model_path: str,
    engine_type: str = "transformers",
    host: str = "0.0.0.0",
    port: int = 8000,
    **kwargs,
):
    """
    启动API服务

    Args:
        model_path: 模型路径
        engine_type: 推理引擎类型
        host: 主机地址
        port: 端口号
        **kwargs: 推理引擎参数
    """
    # 初始化推理管理器
    inference_manager.initialize(model_path, engine_type, **kwargs)

    # 启动服务
    logger.info(f"启动API服务: http://{host}:{port}")
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="LLM API服务")
    parser.add_argument("--model_path", type=str, required=True, help="模型路径")
    parser.add_argument("--engine", type=str, default="transformers", help="推理引擎")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="主机地址")
    parser.add_argument("--port", type=int, default=8000, help="端口号")
    parser.add_argument("--load_in_4bit", action="store_true", help="使用4bit量化")

    args = parser.parse_args()

    start_server(
        model_path=args.model_path,
        engine_type=args.engine,
        host=args.host,
        port=args.port,
        load_in_4bit=args.load_in_4bit,
    )
