import threading
from typing import Optional

import torch
from langchain_openai import ChatOpenAI
from openai import OpenAI
from dotenv import load_dotenv
from pymilvus.model.hybrid import BGEM3EmbeddingFunction
from FlagEmbedding import FlagReranker

from knowledge.utils.client.base import BaseClientManager, logger

load_dotenv()


class AIClients(BaseClientManager):
    """AI 模型类客户端"""

    _openai_client: Optional[OpenAI] = None
    _openai_lock = threading.Lock()

    _openai_llm_response_text_client: Optional[ChatOpenAI] = None
    _openai_llm_response_text_lock = threading.Lock()

    _openai_llm_response_json_client: Optional[ChatOpenAI] = None
    _openai_llm_response_json_lock = threading.Lock()

    _bge_m3_client: Optional[BGEM3EmbeddingFunction] = None
    _bge_m3_lock = threading.Lock()

    _bge_m3_reranker_client: Optional[FlagReranker] = None
    _bge_m3_reranker_lock = threading.Lock()


    # ── LLM
    @classmethod
    def get_vlm_client(cls) -> OpenAI:
        return cls._get_or_create("_openai_client", cls._openai_lock, cls._create_vlm_client)

    @classmethod
    def _create_vlm_client(cls) -> OpenAI:
        try:
            api_key = cls._require_env("OPENAI_API_KEY")
            base_url = cls._require_env("OPENAI_API_BASE")

            client = OpenAI(api_key=api_key, base_url=base_url)
            logger.info(f"OpenAI 客户端初始化成功 (base_url={base_url})")
            return client

        except EnvironmentError:
            raise
        except Exception as e:
            logger.error(f"OpenAI 客户端创建失败: {e}")
            raise ConnectionError(f"OpenAI 连接失败: {e}") from e

    # LLM
    @classmethod
    def get_llm_client(cls,response_format:bool=True) -> ChatOpenAI:
        if response_format:
            return cls._get_or_create("_openai_llm_response_json_client", cls._openai_llm_response_json_lock, lambda: cls._create_llm_json_client(response_format))
        else:
            # lambda 它也不会立马执行，返回的还是一个函数，在父类中factory执行的时候，才会执行, 使用lambda，只是为了传参
            return cls._get_or_create("_openai_llm_response_text_client", cls._openai_llm_response_text_lock, cls._create_llm_client)

    # LLM - text
    @classmethod
    def _create_llm_client(cls) -> ChatOpenAI:
        try:
            api_key = cls._require_env("OPENAI_API_KEY")
            base_url = cls._require_env("OPENAI_API_BASE")
            model_name = cls._require_env("LLM_DEFAULT_MODEL")

            llm_client = ChatOpenAI(
                model_name=model_name,
                temperature=0,
                # openai_api_key=api_key,
                # openai_api_base=base_url,
                api_key=api_key,
                base_url=base_url,
                # 加了这个，此时响应的格式，是纯真的json格式，不是代码块的json格式
            )

            logger.info(f"ChatOpenAI 客户端初始化成功 (base_url={base_url})")
            return llm_client

        except EnvironmentError:
            raise
        except Exception as e:
            logger.error(f"ChatOpenAI 客户端创建失败: {e}")
            raise ConnectionError(f"ChatOpenAI 连接失败: {e}") from e

    # LLM - json
    @classmethod
    def _create_llm_json_client(cls, response_format) -> ChatOpenAI:
        try:
            api_key = cls._require_env("OPENAI_API_KEY")
            base_url = cls._require_env("OPENAI_API_BASE")
            model_name = cls._require_env("LLM_DEFAULT_MODEL")

            model_kwargs = {}
            if response_format:
                model_kwargs["response_format"] = {'type': 'json_object'}

            llm_client = ChatOpenAI(
                model_name=model_name,
                temperature=0,
                # openai_api_key=api_key,
                # openai_api_base=base_url,
                api_key=api_key,
                base_url=base_url,
                # 加了这个，此时响应的格式，是纯真的json格式，不是代码块的json格式
                model_kwargs=model_kwargs
            )

            logger.info(f"ChatOpenAI 客户端初始化成功 (base_url={base_url})")
            return llm_client

        except EnvironmentError:
            raise
        except Exception as e:
            logger.error(f"ChatOpenAI 客户端创建失败: {e}")
            raise ConnectionError(f"ChatOpenAI 连接失败: {e}") from e

    # bge-m3  嵌入模型
    @classmethod
    def get_bge_m3_client(cls) -> BGEM3EmbeddingFunction:
        return cls._get_or_create("_bge_m3_client", cls._bge_m3_lock,cls._create_bge_m3_client)

    @classmethod
    def _create_bge_m3_client(cls) -> BGEM3EmbeddingFunction:
        try:
            model_name = cls._require_env("BGE_M3_PATH")
            device = cls._require_env("BGE_DEVICE")
            use_fp16 = cls._require_env("BGE_FP16")

            fp16 = True if use_fp16.lower() in ("true","1") else False

            bge_m3_ef = BGEM3EmbeddingFunction(
                model_name=model_name,  # Specify the model name
                device=device,  # Specify the device to use, e.g., 'cpu' or 'cuda:0'
                use_fp16=fp16  # Specify whether to use fp16. Set to `False` if `device` is `cpu`.
            )

            logger.info(f"bge_m3 客户端初始化成功")
            return bge_m3_ef

        except EnvironmentError:
            raise
        except Exception as e:
            logger.error(f"bge_m3 客户端创建失败: {e}")
            raise ConnectionError(f"bge_m3 连接失败: {e}") from e

    # bge-m3-reranker 模型
    @classmethod
    def get_bge_m3_reranker_client(cls) -> BGEM3EmbeddingFunction:
        return cls._get_or_create("_bge_m3_reranker_client", cls._bge_m3_reranker_lock, cls._create_bge_m3_reranker_client)

    @classmethod
    def _create_bge_m3_reranker_client(cls) -> BGEM3EmbeddingFunction:
        try:
            model_name = cls._require_env("BGE_RERANKER_LARGE")
            device = cls._require_env("BGE_DEVICE")
            use_fp16 = cls._require_env("BGE_FP16")

            fp16 = True if use_fp16.lower() in ("true", "1") else False

            reranker = FlagReranker(
                model_name_or_path=model_name,  # Specify the model name
                device=device,  # Specify the device to use, e.g., 'cpu' or 'cuda:0'
                use_fp16=fp16  # Specify whether to use fp16. Set to `False` if `device` is `cpu`.
            )

            logger.info(f"bge_m3_reranker 客户端初始化成功")
            return reranker

        except EnvironmentError:
            raise
        except Exception as e:
            logger.error(f"bge_m3_reranker 客户端创建失败: {e}")
            raise ConnectionError(f"bge_m3_reranker 连接失败: {e}") from e

if __name__ == "__main__":
    import json
    llm_client = AIClients.get_llm_client()
    res = llm_client.invoke("请给我说一个笑话，模式为json")

    # ```json
    # {
    #     "joke": "有一天，小明去面试，老板问他：‘你有什么特长？’\n小明回答：‘我会预测未来。’\n老板笑了笑：‘那你预测一下，我什么时候会辞职？’\n小明淡定地说：‘这个嘛……我预测不到，因为我还没被录用。’"
    # }
    # ```
    # res.content
    # 只有response_format=True的时候，才不会报错 json.loads,把json -》str
    result = json.loads(res.content)
    print("res.content===",res.content)
    print("res===",res)