import asyncio
import json
import logging
import queue
from queue import Empty
from typing import Dict, Any, Optional, AsyncGenerator
from fastapi import Request


class SSEEvent:
    PROGRESS = "progress"   # 任务节点进度
    DELTA = "delta"         # LLM 流式输出增量
    FINAL = "final"         # 最终完整答案


# 全局 SSE 任务队列存储
# Key: task_id, Value: queue.Queue
_task_stream: Dict[str, queue.Queue] = {}


def get_sse_queue(task_id: str) -> Optional[queue.Queue]:
    """获取指定任务的队列"""
    return _task_stream.get(task_id)


def create_sse_queue(task_id: str) -> queue.Queue:
    """创建并注册一个新的 SSE 队列"""
    q = queue.Queue()
    _task_stream[task_id] = q
    return q

def remove_sse_queue(task_id: str):
    """移除指定任务的队列
    不存在 key 默认返回 None
    """
    _task_stream.pop(task_id, None)


def _sse_pack(event: str, data: Dict[str, Any]) -> str:
    """打包 SSE 消息格式"""
    payload = json.dumps(data, ensure_ascii=False)
    return f"event: {event}\ndata: {payload}\n\n"


def push_sse_event (task_id: str, event: str, data: Dict[str, Any]):
    """
    通过 task_id 推送事件到 SSE 队列
    """
    # 1. 获取 SSE 队列
    stream_queue = get_sse_queue(task_id)

    # 2. 队列存在
    if stream_queue:
        # 3. 将事件推送到队列
        stream_queue.put({"event": event, "data": data})


async def sse_generator(task_id: str,request: Request) -> AsyncGenerator:
    """
    流式输出结果的消费者
    从SSE队列中获取结果
    封装队列中的数据以及事件类型为SSE协议的数据包格式
    封装好的数据包yield出去
    Args:
        task_id:

    Returns:异步生成器对象

    """

    # 1、根据task_id 获取任务的队列
    sse_queue = _task_stream.get(task_id)

    if sse_queue is None:
        return

    loop = asyncio.get_event_loop()

    # 2、让当前线程一直从队列中获取数据
    try:
        while True:
            try:
                # 判断前端SSE是否已经断开，如果已经断开，则终止  -》 可以通过fastapi的request来感知
                if await request.is_disconnected():
                    return

                # sse_queue.get：阻塞队列
                msg = await loop.run_in_executor(None, sse_queue.get, True,1)
                event = msg.get("event")
                data = msg.get("data")

                # 打包，通过yield返回
                yield _sse_pack(event, data)

            except Empty: #sse_queue.get，由于超出时间，会报错（Empty的error），所以我们在这里捕获异常，继续执行下一次循环
                logging.info("SSE 队列已空, 请等待一下")
                continue
    except (ConnectionResetError,BrokenPipeError) as e:
        # 服务端还在推送数据，客户端关闭了窗口或者浏览器，此时会报错
        return

    except asyncio.CancelledError:
        # 服务端中断，协程被取消，把异常抛给调用方
        remove_sse_queue(task_id)
        raise

    finally:
        remove_sse_queue(task_id)
