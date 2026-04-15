import asyncio
import os.path
from typing import Union

import uvicorn
from fastapi import FastAPI, Depends, BackgroundTasks, Request, HTTPException
from starlette.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from knowledge.core.deps import get_query_service
from knowledge.core.paths import get_front_page_dir
from knowledge.schema.query_schema import QueryRequest, QueryResponse, StreamSubmitResponse
from knowledge.service.query_service import QueryService
from knowledge.utils.sse_util import create_sse_queue, sse_generator
from knowledge.utils.task_utils import get_task_result
from fastapi.responses import StreamingResponse



def create_app():
    # 1、创建fastapi实例
    app = FastAPI(description="掌柜智库导入的应用",version="v1.0")

    # 2、跨域配置
    app.add_middleware(
        CORSMiddleware,
        # 允许所有的源，也可以指定一些源
        allow_origins=["*"],  # ← 和 credentials=True 冲突
        allow_credentials=False, # 自定义 cookies Authorization tsl 客户端证书信息；如果这个为ture，如果有黑客伪造证书，你的其他都是任意访问的，那么很容易获取到你的数据
        allow_methods=["*"],  # ← 和 credentials=True 冲突 get(获取) post(新增) delete(删除) put(更新)
        allow_headers=["*"],  # ← 和 credentials=True 冲突 自定义的头字段 token content-type:application/json
    )

    # 3、挂载静态文件（import.html）
    page_dir = get_front_page_dir()
    if page_dir and os.path.exists(page_dir):
        # 只要在浏览器中输入/front,那么就会到page_dir目录下寻找静态资源文件
        app.mount("/chat", StaticFiles(directory=page_dir))

    # 4、注册路由：将上传请求 以及 查询导入任务的请求注册到fastapi上
    register_router(app)

    # 5、返回fastapi实例
    return app

def register_router(app:FastAPI):
    @app.get("/")
    def hell_world():
        return {"test":123}

    @app.post("/query",response_model=Union[QueryResponse, StreamSubmitResponse])
    async def query(request: QueryRequest,
              background_tasks: BackgroundTasks,
              # Depends：可依赖项，是一个函数 或者 一个类的实例，这里是一个类的实例，在依赖项中，使用单例模式，防止多次创建对象
              query_series: QueryService = Depends(get_query_service)) -> Union[QueryResponse, StreamSubmitResponse]:  #fastapi会自动将json数据转为指定的schema
        """
        处理查询请求
        Args:
            request: 前端发送的请求数据
        Returns:
        """

        # 1、获取session_id
        session_id = request.session_id  if request.session_id else QueryService.generate_session_id()

        # 2、获取任务id
        task_id = QueryService.generate_task_id()

        # 3、判断是否式流式
        if request.is_stream:
            # 3.1 创建sse队列(容器中的任务id和队列的对应关系)
            create_sse_queue(task_id=task_id)

            # 3.2 启动查询任务,是后台任务
            background_tasks.add_task(query_series.run_query_graph, session_id, task_id, request.query,request.is_stream)

            return StreamSubmitResponse(message="查询请求已经提交", session_id=session_id, task_id=task_id)
        else:
            # 事件循环: 每次来了请求,都会执行,遇到await，不会等你，去处理其他请求，await下面的代码不会执行，等到await这里完成(如果不写await，就会阻塞),会告诉事件循环,然后继续往下执行
            # 流式调用也很慢,不能直接调用pipline,否则会阻塞,所以用线程池执行查询任务,跟background_tasks.add_task()效果一致
            # 当前事件循环对象(uvicorn)
            loop = asyncio.get_event_loop()
            # 在线程池中执行查询任务,不写用uvicorn的内部的,也可以自己定义
            await loop.run_in_executor(None, query_series.run_query_graph, session_id, task_id, request.query,request.is_stream)

            # 路由层拿不到state，从_tasks_result中拿答案
            return QueryResponse(message="查询请求已经完成", session_id=session_id, answer=query_series.get_task_result(task_id=task_id))

    @app.get("/stream/{task_id}")
    async def stream(task_id:str,request: Request) -> StreamingResponse:
        """
        返回sse的数据包：流式 + yield使用
        如何返回：利用生成器yield返回SSE数据包
        注意事项："event:自定义\ndata:自定义\n\n"
        Args:
            task_id:

        Returns:
        """

        # 将后端组合的sse协议格式的数据，返回给前端
        # content：异步可迭代对象，生成器也是可以跌代的
        return StreamingResponse(content=sse_generator(task_id,request),media_type="text/event-stream")


    @app.get("/history/{session_id}")
    async def get_history(
            session_id: str, limit: int = 50,
            service: QueryService = Depends(get_query_service),
    ):
        try:
            items = service.get_history(session_id, limit)
            return {"session_id": session_id, "items": items}
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"history error: {e}")

    @app.delete("/history/{session_id}")
    async def clear_chat_history(
            session_id: str,
            service: QueryService = Depends(get_query_service),
    ):
        count = service.clear_history(session_id)
        return {"message": "History cleared", "deleted_count": count}

if __name__ == "__main__":
    # 3、利用uvicorn服务启动fastapi
    uvicorn.run(app=create_app(), host="127.0.0.1", port=8001, log_level="info")