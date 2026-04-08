import os.path
import uvicorn
from fastapi import FastAPI, UploadFile, Depends, BackgroundTasks
from starlette.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from knowledge.core.deps import get_upload_file_service
from knowledge.core.paths import get_front_page_dir
from knowledge.schema.upload_schema import UploadResponse, TaskStatusResponse
from knowledge.service.upload_service import UploadSeries
from knowledge.utils.task_utils import get_task_info


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
        app.mount("/front", StaticFiles(directory=page_dir))

    # 4、注册路由：将上传请求 以及 查询导入任务的请求注册到fastapi上
    register_router(app)

    # 5、返回fastapi实例
    return app

def register_router(app:FastAPI):
    @app.post("/upload",response_model=UploadResponse)
    def upload_endpoint(file: UploadFile,
                        background_tasks: BackgroundTasks,
                        # Depends：可依赖项，是一个函数 或者 一个类的实例，这里是一个类的实例，在依赖项中，使用单例模式，防止多次创建对象
                        upload_series: UploadSeries = Depends(get_upload_file_service),
                        ):
        """
        处理文件的上传: 上传文件的原始的名字
        Returns:
        """
        # 1、将上传的文件，写入到本地的临时目录中，保存到minio中
        task_id,import_file_path,file_dir = upload_series.process_upload_file(file)

        # 2、运行整个导入的图谱，使用background_tasks.add_task 来添加后台任务（整个阶段节点多，比较耗时，放在后台，慢慢做）
        background_tasks.add_task(upload_series.run_import_graph,task_id,import_file_path,file_dir)

        # 3、返回上传后的响应的数据类型
        return UploadResponse(message=f"{file.filename}文件上传成功",task_id=task_id)

    @app.get("/status/{task_id}")
    def get_task_status_endpoint(task_id: str):
        """
        处理查询导入任务: 前端会轮询的调用查询上传任务的状态结构（1.5s 轮询）
        轮询1.5s：1、性能---》相比极短的时间轮询，性能高   2、实时性--》不高
        Returns:
        """
        task_info = get_task_info(task_id)

        return TaskStatusResponse(**task_info)



if __name__ == "__main__":
    # 3、利用uvicorn服务启动fastapi
    uvicorn.run(app=create_app(), host="127.0.0.1", port=8000, log_level="info")