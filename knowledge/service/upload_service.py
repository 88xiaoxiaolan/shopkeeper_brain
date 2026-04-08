from datetime import datetime
import logging
import os.path
import shutil
import time
import uuid
from typing import Tuple

from fastapi import UploadFile
from knowledge.core.paths import get_local_base_dir
from knowledge.processor.import_processor.exceptions import FileProcessingError
from knowledge.processor.import_processor.main_graph import import_app
from knowledge.utils.client.storage_clients import StorageClients
from knowledge.utils.task_utils import update_task_status, TASK_STATUS_PROCESSING, TASK_STATUS_COMPLETED, \
    TASK_STATUS_FAILED, add_running_task, add_done_task, add_node_duration

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class UploadSeries:
    def get_base_dir(self):
        # %Y % m % d : 年月日  小写的y，年份只有两位
        return os.path.join(get_local_base_dir(), datetime.now().strftime("%Y%m%d"))

    """
    处理文件上传相关的逻辑
    """
    def run_import_graph(self,task_id,import_file_path,file_dir):
        # 更新任务状态为处理中
        update_task_status(task_id,TASK_STATUS_PROCESSING)

        init_state = {
            "task_id": task_id,
            "import_file_path": import_file_path,
            "file_dir": file_dir
        }

        try:
            for event in import_app.stream(init_state):
                for node_name, node_state in event.items():
                    logger.info(f"当前正在执行的节点：{node_name}")
            # 更新任务状态为完成
            update_task_status(task_id, TASK_STATUS_COMPLETED)
        except Exception as e:
            logger.error(f"{task_id} 执行导入过程中出现异常,{str(e)}")
            # 更新任务状态为失败
            update_task_status(task_id, TASK_STATUS_FAILED)

    def process_upload_file(self, file: UploadFile) -> Tuple[str,str,str]:
        """
        处理文件上传
        1、将上传的文件保存到临时目录（为了做中转）
        2、将文件上传到minio，（为了持久化）
        3、将file_dir / import_file_path / task_id 返回
        Args:
            file:

        Returns:
        """
        # 1、真正的随机,获取前8个随机数
        task_id = str(uuid.uuid4().hex[:8])

        add_running_task(task_id,"upload_file")
        start_time = time.time()


        # 2、生成日期，把temp_data 与 日期拼接
        base_file_dir = self.get_base_dir()

        # 3、构建完整的文档归属目录
        file_dir = os.path.join(base_file_dir, task_id)

        # 4、保存文件到临时目录
        import_file_path = self.save_upload_file_to_dir(file, file_dir)

        # 5、保存文件到minio
        self.save_upload_file_to_minio(import_file_path,file.filename)

        add_done_task(task_id, "upload_file")
        end_time = time.time()
        add_node_duration(task_id, "upload_file", end_time - start_time)

        # 6、返回图谱的信息
        return task_id,import_file_path,file_dir

    def save_upload_file_to_dir(self, file:UploadFile, file_dir:str) -> str:
        """
        保存文件到临时目录
        Args:
            file: 文件上传对象
            file_dir: 上传文件的目录

        Returns:
        """

        # 1、创建文件的归属目录
        os.makedirs(file_dir,exist_ok=True)

        # 2、构建导入文件的path
        import_file_path = os.path.join(file_dir,file.filename)

        # 3、写入
        try:
            with open(import_file_path,"wb") as f:
                # 不同的操作系统，以及不同python版本，都可以分批次写入（windows以及3.7以上的sdk版本，写1M）
               shutil.copyfileobj(file.file,f)
        except IOError as e:
            logger.info(f"{file.filename}写入临时目录失败：{str(e)}")
            raise FileProcessingError(message=f"{file.filename}写入临时目录失败：{str(e)}")

        # 4、返回导入的文件路径
        return import_file_path

    def save_upload_file_to_minio(self, import_file_path:str, filename:str):
        """

        Args:
            import_file_path: 上传文件的地址
            filename: 上传文件的名字

        Returns:
        """

        # 1、获取minio客户端
        try:
            minio_client = StorageClients.get_minio_client()
        except ConnectionError as e:
            logger.error(f"获取minio地址失败：{str(e)}")
            return

        # 2、获取minio相关的信息
        bucket_name = os.getenv("MINIO_BUCKET_NAME")
        bucket_object_name = f"origin_files/{datetime.now().strftime('%Y%m%d')}/{filename}"

        # 3、上传
        try:
            minio_client.fput_object(bucket_name,bucket_object_name,import_file_path)
        except Exception as e:
            logger.error(f"上传文件到minio失败：{str(e)}")
