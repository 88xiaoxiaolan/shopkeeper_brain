import logging
import uuid
from typing import List, Dict, Any

from knowledge.processor.query_processor.main_graph import query_app
from knowledge.utils.mongo_history_util import get_recent_messages, clear_history
from knowledge.utils.task_utils import update_task_status, TASK_STATUS_PROCESSING, TASK_STATUS_FAILED, \
    TASK_STATUS_COMPLETED, get_task_result

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class QueryService:
    @staticmethod
    def generate_session_id() -> str:
        return str(uuid.uuid4())

    @staticmethod
    def generate_task_id() -> str:
        return str(uuid.uuid4().hex[:12])

    def run_query_graph(self, session_id: str, task_id: str, query: str, is_stream: bool):
        """
        Args:
            session_id: 会话id
            task_id: 任务id
            query: 问题
            is_stream: 是否式流式

        Returns:

        """
        # 修改任务状态为正在执行
        update_task_status(task_id=task_id, status_name=TASK_STATUS_PROCESSING)

        # 1、构建查询初始化参数
        query_init_state = {
            "session_id": session_id,
            "task_id": task_id,
            "original_query": query,
            "is_stream": is_stream,
        }

        try:
            # 2、运行查询状态的pipeline
            query_app.invoke(query_init_state)
            # 3、修改任务状态为完成
            update_task_status(task_id=task_id, status_name=TASK_STATUS_COMPLETED)
        except Exception as e:
            logger.error(f"session_id:{session_id}，task_id:{task_id} 运行查询流程，出现了异常,{str(e)}")
            # 4、修改任务状态为失败
            update_task_status(task_id=task_id, status_name=TASK_STATUS_FAILED)

    def get_task_result(self,task_id:str) -> str:
        return get_task_result(task_id=task_id, key="answer")

    def get_history(self, session_id: str, limit: int = 50) -> List[Dict[str, Any]]:

        # 1. 根据session_id获取最近的指定条数的历史对话
        records = get_recent_messages(session_id, limit=limit)
        return [
            {
                "_id": str(r.get("_id", "")),
                "session_id": r.get("session_id", ""),
                "role": r.get("role", ""),
                "text": r.get("text", ""),
                "rewritten_query": r.get("rewritten_query", ""),
                "item_names": r.get("item_names", []),
                "ts": r.get("ts"),
            }
            for r in records
        ]

    def clear_history(self, session_id: str) -> int:
        return clear_history(session_id)