"""
导入流程节点基类

定义统一的节点接口规范，提供通用功能
"""
import json
import os
import time
from abc import ABC, abstractmethod
from typing import TypeVar, Optional, Any, Dict
import logging

from knowledge.processor.import_processor.config import ImportConfig, get_config
from knowledge.processor.import_processor.exceptions import ImportProcessError
from knowledge.utils.task_utils import add_running_task, add_done_task, add_node_duration

T = TypeVar("T")  # 泛型状态类型


class BaseNode(ABC):
    """
    导入流程节点基类

    所有节点类都应继承此基类，实现 process 方法。
    基类提供统一的日志、任务追踪和错误处理。

    使用示例:
        class MyNode(BaseNode):
            name = "my_node"

            def process(self, state):
                # 实现具体逻辑
                return state

        # 作为 LangGraph 节点使用
        node = MyNode()
        workflow.add_node("my_node", node)
    """

    name: str = "base_node"  # 节点名称，子类应覆盖

    def __init__(self, config: Optional[ImportConfig] = None):
        """
        初始化节点

        Args:
            config: 配置对象，默认使用全局配置
        """
        self.config = config or get_config()
        self.logger = logging.getLogger(f"import.{self.name}")

    def __call__(self, state: T) -> T:
        """
        节点执行入口

        LangGraph 调用节点时会调用此方法。
        提供统一的日志输出、任务追踪和异常处理。

        Args:
            state: 图状态字典

        Returns:
            更新后的状态字典

        Raises:
            ImportProcessError: 节点执行失败时抛出
        """
        task_id = state.get("task_id","")
        try:
            # 1. 开始准备执行节点
            self.logger.info(f"--- {self.name} 开始 ---")

            # 2. 执行节点
            if task_id:
                add_running_task(task_id,self.name)
            start_time = time.time()

            result = self.process(state)

            # 3. 执行节点成功
            self.logger.info(f"--- {self.name} 完成 ---")

            end_time = time.time()
            if task_id:
                add_done_task(task_id,self.name)
                add_node_duration(task_id,self.name,end_time-start_time)

            return result
        except Exception as e:
            self.logger.error(f"{self.name} 执行失败: {e}")
            raise ImportProcessError(
                message=str(e),
                node_name=self.name,
                cause=e
            )

    @abstractmethod
    def process(self, state: T) -> T:
        """
        节点核心处理逻辑

        子类必须实现此方法。

        Args:
            state: 图状态字典

        Returns:
            更新后的状态字典
        """
        pass

    def log_step(self, step_name: str, message: str = ""):
        """
        记录步骤日志

        Args:
            step_name: 步骤名称
            message: 附加信息
        """
        log_msg = f"[{step_name}]"
        if message:
            log_msg += f" {message}"
        self.logger.info(log_msg)

    def backup_chunks(self, state: T, file_name: str = "chunks.json") -> T:
        """备份chunks数据到JSON文件"""
        file_dir = state.get("file_dir", "")
        chunks = state.get("chunks", [])

        # 1. 检查目录
        if not file_dir:
            self.logger.info("未提供 file_dir，跳过备份")
            return None

        # 2. 检查数据
        if not chunks:
            self.logger.info("chunks 为空，跳过备份")
            return None

        # 3. 处理路径
        if not os.path.isabs(file_dir):
            file_dir = os.path.abspath(file_dir)

        try:
            # 4. 创建目录
            os.makedirs(file_dir, exist_ok=True)

            # 5. 构建输出路径
            output_path = os.path.join(file_dir, file_name)

            # 6. 写入文件
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(chunks, f, ensure_ascii=False, indent=2)

            self.logger.info(f"备份成功: {output_path}, chunks 数量: {len(chunks)}")
            return output_path

        except Exception as e:
            self.logger.warning(f"备份失败: {e}")
            return None

# 配置日志格式
def setup_logging(level: int = logging.INFO):
    """
    配置导入流程日志

    Args:
        level: 日志级别
    """
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
