import base64
import re
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Set, Deque
from logging import Logger

from langchain_openai import OpenAI

from knowledge.processor.import_processor.base import BaseNode, T, setup_logging
from knowledge.processor.import_processor.exceptions import StateFieldError, FileProcessingError
from knowledge.processor.import_processor.state import ImportGraphState
from knowledge.utils.client.ai_clients import AIClients
from knowledge.utils.client.storage_clients import StorageClients


@dataclass
class ImageContext:
    head: str # 标题
    pre_context: str # 上文
    post_context: str # 下文

@dataclass
class ImageInfo:
    """
    一张图片的完整信息
    name: minio用，作为key
    path: 1.vlm中使用  2.minio中用
    image_context: vlm中使用
    """
    name: str # 图片的名称（全名）
    path: str # 图片的路径
    image_context: ImageContext # 图片的上下文信息


class _MdFileHandler:
    def __init__(self,logger:Logger,name:str):
        self.logger = logger
        self.node_name = name

    def validate_and_read_md(self,state:ImportGraphState) -> Tuple[str,Path,Path]:
        """
        核心逻辑：
        1、读取md内容
        2、返回md的路径
        3、返回image的路径
        Args:
            state: 上一个节点最后一次更新后的state

        Returns:
            Tuple[str,Path,Path]
            md_content,
            md_path_obj,
            images_dir_obj

        """
        md_path = state.get("md_path","")

        # 判断md的路径是否为空
        if not md_path:
            raise StateFieldError(node_name=self.node_name,field_name="md_path",expected_type=str,message="md_path为空")

        # 标准化路径
        md_path_obj = Path(md_path)

        # 判断路径是否存在
        if not md_path_obj.exists():
            raise StateFieldError(node_name=self.node_name,field_name="md_path",expected_type=Path,message="md_path不存在")

        # 读取文件内容
        try:
            with open (md_path_obj, "r", encoding="utf-8") as f:
                md_content = f.read()
        except Exception as e:
            self.logger.error(f"{self.node_name} 读取md文件错误: {e}")
            raise FileProcessingError(message="读取md文件错误",node_name=self.node_name)

        # 获取图片的路径
        images_dir_obj = md_path_obj.parent / "images"

        return md_content,md_path_obj,images_dir_obj

    def backup_md(self, md_path_obj: Path, new_md_content: str):
        """
        Args:
            md_path_obj: md内容的地址
            new_md_content: 新的md内容

        Returns:

        """
        # 替换路径中的文件名部分，同时保持目录路径不变
        new_file_path = md_path_obj.with_name(f"{md_path_obj.stem}_new{md_path_obj.suffix}")

        try:
            with open(new_file_path, "w", encoding="utf-8") as f:
                f.write(new_md_content)
            self.logger.info(f"处理好的文件已经备份到{new_file_path}")
        except Exception as e:
            self.logger.error(f"{new_file_path} 备份md文件错误: {e}")
            raise FileProcessingError(
                message="备份md文件错误",
                node_name="md_image_node",
            )

        return str(new_file_path)

class _ImageScanner:
    def __init__(self,logger:Logger):
        self.logger = logger

    def _scan_images_dir(self, images_dir_obj:Path, md_content:str, image_extensions:Set[str], img_content_length:int) -> List[ImageInfo]:
        """
        核心逻辑：
        1、扫描图片目录
        2、找到图片对应的上下文
            找到图片对应在md_content 中的位置
            获取上文信息： 标题 + 上文内容
            获取下文信息： 下文内容
        Args:
            images_dir_obj: 图片目录
            md_content: md内容
            image_extensions: 允许的图片的格式
            img_content_length: 图片最长的上下文字符

        Returns:
            List[ImageInfo] 图像的信息 name / path / image_context
        """
        image_info_list = []

        # 遍历文件目录
        for image_path in images_dir_obj.iterdir():

            # 过滤掉子目录,只保留文件
            if not image_path.is_file():
                self.logger.error(f"{image_path} 不是一个有效的文件，或者是一个目录")
                continue

            # 判断目录中的文件扩展名是否是否在image_extensions中
            if not image_path.suffix.lower() in image_extensions:
                self.logger.error(f"{image_path} 不是允许的图片格式")
                continue

            # 获取图片的上下文，放到容器中
            ctx = self._get_image_context(image_path.name, md_content, img_content_length)
            if not ctx:
                self.logger.info(f"{image_path} 暂无对应的上下文信息")
                continue

            image_info_list.append(
                ImageInfo(
                    name=image_path.name,
                    path=str(image_path),
                    image_context=ctx
                )
            )

        self.logger.info(f"MD中扫描到 {len(image_info_list)} 张图片")
        return image_info_list

    def _get_image_context(self, image_name:str, md_content:str, img_content_length:int) -> Optional[ImageContext]:
        """
        获取每一张图片的上下文信息
        Args:
            image_path: 图片的路径
            md_content: md的内容
            img_content_length: 图片最长的上下文字符

        Returns:
            Context
                head: str # 标题
                pre_context: str # 上文
                post_context: str # 下文
        """

        # 1. 预编译正则规则(主要目的：从MD（很多行）中抓取到当前这个图片)
        # ![](images\xxx.png "abc")
        # 正则在大模型应用中特别多
        # . 任意字符 * 0次或者多次  \[ \] \( \) ?非贪婪模式  escape（a.png）
        pattern = re.compile(r"!\[.*?\]\(.*?" + re.escape(image_name) + r".*?\)")

        # 切割md的内容
        # md_content '# 标题\n这是内容\n\n- 列表项1\n- 列表项2\n'
        md_lines = md_content.split("\n")

        for md_index, md_line in enumerate(md_lines):
            # 匹配不到当前图片的路径
            if not pattern.search(md_line):
                 continue

            # 上文的标题 和 上文的索引
            head,pre_index = self._get_up_context(md_index, md_lines)
            # 上文的所有的内容
            pre_lines = md_lines[pre_index+1:md_index]
            # 对上文的长度做了处理时候，符合条件的上文内容
            pre_context = self._extract_limited_context(pre_lines, img_content_length,direction="up")

            # 下文的索引
            next_index = self._get_down_context(md_index, md_lines)
            # 下文的所有的内容
            next_lines = md_lines[md_index + 1:next_index]
            # 对下文的长度做了处理时候，符合条件的上文内容
            next_context = self._extract_limited_context(next_lines, img_content_length,direction="down")

            return ImageContext(
                head=head,
                pre_context=pre_context,
                post_context=next_context
            )

        # 如何该图片没有对应的上下文信息，就返回None
        return None

    def _get_up_context(self, from_index:int, md_lines:list[str]) -> Tuple[str,int]:
        """
        获取图片的上文信息
        Args:
            from_index: 当前图片的索引
            md_lines: md文件的内容
        Returns:
            head: str # 标题
            pre_index: int # 上文索引
        """
        for i in range(from_index-1,-1,-1):
            if re.match(r"^#{1,6}\s+", md_lines[i]):
                return (md_lines[i], i)
        return "",-1

    def _get_down_context(self, from_index:int, md_lines:list[str]) -> int:
        """
        获取图片的下文索引
        Args:
            from_index: 当前图片的索引
            md_lines: md文件的内容
        Returns:
            next_index: int # 下文索引
        """
        for i in range(from_index+1,len(md_lines)):
            if re.match(r"^#{1,6}\s+", md_lines[i]):
                return i
        return len(md_lines)

    def _extract_limited_context(self, extracted_context_lines:list[str], img_content_length:int, direction:str) -> str:
        """
        Args:
            extracted_context_lines: 上或者下文内容
            img_content_length: 图片最长的上下文字符
            direction: 方向

        Returns:
            符合条件的上或者下文内容
        """
        graphs  = []
        current_graph = []

        for line in extracted_context_lines:
            # 定义自然而然的段落,是一个空行
            blank_graph =  not line.strip()

            # 是一个图片
            is_other_image = re.match(
                r"^!\[.*?\]\(.*?\)$", line.strip()
            )

            # 遇到空格或者图片，把当前的所有内容加入到段落中
            if blank_graph or is_other_image:
                if current_graph:
                    graphs.append("\n".join(current_graph))
                    current_graph = []
                    continue

            # 不是空格和图片,把当前的内容加入到当前段落中
            current_graph.append(line)

        # 处理最后一个段落的内容
        if current_graph:
            graphs.append("\n".join(current_graph))


        # 收集最终的段落, 字符长度 < img_content_length
        selected = []
        total = 0

        # 当是上文的时候,从尾部开始遍历
        if direction == "up":
            graphs.reverse()

        for graph in graphs:
            if total + len(graph) > img_content_length and selected:
                break
            total += len(graph)
            selected.append(graph)

        # 把段落进行反转回去
        if direction == "up":
            selected.reverse()

        # 把最终的段落进行拼接
        return "\n\n".join(selected)

class _VlmSummarizer:
    def __init__(self,logger:Logger,requests_per_minute:int):
        self.logger = logger
        self.requests_per_minute = requests_per_minute


    def _summarize_all(self, document_name: str, image_list_info: List[ImageInfo], vl_model: str) -> Dict[str, str]:
        """
        Args:
            document_name: md文档的名字
            image_list_info: 图片的信息
            vl_model: vlm模型

        Returns:
            Dict[str, str]: 每个图片对应的总结
        """
        summarize = {}
        request_timestamps: Deque[float] = deque()

        # 获取vlm的客户端
        try:
            vlm_client = AIClients.get_vlm_client()
        except Exception as e:
            for image in image_list_info:
                summarize[image.name] = "图片处理失败，暂无摘要"
            return summarize

        # 否则为每一张图片进行总结
        for image in image_list_info:
            self._enforce_rate_limit(request_timestamps, self.requests_per_minute)
            summarize[image.name] = self._summarize_one(vlm_client, document_name, image, vl_model)

        self.logger.info(f"md中{len(summarize)} 图片总结完成")
        return summarize

    def _summarize_one(self, vlm_client: OpenAI, document_name:str, image:ImageInfo, vl_model:str) -> str:
        """
        为每一张图片生成总结
        Args:
            vlm_client: vlm客户端
            document_name: 文档名称
            image: 图片信息
            vl_model: vlm模型的名称

        Returns:
            当前图片的总结
        """

        # 1、构建vlm需要的的head，上文，下文
        tuple_context = (
            image.image_context.head,
            image.image_context.pre_context,
            image.image_context.post_context
        )
        parts = [p for p in tuple_context if p]

        # 2、构建vlm的上下文
        final_context = "\n".join(parts) if parts else "暂无上下文"

        # 3. 根据图片地址获取到图片的内容（二进制字节流）---文本协议认识（base64编码）--->解码（‘utf-8’）--->字符串（文本协议能传输） ---- 根据收到字符串解码（二进制字节流 还原图片内容）
        try:
            with open(image.path, "rb") as f:
                image_data = base64.b64encode(f.read()).decode("utf-8")
        except Exception as e:
            self.logger.error(f"读取图片文件{image.path} 内容失败: {e}")
            return "图片处理失败，暂无图片描述"

        # 4、利用vlm模型，得到摘要
        try:
            resp = vlm_client.chat.completions.create(
                model=vl_model,
                messages=[{
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": (
                                f"任务：为Markdown文档中的图片生成一个简短的中文标题。\n"
                                f"背景信息：\n"
                                f"  1. 所属文档标题：\"{document_name}\"\n"
                                f"  2. 图片上下文：{final_context}\n"
                                f"请结合图片内容和上述上下文信息，"
                                f"用中文简要总结这张图片的内容，"
                                f"生成一个精准的中文标题摘要（不要包含图片二字）。"
                            ),
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{image_data}"
                            },
                        },
                    ],
                }],
            )

            return resp.choices[0].message.content.strip()
        except Exception as e:
            self.logger.error(f"图片摘要生成失败 {image.path}: {e}")
            return "暂无图片描述"


    def _enforce_rate_limit(
            self, timestamps: Deque[float],
            max_requests: int,
            window: int = 60,
    ):
        now = time.time()
        while timestamps and now - timestamps[0] >= window:
            timestamps.popleft()

        if len(timestamps) >= max_requests:
            sleep_dur = window - (now - timestamps[0])
            if sleep_dur > 0:
                self.logger.info(
                    f"达到速率限制，暂停 {sleep_dur:.2f} 秒..."
                )
                time.sleep(sleep_dur)
            now = time.time()
            while timestamps and now - timestamps[0] >= window:
                timestamps.popleft()

        timestamps.append(now)

class _ImageUpload:
    def __init__(self,logger:Logger):
        self.logger = logger

    def uploade_and_replace_images(self, obj_dir_name:str, image_list_info:List[ImageInfo], image_summary:Dict[str,str], md_content:str, minio_url:str, minio_bucket:str):
        """
        把本地的图片上传到minio，得到minio的地址
        替换md中图片的 地址 和 摘要
        Args:
            obj_dir_name: minio中对象的名字
            image_list_info: 图片的内容
            image_summary: 图片的总结
            md_content: md的内容
            minio_url: minio服务器的地址
            minio_bucket: minio的桶名

        Returns:

        """

        remote_urls: Dict[str, str] = self._uploade_all_images(obj_dir_name,image_list_info,minio_url,minio_bucket)

        md_content = self._update_md_content(md_content,remote_urls,image_summary)

        return md_content

    def _uploade_all_images(self, obj_dir_name:str, image_list_info:List[ImageInfo], minio_url:str, minio_bucket:str):
        remote_urls = {}

        # 1、获取minio的客户端
        try:
            minio_client = StorageClients.get_minio_client()
        except Exception as e:
            for image_info in image_list_info:
                remote_urls[image_info.name] = image_info.path
            return remote_urls

        # 遍历图片上次到minio
        for image_info in image_list_info:
            object_name = f"{obj_dir_name}/{image_info.name}"
            try:
                # 上传到minio
                minio_client.fput_object(
                    minio_bucket, object_name, image_info.path,
                )

                self.logger.info(f"图片上传成功: {image_info.name}")

                # 做拼接 # http://192.168.10.128:9000/桶名/对象名
                remote_urls[image_info.name] = f"{minio_url}/{minio_bucket}/{object_name}"
            except Exception as e:
                remote_urls[image_info.name] = image_info.path
                self.logger.error(f"图片上传失败: {image_info.name}: {e}")

        self.logger.info(f"获取到远端的{len(remote_urls)}张图片")
        return remote_urls

    def _update_md_content(self, md_content:str, remote_urls:Dict[str,str], image_summary:Dict[str,str]):
        """
        对md做替换 - 图片 + 摘要
        Args:
            md_content: md内容
            remote_urls: 图片远端的地址
            image_summary: 图片的总结

        Returns:

        """
        # 利用正则寻找(捕获组：()一个捕获组：group(0) 将整个匹配到的内容放进去 group(1)：图片的摘要 group(2):图片地址)
        pattern = re.compile(r"!\[(.*?)\]\((.*?)\)")

        def replacer(match:re.match) -> str:
            for image_name, img_summary in image_summary.items():
                md_image_path = match.group(2)
                md_image_name = Path(md_image_path).name

                if md_image_name == image_name:
                    return f"![{img_summary}]({remote_urls[image_name]})"

            return match.group(0)

        return pattern.sub(replacer,md_content)


class MarkDownImageNode(BaseNode):
    """
    核心流程：得到四个实例对象，调用实例
    """

    name = "markdown_image_node"

    def __init__(self):
        super().__init__()
        self.md_file_handler = _MdFileHandler(self.logger,self.name)
        self.image_scanner = _ImageScanner(self.logger)
        self.vlm_summarizer = _VlmSummarizer(self.logger,self.config.requests_per_minute)
        self.image_upload = _ImageUpload(self.logger)

    def process(self, state: ImportGraphState) -> ImportGraphState:
        config = self.config

        # 操作_MdFileHandler
        self.log_step(step_name="step1----------md_file_handler",message="获取图片的地址")
        md_content,md_path_obj,images_dir_obj = self.md_file_handler.validate_and_read_md(state)

        # self.logger.info(f"md_path_obj:{md_path_obj}")
        # self.logger.info(f"images_dir_obj:{images_dir_obj}")
        # self.logger.info(f"md_content:{md_content}")

        # 操作_ImageScanner
        if not images_dir_obj.exists():
            state["md_content"] = md_content
            return state
        self.log_step(step_name="step2----------image_scanner",message="开始扫描图片")
        image_list_info: List[ImageInfo] = self.image_scanner._scan_images_dir(images_dir_obj,md_content,config.image_extensions,config.img_content_length)

        # 操作_VlmSummarizer
        self.log_step(step_name="step3----------VlmSummarizer", message="开始对图片做总结")
        image_summary: Dict[str,str] = self.vlm_summarizer._summarize_all(md_path_obj.stem,image_list_info,config.vl_model)

        # 操作ImageUpload
        self.log_step(step_name="step4----------ImageUpload", message="把本地图片上传到minio，并且替换本地的md")
        new_md_content = self.image_upload.uploade_and_replace_images(md_path_obj.stem,image_list_info,image_summary,md_content,config.get_minio_base_url(),config.minio_bucket)

        # 做备份
        self.log_step(step_name="step5----------md_file_handler", message="把新的md做备份")
        self.md_file_handler.backup_md(md_path_obj,new_md_content)

        state["md_content"] = new_md_content
        return state

if __name__ == "__main__":
    setup_logging()
    markdown_image_node = MarkDownImageNode()
    state = ImportGraphState()
    state["md_path"] = r"D:\20251208_study\shopkeeper_brain\knowledge\processor\import_processor\temp_dir\万用表的使用\hybrid_auto\万用表的使用.md"
    markdown_image_node(state)