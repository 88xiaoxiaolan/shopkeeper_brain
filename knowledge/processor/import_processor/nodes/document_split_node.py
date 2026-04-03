import json
import os
import re
from typing import Tuple, Dict, Any, List
from langchain_text_splitters import RecursiveCharacterTextSplitter
from knowledge.processor.import_processor.base import BaseNode, T, setup_logging
from knowledge.processor.import_processor.state import ImportGraphState
from knowledge.utils.markdown_util import MarkdownTableLinearizer


class DocumentSplitNode(BaseNode):
    name = "document_split_node"

    def process(self, state: ImportGraphState) -> ImportGraphState:
        config = self.config

        # 参数校验
        self.log_step(step_name="step1---------",message="开始校验参数")
        md_content,file_title,config.max_content_length,config.min_content_length = self._validate_state(state,config)

        # 文档切分
        self.log_step(step_name="step2---------", message="做文档切分")
        section: list[Dict[str,Any]] = self._splite_document(md_content,file_title)

        # 二次切分或者合并
        self.log_step(step_name="step3---------", message="做二次切分或者合并")
        final_section:list[Dict[str,Any]] = self._split_and_merge(section,config.max_content_length,config.min_content_length)

        # 组装成chunk对象
        self.log_step(step_name="step4---------", message="组装成chunk对象")
        final_chunks:list[Dict[str,Any]] = self.assemble_chunk(final_section)

        state["chunks"] = final_chunks

        # 备份
        self.log_step(step_name="step5----------", message="备份")
        self.backup_chunks(state=state,file_name= "chunks.json")

        return state

    def _validate_state(self, state: ImportGraphState,config) -> Tuple[str,str,int,int]:
        # 拿到md_content
        md_content = state.get("md_content")

        # 统一换行
        if md_content:
            md_content = md_content.replace("\r\n", "\n").replace("\r", "\n")

        # 获取文件的标题
        file_title = state.get("file_title")

        # 校验最大值，最小值，是否符合预期
        if config.max_content_length<=0 or config.min_content_length <=0 or config.max_content_length<=config.min_content_length:
            raise ValueError(f"切片长度参数校验失败")

        return md_content,file_title,config.max_content_length,config.min_content_length

    def _splite_document(self, md_content:str, file_title:str) -> list[Dict[str,Any]]:
        """
        根据标题把文档切分成章节
        Args:
            md_content: md的内容
            file_title: 文件的标题

        Returns:
            list[Dict[str,Any]]
             {
             "title": title,
            "parent_title": parent_title,
            "body" : body,
            "file_title": file_title
            }
        """
        
        # 是否是代码块
        in_dense = False
        # 当前章节的body_lines
        body_lines = []
        # 所有章节的数据
        sections = []
        # 当前的title
        current_title = ""
        # 标题等级,作为section的父标题使用
        hierarchy = [""] * 7
        # 当前标题的level
        current_level = 0

        # 1、根据\n 切分md_content
        md_lines = md_content.split("\n")
        
        # 2、定义正则（正则是从md中找到标题#{1,6}） ():捕获组: 产生3个group(0)，group(1):#号的个数 1-6个，group(2):标题的内容
        head_reg = re.compile(r"^\s*(#{1,6})\s+(.+)")

        def _flush() -> List[Dict[str, Any]]:
            body = "\n".join(body_lines)
            # 如果第一个是标题,那么打包的时候,没有 current_title +  body, 所以此时不需要打包
            if current_title or body:
                # 当前标题的父title
                parent_title = ""
                for i in range(current_level-1, 0, -1):
                    # 有可能有跳级,比如一级标题,没有二级标题,直接三级标题
                    if hierarchy[i]:
                        parent_title = hierarchy[i]
                        break

                # 如果是标题 + 段落 + 标题, 到第二个标题的时候, 打包第一个section, 此时没有父标题,只有parent_title
                if not parent_title:
                    parent_title = current_title if current_title else file_title

                sections.append({
                    # 如果一开始没有current_title,走到第一个current_title,那么会flush上一个段落,此时没有current_title,那么就使用file_title
                    # 如果一开始有current_title, 走到第二个current_title,那么会flush上一个段落,此时有current_title,那么就使用file_title
                    "title": current_title if current_title else file_title,
                    "parent_title": parent_title,
                    "body" : body,
                    "file_title": file_title,
                })
        
        # 3、遍历切分后的内容
        for md_line in md_lines:
            # 3.1 看是否是代码块中的注释
            if md_line.strip().startswith("```") or md_line.strip().startswith("~~~"):
                in_dense = not in_dense

            # 3.2 判断是否走正则
            math = head_reg.match(md_line) if not in_dense else None

            # 表示一定是标题,并且一定不是代码块中的标题
            if math:
                # 将body_lines 中收集到的内容加到section中
                _flush()

                # 当前标题
                current_title = md_line
                # 当前标题是几级
                level = len(math.group(1))
                hierarchy[level] = current_title

                # 清空当前行之后的所有行,防止被污染
                for i in range(level+1,7):
                    hierarchy[i] = ""

                body_lines = []
            else:
                # 要么不是标题,要么是代码块中的注释
                body_lines.append(md_line)

        # 防止最后一个段落,没有标题,对它进行打包
        _flush()

        return sections

    def _split_and_merge(self, sections:list[Dict[str,Any]], max_content_length:int, min_content_length:int) -> list[Dict[str,Any]]:
        """
        对超过 max_content_length 做切分
        对小于 min_content_length 做合并
        Args:
            section: 章节
            max_content_length: 最大的长度
            min_content_length: 最小的长度

        Returns:
        list[Dict[str,Any]]
        """

        # 1、切分
        current_section = []
        for section in sections:
            current_section.extend(self._split_long_section(section,max_content_length))


        # 2、合并短内容
        final_sections = self._merge_short_section(current_section,min_content_length)
        return final_sections

    def _split_long_section(self, section: Dict[str, Any], max_content_length: int):
        title = section.get("title")
        body = section.get("body")
        parent_title = section.get("parent_title")
        file_title = section.get("file_title")

        if len(title) > 80:
            title = title[:80] # 防御性编程：title 本身就超长的极端情况

        if "<table>" in body:
            body = MarkdownTableLinearizer.process(body)
            section["body"] = body

        title_prefix = f"{title}\n\n"
        total_len = len(title_prefix) + len(body)

        # 需要拿标题和body的内容来判断是否超过了阈值，因为后面是要用标题和body合并在一起来存放到chunks.json 中去的
        if total_len <= max_content_length:
            return [section]

        # 可以切分的长度
        body_length = max_content_length - len(title_prefix)
        if body_length <= 0:
            return [section] # 防御性编程：title 本身就超长的极端情况

        # 按照递归字符来进行切分
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=body_length,
            chunk_overlap=0, # 这里还算合适，因为已经通过一级标题进行划分了，所以语义基本是完整的
            separators=["\n\n", "\n", "。", "？", "！", "；", ".", "?", "!", ';', " ", ""],
            keep_separator=True
        )
        # 切分后的段落
        body_chunks  = text_splitter.split_text(body)

        # 这种情况基本不会发生，下面的情况可能导致这种情况发生
        # - 字符计数方式不同（中英文混合）
        # - body_length 计算误差
        # - LangChain 版本差异
        if len(body_chunks ) == 1:
            return [section]

        # 对大内容进行切分
        sub_section = []
        for index,chunk in enumerate(body_chunks ):
            sub_section.append({
                "title": f"{title}_{index+1}",
                "body": chunk,
                "parent_title": parent_title,
                "file_title": file_title
            })

        self.logger.info(f"切分后的章节数：{len(sub_section)}")
        return sub_section

    def _merge_short_section(self, current_sections: List[Dict[str, Any]], min_content_length:int) -> List[Dict[str, Any]]:
        """
        把小于阈值，并且同源的段落 进行合并
        Args:
            current_section: 所有的章节
            min_content_length: 最小的阈值

        Returns:

        """
        # 拿第一个当作基准，去向下合并
        current_section = current_sections[0]
        final_sections = []

        for next_section in current_sections[1:]:
            # 如果同源
            parent = current_section.get("parent_title") == next_section.get("parent_title")
            # 并且当前段落 < min_content_length, 那么进行合并
            if parent and len(current_section.get("body")) < min_content_length:
                # 合并body
                current_section["body"] = current_section.get("body").rstrip() + "\n\n" + next_section.get("body").lstrip()
                # 把标题回退到父标题
                current_section["title"] = current_section.get("parent_title")
            else:
                # 不同源，当前内如长度大于min_content_length，不合并
                final_sections.append(current_section)
                # 更换下一个为基准去向下合并
                current_section = next_section

        # 最后一个，由于没有合并进去，所以单独处理
        final_sections.append(current_section)
        self.logger.info(f"合并后的章节数：{len(final_sections)}")
        return final_sections

    def assemble_chunk(self, final_section: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        final_chunk = []
        for section in final_section:
            final_chunk.append({
                "title": section["title"],
                "content": f"{section["title"]}\n\n{section["body"]}",
                "parent_title": section["parent_title"],
                "file_title": section["file_title"],
            })

        self.logger.info(f"最终的chunk数为: {len(final_chunk)}")
        return final_chunk

if __name__ == "__main__":
    setup_logging()

    documentSplitNode = DocumentSplitNode()

    md_path = r"D:\20251208_study\shopkeeper_brain\knowledge\processor\import_processor\temp_dir\万用表的使用\hybrid_auto\万用表的使用_new.md"
    with open (md_path, "r", encoding="utf-8") as f:
        md_content = f.read()

    state = {
        "md_content": md_content,
        "file_title": "万用表的使用",
        "file_dir": r"D:\20251208_study\shopkeeper_brain\knowledge\processor\import_processor\temp_dir"
    }

    documentSplitNode.process(state=state)
