import asyncio
import json
from json import JSONDecodeError
from typing import Tuple, List, Dict, Any, Optional, Union
from agents.mcp import MCPServerStreamableHttp
from knowledge.processor.query_processor.base import BaseNode
from knowledge.processor.query_processor.exceptions import StateFieldError
from knowledge.processor.query_processor.state import QueryGraphState


class WebMcpSearchNode(BaseNode):
    name = "web_mcp_search_node"

    def process(self, state: QueryGraphState) -> Union[QueryGraphState, Dict[str, Any]]:
        # 1、参数校验
        rewritten_query, item_names = self._validate(state)

        # 2、定义并且mcp的调用
        # 调用函数，如何函数内部是async，那么在调用的时候有两种方式：方式1：使用await，是异步的；方式2：使用asyncio.run，放在异步环境中，但是它是同步的
        web_search_result = asyncio.run(self._execute_map_server(rewritten_query))

        if not web_search_result:
            return {}

        # 由于是并行搜索，如果直接返回state，langgraph会认为多个节点在同一个时间对state中的同一个数据做了更改，此时会报错，所以这里直接返回修改的字段既可，内部会更新到state中，以供其他节点使用的
        # 如果是没有更新，直接返回{}
        return {"web_search_docs": web_search_result}

    def _validate(self, state: QueryGraphState) -> Tuple[str, List]:
        rewritten_query = state.get("rewritten_query")

        item_names = state.get("item_names")

        if not rewritten_query and not isinstance(rewritten_query, str):
            raise StateFieldError(node_name=self.name, field_name="rewritten_query", expected_type=str)

        if not item_names and not isinstance(item_names, List):
            raise StateFieldError(node_name=self.name, field_name="item_names", expected_type=list)

        return rewritten_query, item_names

    async def _execute_map_server(self, rewritten_query:str) -> Optional[List[Dict[str, str]]]:
            """
            执行mcp服务
            注意：一个mcp服务下可能有多个工具（函数）
            Args:
                rewritten_query:

            Returns:

            """
            # openai sdk: 提供MCPServerStreamableHttp，call_tool，以及结果
            # 1、定义mcp客户端
            self.http = MCPServerStreamableHttp(
                # mcp客户端的名字
                name="联网搜索",
                # 提供mcp服务的第三方平台的api_key 和 base_url
                params={
                    "url": self.config.mcp_dashscope_base_url,
                    "headers": {"Authorization": f"Bearer {self.config.openai_api_key}"},
                    "timeout": 60,  # 超时时间s
                    "terminate_on_close": True,
                },
                max_retry_attempts=2,  # 重试保护
                cache_tools_list=True,  # 缓存工具列表，加速
            )
            async with self.http as mcp_client:
                # 2、调用工具
                web_search_result = await mcp_client.call_tool(tool_name="bailian_web_search",
                                                               arguments={"query": rewritten_query, "count": 3})

                # 3、解析数据
                text_content = web_search_result.content[0]

                if not text_content:
                    return []

                text_content_text = text_content.text
                if not text_content_text:
                    return []

                try:
                    text_content_text_obj:Dict[str, Any] = json.loads(text_content_text)

                    pages = text_content_text_obj.get("pages",[])
                    if not pages:
                        return []

                    return [{"snippet": page.get("snippet","").strip(), "title": page.get("title","").strip(), "url": page.get("url","").strip()} for page in pages ]


                except JSONDecodeError as e:
                    self.logger.error(f"web_search_result 解析失败，失败信息：{e.msg},失败内容{e.doc},失败位置{e.pos}")



if __name__ == '__main__':
    node = WebMcpSearchNode()
    init_state = {
        "rewritten_query": "RS-12 万用表如何测试直流电压",
        "item_names": ["RS-12 数字万用表"],
    }
    res = node.process(init_state)
    print("res===", res)