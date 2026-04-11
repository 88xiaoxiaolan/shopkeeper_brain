from typing import Tuple, List, Optional
from langchain_core.messages import SystemMessage, HumanMessage
from knowledge.processor.query_processor.exceptions import StateFieldError
from knowledge.processor.query_processor.base import BaseNode, T
from knowledge.processor.query_processor.state import QueryGraphState
from knowledge.propmpt.query_prompt import HYDE_USER_PROMPT_TEMPLATE
from knowledge.utils.client.ai_clients import AIClients
from knowledge.utils.client.storage_clients import StorageClients
from knowledge.utils.embedding_util import generate_bge_m3_hybrid_vectors
from knowledge.utils.milvus_util import _item_name_search, create_hybrid_search_requests, execute_hybrid_search_query


class HydeVectorSearchNode(BaseNode):
    name = "hyde_vector_search_node"

    def process(self, state: QueryGraphState) -> QueryGraphState:
        # 1、参数校验
        rewritten_query, item_names = self._validate(state)

        # 2、利用llm生成原始问题对应的假设性答案，解决跨域不对称问题
        hy_document = self._generate_document(rewritten_query,item_names)

        # 3、判断
        if hy_document is None:
            return state

        # 4、获取嵌入模型
        try:
            bge_m3_client = AIClients.get_bge_m3_client()
        except ConnectionError as e:
            self.logger.error(f"嵌入模型获取失败，{str(e)}")
            return state

        # 5、获取milvus客户端
        try:
            milvus_client = StorageClients.get_milvus_client()
        except ConnectionError as e:
            self.logger.error(f"嵌入milvus客户端获取失败，{str(e)}")
            return state

        # 6、为假设性文档生成嵌入向量
        try:
            hy_document_vector = generate_bge_m3_hybrid_vectors(bge_m3_client, [f"{rewritten_query}\n{hy_document}"])
        except Exception as e:
            self.logger.error(f"获取查询的嵌入向量失败，{str(e)}")
            return state

        # 7、创建search请求，做混合检索
        try:
            # 创建search请求
            expr,expr_params = _item_name_search(item_names)
            hybrid_search = create_hybrid_search_requests(
                dense_vector=hy_document_vector["dense"][0],
                sparse_vector=hy_document_vector["sparse"][0],
                expr=expr,
                expr_params=expr_params,
                limit=5
            )

            # 执行混合搜索
            hybrid_search_result = execute_hybrid_search_query(
                milvus_client=milvus_client,
                collection_name=self.config.chunks_collection,
                search_requests=hybrid_search,
                ranker_weights=(0.5, 0.5),
                norm_score=True,
                limit=5,
                output_fields=["chunk_id", "content", "item_name"],
            )

            # 判断是否有结果
            if not hybrid_search_result or not hybrid_search_result[0]:
                self.logger.info("混合搜索结果为空")
                return state
            else:
                # 更新state
                state["hyde_embedding_chunks"] = hybrid_search_result[0]
                return state

        except Exception as e:
            self.logger.error(f"原始问题{rewritten_query}搜索结果失败，{str(e)}")
            return state

    def _validate(self, state: QueryGraphState) -> Tuple[str,List]:
        rewritten_query = state.get("rewritten_query")

        item_names = state.get("item_names")

        if not rewritten_query and not isinstance(rewritten_query, str):
            raise StateFieldError(node_name=self.name,field_name="rewritten_query",expected_type=str)

        if not item_names and not isinstance(item_names, List):
            raise StateFieldError(node_name=self.name,field_name="item_names",expected_type=list)

        return  rewritten_query,item_names

    def _generate_document(self, rewritten_query:str, item_names:List[str]) -> Optional[str]:
        """

        Args:
            rewritten_query: 用户问题
            item_names: 商品名
        Returns:
            假设性答案
        """

        # 1、获取llm客户端
        try:
            llm_client = AIClients.get_llm_client(response_format=False)
        except ConnectionError as e:
            self.logger.error(f"嵌入llm客户端获取失败，{str(e)}")
            return None

        # 2、获取提示词
        system_prompt = f"您是一位{item_names}的技术文档领域的专家，主要擅长编写技术文档、操作手册、文档规格说明"
        user_prompt = HYDE_USER_PROMPT_TEMPLATE.format(
            item_names=item_names,
            rewritten_query=rewritten_query
        )

        # 3、调用
        try:
            llm_response = llm_client.invoke([
                SystemMessage(content=system_prompt),
                HumanMessage(content=user_prompt)
            ])
        except Exception as e:
            self.logger.error(f"调用llm，生成假设性文档失败，{str(e)}")
            return None

        if not llm_response.content.strip():
            return None

        return llm_response.content.strip()

if __name__ == '__main__':
    node  = HydeVectorSearchNode()
    init_state = {
        "rewritten_query": "RS-12 万用表如何测试直流电压",
        "item_names": ["RS-12 数字万用表"],
    }
    res = node.process(init_state)
    print("res===",res)