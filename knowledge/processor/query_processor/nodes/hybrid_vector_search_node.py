from typing import List, Tuple, Dict, Any

from knowledge.processor.query_processor.base import BaseNode
from knowledge.processor.query_processor.exceptions import StateFieldError
from knowledge.processor.query_processor.state import QueryGraphState
from knowledge.utils.client.ai_clients import AIClients
from knowledge.utils.client.storage_clients import StorageClients
from knowledge.utils.embedding_util import generate_bge_m3_hybrid_vectors
from knowledge.utils.milvus_util import create_hybrid_search_requests, execute_hybrid_search_query, _item_name_search


class HybridVectorSearchNode(BaseNode):

    name = "hybrid_vector_search_node"

    def process(self, state: QueryGraphState) -> QueryGraphState:
        # 1、参数校验
        rewritten_query,item_names = self._validate(state)

        # 2、获取嵌入模型
        try:
            bge_m3_client = AIClients.get_bge_m3_client()
        except ConnectionError as e:
            self.logger.error(f"嵌入模型获取失败，{str(e)}")
            return state

        # 3、获取milvus客户端
        try:
            milvus_client = StorageClients.get_milvus_client()
        except ConnectionError as e:
            self.logger.error(f"嵌入milvus客户端获取失败，{str(e)}")
            return state

        # 4、获取查询的嵌入向量
        try:
            embed_query_vector = generate_bge_m3_hybrid_vectors(bge_m3_client, [rewritten_query])
        except Exception as e:
            self.logger.error(f"获取查询的嵌入向量失败，{str(e)}")
            return  state

        try:
            # 创建search请求
            expr,expr_params = _item_name_search(item_names)
            hybrid_search = create_hybrid_search_requests(
                dense_vector=embed_query_vector["dense"][0],
                sparse_vector=embed_query_vector["sparse"][0],
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
                state["embedding_chunks"] = hybrid_search_result[0]

                return state

        except Exception as e:
            self.logger.error(f"创建混合搜索失败，{str(e)}")
            return state

    def _validate(self, state: QueryGraphState) -> Tuple[str,List]:
        rewritten_query = state.get("rewritten_query")

        item_names = state.get("item_names")

        if not rewritten_query and not isinstance(rewritten_query, str):
            raise StateFieldError(node_name=self.name,field_name="rewritten_query",expected_type=str)

        if not item_names and not isinstance(item_names, List):
            raise StateFieldError(node_name=self.name,field_name="item_names",expected_type=list)

        return  rewritten_query,item_names

if __name__ == '__main__':
    node  = HybridVectorSearchNode()
    init_state = {
        "rewritten_query": "万用表如何使用电阻",
        "item_names": ["RS-12 数字万用表"],
    }
    res = node.process(init_state)
    print("res===",res)