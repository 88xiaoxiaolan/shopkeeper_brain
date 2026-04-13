import asyncio
import json
from json import JSONDecodeError
from typing import Tuple, List, Dict, Any, Optional
from agents.mcp import MCPServerStreamableHttp
from knowledge.processor.query_processor.base import BaseNode
from knowledge.processor.query_processor.exceptions import StateFieldError
from knowledge.processor.query_processor.state import QueryGraphState

class RrfMergeNode(BaseNode):
    name = "rrf_merge_node"

    def process(self, state: QueryGraphState) -> QueryGraphState:
        # 1、获取本地搜索的结果
        embedding_chunks = state.get("embedding_chunks") or []
        hyde_embedding_chunks = state.get("hyde_embedding_chunks") or []

        # 2、定义两路检索结果和对应路的权重映射表（等权测试，后续拿到一批query之后进行观察调整，但是不必过于纠结，rrf影响最大的是k）
        search_result_weight = {
            "embedding_chunks": (self._validate_search_result(embedding_chunks), 1.0),
            "hyde_embedding_chunks": (self._validate_search_result(hyde_embedding_chunks), 1.0),
        }

        # 3、收集映射表中搜索结果和权重
        rrf_inputs = list(search_result_weight.values())

        # 4、利用rrf计算两路文档的分数
        merged_rrf_result_tuple: List[Tuple[Dict[str, Any], float]] = self._merge_docs(rrf_inputs, self.config.rrf_k,
                                                                                 self.config.rrf_max_results)
        merged_rrf_result = [doc for doc,_ in merged_rrf_result_tuple]

        # 5、更新state
        state["rrf_chunks"] = merged_rrf_result

        # 6、返回state
        return state

    def _validate_search_result(self, chunks: List[Dict[str, Any]]):
        """
        Args:
            chunks:
        Returns:
        """
        if not chunks:
            return []

        search_result = []
        for chunk in chunks:
            if not chunk or not isinstance(chunk, dict):
                continue


            entity = chunk.get("entity")

            if not entity or not isinstance(entity, dict):
                continue


            search_result.append(entity)

        return search_result

    def _merge_docs(self, rrf_inputs: List[Tuple[List[Dict[str, Any]], float]], rrf_k:int, rrf_max_results:int):
        """
        rrf 经过多路检索返回的文档得分
        Args:
            rrf_inputs: 多路检索的文档 + 对应权重
            rrf_k: 平滑参数
            rrf_max_results: 最大返回的个数

        Returns:
            多路检索文档的对象 以及 经过RRF计算之后文档的得分
            Tuple[Dict,float]: dict 文档对象，float 对应分数

        List[Tuple[Dict[str, Any], float]]
        """

        chunk_score = {}
        chunk_data = {}
        # 遍历所有路检索结果
        for search_result, weight in rrf_inputs:
            # 遍历某一路的结果
            for rank,entity in enumerate(search_result,1):
                # 拿到chunk_id
                chunk_id = entity.get("chunk_id")

                if not chunk_id:
                    continue

                # 某一个chunk的得分
                rrf_score = weight / (rrf_k + rank)

                # 某一个chunk累加的得分
                chunk_score[chunk_id] = chunk_score.get(chunk_id,float(0)) + rrf_score

                # setdefault 可以根据key去重
                chunk_data.setdefault(chunk_id,entity)


        # 排序以及构建chunk和得分的结果
        final_rrf_result = sorted([(chunk_data.get(chunk_id), score) for chunk_id, score in chunk_score.items()],key=lambda x:x[1],reverse=True)

        return final_rrf_result[:rrf_max_results] if rrf_max_results else final_rrf_result


if __name__ == '__main__':
    print("=" * 60)
    print("开始测试: RRF 融合节点")
    print("=" * 60)

    # 模拟两路检索结果
    # chunk_1 命中 2 路（预期最高分）
    # chunk_2 命中 2 路
    # chunk_3, chunk_4 各命中 1 路
    mock_state = {
        "embedding_chunks": [
            {"entity": {"chunk_id": "chunk_1", "content": "向量搜索结果#1"}},
            {"entity": {"chunk_id": "chunk_2", "content": "向量搜索结果#2"}},
            {"entity": {"chunk_id": "chunk_3", "content": "向量搜索结果#3"}},
        ],
        "hyde_embedding_chunks": [
            {"entity": {"chunk_id": "chunk_2", "content": "HyDE搜索结果#1"}},
            {"entity": {"chunk_id": "chunk_1", "content": "HyDE搜索结果#2"}},
            {"entity": {"chunk_id": "chunk_4", "content": "HyDE搜索结果#3"}},
        ],
    }

    print("【输入状态】:")
    print(f"  embedding_chunks: {len(mock_state['embedding_chunks'])} 条")
    print(f"  hyde_embedding_chunks: {len(mock_state['hyde_embedding_chunks'])} 条")
    print("-" * 60)

    rrf_node = RrfMergeNode()
    result = rrf_node.process(mock_state)
    print("result===",result)

