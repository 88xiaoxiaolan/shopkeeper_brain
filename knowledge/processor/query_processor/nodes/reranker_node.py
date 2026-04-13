import math
from typing import List, Dict, Any

from knowledge.processor.query_processor.base import BaseNode
from knowledge.processor.query_processor.state import QueryGraphState
from knowledge.utils.client.ai_clients import AIClients


class RerankerNode(BaseNode):
    name = "reranker_node"

    def process(self, state: QueryGraphState) -> QueryGraphState:
        # 1、获取用户问题
        user_query = state.get("rewritten_query") or state.get("original_query")

        # 2、获取两路搜索的结果
        reranker_outputs: List[Dict[str, Any]] = self._collect_reranker_inputs(state)

        # 3、精排
        refine_results:List[Dict[str, Any]] = self._refine_search(user_query, reranker_outputs)

        # 4、动态截取
        reranked_docs = self._cliff_cutoff(refine_results,self.config.rerank_min_top_k,self.config.rerank_max_top_k)

        state["reranked_docs"] = reranked_docs

        return state

    def _collect_reranker_inputs(self, state: QueryGraphState) -> List[Dict[str, Any]]:
        final_docs = []

        # 1、获取本地搜索结果
        rrf_chunks = state.get("rrf_chunks") or []
        for chunk in rrf_chunks:
            if not chunk or not isinstance(chunk, dict):
                continue

            content = chunk.get("content","")
            if not content:
                continue

            title = chunk.get("title","")

            chunk_id = chunk.get("chunk_id")

            format_local_doc = self.format_doc(chunk_id=chunk_id,content=content,title=title,source="local")
            final_docs.append(format_local_doc)

        # 获取网络检索内容
        web_search_docs = state.get("web_search_docs") or []
        for doc in web_search_docs:
            if not doc or not isinstance(doc, dict):
                continue

            content = doc.get("snippet","")
            title = doc.get("title","")
            url = doc.get("url","")

            format_web_doc = self.format_doc(content=content,title=title,url=url,source="web")
            final_docs.append(format_web_doc)

        self.logger.info(f"最终的reranker输入个数为：{len(final_docs)}")
        return final_docs

    def format_doc(self, chunk_id:int=None, content:str="", title:str="", url:str="",source:str=""):
        """

        Args:
            chunk_id:
            content:
            title:
            source:
            url:

        Returns:

        """
        return {
            "chunk_id": chunk_id,
            "content": content,
            "title": title,
            "url": url,
            "source": source,
        }

    def _refine_search(self, user_query:str, reranker_outputs:List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        reranker 进行精排打分 & 排序
        Args:
            user_query: 用户的查询
            reranker_outputs: 本地和网络搜索结果

        Returns:
        """
        # 1、获取重排序模型
        try:
            reranker_client = AIClients.get_bge_m3_reranker_client()
        except ConnectionError as e:
            self.logger.error(f"获取重排序模型失败: 原因{str(e)}")
            return [{**doc, "score": None} for doc in reranker_outputs]

        # 2、构建q->d 的pair对
        query_doc_pairs = [(user_query,doc.get("content","")) for doc in reranker_outputs]

        # 3、计算,bge-m3-reranker 计算出来的分数，是（负无穷，正无穷）
        try:
            scores = reranker_client.compute_score(query_doc_pairs)

            doc_cors = [{**doc, "score": self._sigmoid(float(score))} for doc,score in zip(reranker_outputs,scores)]

            sorted_docs = sorted(doc_cors,key=lambda x:x["score"],reverse=True)

            return sorted_docs
        except Exception as e:
            self.logger.error(f"重排序模型计算分数失败: 原因{str(e)}")
            return [{**doc,"score": None} for doc in reranker_outputs]

    def _cliff_cutoff(self, refine_results:List[Dict[str, Any]], min_top_k:int, max_top_k:int) -> List[Dict[str, Any]]:
        """
        动态截取
        Args:
            refine_results: 精排结果
            min_top_k: 最小topk
            max_top_k: 最大topk

        Returns:

        """
        # 1、定义两个索引
        upper_bound = min(max_top_k,len(refine_results))
        lower_bound = min(min_top_k,upper_bound)
        cut_off = upper_bound
        max_gap = 0

        # 2、遍历
        for i in range(0,upper_bound-1):
            current_score = refine_results[i].get("score")
            next_score = refine_results[i+1].get("score")

            if not current_score or not next_score:
                continue

            gab = abs(current_score - next_score)

            if gab >= 0.15 and gab > max_gap:
                max_gap = gab
                cut_off = i+1
                self.logger.info(f"位置{i + 1}发生断崖")

        # 不管在哪里断崖，至少保留lower_bound个
        cut_off = max(lower_bound,cut_off)
        cut_docs = refine_results[:cut_off]

        return  cut_docs

    @staticmethod
    def _sigmoid(score: float):
        return 1.0 / (1.0 + math.exp(-score))


if __name__ == '__main__':
    mock_state = {
        "rewritten_query": "怎么测这块主板的短路问题？",
        "rrf_chunks": [
            {"chunk_id": "local_1", "title": "主板维修手册",
             "content": "主板短路通常表现为通电后风扇转一下就停，可以使用万用表的蜂鸣档测量。"},
            {"chunk_id": "local_2", "title": "闲聊",
             "content": "今天中午去吃猪脚饭吧，这块主板外观很漂亮。"},
        ],
        "web_search_docs": [
            {"url": "https://example.com/repair", "title": "短路查修指南",
             "snippet": "主板通电前先打各主供电电感的对地阻值，阻值偏低就是短路。"},
            {"url": "https://example.com/news", "title": "科技新闻",
             "snippet": "苹果发布新款手机，A系列芯片性能提升20%。"},
        ],
    }

    print("【输入状态】:")
    print(f"  查询: {mock_state['rewritten_query']}")
    print(f"  本地文档: {len(mock_state['rrf_chunks'])} 篇")
    print(f"  网络文档: {len(mock_state['web_search_docs'])} 篇")
    print("-" * 60)

    node = RerankerNode()
    result = node.process(mock_state)
