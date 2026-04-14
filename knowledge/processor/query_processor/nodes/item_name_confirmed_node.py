import json
import logging
import re
from json import JSONDecoder, JSONDecodeError
from typing import Dict, Any, List, Tuple

from langchain_core.messages import SystemMessage, HumanMessage

from knowledge.processor.query_processor.base import BaseNode, T
from knowledge.processor.query_processor.config import get_config
from knowledge.processor.query_processor.state import QueryGraphState
from knowledge.propmpt.query_prompt import ITEM_NAME_USER_EXTRACT_TEMPLATE
from knowledge.utils.client.ai_clients import AIClients
from knowledge.utils.client.storage_clients import StorageClients
from knowledge.utils.embedding_util import generate_bge_m3_hybrid_vectors
from knowledge.utils.milvus_util import create_hybrid_search_requests, execute_hybrid_search_query
from knowledge.utils.mongo_history_util import get_recent_messages

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class _ItemNameExtractor:
    def extract_item_name(self,original_query:str, history_context:str) -> Dict[str,Any]:
        """
        提取商品名
        Args:
            original_query: 用户原始查询
            history_context: 历史对话
        Returns:
            Dict[str,Any]:
                "item_names":商品名
                "rewritten_query":重写后的查询
        """
        # 1、定义llm输出的默认结果
        llm_result = {
            "item_names": [],
            "rewritten_query": original_query
        }


        # 2、获取llm的客户端
        try:
            llm_client = AIClients.get_llm_client()
        except ConnectionError as e:
            logger.error(f"获取llm客户端失败: {str(e)}")
            return llm_result

        # 3、获取商品名提取的提示词
        item_name_system_prompt = "你是一名商品提示词专家，请你从用户的问题以及历史对话中提取相关的商品名以及改写原始查询"
        item_name_user_prompt = ITEM_NAME_USER_EXTRACT_TEMPLATE.format(
            history_text = history_context if history_context else "暂无历史上下文",
            query = original_query
        )

        # 4、调用llm
        try:
            llm_response = llm_client.invoke([
                    SystemMessage(content=item_name_system_prompt),
                    HumanMessage(content=item_name_user_prompt)
                ])
        except Exception as e:
            logger.error(f"调用llm模型提取商品名失败: {str(e)}")
            return llm_result

        # 5、获取llm的回复
        # "{\n item_names”:["RS-12数字万用表"]\n "rewritten_query”:"RS-12数字万用表的使用方法是什么?"\n)}'
        llm_response_content = llm_response.content

        # 6、判断llm的回复是否为空
        if not llm_response_content:
            return llm_result

        # 7、llm的回复如果存在，则对数据进行清洗和反序列化
        parsed_result: Dict[str, Any] = self._clean_and_parse_result(llm_response_content)

        logger.info(f"llm模型提取商品名经过解析后的结果: {parsed_result}")
        return parsed_result

    def _clean_and_parse_result(self, llm_response_content:str) -> Dict[str, Any]:
        """
        对数据进行清洗 和 反序列化
        Args:
            llm_response_content: llm回复的内容
        Returns:
            {
                "item_names": ["商品A", "商品B"],
                "rewritten_query": "关于商品A和商品B，..."
            }
        """

        # 1、去除json代码块围栏标记（在这里正常我们不会有，因为在调用llm的时候，返回的是json格式的字符串，但是如果换了模型，或者底层API升级，可能会有json代码块围栏标记，这里做防御编程）
        # 第一行：去除开头的 ```json 和后面的空白
        cleaned = re.sub(r"^```(?:json)?\s*", "", llm_response_content.strip())
        # 第二行：去除结尾的 ``` 和前面的空白
        content = re.sub(r"\s*```$", "", cleaned)

        # 2、解析 字符串 -》 字典
        try:
            json_content: Dict[str, Any] = json.loads(content)

            # 2.1拿item_names 字段
            row_item_names = json_content.get("item_names")

            # 2.2 判断item_name字段
            if not isinstance(row_item_names, list):
                item_names = []
            else:
                item_names = [item_name.strip() for item_name in row_item_names if isinstance(item_name, str) and item_name.strip()]

            # 2.3 rewritten_query 字段
            row_rewritten_query = json_content.get("rewritten_query")

            # 2.4 判断rewritten_query字段
            if not isinstance(row_rewritten_query, str):
                rewritten_query = ""
            else:
                rewritten_query = row_rewritten_query.strip()

            # 返回
            return {
                "item_names": item_names,
                "rewritten_query" : rewritten_query
            }

        except JSONDecodeError as e:
            logger.error(f"数据{llm_response_content}, 清洗和反序列化失败: {str(e)}")
            raise JSONDecodeError(msg=e.msg,doc=e.doc,pos=e.pos)

class _ItemNameAligner:
    def __init__(self):
        self._config = get_config()

    def search_align_item_name(self, item_names:List[str]) -> Tuple[List[str],List[str]]:
        """
              根据商品名 到 milvus中搜索，并且对齐，返回确定的商品列表 和 模糊的商品列表
              Args:
                  item_names: 商品名
              Returns:
        """

        # 1、混合搜索
        search_result:List[Dict[str,Any]] = self._search_vector(item_names)
        if not search_result:
            return [],[]

        # 2、根据混合搜索的结果，做对齐
        confirmed, options = self._align(search_result)

        # 3、分数差异化过滤
        if len(confirmed) > 1:
            confirmed = self._item_name_filter_score(confirmed,search_result)

        # 4、返回confirmed 和 options 容器
        return confirmed,options

    def _search_vector(self, item_names:List[str]) -> List[Dict[str,Any]]:
        """
        根据llm提取到的商品名，到milvus中对齐商品名
        Args:
            item_names: llm提起的商品名

        Returns:
        [{"extracted_name":llm提取的商品名1，"matches":[{score1:0.78, item_name:商品名1},{score12:0.55, item_name:商品名12}]},
        {"extracted_name":llm提取的商品名2，"matches":[{score2:0.78, item_name:商品名2},{score22:0.55, item_name:商品名22}]}]

        向量数据库中查询到的文档1: {}
        """

        # 1、获取milvus客户端
        try:
            milvus_client = StorageClients.get_milvus_client()
        except ConnectionError as e:
            logger.error(f"获取milvus客户端失败: {str(e)}")
            return []

        # 2、获取bge-m3 模型
        try:
            bge_m3_client = AIClients.get_bge_m3_client()
        except ConnectionError as e:
            logger.error(f"获取bge_m3_客户端失败: {str(e)}")
            return []

        # 3、商品名向量化,得到所有商品的向量 {dense:[商品1[]，商品2[]],sparse:[商品1{token_id:1,weight:0.2},商品2{token_id:11,weight:0.12}]}
        try:
            hybrid_vector_result = generate_bge_m3_hybrid_vectors(bge_m3_client, item_names)
        except Exception as e:
            logger.error(f"商品名向量化失败: {str(e)}")
            return []

        # 4、混合向量上搜索
        # 4.1 构建稠密以及稀疏向量 的AnnSearchRequest
        final_search_result = []
        for index, item_name in enumerate(item_names):
            hybrid_requests = create_hybrid_search_requests(dense_vector=hybrid_vector_result["dense"][index],sparse_vector=hybrid_vector_result["sparse"][index])

            # 4.2 构建WeightedReranker实例
            # 4.3 调用hybrid__search，返回AnnSearchResult
            hybrid_request_result = execute_hybrid_search_query(
                milvus_client = milvus_client,
                collection_name = self._config.item_name_collection,
                search_requests = hybrid_requests,
                ranker_weights = (0.5, 0.5),  # 跟search_requests顺序有关
                norm_score = True,
                limit = 5,
                output_fields = ["item_name"]
            )

            # 拿到解析结果
            matches = [{"score": item["distance"], "item_name": item["entity"]["item_name"]} for item in
                       (hybrid_request_result[0] if hybrid_request_result else [])]

            # 添加到结果集中
            final_search_result.append({
                "extracted_name": item_name,
                "matches" : matches
            })

        # 返回结果
        return final_search_result

    def _align(self, search_result:List[Dict[str,Any]]) -> Tuple[List[str],List[str]]:
        """
        主要职责：
            把什么样子的item_name 放入到conformed / options 中
            制定规则：
               1、如果商品名的分数>0.7，放到confirmed中
               2、0.45 < 商品名的分数 <= 0.7, 放入到options中
               3、 商品名的分数 <= 0.45，都不放
            规则中的阈值：0.7 / 0.45，根据压测得到（构建数据集 1、询问的方式，llm提取到的商品名 2、构建阈值列表 3、跑完整个流程，看哪个阈值最好）

            放入confirmed的条件：
                商品名的分数>0.7；从milvus中搜索到的商品名=llm提取到的商品名 或者 从milvus中搜索到的商品名个数=1；当前满足条件的商品名不在confirmed中,那么把商品名放入到confirmed中
                （因为商品名只要在confirmed中就会走搜索，不用管options了，所以在判断的时候，也不用管是否在options中）
            放入options的条件：
                如果商品名的分数>0.7，如果第一个跟第二个的分数之差 <0.15, 如果不在options 和 confirmed中，那么取前三个都放入到options中
                0.45 < 商品名的分数 <= 0.7，并且商品名不在options 和 confirmed中，提取前三个，放入到options中
                （由于如果商品名在confirmed中，那么就不需要options了，就直接走搜索了，所以每次把商品名添加到options的时候，需要看是否在confirmed中，并且为了重复，也需要看是否在options中）


        Args:
            search_result: 从向量数据库中检索到的向量
            [{"extracted_name":llm提取的商品名1，"matches":[{score1:0.78, item_name:商品名1},{score12:0.55, item_name:商品名12}]},
            {"extracted_name":llm提取的商品名2，"matches":[{score2:0.78, item_name:商品名2},{score22:0.55, item_name:商品名22}]}]

        Returns:
            confirmed:[商品名1],options:[商品名2]
        """

        confirmed, options = [], []

        # 遍历检索结果
        for item_search_result in search_result:
            # 拿到llm提取到的商品名
            llm_item_name = item_search_result.get("extracted_name")

            # 拿到每一个商品名在milvus搜索到的结果
            item_name_matches = item_search_result.get("matches")

            # 对每一个商品名在milvus搜索到的结果进行排序
            sorted_item_name_matches = sorted(item_name_matches,key=lambda x: x["score"], reverse=True)

            # 收集分数 > 0.7 的，是高置信度
            # sorted_item_name_matches -> [{score1: 0.78, item_name: 商品名1}, {score12: 0.55, item_name: 商品名12}]
            high = [item for item in sorted_item_name_matches if item["score"] > self._config.item_name_high_confidence]

            # 是高置信度
            if high:
                # 看是否跟llm提取的商品名一致，如果一致，最可信  -> 一般不多
                extract = next((h for h in high if h.get("item_name") == llm_item_name),None)

                if extract:
                    picked = extract.get("item_name")
                    if picked not in confirmed:
                        confirmed.append(picked)
                elif len(high) == 1:
                    picked = high[0].get("item_name")
                    if picked not in confirmed:
                        confirmed.append(picked)
                else:
                    # 有多个分数比较高的商品 [0.90，0.8，0.76]
                    # 看第一个 跟 第二个 的gap 是否大于阈值，如果大于阈值0.15，则添加到confirmed中，否则最多只取3个放到options中
                    top_score = high[0]["score"]
                    if top_score - high[1]["score"]  > self._config.item_name_score_gap:
                        picked = high[0].get("item_name")
                        if picked not in confirmed:
                            confirmed.append(picked)
                    else:
                        for m in high[:self._config.item_name_max_options]:
                            picked = m.get("item_name")
                            if picked not in options and picked not in confirmed:
                                options.append(picked)

            else:
                # 分数 > 0.45, 不在options中，也不在confirm中，才会添加到options中
                mid = [item for item in sorted_item_name_matches if
                       item["score"] >= self._config.item_name_mid_confidence and item.get(
                           "item_name") not in options and
                       item.get("item_name") not in confirmed]
                if mid:
                    for m in mid[:self._config.item_name_max_options]:
                        options.append(m.get("item_name"))

        # 最终options中只保留3个给前端看，因为如果太多了，没有必要
        return confirmed,options[:self._config.item_name_max_options]

    def _item_name_filter_score(self, confirmed:List[str], search_result:List[Dict[str, Any]]) -> List[str]:
        """
        把最终得到的confirmed根据分数来进行过滤，防止llm误加了一些商品名
        Args:
            confirmed: 确认的商品名
            search_result: 根据milvus搜索到的商品
            [{"extracted_name":llm提取的商品名1，"matches":[{score1:0.78, item_name:商品名1},{score12:0.55, item_name:商品名12}]},
            {"extracted_name":llm提取的商品名2，"matches":[{score2:0.78, item_names:商品名2},{score22:0.55, item_name:商品名22}]}]

        Returns:
        把confirmed根据分数来进行过滤
        """
        item_name_score = {}
        for item in search_result:
            matches = item.get("matches",[])
            for m in matches:
                score = m.get('score', 0)
                item_name = m.get("item_name",0)
                if item_name in confirmed:
                    item_name_score[item_name] = max(item_name_score.get(item_name, 0), score)

        # 防御性编程，如果没有收集到任何的评分，则直接返回confirmed
        if not item_name_score:
            return confirmed

        max_score = max(item_name_score.values())

        # 如果分数max - 当前的分数 < 0.15, 那么认为该商品名是可信的
        return [ name for name,score in item_name_score.items() if max_score - score <= self._config.item_name_score_gap]

class ItemNameConfirmedNode(BaseNode):
    name = "item_name_confirmed_node"

    def __init__(self):
        super().__init__()
        self._extractor = _ItemNameExtractor()
        self._aligner = _ItemNameAligner()

    def process(self, state: QueryGraphState) -> QueryGraphState:
        """
        主要职责：
        1、利用llm从用户原始查询中提取商品名，以及改写原始查询
        1.1、如果llm提取到了商品名，才进行第二步，去milvus中对齐
        1.2、如果llm提取到了商品名，直接返回
        2、根据Milvus中存储的商品名来对齐（为了检索更加准确：三路检索都会利用该节点提取到的商品名，因此，如果直接用llm提取到的商品，下游的三路检索在过滤的时候，条件极其精准，导致检索到的噪音很多，llm最终输出幻觉很大）
        最终不是要llm的商品名，而是milvus中的商品名，因为milvus中每一个chunk都会关联milvus自己的商品名
        3、决策（该走下去，还是回头）

        两个容器：
        1、confirmed: 如果是精确的商品名， ----> 向confirm中添加精确
        2、options：如果不是精确的商品名，可能找到多个相似的 ----> 向options中添加不精确的
        利用两个容器产生三个分支：检索 / 用户确认 / 抱歉

        confirmed有，options没有:
            state["answer"] 不给，直接三路检索
            获取到三路检索结果
            把三路检索到的结果（RRF、ReRanker）给llm
            llm生成答案
            更新state["answer"]

        confirmed没有，options有:
            state["answer"]返回：
            1、返回候选商品名【不精准】，给用户下一步确认使用

        confirmed没有，options没有:
            2、没有任何商品名，返回抱歉，没有找到你询问的关于任何商品的名字

        Args:
            state:

        Returns:
        """

        # 1、获取用户的原始问题
        original_query = state.get("original_query")

        # 2、获取历史对话（mongodb）
        session_id = state.get("session_id")
        # limit=10, 条数是10条，但是只有5轮，一轮有query 和 answer 两条数据
        history_context = get_recent_messages(session_id=session_id,limit=10)
        formatted_history = []
        for h in history_context:
            role = h.get("role")
            text = h.get("text")
            content = f"角色:{role},内容:{text}"
            formatted_history.append(content)
        formatted_history_str = " ".join(formatted_history)

        # 直接从mondb中获取的历史对话
        state["history"] = history_context

        # 3、利用llm进行商品名提取和查询重写
        # {'item_names': ['RS-12 数字万用表'], 'rewritten_query': 'RS-12 数字万用表的使用方法是什么？'}
        llm_result: Dict[str:Any] = self._extractor.extract_item_name(original_query, formatted_history_str)

        # 3.1、获取llm的结果
        item_names = llm_result.get("item_names")
        rewritten_query = llm_result.get("rewritten_query")

        # 4、根据item_names 做判断
        if item_names:
            confirmed,options = self._aligner.search_align_item_name(item_names)
        else:
            confirmed,options = [],[]

        # 5、决策
        self._decide(confirmed,options,state,rewritten_query)

        self.logger.info(f"确认商品名列表：{confirmed}，候选商品列表：{options}")
        return state

    def _decide(self, confirmed:List[str], options:List[str], state:QueryGraphState, rewritten_query:str):
        """
        根据confirmed、options 来判断继续检索还是继续返回用户的信息
        Args:
            confirmed: 已确认的商品列表
            options: 候选的商品列表
            state: 查询状态
            rewritten_query: 问题重写
        Returns:
        """

        if confirmed:
            state["item_names"] = confirmed
            state["rewritten_query"] = rewritten_query
        elif options:
            state["answer"] = (
                f"我不确定您指的是哪款产品。"
                f"您是在询问以下产品吗：{'、'.join(options)}？"
            )
        else:
            state["answer"] = (
                "抱歉，我无法识别您询问的具体产品名称，请提供更准确的产品名称或型号。"
            )


if __name__ == '__main__':
    item_name_confirmed_node = ItemNameConfirmedNode()

    init_state = {
        # "original_query": "请问一哈下RS-12数字万用表怎么使用的呢？"

        # 确认商品名列表：['RS-12 数字万用表']，候选商品列表：[]
        # llm模型提取商品名经过解析后的结果: {'item_names': ['RS-12 数字万用表'], 'rewritten_query': 'RS-12 数字万用表的使用方法是什么？'}
        # "original_query": "请问一哈下RS-12数字万用表 和 数字万用表怎么使用的呢？"

        # 确认商品名列表：['RS-12 数字万用表', 'H3C LA2608 室内无线网关']，候选商品列表：[]
        # ['RS-12 数字万用表', 'H3C LA2608室内无线网关'], 'rewritten_query': 'RS-12 数字万用表和 H3C LA2608室内无线网关的使用方法是什么？'}
        "original_query": "请问一哈下RS-12数字万用表 和 H3C LA2608室内无线网关 怎么使用的"
    }
    res  = item_name_confirmed_node.process(init_state)
    print("res====",res)

