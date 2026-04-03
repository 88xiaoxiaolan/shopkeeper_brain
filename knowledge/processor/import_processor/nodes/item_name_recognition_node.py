import json
from typing import Tuple, List, Dict, Any, Optional

from langchain_core.messages import SystemMessage, HumanMessage
from langchain_openai import ChatOpenAI
from pymilvus import MilvusClient, DataType
from pymilvus.model.hybrid import BGEM3EmbeddingFunction

from knowledge.processor.import_processor.base import BaseNode, T, setup_logging
from knowledge.processor.import_processor.exceptions import StateFieldError, ValidationError, LLMError
from knowledge.processor.import_processor.state import ImportGraphState
from knowledge.propmpt.import_prompt import ITEM_NAME_SYSTEM_PROMPT, ITEM_NAME_USER_PROMPT_TEMPLATE
from knowledge.utils.client.ai_clients import AIClients
from knowledge.utils.client.storage_clients import StorageClients


class ItemNameRecognitionNode(BaseNode):
    name = "item_name_recognition_node"
    def process(self, state: ImportGraphState) -> ImportGraphState:
        # 1、构建商品名识别的LLM的上下文
        # 2、调用LLM模型进行商品名名称
        # 3、将LLM提取到的商品进行嵌入
        # 4、存入Milvus（有很多商品名的向量值 A（xxA） B（XXB） C（xxC））


        # 查询
        # 1、让llm对用于的问题提取商品名（产品名）--- xxxxA
        # 2、商品名向量化
        # 3、查询Milvus对齐导入阶段存入产品名
        # 4、查询到了 -》1（0.8） 2（0.6） 3（0.5）   0.7 ---》xxxxA（检索）
        # 4.1、查询到了 -》1（0.6） 2（0.61） 3（0.58）   0.6 ---》拿出1和2，给用户确认
        # 4.2、查询到了 -》1（0.2） 2（0.1） 3（0.2）   0.6 ---》没有检索到任何商品，告知用户无法识别名称

        """
        职责：
        利用llm提取商品具体的型号
        嵌入
        存入Milvus(MySql：模糊查询的时候不会考虑语义)
        Args:
            state:

        Returns:

        """

        # 1、参数校验
        file_title,chunks,item_name_chunk_k = self._validate_params(state)

        # 2、构建上下文
        item_context = self._prepare_llm_context(chunks,item_name_chunk_k)

        # 3、调用llm模型，提取商品名
        item_name = self._recognition_item_name(item_context,file_title)

        # 4、向量化（嵌入模型：1、OpenAIEmbedding 2、文本嵌入模型（text-embedding-v(x)）-> 阿里灵积服务平台 dashscope 3、bge（bge-m3）-> 混合向量【稠密-》语义相似度】，【稀疏 -》精确】）
        dense_vector, spase_vector = self._embedding_item_name(item_name)

        # 5、入库
        self._insert_milvus(dense_vector,spase_vector,item_name,file_title)

        # 6、回填
        self._fill_item_name(state,chunks,item_name)

        #7、 备份
        self.log_step(step_name="step5----------", message="备份")
        self.backup_chunks(state=state, file_name="chunks_item_name.json")

        return state

    def _validate_params(self, state: ImportGraphState) -> Tuple[str, List, int]:
        """
        参数校验
        Args:
            state:

        Returns:

        """
        # 使用文档标题来兜底
        file_title = state.get("file_title")

        if not file_title:
            raise StateFieldError(node_name=self.name, field_name="file_title", expected_type=str,message="文件名不能为空")

        # 获取chunks来提供vlm的上下文
        chunks = state.get("chunks")

        if not chunks or not isinstance(chunks, list):
            raise StateFieldError(node_name=self.name, field_name="chunks", expected_type=list,message="chunks不能为空")

        # item_name_chunk_k = 3, item_name_chunk_size = 2500
        item_name_chunk_k = self.config.item_name_chunk_k

        if not item_name_chunk_k or item_name_chunk_k <= 0:
            raise ValidationError(node_name=self.name, message="商品名的辅助切片数不合法")

        return file_title,chunks,item_name_chunk_k

    def _prepare_llm_context(self, chunks:list, item_name_chunk_k:int) -> str:
        """
        构建上下文
        Args:
            chunks: 该文档所有的块
            item_name_chunk_k: 使用的块

        Returns:
            上下文信息
        """
        final_context = []
        for index,chunk in enumerate(chunks[:item_name_chunk_k]):
            # 如果不是字典，则跳过
            if not isinstance(chunk,dict):
                continue

            # 获取content作为上下文
            content = chunk.get("content")

            splice_context = f"【切片】 - f{index}-{content}"

            final_context.append(splice_context)

        return "\n".join(final_context)

    def _recognition_item_name(self, item_context:str, file_title:str) -> str:
        """

        Args:
            item_context:
            file_title:

        Returns:

        """
        # 获取llm模型的客户端
        try:
            llm_client: ChatOpenAI = AIClients.get_llm_client(response_format=False)
        except ConnectionError as e:
            self.logger.error(f"openAI LLM 模型提取商品名称失败,降级使用文件标题{file_title}作为商品名称: {str(e)}")
            return file_title

        # 获取llm的系统提示词
        sys_prompt = ITEM_NAME_SYSTEM_PROMPT

        # 获取llm的商品名提示词模板,format 是字符串格式化
        user_prompt = ITEM_NAME_USER_PROMPT_TEMPLATE.format(file_title=file_title,context=item_context)

        # 调用llm模型
        try:
            # 返回AiMessage对象
            llm_response = llm_client.invoke([
                # content里面不能有变量
                SystemMessage(content=sys_prompt),
                HumanMessage(content=user_prompt)
            ])

            # 具体回复
            result = llm_response.content.strip()
            self.logger.info(f"为{file_title}提取的商品名称{result}")

            if not result or result == "UNKNOWN":
                self.logger.error(f"openAI LLM 模型提取商品名称失败,降级使用文件标题{file_title}作为商品名称")
            return result
        except Exception as e:
            self.logger.error(f"openAI LLM 模型提取商品名称失败,降级使用文件标题{file_title}作为商品名称: {str(e)}")
            return file_title

    def _embedding_item_name(self, item_name:str) -> Tuple[Optional[List[float]], Optional[Dict[str,Any]]]:
        """

        Args:
            item_name: 商品名

        Returns:
            dense_vector = Tuple[List[float]
            sparse_vector = Dict[str,Any]]
        """

        try:
            bge_m3_ef: BGEM3EmbeddingFunction = AIClients.get_bge_m3_client()
        except ConnectionError as e:
            self.logger.error(f"获取bge_m3_client失败: {str(e)}")
            return None,None

        try:
            vector_result = bge_m3_ef.encode_documents(documents=[item_name])

            # 获取稠密向量
            dense_vector = vector_result.get("dense")[0].tolist()
            # 获取稀疏向量矩阵
            sparse_array = vector_result.get("sparse")
            # 获取行索引
            start_index = sparse_array.indptr[0]
            end_index = sparse_array.indptr[1]
            # 获取token_id
            token_id = sparse_array.indices[start_index:end_index].tolist()
            weight = sparse_array.data[start_index:end_index].tolist()
            sparse_vector = dict(zip(token_id, weight))

            self.logger.info(f"嵌入模型bge_m3，稠密向量的维度：{len(dense_vector)}")

            return dense_vector,sparse_vector
        except Exception as e:
            self.logger.error(f"嵌入模型bge_m3，计算嵌入向量失败 {str(e)}")
            return None,None

    def _insert_milvus(self, dense_vector:Optional[List[float]], spase_vector:Optional[List[float]], item_name:str, file_title:str):
        """
        milvus中每一行的数据：{dense_vector,spase_vector,item_name,file_title}
        Args:
            dense_vector: 稠密向量
            spase_vector: 稀疏向量
            item_name: 商品名
            file_title: 文件名

        Returns:

        """
        # 1、判断稠密和稀疏向量是否存在
        if not dense_vector or not spase_vector:
            self.logger.info("dense_vector or spase_vector is None")
            return

        # 2、获取milvus的客户端
        try:
            milvus_client = StorageClients.get_milvus_client()
        except ConnectionError as e:
            self.logger.error(f"获取Milvus客户端失败: {str(e)}")
            return

        # milvus （集合 Collection[集合名、schema、index]）
        item_name_collection = self.config.item_name_collection
        try:
            if not milvus_client.has_collection(item_name_collection):
                # 3、创建milvus集合
                self._create_item_name_collection(item_name_collection,milvus_client)

            # 4、构建行
            row = {
                "file_title": file_title,
                "item_name" : item_name,
                "dense_vector" : dense_vector,
                "sparse_vector" : spase_vector,
            }

            # 5、拆入数据
            try:
                insert_result = milvus_client.insert(item_name_collection,row)
                self.logger.info(f"插入Milvus成功，行数：{insert_result},插入结果{insert_result}, 主键值：{insert_result.get("ids")}")
            except Exception as e:
                self.logger.error(f"Milvus 插入失败: {str(e)}")
        except Exception as e:
            self.logger.error(f"构建Milvus失败: {str(e)}")

    def _create_item_name_collection(self, item_name_collection:str, milvus_client:MilvusClient):
        """

        Args:
            item_name_collection: 集合名
            milvus_client: 客户端

        Returns:

        """

        # 1、创建schema
        schema = milvus_client.create_schema()
        # 1.1、创建主键约束
        schema.add_field(field_name="pk", datatype=DataType.INT64, is_primary=True,max_length=10, auto_id = True)
        # 1.2创建标量
        schema.add_field(field_name="item_name", datatype=DataType.VARCHAR, max_length=65535)
        schema.add_field(field_name="file_title", datatype=DataType.VARCHAR, max_length=65535)
        # 1.3创建向量
        schema.add_field(field_name="dense_vector", datatype=DataType.FLOAT_VECTOR, dim=1024)
        schema.add_field(field_name="sparse_vector", datatype=DataType.SPARSE_FLOAT_VECTOR)

        # 2、创建index
        index_params = milvus_client.prepare_index_params()
        # milvus 计算处理的稠密向量已经归一化，所以使用COSINE = IP，没有经过归一化，COSINE != IP
        index_params.add_index(field_name="dense_vector", index_name="dense_vector_index",index_type="AUTOINDEX", metric_type="COSINE")
        index_params.add_index(field_name="sparse_vector", index_name="sparse_vector_index",index_type="SPARSE_INVERTED_INDEX", metric_type="IP")

        milvus_client.create_collection(
            collection_name=item_name_collection,
            schema=schema,
            index_params=index_params
        )

        self.logger.info(f"Milvus 创建集合{item_name_collection}成功")

    def _fill_item_name(self, state:ImportGraphState, chunks:List[Dict[str, Any]], item_name:str):
        """
        回填数据
        1、在chunk中
        2、在State中
        Args:
            state:
            chunks:
            item_name:

        Returns:

        """
        for chunk in chunks:
            chunk["item_name"] = item_name

        state["item_name"] = item_name


if __name__ == "__main__":
    setup_logging()
    chunk_path = r"D:\20251208_study\shopkeeper_brain\knowledge\processor\import_processor\temp_dir\chunks.json"

    with open(chunk_path, "r", encoding="utf-8") as f:
        # 从文件中读取json文件数据，并且解析为python对象
        chunks = json.load(f) # 读取的是List
        # chunks = f.read() # 读取的是str

    init_state = {
        "file_dir": r"D:\20251208_study\shopkeeper_brain\knowledge\processor\import_processor\temp_dir",
        "file_title": "万用表的使用",
        "chunks": chunks
    }
    res = ItemNameRecognitionNode().process(init_state)

