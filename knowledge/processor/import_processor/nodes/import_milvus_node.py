import json
from dataclasses import dataclass
from typing import Tuple, List, Dict, Any, Sequence, Optional

from pymilvus import MilvusClient, DataType

from knowledge.processor.import_processor.base import BaseNode, setup_logging
from knowledge.processor.import_processor.exceptions import StateFieldError, ValidationError
from knowledge.processor.import_processor.state import ImportGraphState
from knowledge.utils.client.storage_clients import StorageClients

@dataclass
class _SCALAR_FIELD_SPC:
    field_name: str
    datatype: DataType
    max_length: Optional[int] = None

_SCALAR_FIELD:Sequence[_SCALAR_FIELD_SPC] = (
    _SCALAR_FIELD_SPC(field_name="content", datatype=DataType.VARCHAR, max_length=65535),
    _SCALAR_FIELD_SPC(field_name="title", datatype=DataType.VARCHAR, max_length=65535),
    _SCALAR_FIELD_SPC(field_name="parent_title", datatype=DataType.VARCHAR, max_length=65535),
    _SCALAR_FIELD_SPC(field_name="file_title", datatype=DataType.VARCHAR, max_length=65535),
    _SCALAR_FIELD_SPC(field_name="item_name", datatype=DataType.VARCHAR, max_length=65535),
)

class _MilvusIndexBuilder:
    @staticmethod
    def _build_index(milvus_client: MilvusClient):
        index_params = milvus_client.prepare_index_params()
        # milvus 计算处理的稠密向量已经归一化，所以使用COSINE = IP，没有经过归一化，COSINE != IP
        index_params.add_index(field_name="dense_vector", index_name="dense_vector_index", index_type="AUTOINDEX",
                               metric_type="COSINE")
        index_params.add_index(field_name="sparse_vector", index_name="sparse_vector_index",
                               index_type="SPARSE_INVERTED_INDEX", metric_type="IP")

        return index_params

class _MilvusSchemaBuilder:
    """
    负责处理milvus 约束相关的逻辑
    """
    @staticmethod
    def _build_schema(milvus_client: MilvusClient, dim:int):
        """

        Args:
            milvus_client: 客户端
            dim: 稠密向量的维度

        Returns:
        enable_dynamic_field: 动态添加字段，可以在静态基础上再额外添加字段(是指在插入数据的时候，可以多添加字段)
        静态字段：提前定义好的
        如果不这样子设置，那么当添加了新的字段，需要处理
        """
        # 1、创建schema
        schema = milvus_client.create_schema(enable_dynamic_field=True)

        # 2、添加字段
        # 2.1 添加主键字段的约束 auto_id=True
        # 可以自动生成值，并且在一定时间有顺序；后面如果需要用它来查询，必须设置datatype=DataType.INT64；插入数据的时候，无需传入该字段的值
        schema.add_field(field_name="chunk_id", datatype=DataType.INT64, is_primary=True,auto_id = True)

        # 2.2 添加向量字段的约束
        schema.add_field(field_name="dense_vector", datatype=DataType.FLOAT_VECTOR, dim=dim)
        schema.add_field(field_name="sparse_vector", datatype=DataType.SPARSE_FLOAT_VECTOR)

        # 2.3 添加普通字段的约束[5个标量字段]
        for spec in _SCALAR_FIELD:
            kwargs: Dict[str,Any] = {
                "field_name": spec.field_name,
                "datatype": spec.datatype
            }
            if spec.max_length:
                kwargs["max_length"] = spec.max_length
            schema.add_field(**kwargs)

        return schema

class _MilvusInserter:
    def __init__(self, milvus_client: MilvusClient, collection_name: str):
        self.milvus_client = milvus_client
        self.collection_name = collection_name

    def _insert_row(self, chunk: Dict[str, Any]):
        inserted_result = self.milvus_client.insert(collection_name=self.collection_name, data=chunk)

        # 得到每一个chunk的id
        chunk_ids = inserted_result.get("ids")

        # 回填到chunk中
        for chunk_id, chunk in zip(chunk_ids, chunk):
            chunk["chunk_id"] = chunk_id


class ImportMilvusNode(BaseNode):
    """
    角色：充当门面
    """

    name = "import_milvus_node"

    def process(self, state: ImportGraphState) -> ImportGraphState:
        # 1、校验
        validated_chunks, dim = self._validate_state(state)

        # 2、获取Milvus客户端
        try:
            milvus_client = StorageClients.get_milvus_client()
        except ConnectionError as e:
            self.logger.error(f"获取Milvus客户端失败: {str(e)}")
            raise ConnectionError(node_name=self.name, message=f"获取Milvus客户端失败，请检查Milvus服务是否正常")

        # 3、获取集合名称
        collection_name = self.config.chunks_collection

        # 4、创建集合
        self._create_chunk_collection(milvus_client,collection_name,dim)

        # 5、插入向量
        milvus_inserter = _MilvusInserter(milvus_client,collection_name)
        milvus_inserter._insert_row(validated_chunks)

        # 6、备份
        self.backup_chunks(state=state,file_name="chunks_vector_ids.json")

        return state

    def _validate_state(self, state: ImportGraphState) -> Tuple[List[Dict[str,Any]],int]:
        chunks = state.get("chunks")

        # 校验chunks
        if not chunks or not isinstance(chunks, list):
            raise StateFieldError(node_name=self.name, field_name="chunks", expected_type=list, message="chunks 状态字段缺失或无效")

        # 校验每一个chunk对象
        validated_chunks = []
        for index,chunk in enumerate(chunks):
            if not chunk or not isinstance(chunk, dict):
                raise ValidationError(node_name=self.name,message=f"chunk-{index + 1} 类型和期望的类型不匹配，期望的类型{type(chunk).__name__}")

            if chunk.get("dense_vector") and chunk.get("sparse_vector"):
                validated_chunks.append(chunk)
            else:
                self.logger.warning(f"chunk-{index + 1} 缺少向量信息，跳过")

        if not validated_chunks:
            raise ValidationError(node_name=self.name,message="没有向量信息，无法入库")

        dim = len(validated_chunks[0].get("dense_vector"))
        self.logger.info(f"向量维度为{dim}, 有向量的个数{len(validated_chunks)}")

        return validated_chunks,dim

    def _create_chunk_collection(self, milvus_client: MilvusClient, collection_name:str,dim:int) :
        # 判断集合collections是否有
        if milvus_client.has_collection(collection_name):
            self.logger.info(f"集合{collection_name}已存在,跳过创建")
            return

        schema = _MilvusSchemaBuilder._build_schema(milvus_client, dim)

        index_params = _MilvusIndexBuilder._build_index(milvus_client)

        # 创建集合
        milvus_client.create_collection(
            collection_name=collection_name,
            schema=schema,
            index_params=index_params
        )


if __name__ == "__main__":
    setup_logging()
    input_path = r"D:\20251208_study\shopkeeper_brain\knowledge\processor\import_processor\temp_dir\chunks_vector.json"
    with open(input_path, "r", encoding="utf-8") as f:
        chunks = json.load(f)

    init_state = {
        "file_dir": r"D:\20251208_study\shopkeeper_brain\knowledge\processor\import_processor\temp_dir",
        "chunks": chunks
    }
    node = ImportMilvusNode().process(init_state)


