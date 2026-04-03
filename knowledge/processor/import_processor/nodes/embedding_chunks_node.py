from typing import List, Dict, Any
from pymilvus.model.hybrid import BGEM3EmbeddingFunction
from pathlib import Path
import json
from knowledge.processor.import_processor.base import BaseNode, setup_logging
from knowledge.processor.import_processor.exceptions import StateFieldError, ValidationError, EmbeddingError
from knowledge.processor.import_processor.state import ImportGraphState
from knowledge.utils.client.ai_clients import AIClients


class EmbeddingChunksNode(BaseNode):

    name = "embedding_chunk_node"

    def process(self, state: ImportGraphState) -> ImportGraphState:

        # 1、校验chunks的状态
        self.log_step("step1",f"校验chunks 的状态")
        chunks:List[Dict[str,Any]] = self._validate_state(state)

        # 2、获取嵌入模型
        self.log_step("step2",f"获取嵌入模型")
        bge_m3_client = self._get_bge_m3_client()

        # 3、批量嵌入
        # 3.1 获取批量阈值
        batch_size = self.config.embedding_batch_size
        # 3.2 获取chunks的总数
        total_chunks = len(chunks)
        final_chunks = []
        for index in range(0, total_chunks, batch_size):
            # 当前的一批
            bath_chunks = chunks[index:index+batch_size]
            # 当前这一批最后一个编号
            bath_end = index+batch_size
            self.logger.info(f"嵌入批次 【{index+1}-{bath_end}】/ {total_chunks}")

            current_chunks = self._embed_chunks(bath_chunks,bge_m3_client)
            final_chunks.extend(current_chunks)

        # 4、更新state中的chunks
        state["chunks"] = final_chunks

        # 5、 备份
        self.log_step(step_name="step3----------", message="备份")
        self.backup_chunks(state=state, file_name="chunks_vector.json")

        # 6、返回
        return state

    def _validate_state(self, state: ImportGraphState) -> List[Dict[str,Any]]:
        # 获取chunks
        chunks = state.get("chunks")

        # 校验chunks
        if not chunks or not isinstance(chunks, list):
            raise StateFieldError(node_name=self.name, field_name="chunks", expected_type=list, message="chunks 状态字段缺失或无效")

        for index,chunk in enumerate(chunks):
            # 校验chunk是否是字典
            if not chunk or not isinstance(chunk, dict):
                raise ValidationError(node_name=self.name, message=f"chunk-{index+1} 类型和期望的类型不匹配，期望的类型{type(chunk).__name__}")

        return chunks

    def _get_bge_m3_client(self):
        try:
            bge_m3_client = AIClients.get_bge_m3_client()
        except ConnectionError as e:
            self.logger.info(f"获取嵌入模型bge-m3失败,{str(e)}")
            raise EmbeddinigError(node_name=self.name, message=f"获取嵌入模型失败，请检查模型服务是否正常")
        return bge_m3_client

    def _embed_chunks(self,bath_chunks:List[Dict[str,Any]],bge_m3_client:BGEM3EmbeddingFunction):
       """

       Args:
           bath_chunks:
           bge_m3_client:

       Returns:

       """
       # 1、获取嵌入的内容
       embedding_documents = [f"{chunk.get("item_name",'')}\n{chunk.get("content")}" for chunk in bath_chunks]

       # 2、嵌入
       try:
           vector_result = bge_m3_client.encode_documents(documents=embedding_documents)
       except Exception as e:
           raise EmbeddingError(node_name=self.name, message=f"嵌入失败，请检查模型服务是否正常")

       if not vector_result:
           raise EmbeddingError(node_name=self.name, message=f"嵌入失败，没有获取到嵌入向量")

       # 3、获取稠密向量
       sparse_csr = vector_result.get("sparse")
       for i, chunk in enumerate(bath_chunks):
           chunk["dense_vector"] = vector_result.get("dense")[i].tolist()
           chunk["sparse_vector"] = self._extract_sparse_vector(sparse_csr, i)

       return bath_chunks


    def _extract_sparse_vector(self,sparse_csr, index: int) -> Dict[str, Any]:
        # 获取行索引
        start_index = sparse_csr.indptr[index]
        end_index = sparse_csr.indptr[index+1]
        # 获取token_id
        token_id = sparse_csr.indices[start_index:end_index].tolist()
        weight = sparse_csr.data[start_index:end_index].tolist()
        # 返回单个向量的稀疏向量
        sparse_vector = dict(zip(token_id, weight))
        return sparse_vector

if __name__ == "__main__":
    setup_logging()

    base_dir = Path(
        r"D:\20251208_study\shopkeeper_brain\knowledge\processor\import_processor\temp_dir"
    )
    input_path = base_dir / "chunks_item_name.json"
    # output_path = base_dir / "chunks_vector.json"

    with open(input_path, "r", encoding="utf-8") as f:
        chunks = json.load(f)

    state = {
        "file_dir":  r"D:\20251208_study\shopkeeper_brain\knowledge\processor\import_processor\temp_dir",
        "chunks": chunks
    }

    chunks_vectors = EmbeddingChunksNode().process(state)
    #
    # with open(output_path, "w", encoding="utf-8") as f:
    #     json.dump(chunks_vectors, f, ensure_ascii=False, indent=4)
    #
    # print(f"向量生成完成，已经保存到{output_path}")
