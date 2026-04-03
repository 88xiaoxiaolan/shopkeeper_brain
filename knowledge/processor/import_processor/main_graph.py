"""
编排节点

定义节点
定义条件边
定义顺序边
运行整个pineline图谱的各个节点

"""
import json

from langgraph.constants import END
from langgraph.graph import StateGraph

from knowledge.processor.import_processor.base import setup_logging
from knowledge.processor.import_processor.nodes.document_split_node import DocumentSplitNode
from knowledge.processor.import_processor.nodes.embedding_chunks_node import EmbeddingChunksNode
from knowledge.processor.import_processor.nodes.entry_node import EntryNode
from knowledge.processor.import_processor.nodes.import_milvus_node import ImportMilvusNode
from knowledge.processor.import_processor.nodes.item_name_recognition_node import ItemNameRecognitionNode
from knowledge.processor.import_processor.nodes.md_to_image_node import MarkDownImageNode
from knowledge.processor.import_processor.nodes.pdf_to_md_node import PdfToMdNode
from knowledge.processor.import_processor.state import ImportGraphState



def import_router(state: ImportGraphState):
    if state.get("is_pdf_read_enabled"):
        return "pdf_to_md_node"
    elif state.get("is_md_read_enabled"):
        return "md_to_image_node"
    else:
        return END


def import_graph():
        # 定义运行时候的图状态
        workflow = StateGraph(ImportGraphState) # type: ignore

        # 定义入口节点
        workflow.set_entry_point("entry_node")

        # 添加其他节点
        node_map = {
            "entry_node": EntryNode(),
            "pdf_to_md_node": PdfToMdNode(),
            "md_to_image_node": MarkDownImageNode(),
            "document_split_node": DocumentSplitNode(),
            "item_name_recognition_node": ItemNameRecognitionNode(),
            "embedding_chunk_node": EmbeddingChunksNode(),
            "import_milvus_node": ImportMilvusNode()
        }

        for node_name, node_obj in node_map.items():
            workflow.add_node(node_name, node_obj)

        # 添加条件边
        workflow.add_conditional_edges("entry_node",import_router,{
            "pdf_to_md_node": "pdf_to_md_node",
            "md_to_image_node": "md_to_image_node",
            END: END
        })

        # 添加边
        workflow.add_edge("pdf_to_md_node","md_to_image_node")
        workflow.add_edge("md_to_image_node","document_split_node")
        workflow.add_edge("document_split_node", "item_name_recognition_node")
        workflow.add_edge("item_name_recognition_node", "embedding_chunk_node")
        workflow.add_edge("embedding_chunk_node", "import_milvus_node")
        workflow.add_edge("import_milvus_node", END)

        # 编译
        compiled_graph = workflow.compile()

        return compiled_graph

import_app = import_graph()

def run_import_graph():
    init_state = {
        "import_file_path": r"D:\20251208_study\shopkeeper_brain\knowledge\processor\import_processor\temp_dir\万用表的使用.pdf",
        # "import_file_path": r"D:\20251208_study\shopkeeper_brain\knowledge\processor\import_processor\temp_dir\万用表的使用\hybrid_auto\万用表的使用.md",
        "file_dir": r"D:\20251208_study\shopkeeper_brain\knowledge\processor\import_processor\temp_dir"
    }

    for event in import_app.stream(init_state):
        final_state = {}
        for node_name, node_state in event.items():
            print(f"当前正在执行的节点：{node_name}")
            print(f"当前正在执行的节点状态:{node_state}")

            final_state = node_state

    return final_state

if __name__ == "__main__":
    setup_logging()
    final_state = run_import_graph()
    print(json.dumps(final_state, ensure_ascii=False, indent=4))

    # 整个执行的状态图(方便观察) ascii
    print(import_app.get_graph().print_ascii())
