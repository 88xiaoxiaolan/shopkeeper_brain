from typing import List, Dict, Any, Tuple

from langchain_openai import ChatOpenAI

from knowledge.processor.query_processor.base import BaseNode, T
from knowledge.processor.query_processor.state import QueryGraphState
from knowledge.propmpt.query_prompt import ANSWER_PROMPT
from knowledge.utils.client.ai_clients import AIClients
from knowledge.utils.mongo_history_util import save_chat_message
from knowledge.utils.sse_util import push_sse_event, SSEEvent
from knowledge.utils.task_utils import set_task_result


class AnswerOutputNode(BaseNode):

    name = "answer_output_node"

    def process(self, state: QueryGraphState) -> QueryGraphState:
        """
        核心逻辑：
        1、在state中获取answer
        1.1 有answer -》没有进行三路检索【不用再生成答案，直接返回】--》答案如何推送给前端【流式/非流式：直接将已经生成的内容都推给前端】
        1.2 没有答案 -》进行三路检索【需要llm来生成答案，再返回】--》答案如何推送给前端【流式（SSE） / 非流式（传统）】
        Args:
            state:

        Returns:

        """
        # 1、获取是否是流式
        is_stream = state.get("is_stream","")
        # 2、任务id，一次查询，就是一次任务任何
        task_id = state.get("task_id","")
        # 3、answer
        answer = state.get("answer","")

        if answer:
            # 将答案推送出去
            self._push_exist_answer(task_id,is_stream,answer)

            # 流式是否已经推送
            is_streamed = False
        else:
            # 调用llm生成答案，再选择使用流式/非流式推送
            # 组装提示词
            prompt = self._build_prompt(state)
            state["prompt"] = prompt

            # 调用llm
            self._generate_answer(prompt,is_stream,task_id,state)

            is_streamed = is_stream

        # 4、保存历史对话 - 只要问了问题，就会保存历史对话
        self._save_history_context(state)

        # 5、告诉前端，可以关闭sse通道（event为FINAL）
        if is_stream:
            # data = {} llm生成的答案，已经流过
            # data = {"answer": answer}, 先前生成的答案，没有流过
            data = {} if is_streamed else {"answer": answer}
            push_sse_event(task_id=task_id, event=SSEEvent.FINAL, data=data)

        return state

    def _push_exist_answer(self, task_id:str, is_stream: bool,answer: str):
        # 1、非流式（普通任务队列【任务结果：_tasks_result】）
        if not is_stream:
            set_task_result(task_id=task_id, key="answer", value=answer)
        # else:
        #     # 2、流式（普通任务队列 + SSE）
        #     push_sse_event(task_id=task_id, event=SSEEvent.FINAL, data={"answer": answer})

    def _build_prompt(self, state:QueryGraphState) -> str:
        max_context_chars = self.config.max_context_chars

        # 1、获取用户改写后的问题/商品名
        rewritten_query = state.get("rewritten_query")
        item_names = state.get("item_names") or []

        # 2、构建检索的上下文
        retrieval_context = state.get("reranked_docs") or []
        format_context,usage_chars = self._format_retrieval_context(retrieval_context,max_context_chars)

        # 3、构建历史上下文
        chat_history = state.get("history") or [] #从内存中获取历史对话
        format_history_context = self._format_chat_history(chat_history,usage_chars)

        # 4、格式化提示词模板
        prompt = ANSWER_PROMPT.format(
            context=format_context if format_context else "暂无检索到的上下文",
            history=format_history_context if format_history_context else "暂无检索到的上下文",
            item_names="\n".join(item_names) if item_names else "暂无商品名称",
            question=rewritten_query
        )

        return  prompt

    def _format_retrieval_context(self, retrieval_context:List[Dict[str,Any]],max_context_chars:int) -> Tuple[str,int]:
        """
        格式化检索到的上下文
        自己拼接一些原数据，供llm学习，回答的答案更准确
        Args:
            retrieval_context: 检索到的结果
            max_context_chars: 最大的上下文长度

        Returns:

        """
        formatted_lines = []
        usage = 0
        for index,context in enumerate(retrieval_context,1):
            # 1、获取内容
            content = context.get("content","")
            if not content:
                continue

            # 2、获取元数据
            metadata_content = [f"[文档:{index}]"]

            # 3、定义其他原数据
            for meta_field, template in [("chunk_id","[chunk_id={}]"),("title","[title={}]"),("source","[source={}]"),("url","[url={}]")]:
                field_value = str(context.get(meta_field,"")).strip()

                if field_value:
                    metadata_content.append(template.format(field_value))

            # 4、获取得分
            score = context.get("score")
            if score is not None:
                metadata_content.append(f"[score={float(score):.4f}]")

            # 拼接元数据+内容
            formatted_line = " ".join(metadata_content) + "\n" + content

            # 计算行与行的字符数（\n\n）,这个是已经先预判了要把formatted_lines转化为字符串，里面的数据用\n\n来做段落来分隔
            sep_chars = 2 if formatted_lines else 0

            total_length = len(formatted_line) + sep_chars

            if  total_length + usage > max_context_chars:
                break
            else:
                formatted_lines.append(formatted_line)
                usage += total_length

        return "\n\n".join(formatted_lines), max_context_chars - usage

    def _generate_answer(self, prompt:str, is_stream:bool, task_id:str,state:QueryGraphState):
        # 1、获取llm客户端
        try:
            llm_client = AIClients.get_llm_client(response_format=False)
        except ConnectionError as e:
            self.logger.error(f"获取LLM客户端失败，{str(e)}")
            state["answer"] = "llm暂无任何内容回答"
            return

        # 判断流式
        if is_stream:
            llm_stream_result = self._invoke_stream_llm(task_id,prompt,llm_client)
            state["answer"] = llm_stream_result
        else:
            llm_result = self._invoke_llm(prompt,llm_client)
            # 写入到任务结果中，非流失模式
            set_task_result(task_id=task_id, key="answer", value=llm_result)
            state["answer"] = llm_result

    def _invoke_llm(self, prompt:str, llm_client:ChatOpenAI):
        try:
            llm_response = llm_client.invoke(prompt)
        except Exception as e:
            self.logger.error(f"调用llm失败，{str(e)}")
            return "llm暂无任何内容回答"

        # or "" -> llm_response中的llm_response的值为None / （llm_response,"content",""）-> llm_response的content为None
        llm_content = getattr(llm_response,"content","") or ""

        if not llm_content:
            return "llm暂无任何内容回答"
        return llm_content

    def _invoke_stream_llm(self, task_id:str,prompt:str, llm_client:ChatOpenAI) -> str:
        """

        Args:
            prompt:
            llm_client:

        Returns:

        """
        # 获取全量内容
        accelerate_delta = ""
        # chunk并不是一个，而是分词的一个词
        try:
            for chunk in llm_client.stream(prompt):
                delta = getattr(chunk,"delta","") or ""

                # 把增量放到sse队列中
                push_sse_event(task_id=task_id, event=SSEEvent.DELTA, data={"delta": delta})

                accelerate_delta += delta
        except Exception as e:
            self.logger.error(f"调用llm流失式失败，{str(e)}")
            return "llm暂无任何内容回答"

        return accelerate_delta

    def _save_history_context(self, state:QueryGraphState):
        """
        保存历史对话到mongodb：kb001中的chat_message
        Args:
            state:
        Returns:

        """
        session_id = state.get("session_id")
        rewritten_query = state.get("rewritten_query")
        original_query = state.get("original_query")
        item_names = state.get("item_names") or []
        try:
            # 保存用户角色的历史对话
            save_chat_message(
                session_id=session_id,
                role="user",
                text=original_query,
                rewritten_query=rewritten_query,
                item_names=item_names
            )

            # 保存AI角色的信息
            save_chat_message(
                session_id=session_id,
                role="assistance",
                text=state.get("answer"),
                rewritten_query=rewritten_query,
                item_names=item_names
            )
        except Exception as e:
            self.logger.error(f"保存历史对话到mongodb失败，{str(e)}")

    def _format_chat_history(self, chat_history:List[Dict[str,Any]], usage_chars:int):
        """
        格式化历史上下文
        Args:
            chat_history:
            usage_chars:

        Returns:
        """
        role_map = {"role": "用户", "assistance": "AI"}
        formatted_lines = []
        usage = 0
        for msg in chat_history:
            role = msg.get("role")
            text = msg.get("text")

            if not role or role not in role_map:
                continue

            formatted_line = f"{role_map[role]}: {text}"

            seperator_length = 1 if formatted_lines else 0

            total_length = len(formatted_line) + seperator_length

            if usage + total_length > usage_chars:
                break
            else:
                formatted_lines.append(formatted_line)
                usage += total_length

        return "\n".join(formatted_lines)
