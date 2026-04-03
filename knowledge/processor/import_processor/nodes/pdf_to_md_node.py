import json
import subprocess
import time
from pathlib import Path
from typing import Tuple
from knowledge.processor.import_processor.base import BaseNode, setup_logging
from knowledge.processor.import_processor.exceptions import StateFieldError, PdfConversionError
from knowledge.processor.import_processor.state import ImportGraphState


class PdfToMdNode(BaseNode):
    name = "pdf_to_md_node"

    # 将pdf文件转化为markdown文件
    def process(self, state: ImportGraphState) -> ImportGraphState:
        # 1、获取输入和输出路径
        import_file_path, file_dir = self._validate_state(state)

        # 2、pdf -》 md 解析
        parse_code = self._execute_mineru_parse(import_file_path, file_dir)
        if parse_code !=0:
            raise PdfConversionError(node_name=self.name, message="pdf 转换为 md 失败")

        # 3、得到md文件
        md_path = self._get_md_path(import_file_path,file_dir)

        # 4、更新参数
        state["md_path"] = md_path

        return state


    def _validate_state(self, state: ImportGraphState) -> Tuple[Path, Path]:
        """
        :param state: 导入图谱的节点状态
        :return: 输入和输出路径 Tuple[Path, Path]
        """

        self.log_step(step_name="pdf_to_md_node",message="开始校验文件")

        # 获取输入和输出路径
        import_file_path = state.get("import_file_path","")
        file_dir = state.get("file_dir","")

        # 判断导入路径是否是空
        if not import_file_path:
            raise StateFieldError(node_name=self.name, field_name="import_file_path",expected_type=str,message="导入文件路径不能为空")

        # 转话为Path对象
        import_file_path_obj = Path(import_file_path)

        # 判断文件是否是一个正常的路径
        if not import_file_path_obj.exists():
            raise StateFieldError(node_name=self.name, field_name="import_file_path",expected_type=Path,message="导入文件路径不存在")

        # 判断输出路径是否是空
        if not file_dir:
            raise StateFieldError(node_name=self.name, field_name="file_dir", expected_type=str,message="输出文件路径不能为空")

        # 转化为Path对象
        file_dir_obj = Path(file_dir)

        # 判断文件是否是一个正常的路径
        if not file_dir_obj.exists():
            raise StateFieldError(node_name=self.name, field_name="file_dir", expected_type=Path,message="输出文件路径不存在")


        self.logger.info(f"解析文件的路径: {import_file_path_obj}")
        self.logger.info(f"输出文件路径: {file_dir_obj}")

        return import_file_path_obj, file_dir_obj

    def _execute_mineru_parse(self, import_file_path: Path, file_dir: Path) -> int:
        """
        :param import_file_path: 输入路径
        :param file_dir: 输出路径
        :return: 是否解析成功
        0：解析成功（mineru底层是这么设计的，所以0才是成功）
        非0：解析失败
        """

        self.log_step(step_name="pdf_to_md_node", message="开始执行pdf到md转化")

        # 定义command
        command = [
            "mineru",
            "-p",
            str(import_file_path),
            "-o",
            str(file_dir),
            "--source",
            "local",
            # "-b",
            # "hybrid-auto-engine"  # 混合后端，自动调度 CPU/GPU
        ]
        start_time = time.time()

        # 利用子进行解析pdf
        # 我们用一根管子，把正常日志 和 错误日志，放在一起，输出
        process = subprocess.Popen(
            args=command,
            bufsize=1, #实时输出，每次遇到\n就会把日志输出
            stdout=subprocess.PIPE, # 正常的日志
            stderr=subprocess.STDOUT, # 错误的日志
            encoding="utf-8", # 字符串的编码
            text=True, # 输出是字符串，默认是二进制流
            errors="replace" # 当遇到特殊的字符，处理成? 或者 菱形
        )

        # 实时打印日志
        for line in process.stdout:
            self.logger.info(f"mineru产生的日志: {line}")


        # 得到解析的结果
        # 等到子线程执行完毕，再执行主线程
        process_code = process.wait()

        end_time = time.time()
        self.logger.info(f"mineru 执行时间: {end_time - start_time}s")

        if process_code != 0:
            self.logger.info(f"mineru 解析失败")
        else:
            self.logger.info(f"mineru 解析成功")

        return process_code

    def _get_md_path(self, import_file_path: Path, file_dir: Path) -> str:
        """
        :param import_file_path: 输入路径
        :param file_dir: 输出路径
        :return: md文件路径
        """
        # path 的三个属性  stem 文件名 .suffix 后缀名 name 文件名+ 后缀名

        name = import_file_path.stem
        return str(file_dir / name / "hybrid_auto" / f"{name}.md")


if __name__ == "__main__":
    setup_logging()
    pdf_to_md_node = PdfToMdNode()

    init_state = {
        "import_file_path": r"D:\20251208_study\shopkeeper_brain\knowledge\processor\import_processor\temp_dir\万用表的使用.pdf",
        "file_dir": r"D:/20251208_study/shopkeeper_brain/knowledge/processor/import_processor/temp_dir"
    }

    res = pdf_to_md_node(state=init_state)

    # 序列化（对象 -》字符串）
    res_str = json.dumps(res, indent=4,ensure_ascii=False)
    print("res_str----",res_str)


