from langchain_text_splitters import RecursiveCharacterTextSplitter


text = "我是一个句子  我是一个颜色  我 是一个哈哈哈"

spliter = RecursiveCharacterTextSplitter(
    chunk_size=15,
    chunk_overlap=0,
    keep_separator=False
)

res = spliter.split_text(text)
print(res)