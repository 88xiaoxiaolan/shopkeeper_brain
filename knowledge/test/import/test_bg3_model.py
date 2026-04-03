from pymilvus.model.hybrid import BGEM3EmbeddingFunction

bge_m3_ef = BGEM3EmbeddingFunction(
    model_name=r"D:\models\BAAIbge-m3",
    device='cuda:0', # Specify the device to use, e.g., 'cpu' or 'cuda:0'
    use_fp16=True # Specify whether to use fp16. Set to `False` if `device` is `cpu`.
)

result = bge_m3_ef.encode_queries(["我是中国人","你是美国人"])
print(result)
print("dense==",result.get("dense")[0].tolist())
print("sparse==",result.get("sparse").data)

# print(dict(zip([1,2,3],[4,5,6])))
