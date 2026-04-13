from modelscope import snapshot_download

local_dir = snapshot_download(model_id="BAAI/bge-reranker-large",local_dir=r"D:\models\BAAI_bge-rerankder-large")

print(local_dir)