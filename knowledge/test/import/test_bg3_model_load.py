from modelscope import snapshot_download

local_dir = snapshot_download(model_id="BAAI/bge-m3",local_dir=r"D:\models")

print(local_dir)