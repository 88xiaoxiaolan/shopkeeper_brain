from functools import cache, lru_cache

from knowledge.service.query_service import QueryService
from knowledge.service.upload_service import UploadSeries

@cache #缓存（将实例对象缓存一份：可能会出现oom:out of memory）单例实现效果
# @lru_cache 缓存（淘汰机制：当缓存的数据达到一定数量时，会自动淘汰掉一部分数据）
def get_upload_file_service():
    return UploadSeries()

@cache
def get_query_service():
    return QueryService()