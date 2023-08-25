import urllib
from server.utils import BaseResponse, ListResponse
from server.knowledge_base.utils import validate_kb_name
from server.knowledge_base.kb_service.base import KBServiceFactory
from server.db.repository.knowledge_base_repository import list_kbs_from_db
from configs.model_config import EMBEDDING_MODEL
from fastapi import Body


async def list_kbs():
    # Get List of Knowledge Base
    return ListResponse(data=list_kbs_from_db())


async def create_kb(knowledge_base_name: str = Body(..., examples=["samples"]),
                    vector_store_type: str = Body("faiss"),
                    embed_model: str = Body(EMBEDDING_MODEL),
                    ) -> BaseResponse:
    # Create selected knowledge base
    if not validate_kb_name(knowledge_base_name):
        return BaseResponse(code=403, msg="Don't attack me")
    if knowledge_base_name is None or knowledge_base_name.strip() == "":
        return BaseResponse(code=404, msg="知识库名称不能为空，请重新填写知识库名称")

    print("创建kb", knowledge_base_name, vector_store_type, embed_model)

    kb = KBServiceFactory.get_service_by_name(knowledge_base_name)
    if kb is not None:
        return BaseResponse(code=404, msg=f"已存在同名知识库 {knowledge_base_name}")

    kb = KBServiceFactory.get_service(knowledge_base_name, vector_store_type, embed_model)
    try:
        kb.create_kb()
    except Exception as e:
        print(e)
        return BaseResponse(code=500, msg=f"创建知识库出错： {e}")

    return BaseResponse(code=200, msg=f"已新增知识库 {knowledge_base_name}")


async def delete_kb(
        knowledge_base_name: str = Body(..., examples=["samples"])
    ) -> BaseResponse:
    # Delete selected knowledge base
    if not validate_kb_name(knowledge_base_name):
        return BaseResponse(code=403, msg="Don't attack me")
    knowledge_base_name = urllib.parse.unquote(knowledge_base_name)

    kb = KBServiceFactory.get_service_by_name(knowledge_base_name)

    if kb is None:
        return BaseResponse(code=404, msg=f"未找到知识库 {knowledge_base_name}")

    try:
        print("删除知识库接口")
        status = kb.clear_vs()
        status = kb.drop_kb()
        if status:
            return BaseResponse(code=200, msg=f"成功删除知识库 {knowledge_base_name}")
    except Exception as e:
        print(e)
        return BaseResponse(code=500, msg=f"删除知识库时出现意外： {e}")

    return BaseResponse(code=500, msg=f"删除知识库失败 {knowledge_base_name}")
