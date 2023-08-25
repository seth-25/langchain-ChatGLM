import os
from typing import List, Optional

from langchain.embeddings.base import Embeddings
from langchain.schema import Document
from sqlalchemy import text

from configs.model_config import EMBEDDING_DEVICE, kbs_config, EMBEDDING_MODEL
from server.knowledge_base.kb_service.base import SupportedVSType, KBService
from server.knowledge_base.utils import load_embeddings, KnowledgeFile
from urllib.parse import quote_plus
from server.utils import torch_gc
from .AnalyticDB import AnalyticDB


class ADBKBService(KBService):
    adb_vector: AnalyticDB

    def vs_type(self) -> str:
        return SupportedVSType.ADB

    def _init_adb_vector(self, embed_model: str = EMBEDDING_MODEL,
                         embedding_device: str = EMBEDDING_DEVICE,
                         embeddings: Optional[Embeddings] = None):
        if embeddings is None:
            embeddings = load_embeddings(embed_model, embedding_device)

        connection_string = AnalyticDB.connection_string_from_db_params(
            driver=kbs_config.get("adb").get("PG_DRIVER"),
            host=kbs_config.get("adb").get("PG_HOST"),
            port=int(kbs_config.get("adb").get("PG_PORT")),
            database=kbs_config.get("adb").get("PG_DATABASE"),
            user=kbs_config.get("adb").get("PG_USER"),
            password=quote_plus(kbs_config.get("adb").get("PG_PASSWORD")),
        )
        self.adb_vector = AnalyticDB(embedding_function=embeddings,
                                     connection_string=connection_string)
        self.adb_vector.set_collection_name(self.kb_name)

    def _load_vector_store(self):
        if self.adb_vector.get_collection_name() != self.kb_name:
            self.adb_vector.set_collection_name(self.kb_name)

    def do_init(self):
        self._init_adb_vector()

    def do_create_kb(self):
        self.adb_vector.create_table_if_not_exists(self.kb_name)

    def do_drop_kb(self):
        self._load_vector_store()
        self.adb_vector.delete_collection()

    def do_search(self, query: str, top_k: int, score_threshold: float, embeddings: Embeddings):
        # todo 支持score_threshold
        if embeddings:
            self.adb_vector.set_embedding(embeddings)
        self._load_vector_store()
        return self.adb_vector.similarity_search_with_score(query, k=top_k)

    def do_add_doc(self, docs: List[Document], embeddings: Embeddings, **kwargs):
        if embeddings:
            self.adb_vector.set_embedding(embeddings)
        self._load_vector_store()
        self.adb_vector.add_documents(docs)
        torch_gc()

    def do_delete_doc(self, kb_file: KnowledgeFile, **kwargs):
        self._load_vector_store()
        self.adb_vector.delete_doc(kb_file.filename)

    def do_clear_vs(self):
        pass

    def exist_doc(self, file_name: str):
        if super().exist_doc(file_name):
            return "in_db"

        content_path = os.path.join(self.kb_path, "content")
        if os.path.isfile(os.path.join(content_path, file_name)):
            return "in_folder"
        else:
            return False


if __name__ == '__main__':
    from server.db.base import Base, engine

    Base.metadata.create_all(bind=engine)
    aDBKBService = ADBKBService("test")
    aDBKBService.create_kb()
    aDBKBService.add_doc(KnowledgeFile("README.md", "test"))
    aDBKBService.delete_doc(KnowledgeFile("README.md", "test"))
    aDBKBService.drop_kb()
    print(aDBKBService.search_docs("测试"))
