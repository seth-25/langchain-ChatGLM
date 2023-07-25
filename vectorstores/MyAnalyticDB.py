from __future__ import annotations

from langchain.vectorstores.analyticdb import Base
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple, Type
from sqlalchemy import Column, String, Table, create_engine, insert, text, select, Integer, func, and_
from sqlalchemy.dialects.postgresql import ARRAY, JSON, TEXT, REAL

from langchain.docstore.document import Document
from langchain.embeddings.base import Embeddings
from langchain.utils import get_from_dict_or_env
from langchain.vectorstores.base import VectorStore

from configs.model_config import *

try:
    from sqlalchemy.orm import declarative_base
except ImportError:
    from sqlalchemy.ext.declarative import declarative_base

Base = declarative_base()  # type: Any


class MyAnalyticDB(VectorStore):
    """
    VectorStore implementation using AnalyticDB.
    AnalyticDB is a distributed full PostgresSQL syntax cloud-native database.
    - `connection_string` is a postgres connection string.
    - `embedding_function` any embedding function implementing
        `langchain.embeddings.base.Embeddings` interface.
    - `knowledge_name` is the name of the collection to use. (default: langchain_document)
        - NOTE: This is not the name of the table, but the name of the collection.
            The tables will be created when initializing the store (if not exists)
            So, make sure the user has the right permissions to create tables.
    - `pre_delete_collection` if True, will delete the collection if it exists.
        (default: False)
        - Useful for testing.
    """
    def __init__(
            self,
            connection_string: str,
            embedding_function: Embeddings,
            embedding_dimension: int = LANGCHAIN_DEFAULT_EMBEDDING_DIM,
            pre_delete_collection: bool = False,
            logger: Optional[logging.Logger] = None,
            engine_args: Optional[dict] = None,
    ) -> None:
        self.connection_string = connection_string
        self.embedding_function = embedding_function
        self.embedding_dimension = embedding_dimension

        self.pre_delete_collection = pre_delete_collection
        self.logger = logger or logging.getLogger(__name__)

        self.collection_name = None
        self.collections_set = None
        self.collection_table = None
        # todo  self.__collection_name
        self.Base = Base

        self.score_threshold = VECTOR_SEARCH_SCORE_THRESHOLD
        self.chunk_size = CHUNK_SIZE

        self.__post_init__(engine_args)


    # def __del__(self):
    #     self.Base.metadata.clear()

    def __post_init__(
            self,
            engine_args: Optional[dict] = None,
    ) -> None:
        """
        Initialize the store.
        """

        _engine_args = engine_args or {}

        if (
                "pool_recycle" not in _engine_args
        ):  # Check if pool_recycle is not in _engine_args
            _engine_args[
                "pool_recycle"
            ] = 3600  # Set pool_recycle to 3600s if not present

        self.engine = create_engine(self.connection_string, **_engine_args)
        self.init_collection()

    def init_collection(self) -> None:
        if self.pre_delete_collection:
            self.delete_collection()
        self.collections_set = self.create_collections_if_not_exists()
        # self.collection_table, table_is_exist = self.create_table_if_not_exists()

    def create_collections_if_not_exists(self) -> Table:
        # Define the dynamic collections set table
        collections_table = Table(
            LANGCHAIN_DEFAULT_COLLECTIONS_NAME,
            self.Base.metadata,
            Column('id', TEXT, primary_key=True, default=uuid.uuid4),
            Column('collection_name', String),
            extend_existing=True,
        )
        with self.engine.connect() as conn:
            with conn.begin():
                # Create the table
                self.Base.metadata.create_all(conn)
        return collections_table

    def check_collection_if_exists(self, collection_name) -> bool:
        # Check if the collection in collections set
        with self.engine.connect() as conn:
            with conn.begin():
                collection_query = text(
                    f"""
                        SELECT 1
                        FROM {LANGCHAIN_DEFAULT_COLLECTIONS_NAME}
                        WHERE collection_name = '{collection_name}';
                    """
                )
                collection_result = conn.execute(collection_query).scalar()
        if collection_result:
            return True
        else:
            return False

    def create_table_if_not_exists(self, collection_name=None) -> [Table, bool]:
        """
        返回创建的Table对象和bool类型的table_is_exist，用于判断创建的表是否存在
        """
        if collection_name is None:
            collection_name = self.collection_name
        if collection_name == LANGCHAIN_DEFAULT_COLLECTIONS_NAME:
            raise Exception(f"知识库表名不能和统计知识库的表名{LANGCHAIN_DEFAULT_COLLECTIONS_NAME}相同")

        # Define the dynamic collection embedding table
        collection_table = Table(
            collection_name,
            self.Base.metadata,
            Column("id", Integer, primary_key=True, autoincrement=True),
            Column("uid", TEXT, default=uuid.uuid4),
            Column("embedding", ARRAY(REAL)),
            Column("document", String, nullable=True),
            Column("metadata", JSON, nullable=True),
            Column("source", TEXT, nullable=True),
            extend_existing=True,
        )
        table_is_exist = True
        with self.engine.connect() as conn:
            with conn.begin():
                # Create the table
                self.Base.metadata.create_all(conn)

                # Check if the index exists
                index_name = f"{collection_name}_embedding_idx"
                index_query = text(
                    f"""
                     SELECT 1
                     FROM pg_indexes
                     WHERE indexname = '{index_name}';
                 """
                )
                result = conn.execute(index_query).scalar()

                # Create the index if it doesn't exist
                if not result:
                    index_statement = text(
                        f"""
                         CREATE INDEX {index_name}
                         ON {collection_name} USING ann(embedding)
                         WITH (
                             "dim" = {self.embedding_dimension},
                             "hnsw_m" = 100
                         );
                     """
                    )
                    conn.execute(index_statement)

                # Check if the collection in collections set
                collection_query = text(
                    f"""
                       SELECT 1
                       FROM {LANGCHAIN_DEFAULT_COLLECTIONS_NAME}
                       WHERE collection_name = '{collection_name}';
                   """
                )
                collection_result = conn.execute(collection_query).scalar()
                # Add the collection in collections set if it doesn't exist
                if not collection_result:
                    table_is_exist = False
                    ins = self.collections_set.insert().values(collection_name=collection_name)
                    conn.execute(ins)
        return collection_table, table_is_exist

    def delete_collection(self) -> None:
        if self.collection_table is None:
            raise Exception("尚未绑定知识库")
        self.logger.debug("Trying to delete knowledge")
        drop_statement = text(f"DROP TABLE IF EXISTS {self.collection_name};")
        self.Base.metadata.remove(self.collection_table)
        delete_collection_record = text(
            f"DELETE FROM {LANGCHAIN_DEFAULT_COLLECTIONS_NAME} WHERE collection_name = '{self.collection_name}';")
        with self.engine.connect() as conn:
            with conn.begin():
                conn.execute(drop_statement)
                conn.execute(delete_collection_record)

    def get_collections(self) -> List[str]:
        collections_query = text(
            f"SELECT collection_name FROM {LANGCHAIN_DEFAULT_COLLECTIONS_NAME};")
        with self.engine.connect() as conn:
            with conn.begin():
                result = conn.execute(collections_query)
                return [record.collection_name for record in result.all()]

    def set_collection_name(self, collection_name):
        if self.collection_table is not None:
            self.Base.metadata.remove(self.collection_table)
        self.collection_name = collection_name
        self.collection_table, table_is_exist = self.create_table_if_not_exists()

    def delete(self, ids: Optional[List[str]] = None, **kwargs: Any) -> Optional[bool]:
        """Delete by vector IDs.

        Args:
            ids: List of ids to delete.
        """
        if self.collection_table is None:
            raise Exception("尚未绑定知识库")
        if ids is None:
            raise ValueError("No ids provided to delete.")

        try:
            with self.engine.connect() as conn:
                with conn.begin():
                    delete_condition = self.collection_table.c.id.in_(ids)
                    conn.execute(self.collection_table.delete().where(delete_condition))
                    return True
        except Exception as e:
            print("Delete operation failed:", str(e))
            return False

    def delete_doc(self, source: str or List[str]):
        if self.collection_table is None:
            raise Exception("尚未绑定知识库")
        try:
            result = []
            # 查出文件路径等于给定的source的记录的id
            with self.engine.connect() as conn:
                with conn.begin():
                    if isinstance(source, str):
                        select_condition = self.collection_table.c.source == source
                        # select_condition = self.collection_table.c.metadata.op("->>")("source") == source
                        s = select(self.collection_table.c.id).where(select_condition)
                        result = conn.execute(s).fetchall()
                    else:
                        for src in source:
                            select_condition = self.collection_table.c.source == src
                            # select_condition = self.collection_table.c.metadata.op("->>")("source") == src
                            s = select(self.collection_table.c.id).where(select_condition)
                            result.extend(conn.execute(s).fetchall())

            ids = [i[0] for i in result]
            if len(ids) == 0:
                return f"docs delete fail"
            else:
                self.delete(ids)

                # self.save_local(vs_path)
                return f"docs delete success"
        except Exception as e:
            print(e)
            return f"docs delete fail"

    def update_doc(self, source, new_docs):
        try:
            delete_len = self.delete_doc(source)
            ls = self.add_documents(new_docs)
            return f"docs update success"
        except Exception as e:
            print(e)
            return f"docs update fail"

    def list_docs(self):
        if self.collection_table is None:
            raise Exception("尚未绑定知识库")
        with self.engine.connect() as conn:
            with conn.begin():
                s = select(self.collection_table.c.source).group_by(self.collection_table.c.source)
                results = conn.execute(s).fetchall()
        return list(result[0] for result in results)

    def add_texts(
            self,
            texts: Iterable[str],
            metadatas: Optional[List[dict]] = None,
            ids: Optional[List[str]] = None,
            batch_size: int = 500,
            **kwargs: Any,
    ) -> List[str]:
        if self.collection_table is None:
            raise Exception("尚未绑定知识库")

        if ids is None:
            ids = [str(uuid.uuid1()) for _ in texts]

        embeddings = self.embedding_function.embed_documents(list(texts))

        if not metadatas:
            metadatas = [{} for _ in texts]

        # 导入的文件metadata必须要有source，才可以显示和删除
        try:
            sources = [metadata["source"] for metadata in metadatas]
        except KeyError:
            raise KeyError("导入的文本没有source，请检查load_file")

        chunks_table_data = []
        with self.engine.connect() as conn:
            with conn.begin():
                for document, metadata, chunk_id, embedding, source in zip(
                        texts, metadatas, ids, embeddings, sources
                ):
                    chunks_table_data.append(
                        {
                            "uid": chunk_id,
                            "embedding": embedding,
                            "document": document,
                            "metadata": metadata,
                            "source": source,
                        }
                    )

                    # Execute the batch insert when the batch size is reached
                    if len(chunks_table_data) == batch_size:
                        conn.execute(insert(self.collection_table).values(chunks_table_data))
                        # Clear the chunks_table_data list for the next batch
                        chunks_table_data.clear()

                # Insert any remaining records that didn't make up a full batch
                if chunks_table_data:
                    conn.execute(insert(self.collection_table).values(chunks_table_data))

        return ids

    def similarity_search(
            self,
            query: str,
            k: int = 4,
            filter: Optional[dict] = None,
            **kwargs: Any,
    ) -> List[Document]:
        embedding = self.embedding_function.embed_query(text=query)
        return self.similarity_search_by_vector(
            embedding=embedding,
            k=k,
            filter=filter,
        )

    def similarity_search_by_vector(
            self,
            embedding: List[float],
            k: int = 4,
            filter: Optional[dict] = None,
            **kwargs: Any,
    ) -> List[Document]:
        docs_and_scores = self.my_similarity_search_with_score_by_vector_context(
            embedding=embedding, k=k, filter=filter
        )
        return [doc for doc, _ in docs_and_scores]

    def my_similarity_search_with_score_by_vector_context(
            self,
            embedding: List[float],
            k: int = 4,
            filter: Optional[dict] = None,
    ) -> List[Tuple[Document, float]]:
        """
        带上下文的相似性搜索
        """
        if self.collection_table is None:
            raise Exception("尚未绑定知识库")

        try:
            from sqlalchemy.engine import Row
        except ImportError:
            raise ImportError(
                "Could not import Row from sqlalchemy.engine. "
                "Please 'pip install sqlalchemy>=1.4'."
            )

        filter_condition = ""
        if filter is not None:
            conditions = [
                f"metadata->>{key!r} = {value!r}" for key, value in filter.items()
            ]
            filter_condition = f"WHERE {' AND '.join(conditions)}"

        # Define the base query
        sql_query = f"""
                  SELECT *, l2_distance(embedding, :embedding) as distance
                  FROM {self.collection_name}
                  {filter_condition}
                  ORDER BY embedding <-> :embedding
                  LIMIT :k
              """

        # Set up the query parameters
        params = {"embedding": embedding, "k": k}

        # Execute the query and fetch the results
        with self.engine.connect() as conn:
            with conn.begin():
                results: Sequence[Row] = conn.execute(text(sql_query), params).fetchall()
                max_id = conn.execute(select(func.max(self.collection_table.c.id))).first()[0]  # 获得id最大最小值，以确定区间范围
                min_id = conn.execute(select(func.min(self.collection_table.c.id))).first()[0]
        if max_id == None:
            max_id = 0
        if min_id == None:
            min_id = 0

        id_set = set()
        id_map = {}
        batch_size = 10  # 区间一次拓宽多少

        for result in results:
            # print("查询result", result)
            if 0 < self.score_threshold < result.distance:
                continue

            id_set.add(result.id)
            id_map[result.id] = result
            docs_len = len(result.document)
            # print("docs_len", docs_len)

            last_l = result.id - 1  # 上一次搜索区间范围上界的前一个
            last_r = result.id + 1  # 上一次搜索区间范围下界的下一个
            for width in range(10, max_id + batch_size, batch_size):  # width是区间宽度/2，从10开始，一次向前后分别拓宽batch_size个
                if last_l < min_id and last_r > max_id:  # 区间已经拓展到id范围外
                    break
                break_flag = False
                # print("result.id, width, range", result.id, width, [result.id - width, result.id + width])

                left_range = [result.id - width - 1, last_l]
                right_range = [last_r, result.id + width + 1]

                with self.engine.connect() as conn:  # 查询出上下文
                    with conn.begin():
                        dis_condition = text(f"l2_distance(embedding, :embedding) as distance")
                        file_source_condition = self.collection_table.c.metadata.op("->>")("source") == \
                                                result.metadata["source"]

                        min_id_condition = self.collection_table.c.id >= left_range[0]
                        max_id_condition = self.collection_table.c.id <= left_range[1]
                        s = select(self.collection_table, dis_condition). \
                            where(and_(min_id_condition, max_id_condition)). \
                            order_by(self.collection_table.c.id.desc())
                        left_results = conn.execute(s, {"embedding": embedding}).fetchall()

                        min_id_condition = self.collection_table.c.id >= right_range[0]
                        max_id_condition = self.collection_table.c.id <= right_range[1]
                        s = select(self.collection_table, dis_condition). \
                            where(and_(min_id_condition, max_id_condition)). \
                            order_by(self.collection_table.c.id)
                        right_results = conn.execute(s, {"embedding": embedding}).fetchall()

                # print("left", left_range[0], left_range[1])
                # for lid, l_result in enumerate(left_results):
                #     print(lid, len(l_result.document), l_result)
                # print("right", right_range[0], right_range[1])
                # for rid, r_result in enumerate(right_results):
                #     print(rid, len(r_result.document), r_result)

                for i in range(max(len(left_results), len(right_results))):
                    if i < len(right_results):  # 添加下文id
                        r_result = right_results[i]
                        if docs_len + len(r_result.document) > self.chunk_size or \
                                r_result.metadata["source"] != result.metadata["source"]:
                            break_flag = True
                            break
                        elif r_result.metadata["source"] == result.metadata["source"]:
                            docs_len += len(r_result.document)
                            id_set.add(r_result.id)
                            id_map[r_result.id] = r_result

                    if i < len(left_results):  # 添加上文id
                        l_result = left_results[i]
                        if docs_len + len(l_result.document) > self.chunk_size or \
                                l_result.metadata["source"] != result.metadata["source"]:
                            break_flag = True
                            break
                        elif l_result.metadata["source"] == result.metadata["source"]:
                            docs_len += len(l_result.document)
                            id_set.add(l_result.id)
                            id_map[l_result.id] = l_result
                #     print("docs_len", docs_len, id_set)
                # print("docs_len", docs_len, id_set)
                # print("==================================")
                if break_flag:  # 已经添加足够的上下文，退出
                    break
                last_l = result.id - width - 1
                last_r = result.id + width + 1

        if len(id_set) == 0:
            return []
        print("id_set", id_set)
        id_list = sorted(list(id_set))
        # 连续的id分在一起，成为一个id seq
        id_seqs = []  # 存一个个连续的id seq
        id_seq = [id_list[0]]
        for i in range(1, len(id_list)):
            if id_list[i - 1] + 1 == id_list[i]:
                id_seq.append(id_list[i])
            else:
                id_seqs.append(id_seq)
                id_seq = [id_list[i]]
        id_seqs.append(id_seq)

        # print("id_seqs", id_seqs)

        documents_with_scores = []
        # 将一个连续的id seq拼成一个doc
        for id_seq in id_seqs:
            doc: Document = None
            doc_score = None
            for id in id_seq:
                if id == id_seq[0]:
                    res = id_map[id]
                    doc = Document(
                        page_content=res.document,
                        metadata=res.metadata,
                    )
                    doc_score = res.distance
                else:
                    res = id_map[id]
                    doc.page_content += "\n" + res.document
                    doc_score = min(doc_score, res.distance)
            if not isinstance(doc, Document) or doc_score is None:
                raise ValueError(f"Could not find document, got {doc}")

            # 和langchain不同，chatglm会多一步把score写入metadata
            doc.metadata["score"] = round(doc_score, 3)
            documents_with_scores.append((doc, doc_score))
        return documents_with_scores


    @classmethod
    def from_texts(
            cls: Type[MyAnalyticDB],
            texts: List[str],
            embedding: Embeddings,
            metadatas: Optional[List[dict]] = None,
            embedding_dimension: int = LANGCHAIN_DEFAULT_EMBEDDING_DIM,
            collection_name: str = LANGCHAIN_DEFAULT_KNOWLEDGE_NAME,
            ids: Optional[List[str]] = None,
            pre_delete_collection: bool = False,
            engine_args: Optional[dict] = None,
            **kwargs: Any,
    ) -> MyAnalyticDB:
        """
        Return VectorStore initialized from texts and embeddings.
        Postgres Connection string is required
        Either pass it as a parameter
        or set the PG_CONNECTION_STRING environment variable.
        """

        connection_string = cls.get_connection_string(kwargs)

        store = cls(
            connection_string=connection_string,
            embedding_function=embedding,
            embedding_dimension=embedding_dimension,
            pre_delete_collection=pre_delete_collection,
            engine_args=engine_args,
        )

        store.add_texts(texts=texts, metadatas=metadatas, ids=ids, **kwargs)
        return store

    @classmethod
    def get_connection_string(cls, kwargs: Dict[str, Any]) -> str:
        connection_string: str = get_from_dict_or_env(
            data=kwargs,
            key="connection_string",
            env_key="PG_CONNECTION_STRING",
        )

        if not connection_string:
            raise ValueError(
                "Postgres connection string is required"
                "Either pass it as a parameter"
                "or set the PG_CONNECTION_STRING environment variable."
            )
        return connection_string
    @classmethod
    def connection_string_from_db_params(
            cls,
            driver: str,
            host: str,
            port: int,
            database: str,
            user: str,
            password: str,
    ) -> str:
        """Return connection string from database parameters."""
        return f"postgresql+{driver}://{user}:{password}@{host}:{port}/{database}"
