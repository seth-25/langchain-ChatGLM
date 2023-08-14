from __future__ import annotations

import sys

from langchain.vectorstores.analyticdb import Base
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple, Type
from sqlalchemy import Column, String, Table, create_engine, insert, text, select, Integer, func, and_
from sqlalchemy.dialects.postgresql import ARRAY, JSON, TEXT, REAL

from langchain.docstore.document import Document
from langchain.embeddings.base import Embeddings
from langchain.utils import get_from_dict_or_env
from langchain.vectorstores.base import VectorStore

from configs.model_config import *
from textsplitter.my_markdown_splitter import md_headers
from utils.regular_util import match_brackets_at_start, remove_brackets_at_start
from utils.file_util import get_filename_from_source

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

        self.__collection_name = None
        self.__collections_set = None
        self.__collection_table = None
        self.__base = Base

        self.score_threshold = VECTOR_SEARCH_SCORE_THRESHOLD
        self.chunk_content = True
        self.chunk_size = CHUNK_SIZE
        self.md_title_split = 1 if MD_TITLE_SPLIT < 1 else 6 if MD_TITLE_SPLIT > 6 else MD_TITLE_SPLIT

        self.__post_init__(engine_args)

    def __del__(self):
        self.__base.metadata.clear()

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
        self.__collections_set = self.create_collections_if_not_exists()

        # 初始化MyAnalyticDB不创建collection的Table和绑定self.collection_table，由用户调用接口创建，或者set_collection_name时创建
        # self.collection_table, table_is_exist = self.create_table_if_not_exists()

    def create_collections_if_not_exists(self) -> Table:
        # Define the dynamic collections set table
        collections_table = Table(
            LANGCHAIN_DEFAULT_COLLECTIONS_NAME,
            self.__base.metadata,
            Column('id', TEXT, primary_key=True, default=uuid.uuid4),
            Column('collection_name', String),
            extend_existing=True,
        )
        with self.engine.connect() as conn:
            with conn.begin():
                # Create the table
                self.__base.metadata.create_all(conn)
        return collections_table

    def check_collection_if_exists(self, collection_name) -> bool:
        """ Check if the collection in collections set """
        if collection_name == LANGCHAIN_DEFAULT_COLLECTIONS_NAME:  # 表名不能和collections set相同
            return True
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
        """ 返回创建的Table对象和bool类型的table_is_exist，table_is_exist用于判断创建的表是否存在 """
        if collection_name is None:
            collection_name = self.__collection_name
        if collection_name == LANGCHAIN_DEFAULT_COLLECTIONS_NAME:
            raise Exception(f"知识库名不能和统计知识库的表名{LANGCHAIN_DEFAULT_COLLECTIONS_NAME}相同")

        # Define the dynamic collection embedding table
        collection_table = Table(
            collection_name,
            self.__base.metadata,
            Column("id", Integer, primary_key=True, autoincrement=True),
            Column("uid", TEXT, default=uuid.uuid4),
            Column("embedding", ARRAY(REAL)),
            Column("document", String, nullable=True),
            Column("metadata", JSON, nullable=True),
            Column("source", TEXT, nullable=True),  # 存的是filename
            extend_existing=True,
        )
        table_is_exist = True
        with self.engine.connect() as conn:
            with conn.begin():
                # Create the table
                self.__base.metadata.create_all(conn)

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
                         CREATE INDEX "{index_name}"
                         ON "{collection_name}" USING ann(embedding)
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
                    ins = self.__collections_set.insert().values(collection_name=collection_name)
                    conn.execute(ins)
        return collection_table, table_is_exist

    def delete_collection(self) -> None:
        if self.__collection_table is None:
            raise Exception("尚未绑定知识库")
        self.logger.debug("Trying to delete knowledge")
        drop_statement = text(f"""DROP TABLE IF EXISTS "{self.__collection_name}";""")
        self.__base.metadata.remove(self.__collection_table)
        delete_collection_record = text(
            f"DELETE FROM {LANGCHAIN_DEFAULT_COLLECTIONS_NAME} WHERE collection_name = '{self.__collection_name}';")
        with self.engine.connect() as conn:
            with conn.begin():
                conn.execute(drop_statement)
                conn.execute(delete_collection_record)

    def change_collection(self, new_collection_name) -> None:
        if self.__collection_table is None:
            raise Exception("尚未绑定知识库")
        alert_statement = text(f"""ALTER TABLE "{self.__collection_name}" RENAME TO "{new_collection_name}";""")
        alert_index_statement = text(
            f"""ALTER INDEX "{self.__collection_name}_embedding_idx" RENAME TO "{new_collection_name}_embedding_idx";""")
        alert_id_seq_statement = text(
            f"""ALTER TABLE "{self.__collection_name}_id_seq" RENAME TO "{new_collection_name}_id_seq";""")
        update_collection_record = text(
            f"""UPDATE {LANGCHAIN_DEFAULT_COLLECTIONS_NAME} SET collection_name = '{new_collection_name}' WHERE collection_name = '{self.__collection_name}';""")
        with self.engine.connect() as conn:
            with conn.begin():
                conn.execute(alert_statement)
                conn.execute(alert_index_statement)
                conn.execute(alert_id_seq_statement)
                conn.execute(update_collection_record)
        self.set_collection_name(new_collection_name)

    def get_collections(self) -> List[str]:
        collections_query = text(
            f"SELECT collection_name FROM {LANGCHAIN_DEFAULT_COLLECTIONS_NAME};")
        with self.engine.connect() as conn:
            with conn.begin():
                result = conn.execute(collections_query)
                return [record.collection_name for record in result.all()]

    def set_collection_name(self, collection_name):
        if self.__collection_table is not None:
            self.__base.metadata.remove(self.__collection_table)
        self.__collection_name = collection_name
        self.__collection_table, table_is_exist = self.create_table_if_not_exists()

    def delete(self, ids: Optional[List[str]] = None, **kwargs: Any) -> Optional[bool]:
        """Delete by vector IDs.
        Args:
            ids: List of ids to delete.
        """
        if self.__collection_table is None:
            raise Exception("尚未绑定知识库")
        if ids is None:
            raise ValueError("No ids provided to delete.")

        try:
            with self.engine.connect() as conn:
                with conn.begin():
                    delete_condition = self.__collection_table.c.id.in_(ids)
                    conn.execute(self.__collection_table.delete().where(delete_condition))
                    return True
        except Exception as e:
            print("Delete operation failed:", str(e))
            return False

    def delete_doc(self, source: str or List[str]):
        if self.__collection_table is None:
            raise Exception("尚未绑定知识库")
        try:
            result = []
            # 查出文件路径等于给定的source的记录的id
            with self.engine.connect() as conn:
                with conn.begin():
                    if isinstance(source, str):
                        select_condition = self.__collection_table.c.source == get_filename_from_source(source)
                        # select_condition = self.collection_table.c.metadata.op("->>")("source") == source
                        s = select(self.__collection_table.c.id).where(select_condition)
                        result = conn.execute(s).fetchall()
                    else:
                        for src in source:
                            select_condition = self.__collection_table.c.source == get_filename_from_source(src)
                            # select_condition = self.collection_table.c.metadata.op("->>")("source") == src
                            s = select(self.__collection_table.c.id).where(select_condition)
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
        if self.__collection_table is None:
            raise Exception("尚未绑定知识库")
        with self.engine.connect() as conn:
            with conn.begin():
                s = select(self.__collection_table.c.source).group_by(self.__collection_table.c.source)
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
        if self.__collection_table is None:
            raise Exception("尚未绑定知识库")

        if ids is None:
            ids = [str(uuid.uuid1()) for _ in texts]

        embeddings = self.embedding_function.embed_documents(list(texts))

        if not metadatas:
            metadatas = [{} for _ in texts]

        # 导入的文件metadata必须要有source，才可以显示文件的filename和根据filename删除文件
        try:
            filenames = [get_filename_from_source(metadata["source"]) for metadata in metadatas]
        except KeyError:
            raise KeyError("导入的文本没有source，请检查load_file调用的textsplitter")

        print("插入向量总数", len(embeddings))
        cnt = 0
        chunks_table_data = []
        with self.engine.connect() as conn:
            with conn.begin():
                for document, metadata, chunk_id, embedding, filename in zip(
                        texts, metadatas, ids, embeddings, filenames
                ):
                    chunks_table_data.append(
                        {
                            "uid": chunk_id,
                            "embedding": embedding,
                            "document": document,
                            "metadata": metadata,
                            "source": filename,
                        }
                    )

                    # Execute the batch insert when the batch size is reached
                    if len(chunks_table_data) == batch_size:
                        conn.execute(insert(self.__collection_table).values(chunks_table_data))
                        # Clear the chunks_table_data list for the next batch
                        chunks_table_data.clear()
                        cnt += 1
                        print(f"已经插入 {batch_size * cnt} 条向量")

                # Insert any remaining records that didn't make up a full batch
                if chunks_table_data:
                    conn.execute(insert(self.__collection_table).values(chunks_table_data))

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

    def get_search_result_from_database(self,
                                        embedding: List[float],
                                        k: int = 4,
                                        filter: Optional[dict] = None,
                                        ):
        try:
            from sqlalchemy.engine import Row
        except ImportError:
            raise ImportError(
                "Could not import Row from sqlalchemy.engine. "
                "Please 'pip install sqlalchemy>=1.4'."
            )
            # Add the filter if provided
        filter_condition = ""
        if filter is not None:
            conditions = [
                f"metadata->>{key!r} = {value!r}" for key, value in filter.items()
            ]
            filter_condition = f"WHERE {' AND '.join(conditions)}"

        # Define the base query
        sql_query = f"""
                SELECT *, l2_distance(embedding, :embedding) as distance
                FROM {self.__collection_name}
                {filter_condition}
                ORDER BY embedding <-> :embedding
                LIMIT :k
            """

        # Set up the query parameters
        params = {"embedding": embedding, "k": k}

        # Execute the query and fetch the results
        with self.engine.connect() as conn:
            results: Sequence[Row] = conn.execute(text(sql_query), params).fetchall()
        return results

    def similarity_search_by_vector(
            self,
            embedding: List[float],
            k: int = 4,
            filter: Optional[dict] = None,
            **kwargs: Any,
    ) -> List[Document]:
        if self.__collection_table is None:
            raise Exception("尚未绑定知识库")
        if self.chunk_content:  # 使用上下文
            docs_and_scores = self.my_similarity_search_with_score_by_vector_context(
                embedding=embedding, k=k, filter=filter
            )
        else:
            docs_and_scores = self.similarity_search_with_score_by_vector(
                embedding=embedding, k=k, filter=filter
            )
        return [doc for doc, _ in docs_and_scores]

    def similarity_search_with_score_by_vector(
            self,
            embedding: List[float],
            k: int = 4,
            filter: Optional[dict] = None,
    ) -> List[Tuple[Document, float]]:
        """
        不带上下文的相似性搜索
        """
        if self.__collection_table is None:
            raise Exception("尚未绑定知识库")
        results = self.get_search_result_from_database(embedding, k, filter)

        documents_with_scores = []
        for result in results:
            if 0 < self.score_threshold < result.distance:
                continue
            result.metadata["score"] = round(result.distance, 3) if self.embedding_function is not None else None
            result.metadata["content"] = result.document
            documents_with_scores.append((
                Document(
                    page_content=result.document,
                    metadata=result.metadata,
                ),
                result.distance
            ))
        return documents_with_scores

    def my_similarity_search_with_score_by_vector_context(
            self,
            embedding: List[float],
            k: int = 4,
            filter: Optional[dict] = None,
    ) -> List[Tuple[Document, float]]:
        """
        带上下文的相似性搜索
        """
        if self.__collection_table is None:
            raise Exception("尚未绑定知识库")

        # Execute the query and fetch the results
        results = self.get_search_result_from_database(embedding, k, filter)

        with self.engine.connect() as conn:
            with conn.begin():
                max_id = conn.execute(select(func.max(self.__collection_table.c.id))).first()[0]  # 获得id最大最小值，以确定区间范围
                min_id = conn.execute(select(func.min(self.__collection_table.c.id))).first()[0]
        if max_id is None:
            max_id = 0
        if min_id is None:
            min_id = 0

        id_set = set()
        id_map = {}
        batch_size = 20  # 区间一次拓宽多少

        for result in results:
            # count = 0
            # print("查询result", len(result.document), result)
            if 0 < self.score_threshold < result.distance:
                continue

            id_set.add(result.id)
            id_map[result.id] = result
            docs_len = len(result.document)

            last_l = result.id - 1  # 上一次搜索区间范围上界的前一个
            last_r = result.id + 1  # 上一次搜索区间范围下界的下一个
            for width in range(10, max_id + batch_size, batch_size):  # width是区间宽度/2，从10开始，一次向前后分别拓宽batch_size个
                if last_l < min_id and last_r > max_id:  # 区间已经拓展到id范围外
                    # print("区间已经拓展到id范围外")
                    break

                # print(f"result.id {result.id}, width {width}, range {[result.id - width, result.id + width]}")

                left_range = [result.id - width, last_l]
                right_range = [last_r, result.id + width]

                with self.engine.connect() as conn:  # 查询出上下文
                    with conn.begin():
                        dis_condition = text(f"l2_distance(embedding, :embedding) as distance")
                        file_source_condition = self.__collection_table.c.metadata.op("->>")("source") == \
                                                result.metadata["source"]

                        min_id_condition = self.__collection_table.c.id >= left_range[0]
                        max_id_condition = self.__collection_table.c.id <= left_range[1]
                        s = select(self.__collection_table, dis_condition). \
                            where(and_(min_id_condition, max_id_condition)). \
                            order_by(self.__collection_table.c.id.desc())
                        left_results = conn.execute(s, {"embedding": embedding}).fetchall()

                        min_id_condition = self.__collection_table.c.id >= right_range[0]
                        max_id_condition = self.__collection_table.c.id <= right_range[1]
                        s = select(self.__collection_table, dis_condition). \
                            where(and_(min_id_condition, max_id_condition)). \
                            order_by(self.__collection_table.c.id)
                        right_results = conn.execute(s, {"embedding": embedding}).fetchall()
                        # count += 1

                # print("left", left_range[0], left_range[1])
                # for lid, l_result in enumerate(left_results):
                #     print(lid, len(l_result.document), "(", l_result.id, [l_result.document], ")")
                # print("right", right_range[0], right_range[1])
                # for rid, r_result in enumerate(right_results):
                #     print(rid, len(r_result.document), "(", r_result.id, [r_result.document], ")")

                i = j = 0  # i,j = sys.maxsize表示该方向不再可拼
                if len(left_results) == 0:  # 不存在上文
                    i = sys.maxsize
                if len(right_results) == 0:  # 不存在下文
                    j = sys.maxsize
                while i < len(left_results) or j < len(right_results):
                    if i >= len(left_results):  # 无可拼上文，选择拼下文
                        t_result = right_results[j]
                        j += 1
                        is_left = False
                    elif j >= len(right_results):  # 无可拼下文，选择拼上文
                        t_result = left_results[i]
                        i += 1
                        is_left = True
                    else:
                        if right_results[j].distance <= left_results[i].distance:  # 优先拼距离近的上下文，距离相同拼下文
                            t_result = right_results[j]
                            j += 1
                            is_left = False
                        else:
                            t_result = left_results[i]
                            i += 1
                            is_left = True

                    # 拼上该方向的文本超长度了，或不是同个文件，这个方向不再拼
                    if docs_len + len(t_result.document) > self.chunk_size or \
                            t_result.source != result.source:
                        if is_left:
                            i = sys.maxsize
                        else:
                            j = sys.maxsize
                        continue
                    if t_result.source.lower().endswith(".md"):  # 是markdown
                        is_continue = False
                        for h in range(self.md_title_split):
                            header = md_headers[h][1]
                            if header in t_result.metadata.keys() and header in result.metadata.keys():
                                if t_result.metadata[header] != result.metadata[header]:  # 标题不同则该方向不再拼
                                    if is_left:
                                        i = sys.maxsize
                                    else:
                                        j = sys.maxsize
                                    is_continue = True
                                    break
                        if is_continue:
                            continue

                    if t_result.id in id_set:  # 重叠部分跳过，防止都召回相近的内容，信息量过少
                        continue

                    # 拼接，将id加入id_set
                    docs_len += len(t_result.document)
                    id_set.add(t_result.id)
                    id_map[t_result.id] = t_result
                # print(id_set, docs_len, "i:", i, "j:", j)
                if i == sys.maxsize and j == sys.maxsize:  # 两个方向都无法继续拼了，才退出
                    # print("两个方向都无法继续拼了")
                    break

                last_l = result.id - width - 1
                last_r = result.id + width + 1
            # print("查询次数", count)
        if len(id_set) == 0:
            return []
        # print("id_set", id_set)
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
                    result = id_map[id]
                    doc = Document(
                        page_content=result.document,
                        metadata=result.metadata,
                    )
                    doc.metadata["content"] = doc.page_content
                    doc_score = result.distance
                else:
                    result = id_map[id]
                    result_page_content = result.document
                    last_res = id_map[id - 1]  # 上一个文本
                    # 开启标题增强的情况下，如果当前文本和上一个文本标题相同，去掉当前文本的标题。
                    if match_brackets_at_start(last_res.document) == match_brackets_at_start(result.document):
                        result_page_content = remove_brackets_at_start(result.document)

                    if REMOVE_TITLE:    # 有时候大模型会把标题也混入答案中。可选择去掉标题，但查询可能不全
                        doc.page_content += "\n" + result_page_content
                    else:
                        doc.page_content += "\n" + result.document
                    doc.metadata["content"] += "\n" + result_page_content  # 去除标题，方便在webui显示
                    doc_score = min(doc_score, result.distance)
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
