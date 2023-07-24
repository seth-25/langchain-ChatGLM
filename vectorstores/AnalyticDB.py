from __future__ import annotations

import logging
import os
import uuid
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple, Type

from configs.model_config import *

from sqlalchemy.sql.expression import cast
from sqlalchemy import REAL, Column, String, Table, create_engine, insert, text, Index, ForeignKey, select
from sqlalchemy.dialects.postgresql import ARRAY, JSON, TEXT, UUID
try:
    from sqlalchemy.orm import declarative_base, relationship
except ImportError:
    from sqlalchemy.ext.declarative import declarative_base

from langchain.docstore.document import Document
from langchain.embeddings.base import Embeddings
from langchain.utils import get_from_dict_or_env
from langchain.vectorstores.base import VectorStore


class AnalyticDB(VectorStore):
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
            knowledge_name: str = LANGCHAIN_DEFAULT_KNOWLEDGE_NAME,
            pre_delete_knowledge: bool = False,
            pre_get_knowledge: bool = True,
            logger: Optional[logging.Logger] = None,
            engine_args: Optional[dict] = None,
    ) -> None:
        self.connection_string = connection_string
        self.embedding_function = embedding_function
        self.embedding_dimension = embedding_dimension
        self.knowledge_name = knowledge_name
        self.pre_delete_knowledge = pre_delete_knowledge
        self.pre_get_knowledge = pre_get_knowledge
        self.logger = logger or logging.getLogger(__name__)

        self.collections_set = None
        self.knowledge_table = None
        self.Base = declarative_base()  # type: Any

        self.__post_init__()

        self.score_threshold = VECTOR_SEARCH_SCORE_THRESHOLD

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

        if self.pre_delete_knowledge:
            self.delete_knowledge()
        self.collections_set = self.create_collects_set_if_not_exists()
        if self.pre_get_knowledge:
            self.knowledge_table = self.create_table_if_not_exists()

    def create_collects_set_if_not_exists(self) -> Table:
        # Define the dynamic collections set table
        collections_table = Table(
            LANGCHAIN_DEFAULT_COLLECTIONS_SET_NAME,
            self.Base.metadata,
            Column('id', TEXT, primary_key=True, default=uuid.uuid4),
            Column('knowledge_name', String),
        )
        with self.engine.connect() as conn:
            with conn.begin():
                # Create the table
                self.Base.metadata.create_all(conn)
        return collections_table

    def check_knowledge_if_exists(self, knowledge_name) -> bool:
        # Check if the knowledge in collections set
        with self.engine.connect() as conn:
            with conn.begin():
                knowledge_query = text(
                    f"""
                      SELECT 1
                      FROM {LANGCHAIN_DEFAULT_COLLECTIONS_SET_NAME}
                      WHERE knowledge_name = '{knowledge_name}';
                  """
                )
                knowledge_result = conn.execute(knowledge_query).scalar()
        if knowledge_result:
            return True
        else:
            return False
    def create_table_if_not_exists(self) -> Table:
        # Define the dynamic knowledge embedding table
        knowledge_table = Table(
            self.knowledge_name,
            self.Base.metadata,
            Column("id", TEXT, primary_key=True, default=uuid.uuid4),
            # Column('collection_id', UUID(as_uuid=True), ForeignKey(f"{_LANGCHAIN_DEFAULT_COLLECTIONS_SET_NAME}.id", ondelete="CASCADE")),
            Column("embedding", ARRAY(REAL)),
            Column("document", String, nullable=True),
            Column("metadata", JSON, nullable=True),
            extend_existing=True,
        )

        with self.engine.connect() as conn:
            with conn.begin():
                # Create the table
                self.Base.metadata.create_all(conn)

                # Check if the index exists
                index_name = f"{self.knowledge_name}_embedding_idx"
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
                        ON {self.knowledge_name} USING ann(embedding)
                        WITH (
                            "dim" = {self.embedding_dimension},
                            "hnsw_m" = 100
                        );
                    """
                    )
                    conn.execute(index_statement)

                # Check if the knowledge in collections set
                knowledge_query = text(
                    f"""
                      SELECT 1
                      FROM {LANGCHAIN_DEFAULT_COLLECTIONS_SET_NAME}
                      WHERE knowledge_name = '{self.knowledge_name}';
                  """
                )
                knowledge_result = conn.execute(knowledge_query).scalar()
                # Add the knowledge in collections set if it doesn't exist
                if not knowledge_result:
                    ins = self.collections_set.insert().values(knowledge_name=self.knowledge_name)
                    conn.execute(ins)
        return knowledge_table

    def delete_knowledge(self) -> None:
        self.logger.debug("Trying to delete knowledge")
        drop_statement = text(f"DROP TABLE IF EXISTS {self.knowledge_name};")
        delete_knowledge_record = text(
            f"DELETE FROM {LANGCHAIN_DEFAULT_COLLECTIONS_SET_NAME} WHERE knowledge_name = '{self.knowledge_name}';")
        with self.engine.connect() as conn:
            with conn.begin():
                conn.execute(drop_statement)
                conn.execute(delete_knowledge_record)

    def get_collections(self) -> List[str]:
        collections_query = text(
            f"SELECT knowledge_name FROM {LANGCHAIN_DEFAULT_COLLECTIONS_SET_NAME};")
        with self.engine.connect() as conn:
            with conn.begin():
                result = conn.execute(collections_query)
                return [record.knowledge_name for record in result.all()]

    def set_knowledge_name(self, knowledge_name):
        self.knowledge_name = knowledge_name
        self.create_table_if_not_exists()

    def add_texts(
            self,
            texts: Iterable[str],
            metadatas: Optional[List[dict]] = None,
            ids: Optional[List[str]] = None,
            batch_size: int = 500,
            **kwargs: Any,
    ) -> List[str]:
        """Run more texts through the embeddings and add to the vectorstore.

        Args:
            texts: Iterable of strings to add to the vectorstore.
            metadatas: Optional list of metadatas associated with the texts.
            kwargs: vectorstore specific parameters

        Returns:
            List of ids from adding the texts into the vectorstore.
        """
        if ids is None:
            ids = [str(uuid.uuid1()) for _ in texts]

        embeddings = self.embedding_function.embed_documents(list(texts))

        if not metadatas:
            metadatas = [{} for _ in texts]

        chunks_table_data = []
        with self.engine.connect() as conn:
            with conn.begin():
                for document, metadata, chunk_id, embedding in zip(
                        texts, metadatas, ids, embeddings
                ):
                    chunks_table_data.append(
                        {
                            "id": chunk_id,
                            "embedding": embedding,
                            "document": document,
                            "metadata": metadata,
                        }
                    )

                    # Execute the batch insert when the batch size is reached
                    if len(chunks_table_data) == batch_size:
                        conn.execute(insert(self.knowledge_table).values(chunks_table_data))
                        # Clear the chunks_table_data list for the next batch
                        chunks_table_data.clear()

                # Insert any remaining records that didn't make up a full batch
                if chunks_table_data:
                    conn.execute(insert(self.knowledge_table).values(chunks_table_data))

        return ids

    def similarity_search(
            self,
            query: str,
            k: int = 4,
            filter: Optional[dict] = None,
            **kwargs: Any,
    ) -> List[Document]:
        """Run similarity search with AnalyticDB with distance.

        Args:
            query (str): Query text to search for.
            k (int): Number of results to return. Defaults to 4.
            filter (Optional[Dict[str, str]]): Filter by metadata. Defaults to None.

        Returns:
            List of Documents most similar to the query.
        """
        embedding = self.embedding_function.embed_query(text=query)
        return self.similarity_search_by_vector(
            embedding=embedding,
            k=k,
            filter=filter,
        )

    def similarity_search_with_score(
            self,
            query: str,
            k: int = 4,
            filter: Optional[dict] = None,
    ) -> list[Document]:
        """Return docs most similar to query.

        Args:
            query: Text to look up documents similar to.
            k: Number of Documents to return. Defaults to 4.
            filter (Optional[Dict[str, str]]): Filter by metadata. Defaults to None.

        Returns:
            List of Documents most similar to the query and score for each
        """
        embedding = self.embedding_function.embed_query(query)
        docs = self.similarity_search_with_score_by_vector(
            embedding=embedding, k=k, filter=filter
        )
        return docs

    def similarity_search_with_score_by_vector(
            self,
            embedding: List[float],
            k: int = 4,
            filter: Optional[dict] = None,
    ) -> List[Document]:
        # Add the filter if provided
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
            FROM {self.knowledge_name}
            {filter_condition}
            ORDER BY embedding <-> :embedding
            LIMIT :k
        """

        # Set up the query parameters
        params = {"embedding": embedding, "k": k}

        # Execute the query and fetch the results
        with self.engine.connect() as conn:
            results: Sequence[Row] = conn.execute(text(sql_query), params).fetchall()


        documents = []
        for result in results:
            if 0 < self.score_threshold < result.distance:
                continue
            result.metadata["score"] = int(result.distance) if self.embedding_function is not None else None
            documents.append(Document(
                        page_content=result.document,
                        metadata=result.metadata,
                    ))
        return documents

        # documents_with_scores = [
        #     (
        #         Document(
        #             page_content=result.document,
        #             metadata=result.metadata,
        #         ),
        #         result.distance if self.embedding_function is not None else None,
        #     )
        #     for result in results
        # ]
        # return documents_with_scores

    def similarity_search_by_vector(
            self,
            embedding: List[float],
            k: int = 4,
            filter: Optional[dict] = None,
            **kwargs: Any,
    ) -> List[Document]:
        """Return docs most similar to embedding vector.

        Args:
            embedding: Embedding to look up documents similar to.
            k: Number of Documents to return. Defaults to 4.
            filter (Optional[Dict[str, str]]): Filter by metadata. Defaults to None.

        Returns:
            List of Documents most similar to the query vector.
        """
        docs_and_scores = self.similarity_search_with_score_by_vector(
            embedding=embedding, k=k, filter=filter
        )
        return [doc for doc, _ in docs_and_scores]



    @classmethod
    def from_texts(
            cls: Type[AnalyticDB],
            texts: List[str],
            embedding: Embeddings,
            metadatas: Optional[List[dict]] = None,
            embedding_dimension: int = LANGCHAIN_DEFAULT_EMBEDDING_DIM,
            knowledge_name: str = LANGCHAIN_DEFAULT_KNOWLEDGE_NAME,
            ids: Optional[List[str]] = None,
            pre_delete_collection: bool = False,
            engine_args: Optional[dict] = None,
            **kwargs: Any,
    ) -> AnalyticDB:
        """
        Return VectorStore initialized from texts and embeddings.
        Postgres Connection string is required
        Either pass it as a parameter
        or set the PG_CONNECTION_STRING environment variable.
        """
        connection_string = cls.get_connection_string(kwargs)
        store = cls(
            connection_string=connection_string,
            knowledge_name=knowledge_name,
            embedding_function=embedding,
            embedding_dimension=embedding_dimension,
            pre_delete_knowledge=pre_delete_collection,
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
    def from_documents(
            cls: Type[AnalyticDB],
            documents: List[Document],
            embedding: Embeddings,
            embedding_dimension: int = LANGCHAIN_DEFAULT_EMBEDDING_DIM,
            knowledge_name: str = LANGCHAIN_DEFAULT_KNOWLEDGE_NAME,
            ids: Optional[List[str]] = None,
            pre_delete_collection: bool = False,
            engine_args: Optional[dict] = None,
            **kwargs: Any,
    ) -> AnalyticDB:
        """
        Return VectorStore initialized from documents and embeddings.
        Postgres Connection string is required
        Either pass it as a parameter
        or set the PG_CONNECTION_STRING environment variable.
        """

        texts = [d.page_content for d in documents]
        metadatas = [d.metadata for d in documents]
        connection_string = cls.get_connection_string(kwargs)

        kwargs["connection_string"] = connection_string

        return cls.from_texts(
            texts=texts,
            pre_delete_collection=pre_delete_collection,
            embedding=embedding,
            embedding_dimension=embedding_dimension,
            metadatas=metadatas,
            ids=ids,
            knowledge_name=knowledge_name,
            engine_args=engine_args,
            **kwargs,
        )

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




    def delete(self, ids: Optional[List[str]] = None, **kwargs: Any) -> Optional[bool]:
        """Delete by vector IDs.

        Args:
            ids: List of ids to delete.
        """
        if ids is None:
            raise ValueError("No ids provided to delete.")

        try:
            with self.engine.connect() as conn:
                with conn.begin():
                    delete_condition = self.knowledge_table.c.id.in_(ids)
                    conn.execute(self.knowledge_table.delete().where(delete_condition))
                    return True
        except Exception as e:
            print("Delete operation failed:", str(e))
            return False


    def delete_doc(self, source: str or List[str]):
        try:
            result = []
            # 查出文件路径等于给定的source的记录的id
            with self.engine.connect() as conn:
                with conn.begin():
                    if isinstance(source, str):
                        select_condition = self.knowledge_table.c.metadata.op("->>")("source") == source
                        s = select(self.knowledge_table.c.id).where(select_condition)
                        result = conn.execute(s).fetchall()
                    else:
                        for src in source:
                            select_condition = self.knowledge_table.c.metadata.op("->>")("source") == src
                            s = select(self.knowledge_table.c.id).where(select_condition)
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
        with self.engine.connect() as conn:
            with conn.begin():
                s = select(self.knowledge_table.c.metadata["source"])
                result = conn.execute(s).fetchall()
        return list(set(v[0] for v in result))


    # def list_docs_(self):
    #     with self.engine.connect() as conn:
    #         with conn.begin():
    #             # s = select(self.knowledge_table.c.metadata["source"]).group_by(self.knowledge_table.c.metadata["source"])
    #             s = select().select_from(self.knowledge_table).group_by(self.knowledge_table.c.metadata['source'])
    #             # s = select(self.knowledge_table.c.metadata["source"])
    #             result = conn.execute(s).fetchall()
    #             print(result)
        # return list(set(v[0] for v in result))