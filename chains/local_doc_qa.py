from langchain.chains import RetrievalQA
from langchain.embeddings import OpenAIEmbeddings
from langchain.retrievers import ChatGPTPluginRetriever
from langchain import VectorDBQA, OpenAI
from langchain.prompts import PromptTemplate

# return top-k text chunk from vector store
VECTOR_SEARCH_TOP_K = 3

# LLM input history length
LLM_HISTORY_LEN = 3

class LocalDocQA:
    llm: object = None
    embeddings: object = None

    def init_cfg(self,
                 top_k=VECTOR_SEARCH_TOP_K,
                 ):
        self.llm = OpenAI()
        self.embeddings = OpenAIEmbeddings()
        self.top_k = top_k
        self.history = []

    def get_knowledge_based_answer(self,
                                   query,
                                   chat_history=[], ):
        prompt_template = """基于以下已知信息回答问题。
    如果无法从中得到答案，请说 "根据已知信息无法回答该问题"，不允许在答案中添加编造成分。
    
    已知内容:
    {context}
    
    问题:
    {question}
    
    回答:"""
        prompt = PromptTemplate(
            template=prompt_template,
            input_variables=["context", "question"]
        )
        self.history = chat_history
        retriever = ChatGPTPluginRetriever(url="http://127.0.0.1:8000", bearer_token="eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiIxMjM0NTY3ODkwIiwibmFtZSI6IkpvaG4gRG9lIiwiaWF0IjoxNjgwMTU4ODQ4LCJleHAiOjE2ODA3NjM2NDh9.lcS3kSqIWjOoCdzDyfB87bvr6xUCoOpHNKIuBobuhME", top_k=self.top_k)
        knowledge_chain = RetrievalQA.from_llm(
            llm=self.llm,
            retriever=retriever,
            prompt=prompt
        )
        knowledge_chain.combine_documents_chain.document_prompt = PromptTemplate(
            input_variables=["page_content"], template="{page_content}"
        )

        knowledge_chain.return_source_documents = True

        result = knowledge_chain({"query": query})
        self.history = self.history + [[None, result["result"]]]
        self.history[-1][0] = query
        return result, self.history
