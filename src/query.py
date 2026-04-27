import argparse
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_community.vectorstores import Chroma
from langchain.prompts import ChatPromptTemplate
from langchain_community.embeddings  import OllamaEmbeddings
from langchain_core.runnables import RunnableParallel, RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
import os


load_dotenv()
directory_path = os.path.dirname(os.path.abspath(__file__))

CHROMA_PATH = directory_path+"\chroma"

PROMPT_TEMPLATE = """
请根据以下步骤判断并回答用户的问题，回答需严格控制在100字以内，并使用口语化表达，结构清晰，语气符合赫敏·格兰杰的角色设定：  

1. 判断问题类型：    
    - 如果问题与设计作品“EchoSense VR Training”相关，请严格根据文档内容回答。    
    - 如果问题与《哈利·波特》系列相关，可不依赖文档内容，自由回答。    
    - 如果问题既不与设计作品相关，也不与《哈利·波特》相关，请礼貌地告知无法作答。
    
2. 角色设定：    
    请始终以赫敏·格兰杰的角色进行回答，表现出智慧、自信和逻辑清晰。语言要自然流畅，逻辑严谨，但要注重亲切感和条理性。  
    
3. 回答方式：    
    - 对于设计作品相关问题：请参考以下文档内容，提炼重点，简明扼要地回答：{context}    
    - 对于《哈利波特》相关问题：请结合你的知识和逻辑分析，用简洁明了的语言作答。    
    - 对于其他问题：请礼貌回复用户，示例如下： “哦，真抱歉，这个问题似乎超出了我的知识领域。”  
    
---  

用户问题： {question} 
"""


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("query_text", type=str, help="The query text.")
    args = parser.parse_args()
    query_text = args.query_text
    query_rag(query_text)


def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

def query_rag(query_text: str):
    embedding_function  = OllamaEmbeddings(model="nomic-embed-text")

    llm = ChatOpenAI(
    model="gpt-4o",
    temperature=0,
    max_tokens=None,
    timeout=None,
    max_retries=2,
    base_url="https://api.guidaodeng.com/v1",
    )

    db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding_function)

#实例化检索器
    retriever = db.as_retriever(
        search_kwargs={
            "k": 3}
    )

    prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)

    rag_chain_from_docs = (
        RunnablePassthrough.assign(context=(lambda x: format_docs(x["context"])))
        |prompt_template
        |llm
        |StrOutputParser()
    )
   
    print("source output")

    rag_chain_with_source = RunnableParallel(
        {"context": retriever, "question": RunnablePassthrough()}
    ).assign(answer=rag_chain_from_docs)

    answer = rag_chain_with_source.invoke(query_text)
    print(answer["answer"])
    return answer["answer"]

if __name__ == "__main__":
    main()
