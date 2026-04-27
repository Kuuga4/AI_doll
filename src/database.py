import os
import argparse
import shutil
from langchain_community.document_loaders import PyPDFLoader,Docx2txtLoader,TextLoader
from langchain.schema.document import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings.ollama import OllamaEmbeddings
from langchain_community.vectorstores import Chroma


doc_path = "./document"
#导入文档
def load_documents():
    folder_path = "./document"
    documents = [] 
    for filename in os.listdir(folder_path):
        #print(filename)
        file_path = os.path.join(folder_path, filename)
        file_name, file_extension = os.path.splitext(filename)
        #print(file_name, file_extension)
        if file_extension == ".pdf":
            document_loader = PyPDFLoader(file_path)
            documents.extend(document_loader.load()) 
        elif file_extension == ".docx":
            document_loader = Docx2txtLoader(file_path)
            documents.extend(document_loader.load())  
        elif file_extension ==".txt":
            document_loader = TextLoader(file_path,encoding='utf-8')
            documents.extend(document_loader.load())
    return documents 

documents = load_documents()



#文档分块
def split_documents(documents:list[Document]):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size = 800,
        chunk_overlap = 80,
        length_function = len,
        is_separator_regex =False,
    )
    return text_splitter.split_documents(documents)

chunks = split_documents(documents)
'''for i in range(len(chunks)):
    print(chunks[i])'''

#embedding with ollama
def get_embedding_function():
    embeddings = OllamaEmbeddings(model="nomic-embed-text"
    )
    return embeddings


#创建区块chunks id
def calculate_chunk_ids(chunks):
    last_page_id = None
    current_chunk_index = 0

    #同页加index
    for chunk in chunks:
        source = chunk.metadata.get("source")
        page = chunk.metadata.get("page")
        current_page_id = f"{source}:{page}"

        if current_page_id ==last_page_id:
            current_chunk_index += 1
        else:
            current_chunk_index = 0
        
        chunk_id = f"{current_page_id}:{current_chunk_index}"
        last_page_id = current_page_id

        chunk.metadata["id"] = chunk_id
    
    return chunks


#用chroma创建向量数据库
CHROMA_PATH = "chroma"

def add_to_chroma(chunks:list[Document]):
    db = Chroma(
        persist_directory= CHROMA_PATH,embedding_function=get_embedding_function()
    )
    
    chunks_with_ids = calculate_chunk_ids(chunks)

    existing_items = db.get(include=[])
    existing_ids = set(existing_items["ids"])
    print(f"数据库现有文件数：{len(existing_ids)}")

    #添加新文件
    new_chunks= []
    for chunk in chunks_with_ids:
        if chunk.metadata["id"] not in existing_ids:
            new_chunks.append(chunk)   
    
    if len(new_chunks):
        print(f"添加新文件 ：{len(new_chunks)}")
        new_chunk_ids = [chunk.metadata["id"] for chunk in new_chunks]
        db.add_documents(new_chunks, ids=new_chunk_ids)
        db.persist()
    else:
        print("没有发现新文件")

#清除数据
def clear_database():
    if os.path.exists(CHROMA_PATH):
        shutil.rmtree(CHROMA_PATH)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--reset", action="store_true", help="Reset the database.")
    args = parser.parse_args()
    if args.reset:
        print("清除数据集")
        clear_database()

    documents = load_documents()
    chunks = split_documents(documents)
    add_to_chroma(chunks)

if __name__ == "__main__":
    main()