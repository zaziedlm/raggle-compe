import json
import sys
import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain import callbacks

# 環境変数から、OpenAI API キーを取得する
load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")

#==============================================================================
# !!! 警告 !!!: 以下の変数を変更しないでください。
#==============================================================================
model = "gpt-4o-mini"
pdf_file_urls = [
    "https://storage.googleapis.com/gg-raggle-public/competitions/29676d73-5675-4278-b1a6-d4a9fdd0a0ba/dataset/Architectural_Design_Service_Contract.pdf",
    "https://storage.googleapis.com/gg-raggle-public/competitions/29676d73-5675-4278-b1a6-d4a9fdd0a0ba/dataset/Call_Center_Operation_Service_Contract.pdf",
    "https://storage.googleapis.com/gg-raggle-public/competitions/29676d73-5675-4278-b1a6-d4a9fdd0a0ba/dataset/Consulting_Service_Contract.pdf",
    "https://storage.googleapis.com/gg-raggle-public/competitions/29676d73-5675-4278-b1a6-d4a9fdd0a0ba/dataset/Content_Production_Service_Contract_(Request_Form).pdf",
    "https://storage.googleapis.com/gg-raggle-public/competitions/29676d73-5675-4278-b1a6-d4a9fdd0a0ba/dataset/Customer_Referral_Contract.pdf",
    "https://storage.googleapis.com/gg-raggle-public/competitions/29676d73-5675-4278-b1a6-d4a9fdd0a0ba/dataset/Draft_Editing_Service_Contract.pdf",
    "https://storage.googleapis.com/gg-raggle-public/competitions/29676d73-5675-4278-b1a6-d4a9fdd0a0ba/dataset/Graphic_Design_Production_Service_Contract.pdf",
    "https://storage.googleapis.com/gg-raggle-public/competitions/29676d73-5675-4278-b1a6-d4a9fdd0a0ba/dataset/M&A_Advisory_Service_Contract_(Preparatory_Committee).pdf",
    "https://storage.googleapis.com/gg-raggle-public/competitions/29676d73-5675-4278-b1a6-d4a9fdd0a0ba/dataset/M&A_Intermediary_Service_Contract_SME_M&A_[Small_and_Medium_Enterprises].pdf",
    "https://storage.googleapis.com/gg-raggle-public/competitions/29676d73-5675-4278-b1a6-d4a9fdd0a0ba/dataset/Manufacturing_Sales_Post-Safety_Management_Contract.pdf",
    "https://storage.googleapis.com/gg-raggle-public/competitions/29676d73-5675-4278-b1a6-d4a9fdd0a0ba/dataset/software_development_outsourcing_contracts.pdf",
    "https://storage.googleapis.com/gg-raggle-public/competitions/29676d73-5675-4278-b1a6-d4a9fdd0a0ba/dataset/Technical_Verification_(PoC)_Contract.pdf",
]
#==============================================================================


#==============================================================================
# この関数を編集して、あなたの RAG パイプラインを実装してください。
# !!! 注意 !!!: デバッグ過程は標準出力に出力しないでください。
#==============================================================================
def generate_rag_prompt(question: str) -> str:
    """
    ユーザーからの質問を受け取り、RAGに適したプロンプトをLLMを使って生成する関数。
    """
    # プロンプト生成のためのプロンプト
    # あなたはプロンプト生成の専門家です。以下のユーザーからの質問を、RAG（検索拡張生成）に適した形式に再構成してください。再構成する際には、重要なキーワードを強調し、必要に応じて質問を明確化してください。ただし、ユーザーの意図を変更しないでください。
    prompt_template = """
    あなたは契約文書の専門家です。以下のユーザーからの質問を、法律用語を用いて、回答し易いよう明確かつ簡潔な形式に再構成してください。再構成する際には、関係する契約の種類が明確であれば、それを含めてください。さらに重要なキーワードや契約名、日付、日時、金額、委託契約、受託契約などの具体的な情報を強調してください。ただし、ユーザーの意図を変更せず、質問の焦点を明確にしてください。
    
    ユーザーの質問: {question}

    RAGに適したプロンプト:
    """

    # プロンプトをフォーマット
    formatted_prompt = prompt_template.format(question=question)

    # LLMを使ってプロンプトを生成
    llm = ChatOpenAI(model=model, openai_api_key=openai_api_key)
    response = llm.invoke([HumanMessage(content=formatted_prompt)])

    # 結果をパース
    parser = StrOutputParser()
    result = parser.parse(response)

    # print(result.content)

    return result.content.strip()

def rag_implementation(question: str) -> str:

    # ユーザーの質問をRAGに適したプロンプトに再構成
    rag_ready_question = generate_rag_prompt(question)

    # # RAG参照用ドキュメントを取得する
    # rag_documents = []
    # for pdf_url in pdf_file_urls:
    #     loader = PyPDFLoader(pdf_url)
    #     rag_documents.extend(loader.load())

    # # ドキュメントをチャンク分割する
    # text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=500)
    # rag_texts = text_splitter.split_documents(rag_documents)

    
    # RAG参照用ドキュメントを取得する
    rag_documents = []
    for pdf_url in pdf_file_urls:
        loader = PyPDFLoader(pdf_url)
        documents = loader.load()

        # 1ページ目から契約種類を取得
        first_page_content = documents[0].page_content
        lines = first_page_content.strip().split('\n')
        
        # 「契約」を含む最初の行を取得
        contract_type = "契約以外"  # デフォルト値
        for line in lines:
            line = line.strip()
            if "契約" in line:
                contract_type = line
                break

        # print(contract_type)

        # 各ドキュメントに契約種類をメタデータとして追加
        for doc in documents:
            doc.metadata['contract_type'] = contract_type

        rag_documents.extend(documents)

    # ドキュメントをチャンク分割する
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=500)
    rag_texts = text_splitter.split_documents(rag_documents)

    # 各チャンクの先頭に契約種類を追記
    for doc in rag_texts:
        contract_type = doc.metadata.get('contract_type', '')
        # テキストの先頭に契約種類を追加
        doc.page_content = f"契約種類: {contract_type}\n{doc.page_content}"
 

    # OpenAI埋め込みモデルによるベクトルストア化
    embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
    vector_store = Chroma.from_documents(rag_texts, embeddings)

    # 質問に対するベクトル検索を実行する
    retriever = vector_store.as_retriever(search_kwargs={"k": 5})

    # prompt = hub.pull("rlm/rag-prompt")
    prompt = """
    あなたは質問応答タスクのためのアシスタントです。以下の取得したコンテキストに内容を使用して質問に答えてください。答えがわからない場合は、わからないとだけ言ってください。最大で3文を使用し、簡潔に答えてください。
    Question: {question} 
    Context: {context} 
    Answer: 
    """

    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    # パイプラインの各ステップを明確に定義する
    # retrieved_docs = retriever.invoke(question)
    retrieved_docs = retriever.invoke(rag_ready_question)
    formatted_context = format_docs(retrieved_docs)

    llm = ChatOpenAI(model=model, openai_api_key=openai_api_key, temperature=0.5, top_p=0.5)  # llm 変数を定義

    # プロンプトをフォーマットする
    formatted_prompt = prompt.format(question=question, context=formatted_context)

    # LLM に問い合わせる
    response = llm.invoke([HumanMessage(content=formatted_prompt)])

    # 結果をパースする
    parser = StrOutputParser()
    result = parser.parse(response)

    #print(result.content)

    return result.content.strip()

#==============================================================================


#==============================================================================
# !!! 警告 !!!: 以下の関数を編集しないでください。
#==============================================================================
def main(question: str):
    with callbacks.collect_runs() as cb:
        result = rag_implementation(question)
        run_id = cb.traced_runs[0].id

    output = {"result": result, "run_id": str(run_id)}
    print(json.dumps(output))


if __name__ == "__main__":
    load_dotenv()

    if len(sys.argv) > 1:
        question = sys.argv[1]
        main(question)
    else:
        print("Please provide a question as a command-line argument.")
        sys.exit(1)
#==============================================================================
