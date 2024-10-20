import json
import sys

from dotenv import load_dotenv
from langchain import callbacks


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
def rag_implementation(question: str) -> str:
    import io
    import urllib.request
    from pypdf import PdfReader
    from langchain.text_splitter import RecursiveCharacterTextSplitter
    from rank_bm25 import BM25Okapi
    from sudachipy import dictionary
    from sudachipy import tokenizer as sudachi_tokenizer
    from langchain_core.output_parsers import StrOutputParser
    from langchain_core.messages import HumanMessage
    from langchain_openai import ChatOpenAI
    import os
    from dotenv import load_dotenv

    # 環境変数から、OpenAI API キーを取得する
    load_dotenv()
    openai_api_key = os.getenv("OPENAI_API_KEY")

    # Initialize variables
    all_texts = []

    # Initialize SudachiPy tokenizer
    tokenizer_obj = dictionary.Dictionary().create()
    mode = sudachi_tokenizer.Tokenizer.SplitMode.C

    # Step 0: Refine the question using OpenAI API
    def refine_question_func(question_text):
        """
        ユーザーからの質問を受け取り、RAGに適したプロンプトをLLMを使って生成する関数。
        """
        # プロンプト生成のためのプロンプト
        # あなたはプロンプト生成の専門家です。以下のユーザーからの質問を、RAG（検索拡張生成）に適した形式に再構成してください。再構成する際には、重要なキーワードを強調し、必要に応じて質問を明確化してください。ただし、ユーザーの意図を変更しないでください。
        prompt_template = """
        あなたは契約文書の専門家です。以下のユーザーからの質問を、法律用語を用いて、回答し易いよう明確かつ簡潔な形式に再構成してください。再構成する際には、関係する契約の種類が明確であれば、それを含めてください。さらに重要なキーワードや契約名、日時、金額、委託契約、受託契約などの具体的な情報を強調してください。ただし、ユーザーの意図を変更せず、質問の焦点を明確にしてください。
        
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

    refined_question = refine_question_func(question)

    # Debug: Check the type of refined_question
    # print(f"Type of refined_question: {type(refined_question)}")  # Should be <class 'str'>

    # Step 1: Download PDFs and extract texts
    for url in pdf_file_urls:
        try:
            # Download PDF
            response = urllib.request.urlopen(url)
            pdf_data = response.read()
            # Read PDF
            pdf_reader = PdfReader(io.BytesIO(pdf_data))
            # Extract text from each page
            text = ""
            for page in pdf_reader.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text
            all_texts.append(text)
        except Exception:
            continue  # Skip this PDF if there's an error

    # Step 2: Split texts into chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
        length_function=len,
    )
    documents = []
    for text in all_texts:
        splits = text_splitter.split_text(text)
        documents.extend(splits)

    # Step 3: Tokenize documents
    tokenized_corpus = []
    for doc in documents:
        tokens = [m.surface() for m in tokenizer_obj.tokenize(doc, mode)]
        tokenized_corpus.append(tokens)

    # Step 4: Build BM25 index
    bm25 = BM25Okapi(tokenized_corpus)

    # Step 5: Tokenize refined question
    query_tokens = [m.surface() for m in tokenizer_obj.tokenize(refined_question, mode)]

    # Step 6: Retrieve relevant documents
    scores = bm25.get_scores(query_tokens)
    top_n = 5  # Number of top documents to retrieve
    top_n_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_n]
    retrieved_docs = [documents[i] for i in top_n_indices]

    # Step 7: Construct prompt
    context = "\n".join(retrieved_docs)
    # prompt = f"以下の情報を参考にして、質問に答えてください。\n\n{context}\n\n質問: {question}\n\n答え:"

    prompt = """
        あなたは質問応答タスクのためのアシスタントです。以下の取得したコンテキストを使用して質問に答えてください。答えがわからない場合は、「申し訳ありません。情報が見つからず、現在はわかりません。」とだけ言ってください。最大で3文を使用し、長文にならぬよう簡潔に答えてください。
        Question: {question} 
        Context: {context} 
        Answer: 
        """
    # Step 8: Call LLM
    # llm = OpenAI(model=model, temperature=0, openai_api_key=openai_api_key)
    # answer = llm(prompt)

    # # Return the answer
    # return answer.strip()

    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    # パイプラインの各ステップを明確に定義する
    # retrieved_docs = retriever.invoke(question)
    # retrieved_docs = retriever.invoke(rag_ready_question)
    #formatted_context = format_docs(retrieved_docs)

    llm = ChatOpenAI(model=model, openai_api_key=openai_api_key)  # llm 変数を定義

    # プロンプトをフォーマットする
    formatted_prompt = prompt.format(question=question, context=context)

    # LLM に問い合わせる
    response = llm.invoke([HumanMessage(content=formatted_prompt)])

    # 結果をパースする
    parser = StrOutputParser()
    result = parser.parse(response)
    # print(result.content)

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
