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

    # 初期化
    all_texts = []
    doc_metadata = []

    # SudachiPyのトークナイザーを初期化
    tokenizer_obj = dictionary.Dictionary().create()
    mode = sudachi_tokenizer.Tokenizer.SplitMode.C

    # Step 0: 質問をLLMで精錬する
    def refine_question_func(question_text):
        """
        ユーザーからの質問を受け取り、RAGに適したプロンプトをLLMを使って生成する関数。
        """
        prompt_template = """
あなたは契約文書の専門家です。以下のユーザーからの質問を、法律用語を用いて、回答しやすいよう明確かつ簡潔な形式に再構成してください。再構成する際には、関係する契約の種類が明確であれば、それを含めてください。さらに重要なキーワードや契約名、日時、金額、委託契約、受託契約などの具体的な情報を強調してください。ただし、ユーザーの意図を変更せず、質問の焦点を明確にしてください。

ユーザーの質問: {question}

RAGに適したプロンプト:
"""

        formatted_prompt = prompt_template.format(question=question_text)
        llm = ChatOpenAI(model=model, temperature=0.3, openai_api_key=openai_api_key)
        response = llm.invoke([HumanMessage(content=formatted_prompt)])

        parser = StrOutputParser()
        result = parser.parse(response)

        return result.content.strip()

    refined_question = refine_question_func(question)

    # Step 1: PDFをダウンロードしてテキストを抽出
    for idx, url in enumerate(pdf_file_urls):
        try:
            # PDFをダウンロード
            response = urllib.request.urlopen(url)
            pdf_data = response.read()
            # PDFを読み込む
            pdf_reader = PdfReader(io.BytesIO(pdf_data))
            # 各ページからテキストを抽出
            text = ""
            for page_num, page in enumerate(pdf_reader.pages):
                page_text = page.extract_text()
                if page_text:
                    text += page_text
            all_texts.append(text)

            # メタデータを保存
            # 初期メタデータ
            metadata = {
                'source': f'Document {idx+1}',
                'url': url
            }

            # テキストを行に分割
            lines = text.split('\n')
            # 最初の3行を取得
            first_three_lines = lines[:3]
            # 最初の3行の中で「契約」を含む行を探す
            title = None
            for line in first_three_lines:
                if '契約' in line:
                    title = line.strip()
                    break
            if title:
                # メタデータにタイトルを追加
                metadata['title'] = title

            doc_metadata.append(metadata)

        except Exception:
            continue  # エラーが発生した場合、このPDFをスキップ

    # Step 2: テキストをチャンクに分割
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
        length_function=len,
    )
    documents = []
    metadatas = []
    for idx, text in enumerate(all_texts):
        splits = text_splitter.split_text(text)
        documents.extend(splits)
        # 各チャンクにメタデータを割り当て
        metadatas.extend([doc_metadata[idx]] * len(splits))

    # Step 3: ドキュメントをトークン化
    tokenized_corpus = []
    for doc in documents:
        tokens = [m.surface() for m in tokenizer_obj.tokenize(doc, mode)]
        tokenized_corpus.append(tokens)

    # Step 4: BM25インデックスを構築
    bm25 = BM25Okapi(tokenized_corpus)

    # Step 5: 精錬された質問をトークン化
    query_tokens = [m.surface() for m in tokenizer_obj.tokenize(refined_question, mode)]

    # Step 6: 関連するドキュメントを検索
    scores = bm25.get_scores(query_tokens)
    top_n = 5  # 上位5つのドキュメントを取得
    top_n_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_n]
    retrieved_docs = [documents[i] for i in top_n_indices]
    retrieved_metadatas = [metadatas[i] for i in top_n_indices]

    # Step 7: 出典を含めてプロンプトを構築
    context = ""
    for idx, doc in enumerate(retrieved_docs):
        title = retrieved_metadatas[idx].get('title', 'No Title')
        context += f"出典：{title}\n{doc}\n\n"

    prompt = ("あなたは契約文書の専門家です。以下のコンテキストを参考に、ユーザーの質問に対して"
              "簡潔かつ直接的な回答を提供してください。必要に応じて契約用語を使用し、過度な詳細や冗長な説明は避けてください。"
              "回答の中で参考にした文書を簡潔に明記してください（例: 「（○○○○契約書第○条より）」）。"
              "もし答えがわからない場合は、「申し訳ありません。情報が見つかりませんでした。」とだけ述べてください。\n\n"
              f"質問内容: {question}\n"
              "参考にするコンテキスト:\n"
              f"{context}\n"
              "質問への回答:")

    # Step 8: LLMを呼び出す
    llm = ChatOpenAI(model=model, temperature=0.3, openai_api_key=openai_api_key)

    response = llm.invoke([HumanMessage(content=prompt)])

    parser = StrOutputParser()
    result = parser.parse(response)

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
