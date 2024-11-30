# RAGコンペ参加のソースコード公開  （ 法務RAGシステムの性能改善ハッカソン )
raggle  https://raggle.jp/competition/29676d73-5675-4278-b1a6-d4a9fdd0a0ba


このプロジェクトは、RAGコンペ「法務RAGシステムの性能改善ハッカソン」に参加した際に作成したソースコードです。3回のプログラム投稿で、

main-send01.py, main-send02.py, main-send03.py

の3つのソースを作成しました。これらのソースは順に、回答精度を高めるための改良を施しています。本ドキュメントでは、RAGの生成AIプログラミングを学べるように、各ソースの処理内容を解説しています。

***
# 投稿したソース ３回分

- main-send01.py
- main-send02.py
- main-send03.py
***
# main-send01.py

## （第１回投稿分　埋め込みモデルを利用して、基本的なベクトル検索を実装）

### 概要
基本的なRAG（Retrieval-Augmented Generation）パイプラインを実装したスクリプトです。複数のPDFファイルからテキストを抽出し、ユーザーの質問に対して関連するドキュメントを検索し、回答を生成します。

### 処理内容

1. **データの読み込み**

   - 指定されたPDFファイルをダウンロードし、PyPDFLoaderを使用してテキストを抽出します。

2. **テキストのチャンク化**

   - RecursiveCharacterTextSplitterを使用して、テキストを一定の長さで分割します。

3. **ベクトルストアの構築**

   - OpenAIEmbeddingsを使用してテキストチャンクをベクトル化し、Chromaを用いてベクトルストアを構築します。

4. **質問に対するドキュメント検索**

   - ユーザーの質問をベクトル化し、ベクトルストアから関連するドキュメントを検索します。

5. **回答生成**

   - 検索で得られたコンテキスト情報を基に、ChatOpenAIを使用して回答を生成します。

### 改善点

- 基本的なRAGパイプラインの実装に留まっており、回答の精度や適切性に課題が残ります。
***
# main-send02.py
## （第２回投稿分　BM25アルゴリズムの導入で、回答精度の向上を目指す）

### 概要
main-send01.pyの機能に加えて、BM25アルゴリズムと形態素解析を導入し、日本語の検索精度を向上させました。

### 処理内容

1. **質問の精錬**

   - LLMを使用して、ユーザーの質問を法律用語を用いた明確な形式に再構成します。

2. **データの読み込み**

   - urllibとPyPDFLoaderを使用してPDFをダウンロード・読み込み、テキストを抽出します。

3. **テキストのチャンク化**

   - RecursiveCharacterTextSplitterを使用し、テキストを小さなチャンクに分割します。

4. **形態素解析とBM25による検索**

   - SudachiPyで形態素解析を行い、BM25Okapiを用いて関連するドキュメントを検索します。

5. **回答生成**

   - 検索結果を基に、ChatOpenAIで回答を生成します。

### 改善点

- 日本語の形態素解析とBM25アルゴリズムの導入により、関連ドキュメントの検索精度が向上しました。
- 質問の精錬により、LLMが意図を正確に理解しやすくなりました。
***
# main-send03.py
## （第３回投稿分　メタ情報の有効活用や、プロンプトの精緻化を実施）

### 概要
メタデータの活用やプロンプトの最適化など、さらなる改善を加え、より高度なRAGシステムを実現しました。

### 処理内容

1. **質問の精錬**

   - main-send02.pyと同様に質問を再構成しますが、プロンプトを最適化してより良い精錬を行います。

2. **データの読み込みとメタデータの付与**

   - テキスト抽出時に、契約書のタイトルや契約種類などのメタデータを取得し、各ドキュメントに付与します。

3. **テキストのチャンク化とメタデータの管理**

   - チャンク化したテキストに対してもメタデータを保持し、後の処理で利用します。

4. **形態素解析とBM25による検索**

   - より精錬された質問とドキュメントで形態素解析を行い、BM25で検索を実施します。

5. **出典情報の追加と回答生成**

   - 検索結果に含まれるドキュメントの出典情報（タイトルなど）を回答に含めるようにプロンプトを調整し、ChatOpenAIで回答を生成します。

### 改善点

- **メタデータの活用**: ドキュメントに付与したメタデータを用いて、回答に出典を明記することで信頼性を向上させました。
- **プロンプトの最適化**: LLMへの指示を具体的にし、必要な情報を的確に含めるようにしました。
- **形態素解析の精度向上**: トークナイザーの設定やアルゴリズムの調整により、検索精度をさらに向上させました。
***
# まとめ

3つのソースコードを通じて、RAGシステムの基本から実践的な実装方法を学ぶことができると思います。各段階での工夫や改善点は、生成AIプログラミングを学ぶ上で有用な知見となりそうです。特に、日本語のテキスト処理や固有情報の検索、LLMの効果的な活用方法について理解を深められると幸いです。

ぜひ、各ソースコードを参照しながら、また実際に実行させてみて、LLMシステムの理解や自身の開発にも役立ててみてください。