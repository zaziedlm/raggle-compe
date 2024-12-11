# RAGコンペ(raggle: 法務RAGシステムの性能改善ハッカソン)参加記録のプロジェクト
[raggle : 日本最強のRAGエンジニアを目指そう](https://raggle.jp/)  
[第1回Raggleコンペティション表彰式（YouTube)](https://youtu.be/QO6aGdMLJNE)

## 📚 概要
このリポジトリは、**RAGコンペ** の参加を通じて実装した、RAG (Retrieval-Augmented Generation) システムのソースコードを含みます。本プロジェクトでは、指定された評価指標に基づいた、RAGシステムの性能向上を目指しました。  
[法務RAGシステムの性能改善ハッカソン](https://raggle.jp/competition/29676d73-5675-4278-b1a6-d4a9fdd0a0ba)

---

## 🚀 プロジェクト参加進行とエビデンス

### **参加理由**
- RAGシステムの仕組みを再整理しながら理解し、実践的な実装スキルを向上させるため。
- ベクトル検索とLLMを組み合わせた技術、それ以外の有効な検索技術に触れるため。
- 限られた時間のコンペティションの中で競争し、高スコアを目指すため。

---

### **タイムライン**
| 日付         | マイルストーン                           | 内容                                                                 |
|--------------|------------------------------------------|----------------------------------------------------------------------|
| 2024-10-19   | ベースラインモデル(第1回)の提出                | 埋め込みモデルによる基本的なRAG実装による初回提出。                                      |
| 2024-10-20   | 改善版(第2回)の提出                              | 検索アルゴリズムの見直しとプロンプトエンジニアリングの導入。         |
| 2024-10-20   | 最終版(第3回)の提出                                | メタデータ付与などさらに改善を反映した最適化モデルの提出。                             |

---

## **RAGアプローチと手法**  
## [スコア詳細](images/raggle-score.png) （画像クリックでも拡大表示）

<a href="images/raggle-score.png" target="_blank">
    <img src="images/raggle-score.png" alt="サムネイル画像" width="300">
</a>

### RAGパイプラインの性能改善

ここでは、3回にわたるRAGパイプライン実装の改善ポイントとその影響をまとめています。

#### パフォーマンス比較（スコア評価：運営側で設定した独自評価指標による）

| 指標          | 1回目の投稿 | 2回目の投稿 | 3回目の投稿 |
|---------------|-------------|-------------|-------------|
| 正確性 (Correctness) | 66.67      | 86.67      | 96.67      |
| 有用性 (Helpfulness)  | 50.00      | 63.33      | 63.33      |
| 簡潔性 (Conciseness)  | 66.67      | 80.00      | 93.33      |
| 無害性 (Harmfulness)  | 63.33      | 80.00      | 80.00      |
| 処理速度 (Speed)       | 191.7      | 7.45       | 10.19      |
| APIコスト (API Cost)   | 0.000262   | 0.00018    | 0.000318   |

---

### 改善内容の詳細

#### 1回目 → 2回目

| 改善項目             | 内容                                                                                       | 影響                                                                                             |
|----------------------|------------------------------------------------------------------------------------------|------------------------------------------------------------------------------------------------|
| **データ抽出**        | `PyPDFLoader`から`PyPDF`に変更し、PDFの直接的なテキスト抽出を実装。                          | 柔軟性と処理精度が向上。                                                                        |
| **テキスト分割**      | `RecursiveCharacterTextSplitter`を導入し、効率的なチャンク分割を実現。                        | 文書検索の効率性と関連性が向上。                                                                |
| **質問精錬**          | LLM（ChatOpenAI）による質問精錬ステップを追加。                                              | 「正確性」が66.67から86.67に向上。「有用性」も50から63.33に改善。                                 |
| **BM25検索**         | `BM25Okapi`を用いたドキュメントランキングを実装。                                            | 関連性の高い文書の抽出が可能になり、「正確性」の向上に貢献。                                     |
| **APIコスト削減**     | パイプライン内の不要なステップを削減。                                                        | APIコストが0.000262から0.00018に削減。                                                          |

---

#### 2回目 → 3回目

| 改善項目             | 内容                                                                                       | 影響                                                                                             |
|----------------------|------------------------------------------------------------------------------------------|------------------------------------------------------------------------------------------------|
| **メタデータの追加**   | テキストチャンクに契約タイトルやURLなどのメタデータを付加。                                    | 回答の出典が明確になり、信頼性が向上。                                                          |
| **プロンプト改善**    | プロンプトに「出典を含む回答」を指示する内容を追加。                                          | 「簡潔性」が80から93.33に向上し、より明確で簡潔な回答が実現。                                    |
| **LLMパラメータ調整** | `temperature`の調整により、安定性と一貫性が向上。                                            | 「正確性」が86.67から96.67に向上。他の指標にも影響を与えず、全体の品質が向上。                  |
| **コンテキスト整理**   | 取得した文書をメタデータ情報とともに再構成し、重要情報を強調。                                | 回答の可読性が向上し、ユーザーの信頼感を向上。                                                  |
| **追加処理**          | メタデータを活用した高度なコンテキスト処理を導入。                                            | APIコストは0.00018から0.000318に増加したものの、回答品質の向上に成功。                          |

---

### 主な改善ポイント

1. **データ処理の最適化**: PDF処理の効率化とBM25による関連文書抽出精度の向上。
2. **質問精錬プロセス**: LLMを活用し、正確かつ有用な質問生成を実現。
3. **メタデータの活用**: 出典を明示することで回答の信頼性を強化。
4. **回答の簡潔化**: プロンプト改善により明確で簡潔な回答を提供。

---

### 今後の改善案

- **コスト最適化**: 検索数（`top_n`）やLLMパラメータを調整し、APIコストをさらに削減する。
- **有用性の向上**: 多様な質問を用いたテストを実施し、「有用性」スコアのさらなる向上を目指す。
- **モデルアップグレード**: 最新のLLMモデルを導入し、性能とコストのバランスを最適化する。


## **順位**
- **開発評価期間末での順位**: 5位
- **最終順位**: 不明（なぜだか、最終評価データセットによるリーダーボードスコアは更新されていない）

---

## **振り返り**
### **成功点**
- プロンプト最適化などにより、正確性、有用性が大幅に向上。
- ドキュメントラインキングを効果的に適用し、検索精度を向上。

### **課題**
- ドキュメントの体裁、品質やドメイン領域を扱い処理する難しさ。
- 検索の精度(Precision)とリコール(Recall)のバランス、トレードオフの管理。

### **学び**
- 実際のRAGシステムでの反復的なテストと評価確認の重要性。
- ベクトルデータベースとLLMの統合、BM25に関する理解の深化。

---

## 🗂 リポジトリ構成
3回のプログラム投稿で、
main-send01.py, main-send02.py, main-send03.py
の3つのソースを作成しました。これらのソースは順に、回答精度を高めるための改良を施しています。ここでは、RAGの生成AIプログラミングを学べるように、各ソースの処理内容を解説しています。

***
## 投稿したソース ３回分

- main-send01.py
- main-send02.py
- main-send03.py
***
## main-send01.py

### （第１回投稿分　埋め込みモデルを利用して、基本的なベクトル検索を実装）

#### 概要
基本的なRAG（Retrieval-Augmented Generation）パイプラインを実装したスクリプトです。複数のPDFファイルからテキストを抽出し、ユーザーの質問に対して関連するドキュメントを検索し、回答を生成します。

#### 処理内容

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

#### 改善点

- 基本的なRAGパイプラインの実装に留まっており、回答の精度や適切性に課題が残ります。
***
## main-send02.py
### （第２回投稿分　BM25アルゴリズムの導入で、回答精度の向上を目指す）

#### 概要
main-send01.pyの機能に加えて、BM25アルゴリズムと形態素解析を導入し、日本語の検索精度を向上させました。

#### 処理内容

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

#### 改善点

- 日本語の形態素解析とBM25アルゴリズムの導入により、関連ドキュメントの検索精度が向上しました。
- 質問の精錬により、LLMが意図を正確に理解しやすくなりました。
***
## main-send03.py
### （第３回投稿分　メタ情報の有効活用や、プロンプトの精緻化を実施）

#### 概要
メタデータの活用やプロンプトの最適化など、さらなる改善を加え、より高度なRAGシステムを実現しました。

#### 処理内容

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

#### 改善点

- **メタデータの活用**: ドキュメントに付与したメタデータを用いて、回答に出典を明記することで信頼性を向上させました。
- **プロンプトの最適化**: LLMへの指示を具体的にし、必要な情報を的確に含めるようにしました。
- **形態素解析の精度向上**: トークナイザーの設定やアルゴリズムの調整により、検索精度をさらに向上させました。
***
## まとめ

3つのソースコードを通じて、RAGシステムの基本から実践的な実装方法を学ぶことができると思います。各段階での工夫や改善点は、生成AIプログラミングを学ぶ上で有用な知見となりそうです。特に、日本語のテキスト処理や固有情報の検索、LLMの効果的な活用方法について理解を深められると幸いです。

ぜひ、各ソースコードを参照しながら、また実際に実行させてみて、LLMシステムの理解や自身の開発にも役立ててみてください。
