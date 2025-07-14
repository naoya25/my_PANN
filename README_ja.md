# PANN: Power Electronics Modeling のための Physics-in-Architecture Neural Network

[![LinkedIn](https://img.shields.io/badge/LinkedIn-Connect--Xinze%20Li-blue)](https://www.linkedin.com/in/xinze-li-8199561b0/)
[![ORCID](https://img.shields.io/badge/ORCID-Xinze%20Li-brightgreen)](https://orcid.org/0000-0003-3513-209X)
[![GitHub](https://img.shields.io/badge/Github-XinzeLee-black?logo=github)](https://github.com/XinzeLee)
[![ResearchGate](https://img.shields.io/badge/ResearchGate-Xinze%20Li-cyan)](https://www.researchgate.net/scientific-contributions/Xinze-Li-2167307782)
[![Colab](https://img.shields.io/badge/Colab-PANN--Notebooks-red?logo=google-colab)](https://drive.google.com/drive/folders/1FXr82WQfBOj6xP01h-9RHIZBpiOFZwUC)
<br><br>

- 参考文献 1:<br>
  X. Li ら, “**Temporal Modeling for Power Converters With Physics-in-Architecture Recurrent Neural Network**,” _IEEE Transactions on Industrial Electronics_, vol. 71, no. 11, pp. 14111-14123, 2024 年 11 月.<br>
  [![DOI](https://img.shields.io/badge/DOI-10.1109/TIE.2024.3352119-cyan)](https://doi.org/10.1109/TIE.2024.3352119)
  [![IEEE](https://img.shields.io/badge/IEEE-Xplore-orange)](https://ieeexplore.ieee.org/document/10463542)
  [![ResearchGate](https://img.shields.io/badge/ResearchGate--1-blue)](https://www.researchgate.net/publication/378918445_Temporal_Modeling_for_Power_Converters_With_Physics-in-Architecture_Recurrent_Neural_Network)

- 参考文献 2:<br>
  X. Li, F. Lin, X. Zhang, H. Ma, F. Blaabjerg, “**Data-Light Physics-Informed Modeling for the Modulation Optimization of a Dual-Active-Bridge Converter**,” _IEEE Transactions on Power Electronics_, vol. 39, no. 7, pp. 8770-8785, 2024 年 7 月.<br>
  [![DOI](https://img.shields.io/badge/DOI-10.1109/TPEL.2024.3378184-cyan)](https://doi.org/10.1109/TPEL.2024.3378184)
  [![IEEE](https://img.shields.io/badge/IEEE-Xplore-orange)](https://ieeexplore.ieee.org/document/10473116)
  [![ResearchGate](https://img.shields.io/badge/ResearchGate--2-blue)](https://www.researchgate.net/publication/379104054_Data-Light_Physics-Informed_Modeling_for_the_Modulation_Optimization_of_a_Dual-Active-Bridge_Converter)

- 参考文献 3:<br>
  F. Lin, X. Li, X. Zhang, H. Ma, “**STAR: One-Stop Optimization for Dual-Active-Bridge Converter With Robustness to Operational Diversity**,” _IEEE Journal of Emerging and Selected Topics in Power Electronics_, vol. 12, no. 3, pp. 2758-2773, 2024 年 6 月.<br>
  [![DOI](https://img.shields.io/badge/DOI-10.1109/JESTPE.2024.3392684-cyan)](https://doi.org/10.1109/JESTPE.2024.3392684)
  [![IEEE](https://img.shields.io/badge/IEEE-Xplore-orange)](https://ieeexplore.ieee.org/document/10506915)
  [![ResearchGate](https://img.shields.io/badge/ResearchGate--3-blue)](https://www.researchgate.net/publication/380052824_STAR_One-Stop_Optimization_for_Dual_Active_Bridge_Converter_with_Robustness_to_Operational_Diversity)

- 参考文献 4:<br>
  X. Li ら, “**A Generic Modeling Approach for Dual-Active-Bridge Converter Family via Topology Transferrable Networks**,” _IEEE Transactions on Industrial Electronics_.<br>
  [![DOI](https://img.shields.io/badge/DOI-10.1109/TIE.2024.3406858-cyan)](https://doi.org/10.1109/TIE.2024.3406858)
  [![IEEE](https://img.shields.io/badge/IEEE-Xplore-orange)](https://ieeexplore.ieee.org/document/10627933)
  [![ResearchGate](https://img.shields.io/badge/ResearchGate--4-blue)](https://www.researchgate.net/publication/382930411_A_Generic_Modeling_Approach_for_Dual-Active-Bridge_Converter_Family_via_Topology_Transferrable_Networks)

<br><br>

- Colab ノートブック:<br>
  [![Colab](https://img.shields.io/badge/Colab-PANN--Buck-654062?logo=google-colab)](https://colab.research.google.com/drive/1FDxjR-LZxJBbp4PzsinhxdWMrUI7UjW-)
  [![Colab](https://img.shields.io/badge/Colab-PANN--DAB-B4B4B3?logo=google-colab)](https://colab.research.google.com/drive/1dJ4GvKc03_eF__c8l1msbI7Fq-8a6ScD#scrollTo=2ede7f4b)
  [![Colab](https://img.shields.io/badge/Colab-PANN--Operational--Diversity-26577C?logo=google-colab)](https://colab.research.google.com/drive/1PSpqhUEfGKXEfoSVesYUmhZCy4EpYTX9)
  [![Colab](https://img.shields.io/badge/Colab-PANN--Topology--Transfer-E55604?logo=google-colab)](https://colab.research.google.com/drive/1jXo4uugvnRBgP2948HVPLNRsK8fCh-ge)

<br><br>

## 説明

### I. PANN とその構造

PANN（Physics-in-Architecture Neural Network）は、電力エレクトロニクスシステムのモデリングに特化した物理情報組込みニューラルネットワークです。離散化した状態空間回路方程式を組み込んだ再帰型ニューラル構造を用いることで、物理的バイアスを導入し、データに不変な関係性を直接ネットワーク内部に埋め込みます。PANN のニューラルアーキテクチャを Fig. 1 に示します。
![Structure of PANN.](https://github.com/user-attachments/assets/af90a7b0-3e3e-4fad-bf8e-75bf7ce4efe3)
<br>Fig. 1. PANN の構造。<br>

### II. PANN の推論

PANN の推論は、あらかじめ計算された入力変数と前ステップで推定された状態変数を用い、次の状態変数を再帰的に予測することで実行されます。時間展開された PANN 推論の様子を Fig. 2 に示します。
![PANN Inference](https://github.com/user-attachments/assets/2c056085-9d77-4270-8c6e-fd3ed11ae78f)
<br>Fig. 2. PANN の推論構造。<br>

### III. 電力エレクトロニクスにおける PANN の説明可能性

PANN モデルは回路の物理法則、スイッチング動作、コミュテーションループなどを明示的に捉えることができ、電力エレクトロニクスにおいて高い説明可能性を示します。非共振型 DAB コンバータを例とした洞察を Fig. 3 に示します。
![PANN's Explainability](https://github.com/user-attachments/assets/57593884-9546-4964-9c5a-b8926376df86)
<br>Fig. 3. PANN の物理的説明可能性。<br>

### IV. PANN の学習

PANN の学習ワークフローを Fig. 4、1 エポックの流れを Fig. 5 に示します。
![PANN's training workflow](https://github.com/user-attachments/assets/84258774-3626-46d8-8bfc-27befe24256a)
<br>Fig. 4. PANN の学習ワークフロー。<br><br>
![One training epoch for PANN](https://github.com/user-attachments/assets/c70eb196-d688-4468-be95-5d23e8639ae1)
<br>Fig. 5. PANN における 1 エポックの流れ。<br>

### V. PANN はデータライトかつ軽量

PANN は回路の物理法則をネットワーク構造に直接埋め込むため、厳密な物理的一貫性を保ちつつ、必要なデータ量を 3 桁以上削減できます。理論的には、定義されたコンバータパラメータ数以上の時系列データポイントがあれば十分です。また、単層 RNN に近いパラメータ数しか持たないため、モデル自体も軽量です。これらの利点を Fig. 6 に示します。
![PANN is light AI model](https://github.com/user-attachments/assets/1b5a4366-8fd8-401a-b2cd-b9a7708b5e6f)
<br>Fig. 6. PANN のデータライト・軽量性の利点。<br>

### VI. PANN の柔軟性

PANN は動作条件、変調戦略、性能指標、回路パラメータやトポロジ変種という 4 つの主要観点で高い柔軟性を持ちます。概要を Fig. 7 に示します。
![PANN is flexible](https://github.com/user-attachments/assets/6aededa3-b539-4d1c-90b7-d217ffbf213f)
<br>Fig. 7. PANN の柔軟性。<br>

### VII. あなたの応用・コンバータに合わせた PANN のカスタマイズ

Fig. 8 の手順に従うことで、特定の応用やコンバータに合わせて PANN をカスタマイズできます。例として、非共振型 DAB コンバータのモデリングを示します。
![PANN for DAB](https://github.com/user-attachments/assets/da096d81-b48f-41e4-84df-7c2d604821c6)
<br>Fig. 8. ケーススタディ：DAB コンバータ向け PANN 設計。<br>

<br><br>

## PANN チュートリアル

“**電力エレクトロニクスの次世代 AI: Explainable, Light, and Flexible**” を主題とした包括的なチュートリアルを [PANN_Tutorial.pdf](./tutorials/PANN_Tutorial.pdf) に公開しています（作成者: Xinze Li, Fanfan Lin）。<br><br>
内容は、電力エレクトロニクスにおける AI 応用の基礎、物理情報機械学習手法の概要、PANN の推論と説明可能性、学習と軽量性、そして多様な条件・トポロジにまたがる柔軟性（ドメイン外転移能力）を含みます。<br><br>

## PANN のデプロイ

PC 上で PE-GPT を使うには、まず OpenAI モデルへの API コールを設定してください。詳細は `core/llm/llm.py` を参照してください。
Plecs ソフトウェアと連携して DAB 用の変調をシミュレーションする場合は、Plecs の設定で XML-RPC インターフェースを有効にしてください。

```bash
# GitHub リポジトリをクローン
git clone https://github.com/XinzeLee/PANN

# 作業ディレクトリを移動
cd PANN

# 必要な依存ライブラリをインストール
pip install -r requirements.txt

# カスタマイズされた PANN モデルを import できます。
# 実装前にノートブックを一通り試すことを推奨します。
```

<br><br>

## Google Colab でノートブックを実行

ローカルでの実行を強く推奨します（グラフ操作が容易なため）が、Google Colab 版も用意しました。

- [Google Colab (PyTorch) PANN-Buck](https://colab.research.google.com/drive/1FDxjR-LZxJBbp4PzsinhxdWMrUI7UjW-)
- [Google Colab (PyTorch) PANN-DAB](https://colab.research.google.com/drive/1dJ4GvKc03_eF__c8l1msbI7Fq-8a6ScD#scrollTo=2ede7f4b)
- [Google Colab (PyTorch) PANN-Operational-Diversity](https://colab.research.google.com/drive/1PSpqhUEfGKXEfoSVesYUmhZCy4EpYTX9)
- [Google Colab (PyTorch) PANN-Topology-Transfer](https://colab.research.google.com/drive/1jXo4uugvnRBgP2948HVPLNRsK8fCh-ge)

<br><br>

## コード作者

@code-author:

- Xinze Li（メール: [xinzeli831@gmail.com](mailto:xinzeli831@gmail.com)）
- Fanfan Lin（メール: [fanfanlin31@gmail.com](mailto:fanfanlin31@gmail.com)）

<br><br>

## 注意

本リポジトリは、論文で提案した PE-GPT 手法の簡易版を提供します。簡略化されているものの、提案手法のコアアーキテクチャは保持しています。
現在含まれる機能／ブロック：Retrieval Augmented Generation、LLM エージェント、Model Zoo（DAB コンバータ用 PANN を ONNX エンジンで実装）、メタヒューリスティック最適化、シミュレーション検証、GUI、ナレッジベース（簡易版）。

<br><br>

## ライセンス

本コードは [Apache License Version 2.0](./LICENSE) の下で公開されています。
