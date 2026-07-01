# Failure Cases

## factual_w8_bm25_hybrid
- Query: 解释 BM25、embedding retrieval 和 hybrid search 在 RAG 中分别解决什么问题。
- Family: factual_retrieval
- Recall@k: 0.6, MRR: 1.0, page recall: 0.833
- Likely cause: retrieval coverage, OCR/caption mismatch, or insufficient visual grounding.
- Fix: add expected slide metadata, improve caption generation, or route this family to visual retrieval.

## cross_modal_token_reduction_visual
- Query: 找到 Week 7 中讲 Vision Transformer 或 token reduction 的相关 slide 图片，并概括图中的流程。
- Family: cross_modal_retrieval
- Recall@k: 0.4, MRR: 1.0, page recall: 0.75
- Likely cause: retrieval coverage, OCR/caption mismatch, or insufficient visual grounding.
- Fix: add expected slide metadata, improve caption generation, or route this family to visual retrieval.

## analysis_multimodal_rag_vs_text_rag
- Query: Multimodal RAG 相比 text-only RAG 为什么更难设计和评估？
- Family: analytical_multihop
- Recall@k: 0.5, MRR: 1.0, page recall: 0.333
- Likely cause: retrieval coverage, OCR/caption mismatch, or insufficient visual grounding.
- Fix: add expected slide metadata, improve caption generation, or route this family to visual retrieval.

## personalised_revision_plan
- Query: 根据我的 weak topics 和已经上传的课件，制定一个 3 天复习计划，重点覆盖 RAG 和 token reduction。
- Family: personalised_followup
- Recall@k: 0.4, MRR: 0.143, page recall: None
- Likely cause: retrieval coverage, OCR/caption mismatch, or insufficient visual grounding.
- Fix: add expected slide metadata, improve caption generation, or route this family to visual retrieval.
