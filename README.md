# Banking-LLM-Security  
Prompt Injection Detection Framework for Banking Domain

##  Overview
Banking-LLM-Security is an end-to-end security framework designed to detect, classify, and mitigate **prompt injection attacks** in banking chatbots and financial LLM applications.  
The framework integrates **BERT, RoBERTa, DistilBERT**, and an ensemble meta-classifier to deliver high-accuracy detection of adversarial prompts.

##  Features
- Independent fine-tuning of **BERT, RoBERTa, DistilBERT** on 100k+ banking prompts  
- Weighted cross-entropy loss to handle class imbalance  
- **AdamW optimizer** with LR = 5 × 10⁻⁵  
- Comprehensive evaluation: F1, Precision, Recall, ROC-AUC  
- Ensemble model combining all three fine-tuned embeddings  
- Clean training scripts + reproducible experimental pipeline  
- Ready for integration into banking chatbots & API gateways

##  Project Structure
