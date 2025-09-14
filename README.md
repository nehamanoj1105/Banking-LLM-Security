## Problem Statement

### Multi-Layered Defense Against Prompt Injection and Jailbreaking Attacks in Large Language Models for Secure Banking Applications

The rapid adoption of Large Language Models (LLMs) in banking—through customer service bots, financial advisors, and workflow automation—has increased exposure to adversarial threats such as **prompt injection** and **jailbreaking**.  
These attacks can manipulate model behavior to bypass safeguards, exfiltrate sensitive data, or execute unauthorised actions, leading to **privacy breaches, regulatory violations, and financial fraud**.

Traditional rule-based defenses are inadequate against evolving adversarial tactics.  
To address this, we propose a **multi-layered defense pipeline**:

- **First line of defense:**  
  AI agents that act as monitoring and filtering layers to intercept malicious prompts before they reach the LLM.

- **Second line of defense:**  
  An ensemble of transformer models (**BERT, DeBERTa, DistilBERT**) that provides robust adversarial prompt detection through sequence classification.

By combining **agent-based defenses** with **ensemble deep learning models**, the framework ensures **adaptive, real-time protection** and supports the **secure and compliant deployment of LLMs in mission-critical banking environments**.

