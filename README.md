# Java-Is-All-You-Need
For final java project

1. [Overview](#overview)  
2. [Prerequisites](#prerequisites)  
3. [Directory Structure](#directory-structure)  
4. [Java Source Files](#java-source-files)  
5. [Running the Project](#running-the-project)  
   - [Build](#build)  
   - [Training Mode](#training-mode)  
   - [Generation Mode](#generation-mode)  
   - [Demo Main](#demo-main)  
6. [Testing](#testing)  
7. [Weights & .gitignore](#weights--gitignore)  
8. [Contributing](#contributing)  
9. [License](#license)  

---

## Overview

This project implements:

- A **Tokenizer** that builds a vocabulary from plain-text/JSON corpora and encodes/decodes.
- An **Embedding** layer storing token-embedding weights.
- A **Multi-Head Attention** module.
- A **Feed-Forward** sublayer.
- A **LayerNorm** module.
- A stacked **TransformerEncoderLayer**.
- A simple **LMHead** (linear projection to vocabulary logits).
- A **LanguageModel** wrapper to generate text autoregressively.
- A **CorpusTrainer** that reads a corpus, runs teacher-forcing training with a command-line progress bar and logging.
- A **GenAIApp** CLI with `train` and `gen` modes.
- A small **Main.java** demo illustrating one encoder layer.

---

## Prerequisites

- **Java 24** (JDK 24)  
- **Maven** 3.x  
- A UNIX-style shell for the provided bash menu script  

---

## Directory Structure
JavaIsAllYouNeed
├── data
│   ├── MediumTest
│   │   └── 00c2bfc7-57db-496e-9d5c-d62f8d8119e3.json
│   └── TinyTest
│       └── 00c2bfc7-57db-496e-9d5c-d62f8d8119e3.json
├── dependency-reduced-pom.xml
├── main
├── pom.xml
├── src
│   ├── main
│   │   ├── java
│   │   │   └── org
│   │   │       └── anjali
│   │   │           ├── CorpusTrainer.java
│   │   │           ├── Embedding.java
│   │   │           ├── FeedForward.java
│   │   │           ├── GenAIApp.java
│   │   │           ├── LanguageModel.java
│   │   │           ├── LayerNorm.java
│   │   │           ├── Linear.java
│   │   │           ├── LMHead.java
│   │   │           ├── Main.java
│   │   │           ├── MultiHeadAttention.java
│   │   │           ├── Tokenizer.java
│   │   │           └── TransformerEncoderLayer.java
│   │   └── resources
│   └── test
│       └── java
│           └── org
│               └── anjali
│                   └── TransformerEncoderLayerTest.java
└── target
    ├── classes
    │   └── org
    │       └── anjali
    │           ├── CorpusTrainer$Result.class
    │           ├── CorpusTrainer.class
    │           ├── Embedding.class
    │           ├── FeedForward.class
    │           ├── GenAIApp.class
    │           ├── LanguageModel.class
    │           ├── LayerNorm.class
    │           ├── Linear.class
    │           ├── LMHead.class
    │           ├── Main.class
    │           ├── MultiHeadAttention.class
    │           ├── Tokenizer.class
    │           └── TransformerEncoderLayer.class
    ├── FinalProject2-1.0-SNAPSHOT.jar
    ├── generated-sources
    │   └── annotations
    ├── generated-test-sources
    │   └── test-annotations
    ├── maven-archiver
    │   └── pom.properties
    ├── maven-status
    │   └── maven-compiler-plugin
    │       ├── compile
    │       │   └── default-compile
    │       │       ├── createdFiles.lst
    │       │       └── inputFiles.lst
    │       └── testCompile
    │           └── default-testCompile
    │               ├── createdFiles.lst
    │               └── inputFiles.lst
    ├── original-FinalProject2-1.0-SNAPSHOT.jar
    ├── surefire-reports
    │   ├── org.anjali.TransformerEncoderLayerTest.txt
    │   └── TEST-org.anjali.TransformerEncoderLayerTest.xml
    └── test-classes
        └── org
            └── anjali
                └── TransformerEncoderLayerTest.class

33 directories, 41 files

## Java Source Files

### 1. Overall Architecture  
![image](https://github.com/user-attachments/assets/752db052-0c4c-4bb1-ab3c-8a5ff486ffa8)

- **TransformerEncoderLayer.java** ties together self-attention, residuals & layer norms, and feed-forward blocks exactly as in Sections 3.1–3.4 of Vaswani et al. :contentReference[oaicite:2]{index=2}.

---

### 2. Scaled Dot-Product & Multi-Head Attention  
- **Section 3.2:**  
  > “Scaled dot-product attention” computes  
  > \[ \text{Attention}(Q,K,V)=\text{softmax}(\frac{QK^T}{\sqrt{d_k}}) V \]  
  > Multi-head attention runs this in parallel heads. :contentReference[oaicite:3]{index=3}  
- **Mapping:**  
  - `MultiHeadAttention.java`  
    - Projects inputs into Q, K, V (via four `Linear` layers).  
    - Splits into `numHeads`, applies scaled dot-product, concatenates, and final projection.  

---

### 3. Position-Wise Feed-Forward Networks  
- **Section 3.3:**  
  > Two linear layers with ReLU in between (applied independently to each position). :contentReference[oaicite:4]{index=4}  
- **Mapping:**  
  - `FeedForward.java` implements exactly that:  
    1. `Linear(embedDim→hiddenDim)` + ReLU  
    2. `Linear(hiddenDim→embedDim)`

---

### 4. Residual Connections & Layer Normalization  
- **Section 3.1 & 3.4:**  
  > “Each sub-layer is surrounded by a residual connection followed by layer normalization.” :contentReference[oaicite:5]{index=5}  
- **Mapping:**  
  - `TransformerEncoderLayer.java` calls `add(...)` for residuals, then `LayerNorm.forward(...)`.  

---

### 5. Autoregressive Generation  
- **Greedy Decoding:**  
  > After training, we generate one token at a time by picking the highest-probability next token. This style of autoregressive generation builds on early seq2seq work :contentReference[oaicite:6]{index=6}.  
- **Mapping:**  
  - `LanguageModel.java`:  
    - `encodePrompt(...)` to get initial `[<BOS>, …]`  
    - Loop until `<EOS>` or maxLen: embed → encoder stack → `LMHead.forward(...)` → argmax → append.  
    - `decodeAll(...)` converts IDs back to tokens.

---

### 🔧 Java Source Files

Below is a quick reference for where each major Transformer component lives in the code:

| Paper Concept                                | Java Class                           |
|----------------------------------------------|--------------------------------------|
| Tokenization & Special Tokens                | `Tokenizer.java`                     |
| Embedding Look-Up                            | `Embedding.java`                     |
| Scaled Dot-Product + Multi-Head Attention    | `MultiHeadAttention.java`            |
| Position-Wise Feed-Forward                   | `FeedForward.java`                   |
| Layer Normalization                          | `LayerNorm.java`                     |
| Residual + Sublayer Composition              | `TransformerEncoderLayer.java`       |
| Final LM Head Projection                     | `LMHead.java`                        |
| Full Model + Greedy Generation               | `LanguageModel.java`                 |
| Training Loop / Teacher-Forcing + Progress   | `CorpusTrainer.java`                 |
| CLI Entry-Point (`train` / `gen` modes)      | `GenAIApp.java`                      |
| Minimal demo of a single encoder layer       | `Main.java`                          |


---


