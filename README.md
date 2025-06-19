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

### Overall Architecture  
![image](https://github.com/user-attachments/assets/752db052-0c4c-4bb1-ab3c-8a5ff486ffa8)

I used the transformer archetecture from Attention Is All You Need to build this project: https://arxiv.org/abs/1706.03762

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


