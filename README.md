# Java-Is-All-You-Need
For final java project

1. [Prerequisites](#prerequisites)
2. [Running the Project](#running-the-project)  
   - [Build](#build)  
   - [Training Mode](#training-mode)  
   - [Generation Mode](#generation-mode)  
   - [Demo Main](#demo-main)

3.  [Java Source Files](#java-source-files)  
4.  [Directory Structure](#directory-structure)  

---


## Prerequisites

- **Java 24** (JDK 24)  
- **Maven** 3.x  
- A UNIX-style shell for the provided bash menu script  

---

### Running The Project

---
## Java Source Files

### Overall Architecture  
![image](https://github.com/user-attachments/assets/752db052-0c4c-4bb1-ab3c-8a5ff486ffa8)

I used the transformer archetecture from the paper Attention Is All You Need to build this project: https://arxiv.org/abs/1706.03762

Below is a quick reference for where each major Transformer component lives in the code:

| Concept                                          | Java Class                           |
|--------------------------------------------------|--------------------------------------|
| Tokenization & Vocabulary Building               | `Tokenizer.java`                     |
| Embedding Look-Up and Backpro                    | `Embedding.java`                     |
| Scaled Dot-Product + Multi-Head Attention        | `MultiHeadAttention.java`            |
| Position-Wise Feed-Forward Networks              | `FeedForward.java`                   |
| Layer Normalization                              | `LayerNorm.java`                     |
| Transformer Encoder Layer (Residual + Sublayers) | `TransformerEncoderLayer.java`       |
| Final LM Head Projection                         | `LMHead.java`                        |
| Full Model + Greedy Generation                   | `LanguageModel.java`                 |
| Training Loop + Teacher-Forcing + Progress Bar   | `CorpusTrainer.java`                 |
| CLI Entry-Point (`train` / `gen` modes)          | `GenAIApp.java`                      |
| Demo of a single encoder layer                   | `Main.java`                          |
| Full Model Wrapping & Greedy Decoding            | `LanguageModel.java`                 |
| Unit Tests for Encoder Block                     | `TransformerEncoderLayerTest.java`   |


---
```
## Directory Structure
JavaIsAllYouNeed
├── data
│   ├── MediumTest
│   │   └── 00c2bfc7-57db-496e-9d5c-d62f8d8119e3.json
│   └── TinyTest
│       └── 00c2bfc7-57db-496e-9d5c-d62f8d8119e3.json
├── dependency-reduced-pom.xml
├── main
├── pom.xml
├── src
│   ├── main
│   │   ├── java
│   │   │   └── org
│   │   │       └── anjali
│   │   │           ├── CorpusTrainer.java
│   │   │           ├── Embedding.java
│   │   │           ├── FeedForward.java
│   │   │           ├── GenAIApp.java
│   │   │           ├── LanguageModel.java
│   │   │           ├── LayerNorm.java
│   │   │           ├── Linear.java
│   │   │           ├── LMHead.java
│   │   │           ├── Main.java
│   │   │           ├── MultiHeadAttention.java
│   │   │           ├── Tokenizer.java
│   │   │           └── TransformerEncoderLayer.java
│   │   └── resources
│   └── test
│       └── java
│           └── org
│               └── anjali
│                   └── TransformerEncoderLayerTest.java
└── target
    ├── classes
    │   └── org
    │       └── anjali
    │           ├── CorpusTrainer$Result.class
    │           ├── CorpusTrainer.class
    │           ├── Embedding.class
    │           ├── FeedForward.class
    │           ├── GenAIApp.class
    │           ├── LanguageModel.class
    │           ├── LayerNorm.class
    │           ├── Linear.class
    │           ├── LMHead.class
    │           ├── Main.class
    │           ├── MultiHeadAttention.class
    │           ├── Tokenizer.class
    │           └── TransformerEncoderLayer.class
    ├── FinalProject2-1.0-SNAPSHOT.jar
    ├── generated-sources
    │   └── annotations
    ├── generated-test-sources
    │   └── test-annotations
    ├── maven-archiver
    │   └── pom.properties
    ├── maven-status
    │   └── maven-compiler-plugin
    │       ├── compile
    │       │   └── default-compile
    │       │       ├── createdFiles.lst
    │       │       └── inputFiles.lst
    │       └── testCompile
    │           └── default-testCompile
    │               ├── createdFiles.lst
    │               └── inputFiles.lst
    ├── original-FinalProject2-1.0-SNAPSHOT.jar
    ├── surefire-reports
    │   ├── org.anjali.TransformerEncoderLayerTest.txt
    │   └── TEST-org.anjali.TransformerEncoderLayerTest.xml
    └── test-classes
        └── org
            └── anjali
                └── TransformerEncoderLayerTest.class
```
33 directories, 41 files



