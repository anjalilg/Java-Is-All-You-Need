# Java-Is-All-You-Need
By Anjali Godara
2025-06-18
For final java project

1. [Overview](#overview)
2. [Prerequisites](#prerequisites)
3. [Running the Project](#running-the-project)  
4. [Java Source Files](#java-source-files)  
5. [Directory Structure](#directory-structure)  

---
## Overview
JavaIsAllYouNeed is the java implimentation of a transformer based generative language model. It references the paper "Attention Is All You Need". It includes every piece of the pipeline from tokenization and embedding, thought multi-headed self attention and feed forward layers, to an autoregressive LM head written entierly in Java with no external deeplearning frameworks. The models are trained on JSON corpas, and you can use the provided training datasets `TinyTest`, `MediumTest`, `FullTest`, and you can source your very own corpas to train the model on. Training is done on my very simple to use CLI interface that includes both logging and a progress bar. For testing the model you can run a greedy generation on the same CLI menu.

```

       ░▒▓█▓▒░░▒▓██████▓▒░░▒▓█▓▒░░▒▓█▓▒░░▒▓██████▓▒░       ░▒▓█▓▒░░▒▓███████▓▒░ 
       ░▒▓█▓▒░▒▓█▓▒░░▒▓█▓▒░▒▓█▓▒░░▒▓█▓▒░▒▓█▓▒░░▒▓█▓▒░      ░▒▓█▓▒░▒▓█▓▒░        
       ░▒▓█▓▒░▒▓█▓▒░░▒▓█▓▒░░▒▓█▓▒▒▓█▓▒░░▒▓█▓▒░░▒▓█▓▒░      ░▒▓█▓▒░▒▓█▓▒░        
       ░▒▓█▓▒░▒▓████████▓▒░░▒▓█▓▒▒▓█▓▒░░▒▓████████▓▒░      ░▒▓█▓▒░░▒▓██████▓▒░  
░▒▓█▓▒░░▒▓█▓▒░▒▓█▓▒░░▒▓█▓▒░ ░▒▓█▓▓█▓▒░ ░▒▓█▓▒░░▒▓█▓▒░      ░▒▓█▓▒░      ░▒▓█▓▒░ 
░▒▓█▓▒░░▒▓█▓▒░▒▓█▓▒░░▒▓█▓▒░ ░▒▓█▓▓█▓▒░ ░▒▓█▓▒░░▒▓█▓▒░      ░▒▓█▓▒░      ░▒▓█▓▒░ 
 ░▒▓██████▓▒░░▒▓█▓▒░░▒▓█▓▒░  ░▒▓██▓▒░  ░▒▓█▓▒░░▒▓█▓▒░      ░▒▓█▓▒░▒▓███████▓▒░
                                                                                                                                                                                                                                       

 ░▒▓██████▓▒░░▒▓█▓▒░      ░▒▓█▓▒░             ░▒▓█▓▒░░▒▓█▓▒░░▒▓██████▓▒░░▒▓█▓▒░░▒▓█▓▒░ 
░▒▓█▓▒░░▒▓█▓▒░▒▓█▓▒░      ░▒▓█▓▒░             ░▒▓█▓▒░░▒▓█▓▒░▒▓█▓▒░░▒▓█▓▒░▒▓█▓▒░░▒▓█▓▒░ 
░▒▓█▓▒░░▒▓█▓▒░▒▓█▓▒░      ░▒▓█▓▒░             ░▒▓█▓▒░░▒▓█▓▒░▒▓█▓▒░░▒▓█▓▒░▒▓█▓▒░░▒▓█▓▒░ 
░▒▓████████▓▒░▒▓█▓▒░      ░▒▓█▓▒░              ░▒▓██████▓▒░░▒▓█▓▒░░▒▓█▓▒░▒▓█▓▒░░▒▓█▓▒░ 
░▒▓█▓▒░░▒▓█▓▒░▒▓█▓▒░      ░▒▓█▓▒░                ░▒▓█▓▒░   ░▒▓█▓▒░░▒▓█▓▒░▒▓█▓▒░░▒▓█▓▒░ 
░▒▓█▓▒░░▒▓█▓▒░▒▓█▓▒░      ░▒▓█▓▒░                ░▒▓█▓▒░   ░▒▓█▓▒░░▒▓█▓▒░▒▓█▓▒░░▒▓█▓▒░ 
░▒▓█▓▒░░▒▓█▓▒░▒▓████████▓▒░▒▓████████▓▒░         ░▒▓█▓▒░    ░▒▓██████▓▒░ ░▒▓██████▓▒░ 


░▒▓███████▓▒░░▒▓████████▓▒░▒▓████████▓▒░▒▓███████▓▒░░▒▓█▓▒░ 
░▒▓█▓▒░░▒▓█▓▒░▒▓█▓▒░      ░▒▓█▓▒░      ░▒▓█▓▒░░▒▓█▓▒░▒▓█▓▒░ 
░▒▓█▓▒░░▒▓█▓▒░▒▓█▓▒░      ░▒▓█▓▒░      ░▒▓█▓▒░░▒▓█▓▒░▒▓█▓▒░ 
░▒▓█▓▒░░▒▓█▓▒░▒▓██████▓▒░ ░▒▓██████▓▒░ ░▒▓█▓▒░░▒▓█▓▒░▒▓█▓▒░ 
░▒▓█▓▒░░▒▓█▓▒░▒▓█▓▒░      ░▒▓█▓▒░      ░▒▓█▓▒░░▒▓█▓▒░▒▓█▓▒░ 
░▒▓█▓▒░░▒▓█▓▒░▒▓█▓▒░      ░▒▓█▓▒░      ░▒▓█▓▒░░▒▓█▓▒░       
░▒▓█▓▒░░▒▓█▓▒░▒▓████████▓▒░▒▓████████▓▒░▒▓███████▓▒░░▒▓█▓▒░ 


  [1] Train a model
  [2] Generate text
  [3] Quit

Select an option [1-3]:
```
---
## Prerequisites

- **Java 24** (JDK 24)  
- **Maven** 3.x  
- A UNIX-style shell for the provided bash menu script  

---

## Running The Project

To run the project just download the file JavaIsAllYouNeed zip file
Use a bash terminal (I made this project on archlinux so I just used the linux terminal), and enter into the root directory of the project file
Run the command ./main


### Training
So there are multiple degrees of training depending on which file you select to train from the data file. The TinyTest train would estimatingly run for 5 minutes, MediumTest could run for estimatingly an hour.

However, to run the FullTest, you must download the FullTest file from this repo, and place it into the /data directory in the root project file. The FullTest data was sourced from multiple wikipedia pages, and will give the most accurate testing responses. I must warn though that the FullTest data is well over 20 gigabytes, and can estimatingly run for multiple training hours. The large file size is the reason why I decided to include it seperatley and optionally from the root project file.

The training weights are automatically saved and updated in the root project file every time you run an epoch.


### Testing
To test, just run ./main again in your terminal and select option 2 to train. 


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

## Directory Structure
```
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



