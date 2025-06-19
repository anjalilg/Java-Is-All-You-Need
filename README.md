# Java-Is-All-You-Need
### By Anjali Godara
### 2025-06-18
### For summative final program assignment

1. [Overview](#overview)
2. [Prerequisites](#prerequisites)
3. [Running the Project](#running-the-project)  
4. [Java Source Files](#java-source-files)  
5. [Directory Structure](#directory-structure)  

---
## Overview
JavaIsAllYouNeed is the java implimentation of a transformer based generative language model. It references the paper "Attention Is All You Need". It includes every piece of the pipeline from tokenization and embedding, thought multi-headed self attention and feed forward layers, to an autoregressive LM head written entierly in Java with no external deeplearning frameworks. The models are trained on JSON corpas, and you can use the provided training datasets `TinyTest`, `MediumTest`, `FullTest`, and you can source your very own corpas to train the model on. Training is done on my very simple to use CLI interface that includes both logging and a progress bar. For testing the model you can run a greedy generation on the same CLI menu.

**Importaint Note:** I cleared any training weights from the root project file so the model is completley a blank slate and you MUST train it first before you test it.

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

1. To run the project just download the file JavaIsAllYouNeed.zip
2. Use a bash terminal (I made this project on archlinux so I just used the linux terminal), and enter into the root directory of the project file
3. Run the command `./main`

Running ./main automatically builds the project then gives you a visually usable CLI to train and test the model.


### Training
So there are multiple degrees of training depending on which file you select to train from the data file. The TinyTest train would estimatingly run for 5 minutes, MediumTest could run for estimatingly an hour.

There is a third degree of training called FullTest. To use it: 
1. You must download the file from the following link (https://www.kaggle.com/datasets/ltcmdrdata/plain-text-wikipedia-202011), just press the big 'Download' button then 'Download dataset as zip (8 GB)' 
2. Place it into the /data directory in the root project file.

The FullTest data was sourced from this multiple wikipedia files and converted into JSON files. The FullTest will give the most accurate generative responses during testing. FullTest can estimatingly run for multiple training hours. The large file size + long training hours is the reason why I decided to include it seperatley and optionally from the root project file.

But really, you can use any JSON formatted corpus to train the model on.

**Note:** The training weights are automatically saved and updated in the root project file every time you run an epoch. 


### Testing
To test, just run `./main` again in your terminal and select option 2 to test.


---
## Java Source Files

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



