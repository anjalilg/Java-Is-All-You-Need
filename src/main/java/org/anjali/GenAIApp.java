package org.anjali;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.nio.file.Path;
import java.util.Scanner;

/**
 * GenAIApp
 *
 * Command-line entry point for:
 * - training a model on a chosen corpus directory, or
 * - generating text with the latest saved weights.
 *
 * Usage:
 *   java -jar GenAI.jar train [<corpusDir>]
 *   java -jar GenAI.jar gen
 */
public class GenAIApp {
    private static final Logger logger = LoggerFactory.getLogger(GenAIApp.class);

    public static void main(String[] args) throws Exception {
        if (args.length < 1 || args.length > 2) {
            usageAndExit();
        }
        String mode = args[0].toLowerCase();
        logger.info("Starting GenAIApp in '{}' mode", mode);

        // 1) Build tokenizer & vocab
        Path corpus = args.length == 2
        ? Path.of(args[1])
        : Path.of("data");
        Tokenizer tok = new Tokenizer();
        tok.buildVocab(corpus, 1);              // builds vocab here

        // 2) Construct model: embedDim=128, heads=8, hidden=512, layers=6, maxLen=100
        LanguageModel lm = new LanguageModel(tok, 128, 8, 512, 6, 100);

        switch (mode) {
            case "train":
                logger.info("Training on directory: {}", corpus);
                CorpusTrainer.train(lm, corpus, 1e-3, 3);
                lm.getEmbeddingLayer().save("embed.weights");
                lm.getOutputHead().save("head.weights");
                logger.info("Training complete; weights saved.");
                break;

            case "gen":
                logger.info("Loading saved weights");
                lm.getEmbeddingLayer().setWeights(Embedding.load("embed.weights"));
                LMHead.loadInto(lm.getOutputHead(), "head.weights");

                try (Scanner sc = new Scanner(System.in)) {
                    System.out.println("Enter prompts (type exit to quit):");
                    while (true) {
                        System.out.print("> ");
                        String prompt = sc.nextLine().trim();
                        if (prompt.equalsIgnoreCase("exit")) break;
                        // Generate and print one completion
                        System.out.println(lm.generate(prompt));
                    }
                }
                break;

            default:
                usageAndExit();
        }
    }

    private static void usageAndExit() {
        System.err.println("Usage:");
        System.err.println("  java -jar GenAI.jar train [<corpusDir>]");
        System.err.println("  java -jar GenAI.jar gen");
        System.exit(1);
    }
}
