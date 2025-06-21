// Fixed version of NGramsAnalyzer to prevent OutOfMemoryError
package org.example;

import java.io.*;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.*;
import java.util.concurrent.*;
import java.util.stream.Collectors;
import org.apache.spark.api.java.*;
import org.apache.spark.SparkConf;
import scala.Tuple2;

public class NGramsAnalyzer {

    private static final int AVAILABLE_PROCESSORS = Runtime.getRuntime().availableProcessors();
    private static final long AVAILABLE_MEMORY = Runtime.getRuntime().maxMemory();

    public enum ExecutionMode {
        SEQUENTIAL, PARALLEL, DISTRIBUTED
    }

    private static class NGramResult {
        private final Map<String, Integer> ngramFrequencies;
        private final Map<String, Double> conditionalProbabilities;
        private final long executionTime;
        private final int cycles;

        public NGramResult(Map<String, Integer> frequencies, Map<String, Double> probabilities,
                           long time, int cycles) {
            this.ngramFrequencies = frequencies;
            this.conditionalProbabilities = probabilities;
            this.executionTime = time;
            this.cycles = cycles;
        }

        public Map<String, Integer> getNgramFrequencies() { return ngramFrequencies; }
        public Map<String, Double> getConditionalProbabilities() { return conditionalProbabilities; }
        public long getExecutionTime() { return executionTime; }
        public int getCycles() { return cycles; }
    }

    public static void main(String[] args) {
        Scanner scanner = new Scanner(System.in);

        displayHardwareInfo();
        ExecutionMode mode = getExecutionMode(scanner);
        int n = getNGramSize(scanner);
        String filename = getInputFile(scanner);

        try {
            NGramResult result = executeAnalysis(mode, n, filename);
            displayResults(result, mode, n, filename);
        } catch (Exception e) {
            System.err.println("Error during analysis: " + e.getMessage());
            e.printStackTrace();
        }
    }

    private static void displayHardwareInfo() {
        System.out.println("=== Hardware Information ===");
        System.out.println("Available Processors: " + AVAILABLE_PROCESSORS);
        System.out.println("Available Memory: " + (AVAILABLE_MEMORY / (1024 * 1024)) + " MB");
        System.out.println("Java Version: " + System.getProperty("java.version"));
        System.out.println("OS: " + System.getProperty("os.name"));
        System.out.println("============================\n");
    }

    private static ExecutionMode getExecutionMode(Scanner scanner) {
        System.out.println("Select execution mode:\n1. Sequential\n2. Parallel\n3. Distributed");
        System.out.print("Enter choice (1-3): ");
        switch (scanner.nextInt()) {
            case 1: return ExecutionMode.SEQUENTIAL;
            case 2: return ExecutionMode.PARALLEL;
            case 3: return ExecutionMode.DISTRIBUTED;
            default: return ExecutionMode.SEQUENTIAL;
        }
    }

    private static int getNGramSize(Scanner scanner) {
        System.out.print("Enter n-gram size (n): ");
        int n = scanner.nextInt();
        return Math.max(n, 1);
    }

    private static String getInputFile(Scanner scanner) {
        System.out.print("Enter input file name: ");
        scanner.nextLine();
        return scanner.nextLine().trim();
    }

    private static NGramResult executeAnalysis(ExecutionMode mode, int n, String filename) throws Exception {
        switch (mode) {
            case SEQUENTIAL:
                return sequentialAnalysis(filename, n);
            case PARALLEL:
                return parallelAnalysis(filename, n);
            case DISTRIBUTED:
                return distributedAnalysis(filename, n);
            default:
                throw new IllegalArgumentException("Invalid mode");
        }
    }

    private static NGramResult sequentialAnalysis(String filename, int n) throws IOException {
        long start = System.currentTimeMillis();
        Map<String, Integer> freq = new HashMap<>();
        LinkedList<String> window = new LinkedList<>();

        try (BufferedReader reader = new BufferedReader(new FileReader(filename))) {
            String line;
            while ((line = reader.readLine()) != null) {
                for (String word : preprocessText(line).split(" ")) {
                    if (!word.isEmpty()) {
                        window.add(word);
                        if (window.size() == n) {
                            freq.merge(String.join(" ", window), 1, Integer::sum);
                            window.pollFirst();
                        }
                    }
                }
            }
        }

        return new NGramResult(freq, calculateConditionalProbabilities(freq, n),
                System.currentTimeMillis() - start, 1);
    }

    private static NGramResult parallelAnalysis(String filename, int n) throws IOException, InterruptedException {
        long start = System.currentTimeMillis();
        List<String> lines = Files.readAllLines(Paths.get(filename));
        ExecutorService exec = Executors.newFixedThreadPool(AVAILABLE_PROCESSORS);
        List<Future<Map<String, Integer>>> futures = new ArrayList<>();

        for (List<String> chunk : splitList(lines, AVAILABLE_PROCESSORS)) {
            futures.add(exec.submit(() -> {
                Map<String, Integer> local = new HashMap<>();
                LinkedList<String> window = new LinkedList<>();
                for (String line : chunk) {
                    for (String word : preprocessText(line).split(" ")) {
                        if (!word.isEmpty()) {
                            window.add(word);
                            if (window.size() == n) {
                                local.merge(String.join(" ", window), 1, Integer::sum);
                                window.pollFirst();
                            }
                        }
                    }
                }
                return local;
            }));
        }

        Map<String, Integer> finalMap = new HashMap<>();
        for (Future<Map<String, Integer>> f : futures) {
            try {
                Map<String, Integer> map = f.get();
                map.forEach((k, v) -> finalMap.merge(k, v, Integer::sum));
            } catch (ExecutionException e) {
                e.printStackTrace();
            }
        }

        exec.shutdown();
        return new NGramResult(finalMap, calculateConditionalProbabilities(finalMap, n),
                System.currentTimeMillis() - start, AVAILABLE_PROCESSORS);
    }

    private static NGramResult distributedAnalysis(String filename, int n) throws IOException {
        long start = System.currentTimeMillis();

        SparkConf conf = new SparkConf()
                .setAppName("DistributedNGramAnalyzer")
                .setMaster("local[*]")
                .set("spark.driver.memory", "4g");
        JavaSparkContext sc = new JavaSparkContext(conf);

        JavaRDD<String> lines = sc.textFile(filename);
        JavaRDD<String> words = lines.flatMap(line -> Arrays.asList(preprocessText(line).split(" ")).iterator());

        JavaRDD<String> ngrams = words.mapPartitions(iter -> {
            LinkedList<String> win = new LinkedList<>();
            List<String> localNGrams = new ArrayList<>();
            while (iter.hasNext()) {
                win.addLast(iter.next());
                if (win.size() == n) {
                    localNGrams.add(String.join(" ", win));
                    win.pollFirst();
                }
            }
            return localNGrams.iterator();
        });

        Map<String, Integer> freq = ngrams.mapToPair(g -> new Tuple2<>(g, 1))
                .reduceByKey(Integer::sum)
                .collectAsMap();

        sc.close();
        return new NGramResult(freq, calculateConditionalProbabilities(freq, n),
                System.currentTimeMillis() - start, 1);
    }

    private static Map<String, Double> calculateConditionalProbabilities(Map<String, Integer> freq, int n) {
        Map<String, Double> cond = new HashMap<>();
        Map<String, Integer> prefix = new HashMap<>();

        for (Map.Entry<String, Integer> e : freq.entrySet()) {
            String[] parts = e.getKey().split(" ");
            if (parts.length < n) continue;
            String pfx = String.join(" ", Arrays.copyOfRange(parts, 0, n - 1));
            prefix.merge(pfx, e.getValue(), Integer::sum);
        }

        for (Map.Entry<String, Integer> e : freq.entrySet()) {
            String[] parts = e.getKey().split(" ");
            if (parts.length < n) continue;
            String pfx = String.join(" ", Arrays.copyOfRange(parts, 0, n - 1));
            cond.put(e.getKey(), e.getValue() / (double) prefix.getOrDefault(pfx, 1));
        }

        return cond;
    }

    private static String preprocessText(String text) {
        return text.toLowerCase().replaceAll("[^a-zA-Z\\s]", "").replaceAll("\\s+", " ").trim();
    }

    private static <T> List<List<T>> splitList(List<T> list, int parts) {
        List<List<T>> chunks = new ArrayList<>();
        int size = (int) Math.ceil((double) list.size() / parts);
        for (int i = 0; i < list.size(); i += size) {
            chunks.add(list.subList(i, Math.min(list.size(), i + size)));
        }
        return chunks;
    }

    private static void displayResults(NGramResult result, ExecutionMode mode, int n, String file) {
        System.out.println("=== RESULTS ===");
        System.out.printf("Mode: %s | N: %d | File: %s%n", mode, n, file);
        System.out.printf("Time: %d ms | N-Grams: %d%n", result.getExecutionTime(), result.getNgramFrequencies().size());
        System.out.println("Top N-Grams:");
        result.getNgramFrequencies().entrySet().stream()
                .sorted(Map.Entry.<String, Integer>comparingByValue().reversed())
                .limit(20).forEach(e -> System.out.printf("%s : %d%n", e.getKey(), e.getValue()));
    }
}
