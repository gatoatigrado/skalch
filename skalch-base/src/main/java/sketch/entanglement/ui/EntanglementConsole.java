package sketch.entanglement.ui;

import java.io.InputStream;
import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Scanner;
import java.util.Set;
import java.util.Stack;
import java.util.StringTokenizer;

import sketch.dyn.synth.stack.ScStack;
import sketch.entanglement.DynAngel;
import sketch.entanglement.DynAngelPair;
import sketch.entanglement.EntanglementAnalysis;
import sketch.entanglement.EntanglementAnalysisSetBased;
import sketch.entanglement.EntanglementComparison;
import sketch.entanglement.EntanglementSubsets;
import sketch.entanglement.Event;
import sketch.entanglement.Trace;
import sketch.entanglement.partition.AutoPartition;
import sketch.entanglement.partition.Partition;
import sketch.entanglement.partition.TracePartitioner;
import sketch.result.ScSynthesisResults;
import sketch.util.thread.InteractiveThread;

public class EntanglementConsole extends InteractiveThread {

    private EntanglementAnalysis ea;
    private EntanglementAnalysisSetBased eaSet;
    private InputStream input;
    private Map<Trace, ScStack> traceToStack;
    private ScSynthesisResults results;
    private Stack<List<Partition>> partitionsStack;
    private HashSet<String> commandSet;

    static String commands[] =
            { "compare", "entangled", "subsets", "pull", "push", "update",
                    "partitioners", "partitions", "size", "partition", "autopartition",
                    "choose", "reset", "exit", "remove", "gui", "values", "heuristic",
                    "bitstring", "guess", "help" };

    public EntanglementConsole(InputStream input, ScSynthesisResults results) {
        super(0.05f);
        this.input = input;
        this.results = results;
        traceToStack = new HashMap<Trace, ScStack>();
        partitionsStack = new Stack<List<Partition>>();
        commandSet = new HashSet<String>();
        for (int i = 0; i < commands.length; i++) {
            commandSet.add(commands[i]);
        }
        pullFromStore();
    }

    public void startConsole() {
        Scanner in = new Scanner(input);
        while (true) {
            System.out.print("<<<$ ");

            String line = in.nextLine();
            try {
                StringTokenizer tokens = new StringTokenizer(line);
                if (!tokens.hasMoreTokens()) {
                    continue;
                }

                String command = tokens.nextToken();
                if (!commandSet.contains(command)) {
                    System.out.println("Unknown command: " + command);
                    continue;
                }

                if ("compare".equals(command)) {
                    int staticAngel1 = Integer.parseInt(tokens.nextToken());
                    int execNum1 = Integer.parseInt(tokens.nextToken());
                    int staticAngel2 = Integer.parseInt(tokens.nextToken());
                    int execNum2 = Integer.parseInt(tokens.nextToken());
                    compare(staticAngel1, execNum1, staticAngel2, execNum2);
                } else if ("constant".equals(command)) {
                    printAllConstantDynAngels();
                } else if ("entangled".equals(command)) {
                    printAllEntangledPairs();
                } else if ("subsets".equals(command)) {
                    if (tokens.hasMoreElements()) {
                        int n = Integer.parseInt(tokens.nextToken());
                        printNEntangledSubsets(n);
                    } else {
                        printNEntangledSubsets(2);
                    }
                } else if ("pull".equals(command)) {
                    pullFromStore();
                } else if ("push".equals(command)) {
                    pushFromStore();
                } else if ("update".equals(command)) {
                    updateEA();
                } else if ("partitioners".equals(command)) {
                    printTraceListPartitioners();
                } else if ("partitions".equals(command)) {
                    printPartitions();
                } else if ("size".equals(command)) {
                    printSize();
                } else if ("partition".equals(command)) {
                    int n = Integer.parseInt(tokens.nextToken());
                    String args[] = new String[tokens.countTokens()];
                    for (int i = 0; i < args.length; i++) {
                        args[i] = tokens.nextToken();
                    }
                    if (n < TracePartitioner.partitionTypes.length) {
                        partitionTraces(TracePartitioner.partitionTypes[n], args);
                    }
                } else if ("autopartition".equals(command)) {
                    String args[] = new String[tokens.countTokens()];
                    for (int i = 0; i < args.length; i++) {
                        args[i] = tokens.nextToken();
                    }
                    partitionTraces(new AutoPartition(), args);
                } else if ("choose".equals(command)) {
                    int n = Integer.parseInt(tokens.nextToken());
                    choosePartition(n);
                } else if ("reset".equals(command)) {
                    resetPartitions();
                } else if ("exit".equals(command)) {
                    break;
                } else if ("remove".equals(command)) {
                    results.removeAllStackSolutions();
                } else if ("gui".equals(command)) {
                    if (tokens.hasMoreElements()) {
                        int n = Integer.parseInt(tokens.nextToken());
                        showGui(n);
                    } else {
                        showGui(ea);
                    }
                } else if ("values".equals(command)) {
                    printValues();
                } else if ("bitstring".equals(command)) {
                    if (tokens.hasMoreElements()) {
                        printTraces(tokens.nextToken());
                    } else {
                        printTraces("trace");
                    }
                } else if ("guess".equals(command)) {
                    guess();
                } else if ("help".equals(command)) {
                    for (int i = 0; i < commands.length; i++) {
                        System.out.println(commands[i]);
                    }
                } else {
                    System.out.println("Unknown command: " + command);
                }

            } catch (Exception e) {
                e.printStackTrace();
            }
        }
    }

    private void printValues() {
        HashMap<DynAngel, HashSet<Integer>> dynAngelToValues =
                new HashMap<DynAngel, HashSet<Integer>>();
        for (DynAngel angel : ea.getAngels()) {
            dynAngelToValues.put(angel, new HashSet<Integer>());
        }
        List<Partition> partitions = partitionsStack.peek();
        for (Partition partition : partitions) {
            for (Trace t : partition.getTraces()) {
                for (Event e : t.getEvents()) {
                    dynAngelToValues.get(e.dynAngel).add(e.valueChosen);
                }
            }
        }
        ArrayList<DynAngel> angels = new ArrayList<DynAngel>(dynAngelToValues.keySet());
        Collections.sort(angels);
        for (DynAngel angel : angels) {
            System.out.print(angel.toString() + ": ");
            for (Integer v : dynAngelToValues.get(angel)) {
                System.out.print(v + ",");
            }
            System.out.println();
        }
    }

    private void compare(int staticAngel1, int execNum1, int staticAngel2, int execNum2) {
        DynAngel dynAngel1 = new DynAngel(staticAngel1, execNum1);
        DynAngel dynAngel2 = new DynAngel(staticAngel2, execNum2);

        compareTwoDynAngels(dynAngel1, dynAngel2);
    }

    private void printSize() {
        int size = 0;
        for (Partition partition : partitionsStack.peek()) {
            size += partition.getTraces().size();
        }
        System.out.println(size);
    }

    private void pushFromStore() {
        List<ScStack> stackList = new ArrayList<ScStack>();
        List<Partition> partitions = partitionsStack.peek();
        for (Partition partition : partitions) {
            for (Trace trace : partition.getTraces()) {
                stackList.add(traceToStack.get(trace));
            }
        }
        results.resetStackSolutions(stackList);

    }

    private void printPartitions() {
        List<Partition> curPartitions = partitionsStack.peek();
        for (int i = 0; i < curPartitions.size(); i++) {
            System.out.println("[" + i + "]" + curPartitions.get(i).toString());
        }
    }

    private void resetPartitions() {
        if (partitionsStack.size() > 1) {
            partitionsStack.pop();
        }
    }

    private void choosePartition(int n) {
        List<Partition> curPartitions = partitionsStack.peek();
        if (n < curPartitions.size()) {
            List<Partition> newPartition = new ArrayList<Partition>();
            newPartition.add(curPartitions.get(n));
            partitionsStack.push(newPartition);
        }
    }

    private void partitionTraces(TracePartitioner traceListPartitioner, String args[]) {
        List<Partition> newPartitions = new ArrayList<Partition>();
        for (Partition partition : partitionsStack.peek()) {
            newPartitions.addAll(traceListPartitioner.getPartitions(partition, args));
        }
        partitionsStack.push(newPartitions);
    }

    private void printTraceListPartitioners() {
        TracePartitioner[] partitioners = TracePartitioner.partitionTypes;
        for (int i = 0; i < partitioners.length; i++) {
            System.out.println("[" + i + "]" + partitioners[i].toString());
        }
    }

    private void showGui(int n) {
        List<Partition> curPartitions = partitionsStack.peek();
        if (n < curPartitions.size()) {
            showGui(new EntanglementAnalysis(curPartitions.get(n).getTraces()));
        }
    }

    private void showGui(EntanglementAnalysis analysis) {
        new EntanglementGui(analysis);
    }

    private void pullFromStore() {
        traceToStack.clear();
        ArrayList<ScStack> solutions = results.getSolutions();
        for (ScStack solution : solutions) {
            traceToStack.put(solution.getExecutionTrace(), solution);
        }
        List<Partition> initialPartition = new ArrayList<Partition>();
        initialPartition.add(new Partition(new ArrayList<Trace>(traceToStack.keySet()),
                "init", null));
        partitionsStack.add(initialPartition);
    }

    private void updateEA() {
        Set<Trace> traces = new HashSet<Trace>();
        for (Partition partition : partitionsStack.peek()) {
            traces.addAll(partition.getTraces());
        }
        ea = new EntanglementAnalysis(traces);
        eaSet = new EntanglementAnalysisSetBased(traces);
    }

    private void compareTwoDynAngels(DynAngel angel1, DynAngel angel2) {
        Set<DynAngel> proj1 = new HashSet<DynAngel>();
        proj1.add(angel1);

        Set<DynAngel> proj2 = new HashSet<DynAngel>();
        proj2.add(angel2);

        EntanglementComparison result = ea.compareTwoSubtraces(proj1, proj2, true);
        printEntanglementResults(result);
    }

    private void printEntanglementResults(EntanglementComparison result) {
        int[][] correlationMap = result.correlationMap;

        int numRows = 1 + correlationMap.length;
        int numColumns = 1 + correlationMap[0].length;

        String[][] grid = new String[numRows][numColumns];
        for (int i = 0; i < grid.length; i++) {
            for (int j = 0; j < grid[0].length; j++) {
                grid[i][j] = "-";
            }
        }

        int maxStringSize = 0;

        for (int i = 1; i < grid.length; i++) {
            grid[i][0] = "" + (i - 1);
            maxStringSize = Math.max(maxStringSize, grid[i][0].length());
        }

        for (int i = 1; i < grid[0].length; i++) {
            grid[0][i] = "" + (i - 1);
            maxStringSize = Math.max(maxStringSize, grid[0][i].length());
        }

        for (int i = 0; i < correlationMap.length; i++) {
            for (int j = 0; j < correlationMap[0].length; j++) {
                if (correlationMap[i][j] != 0) {
                    grid[i + 1][j + 1] = "" + correlationMap[i][j];
                    maxStringSize = Math.max(maxStringSize, grid[i + 1][j + 1].length());
                }
            }
        }

        for (int i = 0; i < grid.length; i++) {
            for (int j = 0; j < grid[0].length; j++) {
                System.out.print(grid[i][j]);
                for (int k = 0; k < maxStringSize + 1 - grid[i][j].length(); k++) {
                    System.out.print(' ');
                }
            }
            System.out.println();
        }

        List<Trace> proj1Values = result.proj1Values;
        List<Trace> proj2Values = result.proj2Values;

        System.out.println("Projection 1 Values:");
        for (int i = 0; i < proj1Values.size(); i++) {
            System.out.println("" + i + ":" + proj1Values.get(i));
        }
        System.out.println("Projection 2 Values:");
        for (int i = 0; i < proj2Values.size(); i++) {
            System.out.println("" + i + ":" + proj2Values.get(i));
        }
        System.out.println();

    }

    private void printAllEntangledPairs() {
        List<DynAngelPair> entangledAngels =
                new ArrayList<DynAngelPair>(ea.getEntangledPairs());
        Collections.sort(entangledAngels);
        System.out.println("Entangled angels");
        for (DynAngelPair entangledAngelPair : entangledAngels) {
            System.out.println(entangledAngelPair);
        }

    }

    private void printAllConstantDynAngels() {
        List<DynAngel> dynAngels = new ArrayList<DynAngel>(ea.getConstantAngels());
        Collections.sort(dynAngels);
        System.out.println("Constant angels");
        for (DynAngel dynAngel : dynAngels) {
            System.out.println("Location: " + dynAngel);
        }
    }

    private void printNEntangledSubsets(int n) {
        EntanglementSubsets subsets = ea.getNEntangledSubsets(n);
        printSubsets(subsets);
    }

    private void printSubsets(EntanglementSubsets subsets) {
        System.out.println("----Entangled----");
        for (Set<DynAngel> subset : subsets.entangledSubsets) {
            List<DynAngel> subsetList = new ArrayList<DynAngel>(subset);
            System.out.println("--------------");
            Collections.sort(subsetList);
            for (DynAngel dynAngel : subsetList) {
                System.out.println("Location: " + dynAngel);
            }
        }
        System.out.println("----Unentangled----");
        for (Set<DynAngel> subset : subsets.unentangledSubsets) {
            List<DynAngel> subsetList = new ArrayList<DynAngel>(subset);
            System.out.println("--------------");
            Collections.sort(subsetList);
            for (DynAngel dynAngel : subsetList) {
                System.out.println("Location: " + dynAngel);
            }
        }
    }

    public void printTraces(String fileName) {
        eaSet.printBitStrings(fileName);
    }

    private void guess() {
        Set<DynAngel> entangledAngels = eaSet.findEntanglement();
        for (DynAngel dynAngel : entangledAngels) {
            System.out.println("Location: " + dynAngel);
        }
    }

    @Override
    public void run_inner() {
        startConsole();
    }
}
