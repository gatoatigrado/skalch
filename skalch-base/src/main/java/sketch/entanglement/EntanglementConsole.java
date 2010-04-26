package sketch.entanglement;

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
import sketch.entanglement.partition.Partition;
import sketch.entanglement.partition.TraceListPartitioner;
import sketch.result.ScSynthesisResults;
import sketch.util.thread.InteractiveThread;

public class EntanglementConsole extends InteractiveThread {

    private EntanglementAnalysis ea;
    private InputStream input;
    private Map<Trace, ScStack> traceToStack;
    private ScSynthesisResults results;
    private Stack<List<Partition>> partitionsStack;

    public EntanglementConsole(InputStream input, ScSynthesisResults results) {
        super(0.05f);
        this.input = input;
        this.results = results;
        traceToStack = new HashMap<Trace, ScStack>();
        partitionsStack = new Stack<List<Partition>>();
        pullFromStore();
    }

    public void startConsole() {
        Scanner in = new Scanner(input);
        while (true) {
            System.out.print("<<<");

            String line = in.nextLine();
            try {
                StringTokenizer tokens = new StringTokenizer(line);
                String command = tokens.nextToken();

                if ("compare".equals(command)) {
                    int staticAngel1 = Integer.parseInt(tokens.nextToken());
                    int execNum1 = Integer.parseInt(tokens.nextToken());
                    int staticAngel2 = Integer.parseInt(tokens.nextToken());
                    int execNum2 = Integer.parseInt(tokens.nextToken());
                    DynAngel dynAngel1 = new DynAngel(staticAngel1, execNum1);
                    DynAngel dynAngel2 = new DynAngel(staticAngel2, execNum2);

                    compareTwoDynAngels(dynAngel1, dynAngel2);
                } else if ("constant".equals(command)) {
                    printAllConstantDynAngels();
                } else if ("entangled".equals(command)) {
                    printAllEntangledPairs();
                } else if ("subsets".equals(command)) {
                    if (tokens.hasMoreElements()) {
                        int n = Integer.parseInt(tokens.nextToken());
                        printNEntangledSubsets(n);
                    } else {
                        printNEntangledSubsets(1);
                    }
                } else if ("pull".equals(command)) {
                    pullFromStore();
                } else if ("update".equals(command)) {
                    updateEA();
                } else if ("partitioners".equals(command)) {
                    printTraceListPartitioners();
                } else if ("partitions".equals(command)) {
                    printPartitions();
                } else if ("partition".equals(command)) {
                    int n = Integer.parseInt(tokens.nextToken());
                    partitionTraces(TraceListPartitioner.getPartitionerTypes().get(n));
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
                } else {
                    System.out.println("Unknown command: " + command);
                }

            } catch (Exception e) {
                e.printStackTrace();
            }
        }
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

    private void partitionTraces(TraceListPartitioner traceListPartitioner) {
        List<Partition> newPartitions = new ArrayList<Partition>();
        for (Partition partition : partitionsStack.peek()) {
            newPartitions.addAll(traceListPartitioner.getTraceListPartition(partition));
        }
        partitionsStack.push(newPartitions);
    }

    private void printTraceListPartitioners() {
        List<TraceListPartitioner> partitioners =
                TraceListPartitioner.getPartitionerTypes();
        for (int i = 0; i < partitioners.size(); i++) {
            System.out.println("[" + i + "]" + partitioners.get(i).toString());
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
        updateEA();
    }

    private void updateEA() {
        Set<Trace> traces = new HashSet<Trace>();
        for (Partition partition : partitionsStack.peek()) {
            traces.addAll(partition.getTraces());
        }
        ea = new EntanglementAnalysis(traces);
    }

    private void compareTwoDynAngels(DynAngel angel1, DynAngel angel2) {
        Set<DynAngel> proj1 = new HashSet<DynAngel>();
        proj1.add(angel1);

        Set<DynAngel> proj2 = new HashSet<DynAngel>();
        proj2.add(angel2);

        EntanglementComparison result = ea.compareTwoSubtraces(proj1, proj2);
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
                new ArrayList<DynAngelPair>(ea.getAllEntangledPairs());
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

    @Override
    public void run_inner() {
        startConsole();
    }
}
