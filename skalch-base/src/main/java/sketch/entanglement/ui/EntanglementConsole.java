package sketch.entanglement.ui;

import java.io.InputStream;
import java.util.ArrayList;
import java.util.Collection;
import java.util.Collections;
import java.util.Comparator;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Iterator;
import java.util.List;
import java.util.Map;
import java.util.Scanner;
import java.util.Set;
import java.util.Stack;
import java.util.StringTokenizer;

import sketch.dyn.synth.stack.ScStack;
import sketch.entanglement.DynAngel;
import sketch.entanglement.EntanglementComparison;
import sketch.entanglement.Event;
import sketch.entanglement.SimpleEntanglementAnalysis;
import sketch.entanglement.Trace;
import sketch.entanglement.partition.SubsetOfTraces;
import sketch.entanglement.partition.TracePartitioner;
import sketch.entanglement.sat.SATEntanglementAnalysis;
import sketch.entanglement.sat.SubsetTraceFilter;
import sketch.result.ScSynthesisResults;
import sketch.util.thread.InteractiveThread;
import entanglement.EntanglementDetector;
import entanglement.MaxSupportFinder;
import entanglement.trace.Traces;

public class EntanglementConsole extends InteractiveThread {

    private SimpleEntanglementAnalysis ea;
    private SATEntanglementAnalysis satEA;

    private InputStream input;

    private Map<Trace, ScStack> traceToStack;
    private ScSynthesisResults results;

    private Stack<List<SubsetOfTraces>> subsetsStack;

    private HashSet<String> commandSet;

    private Map<Integer, List<SubsetOfTraces>> traceSetStorage;

    static String commands[] =
            { "compare", "constant", "entanglement", "info", "pull", "push", "store",
                    "restore", "showstored", "issubset", "update", "partitioners",
                    "subsets", "size", "partition", "choose", "remove", "reset", "exit",
                    "values", "help", "color", "nocolor", "grow" };

    public EntanglementConsole(InputStream input, ScSynthesisResults results) {
        super(0.05f);
        this.input = input;
        this.results = results;
        traceToStack = new HashMap<Trace, ScStack>();
        subsetsStack = new Stack<List<SubsetOfTraces>>();
        commandSet = new HashSet<String>();
        traceSetStorage = new HashMap<Integer, List<SubsetOfTraces>>();

        for (int i = 0; i < commands.length; i++) {
            commandSet.add(commands[i]);
        }
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
                } else if ("update".equals(command)) {
                    updateEntanglement();
                } else if ("pull".equals(command)) {
                    pullFromResults();
                } else if ("push".equals(command)) {
                    pushToResults();
                } else if ("subsets".equals(command)) {
                    printSubsets();
                } else if ("store".equals(command)) {
                    if (tokens.countTokens() > 0) {
                        int n = Integer.parseInt(tokens.nextToken());
                        storeSubsets(n);
                    } else {
                        storeSubsets(0);
                    }
                } else if ("restore".equals(command)) {
                    if (tokens.countTokens() > 0) {
                        int n = Integer.parseInt(tokens.nextToken());
                        restoreSubsets(n);
                    }
                } else if ("showstored".equals(command)) {
                    showStoredSubsets();
                } else if ("partitioners".equals(command)) {
                    printTracePartitioners();
                } else if ("partition".equals(command)) {
                    int n = Integer.parseInt(tokens.nextToken());
                    String args[] = new String[tokens.countTokens()];
                    for (int i = 0; i < args.length; i++) {
                        args[i] = tokens.nextToken();
                    }
                    if (n < TracePartitioner.partitionTypes.length) {
                        createTraceSubsets(TracePartitioner.partitionTypes[n], args);
                    }
                } else if ("choose".equals(command)) {
                    int n = Integer.parseInt(tokens.nextToken());
                    chooseSubset(n);
                } else if ("remove".equals(command)) {
                    int n = Integer.parseInt(tokens.nextToken());
                    removeSubset(n);
                } else if ("reset".equals(command)) {
                    resetSubsets();
                } else if ("issubset".equals(command)) {
                    int n = Integer.parseInt(tokens.nextToken());
                    printIsSubset(n);
                } else if ("info".equals(command)) {
                    printInfo();
                } else if ("compare".equals(command)) {
                    int staticAngel1 = Integer.parseInt(tokens.nextToken());
                    int execNum1 = Integer.parseInt(tokens.nextToken());
                    int staticAngel2 = Integer.parseInt(tokens.nextToken());
                    int execNum2 = Integer.parseInt(tokens.nextToken());
                    compare(staticAngel1, execNum1, staticAngel2, execNum2);
                } else if ("entanglement".equals(command)) {
                    printPartitioning(satEA);
                } else if ("constant".equals(command)) {
                    printConstantAngels();
                } else if ("size".equals(command)) {
                    printSize();
                } else if ("values".equals(command)) {
                    printValues(ea, getAllTraces(subsetsStack.peek()));
                } else if ("color".equals(command)) {
                    colorTraces();
                } else if ("nocolor".equals(command)) {
                    removeColorTraces();
                } else if ("grow".equals(command)) {
                    int n = Integer.parseInt(tokens.nextToken());
                    growTraces(n);
                } else if ("exit".equals(command)) {
                    break;
                } else if ("help".equals(command)) {
                    for (int i = 0; i < commands.length; i++) {
                        System.out.println(commands[i]);
                    }
                } else {
                    System.out.println("Unknown command: " + command);
                }

            } catch (Throwable t) {
                t.printStackTrace();
            }
        }
    }

    private void updateEntanglement() {
        Set<Trace> traces = getAllTraces(subsetsStack.peek());
        ea = new SimpleEntanglementAnalysis(traces);
        satEA = new SATEntanglementAnalysis(traces);
    }

    private void pullFromResults() {
        traceToStack.clear();
        ArrayList<ScStack> solutions = results.getSolutions();
        for (ScStack solution : solutions) {
            traceToStack.put(solution.getExecutionTrace(), solution);
        }
        List<SubsetOfTraces> initialPartition = new ArrayList<SubsetOfTraces>();
        initialPartition.add(new SubsetOfTraces(new ArrayList<Trace>(
                traceToStack.keySet()), "init", null));
        subsetsStack.push(initialPartition);
        storeSubsets(0);
        updateEntanglement();
    }

    private void pushToResults() {
        List<ScStack> stackList = new ArrayList<ScStack>();
        List<SubsetOfTraces> partitions = subsetsStack.peek();
        for (SubsetOfTraces partition : partitions) {
            for (Trace trace : partition.getTraces()) {
                stackList.add(traceToStack.get(trace));
            }
        }
        results.resetStackSolutions(stackList);
    }

    private void printTracePartitioners() {
        TracePartitioner[] partitioners = TracePartitioner.partitionTypes;
        for (int i = 0; i < partitioners.length; i++) {
            System.out.println("[" + i + "]" + partitioners[i].toString());
        }
    }

    private void createTraceSubsets(TracePartitioner traceListPartitioner, String args[])
    {
        List<SubsetOfTraces> newPartitions = new ArrayList<SubsetOfTraces>();
        for (SubsetOfTraces partition : subsetsStack.peek()) {
            newPartitions.addAll(traceListPartitioner.getSubsets(partition, args));
        }
        subsetsStack.push(newPartitions);
    }

    private void chooseSubset(int n) {
        List<SubsetOfTraces> curPartitions = subsetsStack.peek();
        if (n < curPartitions.size()) {
            List<SubsetOfTraces> newPartition = new ArrayList<SubsetOfTraces>();
            newPartition.add(curPartitions.get(n));
            subsetsStack.push(newPartition);
            updateEntanglement();
        }
    }

    private void removeSubset(int n) {
        List<SubsetOfTraces> curPartitions = subsetsStack.peek();
        if (n < curPartitions.size()) {
            List<SubsetOfTraces> newPartition = new ArrayList<SubsetOfTraces>();
            newPartition.addAll(curPartitions);
            newPartition.remove(curPartitions.get(n));
            subsetsStack.push(newPartition);
            updateEntanglement();
        }
    }

    private void resetSubsets() {
        if (subsetsStack.size() > 1) {
            subsetsStack.pop();
            updateEntanglement();
        }
    }

    private void printSubsets() {
        List<SubsetOfTraces> curSubsets = subsetsStack.peek();
        for (int i = 0; i < curSubsets.size(); i++) {
            System.out.println("[" + i + "]" + curSubsets.get(i).toString());
        }
    }

    private void storeSubsets(int n) {
        if (n < 0) {
            return;
        }
        traceSetStorage.put(n, subsetsStack.peek());
    }

    private void restoreSubsets(int n) {
        if (traceSetStorage.containsKey(n)) {
            List<SubsetOfTraces> s = traceSetStorage.get(n);
            subsetsStack.push(s);
            updateEntanglement();
        }
    }

    private void showStoredSubsets() {
        List<Integer> keys = new ArrayList<Integer>(traceSetStorage.keySet());
        Collections.sort(keys);
        for (Integer key : keys) {
            StringBuilder result = new StringBuilder("[" + key + "] {");
            List<SubsetOfTraces> curPartitions = traceSetStorage.get(key);
            for (int i = 0; i < curPartitions.size(); i++) {
                result.append(curPartitions.get(i).toString());
                if (i != curPartitions.size() - 1) {
                    result.append(',');
                }
            }
            result.append('}');
            System.out.println(result);
        }
    }

    private void printIsSubset(int n) {
        if (traceSetStorage.containsKey(n)) {
            List<SubsetOfTraces> s = traceSetStorage.get(n);
            if (isSubset(s, subsetsStack.peek())) {
                System.out.println(n + " is a subset of the current traces.");
            } else {
                System.out.println(n + " is not a subset of the current traces.");
            }
        }
    }

    private boolean isSubset(List<SubsetOfTraces> subset, List<SubsetOfTraces> superset) {
        Set<Trace> subsetTraces = getAllTraces(subset);
        Set<Trace> supersetTraces = getAllTraces(superset);

        return supersetTraces.containsAll(subsetTraces);
    }

    private void printInfo() {
        List<SubsetOfTraces> subsets = subsetsStack.peek();
        for (SubsetOfTraces subset : subsets) {
            System.out.println("****** " + subset.getPartitionName() + " ******");
            List<Trace> traces = subset.getTraces();
            System.out.println("Size: " + traces.size());
            printPartitioning(new SATEntanglementAnalysis(new HashSet<Trace>(traces)));

            printValues(new SimpleEntanglementAnalysis(traces),
                    new HashSet<Trace>(traces));
        }
    }

    private void compare(int staticAngel1, int execNum1, int staticAngel2, int execNum2) {
        DynAngel dynAngel1 = new DynAngel(staticAngel1, execNum1);
        DynAngel dynAngel2 = new DynAngel(staticAngel2, execNum2);

        compareTwoDynAngels(dynAngel1, dynAngel2);
    }

    private void compareTwoDynAngels(DynAngel angel1, DynAngel angel2) {
        Set<DynAngel> proj1 = new HashSet<DynAngel>();
        proj1.add(angel1);

        Set<DynAngel> proj2 = new HashSet<DynAngel>();
        proj2.add(angel2);

        EntanglementComparison result = ea.entangledInfo(proj1, proj2, true);
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

    private void printPartitioning(SATEntanglementAnalysis ea) {

        Comparator<Collection<DynAngel>> dynAngelsComparator =
                new Comparator<Collection<DynAngel>>() {

                    public int compare(Collection<DynAngel> angelCol1,
                            Collection<DynAngel> angelCol2)
                    {
                        DynAngel smallestAngel1 = null;
                        for (DynAngel da : angelCol1) {
                            if (smallestAngel1 == null ||
                                    smallestAngel1.compareTo(da) > 0)
                            {
                                smallestAngel1 = da;
                            }
                        }
                        DynAngel smallestAngel2 = null;
                        for (DynAngel da : angelCol2) {
                            if (smallestAngel2 == null ||
                                    smallestAngel2.compareTo(da) > 0)
                            {
                                smallestAngel2 = da;
                            }
                        }
                        return smallestAngel1.compareTo(smallestAngel2);
                    }
                };

        Set<Set<DynAngel>> subsets = ea.getEntangledPartitions();

        List<Set<DynAngel>> unentangledSubsets = new ArrayList<Set<DynAngel>>(subsets);
        Collections.sort(unentangledSubsets, dynAngelsComparator);

        for (Set<DynAngel> subset : unentangledSubsets) {
            List<DynAngel> subsetList = new ArrayList<DynAngel>(subset);
            System.out.println(toStringRep(subsetList));
        }
    }

    private void printSize() {
        int size = getAllTraces(subsetsStack.peek()).size();
        System.out.println("Size: " + size);
    }

    private void printConstantAngels() {
        List<DynAngel> dynAngels = new ArrayList<DynAngel>(ea.getConstantAngels());
        System.out.println(toStringRep(dynAngels));
    }

    private void printValues(SimpleEntanglementAnalysis ea, Set<Trace> traces) {
        HashMap<DynAngel, HashSet<Integer>> dynAngelToValues =
                new HashMap<DynAngel, HashSet<Integer>>();
        for (DynAngel angel : ea.getAngels()) {
            dynAngelToValues.put(angel, new HashSet<Integer>());
        }

        for (Trace t : traces) {
            for (Event e : t.getEvents()) {
                dynAngelToValues.get(e.dynAngel).add(e.valueChosen);
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

    private String toStringRep(List<DynAngel> angels) {
        Collections.sort(angels);
        StringBuilder result = new StringBuilder("{ ");
        for (int j = 0; j < angels.size(); j++) {
            DynAngel angel = angels.get(j);
            result.append(angel);
            if (j != angels.size() - 1) {
                result.append(", ");
            }
        }
        result.append(" }");
        return result.toString();
    }

    private Set<Trace> getAllTraces(List<SubsetOfTraces> subsetsList) {
        Set<Trace> traces = new HashSet<Trace>();
        for (SubsetOfTraces partition : subsetsList) {
            traces.addAll(partition.getTraces());
        }
        return traces;
    }

    private void removeColorTraces() {
        setCntMatrix(null);
    }

    private void colorTraces() {
        Set<DynAngel> angels = ea.getAngels();
        int maxStaticAngel = -1;
        for (DynAngel angel : angels) {
            if (angel.staticAngelId > maxStaticAngel) {
                maxStaticAngel = angel.staticAngelId;
            }
        }
        int[] maxExecNum = new int[maxStaticAngel + 1];
        for (DynAngel angel : angels) {
            if (angel.execNum > maxExecNum[angel.staticAngelId]) {
                maxExecNum[angel.staticAngelId] = angel.execNum;
            }
        }
        int[][] cntMatrix = new int[maxStaticAngel + 1][];
        for (int i = 0; i < cntMatrix.length; i++) {
            cntMatrix[i] = new int[maxExecNum[i] + 1];
        }

        int val = 4;
        int inc = 4;
        int constant = 0;
        Set<Set<DynAngel>> partitions = satEA.getEntangledPartitions();
        for (Set<DynAngel> partition : partitions) {
            if (partition.size() == 1) {
                DynAngel d = partition.iterator().next();
                cntMatrix[d.staticAngelId][d.execNum] = constant;
            } else {
                for (DynAngel d : partition) {
                    cntMatrix[d.staticAngelId][d.execNum] = val;
                }
                val += inc;
            }
        }
        setCntMatrix(cntMatrix);
    }

    private void setCntMatrix(int[][] cntMatrix) {
        for (Trace t : traceToStack.keySet()) {
            traceToStack.get(t).setCnt(cntMatrix);
        }
    }

    private void growTraces(int n) {
        if (traceSetStorage.containsKey(n)) {
            List<SubsetOfTraces> s = traceSetStorage.get(n);
            Set<Trace> allTraces = getAllTraces(s);
            SATEntanglementAnalysis allTraceEA = new SATEntanglementAnalysis(allTraces);
            Set<Trace> subsetTraces = getAllTraces(subsetsStack.peek());

            Traces subset =
                    allTraceEA.getTraceConverter().getTraces().restrict(
                            new SubsetTraceFilter(subsetTraces,
                                    allTraceEA.getTraceConverter()));
            System.out.println(subset.size());
            System.out.println(allTraceEA.getEntangledIntSets());
            System.out.println(EntanglementDetector.entanglement(subset));

            Iterator<Traces> it =
                    MaxSupportFinder.findMaximalSupports(
                            allTraceEA.getTraceConverter().getTraces(),
                            allTraceEA.getEntangledIntSets(), subset,
                            EntanglementDetector.entanglement(subset));

            System.out.println("here");
            List<SubsetOfTraces> partitions = new ArrayList<SubsetOfTraces>();
            int index = 0;
            while (it.hasNext()) {

                Traces traces = it.next();
                System.out.println("new trace set" + traces.size());

                List<Trace> sketchTraces = allTraceEA.getTraceConverter().convert(traces);
                SubsetOfTraces partition =
                        new SubsetOfTraces(sketchTraces, "grow" + index, null);
                partitions.add(partition);
                index++;
            }
            subsetsStack.push(partitions);
            updateEntanglement();
        }

    }

    @Override
    public void run_inner() {
        startConsole();
    }
}
