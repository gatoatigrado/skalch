package sketch.entanglement.ui;

import java.io.FileNotFoundException;
import java.io.InputStream;
import java.io.PrintStream;
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

import sketch.dyn.main.ScDynamicSketchCall;
import sketch.dyn.synth.stack.ScStack;
import sketch.entanglement.DynAngel;
import sketch.entanglement.EntanglementComparison;
import sketch.entanglement.Event;
import sketch.entanglement.HeuristicSearch;
import sketch.entanglement.SimpleEntanglementAnalysis;
import sketch.entanglement.Trace;
import sketch.entanglement.partition.TracePartitioner;
import sketch.entanglement.partition.TraceSubset;
import sketch.entanglement.sat.SATEntanglementAnalysis;
import sketch.entanglement.sat.SubsetTraceFilter;
import sketch.result.ScSynthesisResults;
import sketch.ui.sourcecode.ScSourceConstruct;
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

    private Stack<List<TraceSubset>> subsetsStack;

    private HashSet<String> commandSet;

    private Map<Integer, List<TraceSubset>> traceSetStorage;
    private ScDynamicSketchCall<?> sketch;
    private Set<ScSourceConstruct> sourceCodeInfo;

    static String commands[] =
            { "compare", "constant", "entanglement", "info", "pull", "push", "store",
                    "restore", "showstored", "issubset", "update", "partitioners",
                    "subsets", "size", "partition", "choose", "remove", "reset", "exit",
                    "values", "help", "grow", "gui", "summary", "auto", "invert" };

    public EntanglementConsole(InputStream input, ScSynthesisResults results,
            ScDynamicSketchCall<?> sketch, Set<ScSourceConstruct> sourceCodeInfo)
    {
        super(0.05f);
        this.input = input;
        this.results = results;
        this.sketch = sketch;
        this.sourceCodeInfo = sourceCodeInfo;

        traceToStack = new HashMap<Trace, ScStack>();
        subsetsStack = new Stack<List<TraceSubset>>();
        commandSet = new HashSet<String>();
        traceSetStorage = new HashMap<Integer, List<TraceSubset>>();

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
                    if (tokens.hasMoreTokens()) {
                        String angelList = "";
                        while (tokens.hasMoreTokens()) {
                            angelList += tokens.nextToken() + " ";
                        }
                        Set<Trace> values =
                                ea.getValues(new HashSet<DynAngel>(
                                        DynAngel.parseDynAngelList(angelList)));
                        System.out.println(values.size());
                    } else {
                        printValues(ea, getAllTraces(subsetsStack.peek()));
                    }
                } else if ("grow".equals(command)) {
                    int n = Integer.parseInt(tokens.nextToken());
                    growTraces(n);
                } else if ("exit".equals(command)) {
                    break;
                } else if ("help".equals(command)) {
                    for (int i = 0; i < commands.length; i++) {
                        System.out.println(commands[i]);
                    }
                } else if ("gui".equals(command)) {
                    EntanglementGui gui =
                            new EntanglementGui(subsetsStack.peek(), traceToStack,
                                    sketch, sourceCodeInfo);
                    gui.pack();
                    gui.setVisible(true);
                } else if ("summary".equals(command)) {
                    Trace t = traceToStack.keySet().iterator().next();
                    List<DynAngel> angelList = new ArrayList<DynAngel>();
                    for (Event e : t.getEvents()) {
                        angelList.add(e.dynAngel);
                    }
                    printSummary(angelList, satEA, ea);
                } else if ("auto".equals(command)) {
                    Trace trace = traceToStack.keySet().iterator().next();
                    List<DynAngel> angelList = new ArrayList<DynAngel>();
                    for (Event e : trace.getEvents()) {
                        angelList.add(e.dynAngel);
                    }
                    autoPartition(angelList, satEA, ea);
                } else if ("invert".equals(command)) {
                    Set<Trace> curTraces =
                            new HashSet<Trace>(
                                    getAllTraces(subsetsStack.get(subsetsStack.size() - 1)));
                    Set<Trace> prevTraces =
                            new HashSet<Trace>(
                                    getAllTraces(subsetsStack.get(subsetsStack.size() - 2)));
                    prevTraces.removeAll(curTraces);
                    TraceSubset invert =
                            new TraceSubset(new ArrayList<Trace>(prevTraces), "invert",
                                    subsetsStack.peek().get(0));
                    ArrayList<TraceSubset> singleton = new ArrayList<TraceSubset>();
                    singleton.add(invert);
                    subsetsStack.push(singleton);
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
        // HashMap<Trace, ScStack> traceToStackBuffer = new HashMap<Trace, ScStack>();
        for (ScStack solution : solutions) {
            traceToStack.put(solution.getExecutionTrace(), solution);
            // if (traceToStackBuffer.size() >= 10000) {
            // System.out.println(traceToStack.size());
            // traceToStack.putAll(traceToStackBuffer);
            // traceToStackBuffer.clear();
            // }
        }
        List<TraceSubset> initialPartition = new ArrayList<TraceSubset>();
        initialPartition.add(new TraceSubset(new ArrayList<Trace>(traceToStack.keySet()),
                "init", null));
        subsetsStack.push(initialPartition);
        storeSubsets(0);
        updateEntanglement();
    }

    private void pushToResults() {
        List<ScStack> stackList = new ArrayList<ScStack>();
        List<TraceSubset> partitions = subsetsStack.peek();
        for (TraceSubset partition : partitions) {
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
        List<TraceSubset> newPartitions = new ArrayList<TraceSubset>();
        for (TraceSubset partition : subsetsStack.peek()) {
            newPartitions.addAll(traceListPartitioner.getSubsets(partition, args));
        }
        subsetsStack.push(newPartitions);
    }

    private void chooseSubset(int n) {
        List<TraceSubset> curPartitions = subsetsStack.peek();
        if (n < curPartitions.size()) {
            List<TraceSubset> newPartition = new ArrayList<TraceSubset>();
            newPartition.add(curPartitions.get(n));
            subsetsStack.push(newPartition);
            updateEntanglement();
        }
    }

    private void removeSubset(int n) {
        List<TraceSubset> curPartitions = subsetsStack.peek();
        if (n < curPartitions.size()) {
            List<TraceSubset> newPartition = new ArrayList<TraceSubset>();
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
        List<TraceSubset> curSubsets = subsetsStack.peek();
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
            List<TraceSubset> s = traceSetStorage.get(n);
            subsetsStack.push(s);
            updateEntanglement();
        }
    }

    private void showStoredSubsets() {
        List<Integer> keys = new ArrayList<Integer>(traceSetStorage.keySet());
        Collections.sort(keys);
        for (Integer key : keys) {
            StringBuilder result = new StringBuilder("[" + key + "] {");
            List<TraceSubset> curPartitions = traceSetStorage.get(key);
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
            List<TraceSubset> s = traceSetStorage.get(n);
            if (isSubset(s, subsetsStack.peek())) {
                System.out.println(n + " is a subset of the current traces.");
            } else {
                System.out.println(n + " is not a subset of the current traces.");
            }
        }
    }

    private boolean isSubset(List<TraceSubset> subset, List<TraceSubset> superset) {
        Set<Trace> subsetTraces = getAllTraces(subset);
        Set<Trace> supersetTraces = getAllTraces(superset);

        return supersetTraces.containsAll(subsetTraces);
    }

    private void printInfo() {
        List<TraceSubset> subsets = subsetsStack.peek();
        for (TraceSubset subset : subsets) {
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

        long startTime = System.currentTimeMillis();
        Set<Set<DynAngel>> subsets = ea.getEntangledPartitions();
        long endTime = System.currentTimeMillis();
        System.out.println("Time to compute entanglement(ms): " + (endTime - startTime));

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

    private Set<Trace> getAllTraces(List<TraceSubset> subsetsList) {
        Set<Trace> traces = new HashSet<Trace>();
        for (TraceSubset partition : subsetsList) {
            traces.addAll(partition.getTraces());
        }
        return traces;
    }

    private void growTraces(int n) {
        if (traceSetStorage.containsKey(n)) {
            List<TraceSubset> s = traceSetStorage.get(n);
            Set<Trace> allTraces = getAllTraces(s);
            SATEntanglementAnalysis allTraceEA = new SATEntanglementAnalysis(allTraces);
            Set<Trace> subsetTraces = getAllTraces(subsetsStack.peek());

            Traces subset =
                    allTraceEA.getTraceConverter().getTraces().restrict(
                            new SubsetTraceFilter(subsetTraces,
                                    allTraceEA.getTraceConverter(), false));
            System.out.println(subset.size());
            System.out.println(allTraceEA.getEntangledIntSets());
            System.out.println(EntanglementDetector.entanglement(subset));

            Iterator<Traces> it =
                    MaxSupportFinder.findMaximalSupports(
                            allTraceEA.getTraceConverter().getTraces(),
                            allTraceEA.getEntangledIntSets(), subset,
                            EntanglementDetector.entanglement(subset));

            System.out.println("here");
            List<TraceSubset> partitions = new ArrayList<TraceSubset>();
            int index = 0;
            while (it.hasNext()) {

                Traces traces = it.next();
                System.out.println("new trace set" + traces.size());

                List<Trace> sketchTraces = allTraceEA.getTraceConverter().convert(traces);
                TraceSubset partition =
                        new TraceSubset(sketchTraces, "grow" + index, null);
                partitions.add(partition);
                index++;
            }
            subsetsStack.push(partitions);
            updateEntanglement();
        }

    }

    private void printSummary(List<DynAngel> angelList, SATEntanglementAnalysis satEA,
            SimpleEntanglementAnalysis ea)
    {
        Set<Set<DynAngel>> partitions = satEA.getEntangledPartitions();

        Map<Set<DynAngel>, Set<Trace>> subtraces =
                new HashMap<Set<DynAngel>, Set<Trace>>();
        int length = 0;

        for (Set<DynAngel> partition : partitions) {
            Set<Trace> values = ea.getValues(partition);
            length += values.size();
            subtraces.put(partition, values);
        }

        Map<DynAngel, Integer> angelToRow = new HashMap<DynAngel, Integer>();
        for (int i = 0; i < angelList.size(); i++) {
            angelToRow.put(angelList.get(i), i);
        }

        HashSet<DynAngel> seenAngels = new HashSet<DynAngel>();
        SparseMatrix<String> matrix = new SparseMatrix<String>(angelList.size(), length);
        int column = 0;

        for (DynAngel angel : angelList) {
            if (!seenAngels.contains(angel)) {
                Set<DynAngel> angelPartition = null;
                for (Set<DynAngel> partition : partitions) {
                    if (partition.contains(angel)) {
                        angelPartition = partition;
                        break;
                    }
                }

                seenAngels.addAll(angelPartition);

                Set<Trace> values = subtraces.get(angelPartition);
                for (Trace value : values) {
                    for (Event event : value.getEvents()) {
                        matrix.put(angelToRow.get(event.dynAngel), column, "" +
                                event.valueChosen);
                    }
                    column++;
                }
            }
        }

        try {
            PrintStream out = new PrintStream("traces.txt");
            for (int i = 0; i < matrix.getNumRows(); i++) {
                for (int j = 0; j < matrix.getNumCols(); j++) {
                    String val = matrix.get(i, j);
                    if (val != null) {
                        out.print(val);
                    }
                    out.print('\t');
                }
                out.println();
            }
            out.close();
        } catch (FileNotFoundException e) {
            // TODO Auto-generated catch block
            e.printStackTrace();
        }
    }

    private void autoPartition(List<DynAngel> angelOrder, SATEntanglementAnalysis satEA,
            SimpleEntanglementAnalysis ea)
    {
        HeuristicSearch hs = new HeuristicSearch(angelOrder, satEA, ea);
        hs.partition();
    }

    @Override
    public void run_inner() {
        startConsole();
    }
}
