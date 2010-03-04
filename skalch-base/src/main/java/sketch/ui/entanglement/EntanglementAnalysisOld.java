package sketch.ui.entanglement;

import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;
import java.util.List;
import java.util.Scanner;
import java.util.StringTokenizer;

import sketch.util.DebugOut;

public class EntanglementAnalysisOld {

    // private List<ExecutionTrace> traces;

    private HashMap<DynAngel, ArrayList<Integer>> angelsToValueMap;

    public EntanglementAnalysisOld(List<Trace> traces) {
        // this.traces = traces;
        angelsToValueMap = getAngelsToValuesMap(traces);
    }

    private HashMap<DynAngel, ArrayList<Integer>> getAngelsToValuesMap(
            List<Trace> traces)
    {
        HashMap<DynAngel, ArrayList<Integer>> dynamicAngelsToValues =
                new HashMap<DynAngel, ArrayList<Integer>>();

        // go through every trace
        for (int i = 0; i < traces.size(); i++) {
            // go through every dynamic angelic call
            Trace trace = traces.get(i);
            for (Event angelicCall : trace.events) {
                DynAngel location = angelicCall.dynAngel;
                ArrayList<Integer> values;
                // add valueChosen to correct list
                if (dynamicAngelsToValues.containsKey(location)) {
                    values = dynamicAngelsToValues.get(location);
                } else {
                    values = new ArrayList<Integer>();
                    dynamicAngelsToValues.put(location, values);
                }
                // if the dynamic angel was not accessed in the previous traces,
                // then we need to pad the list until the index
                while (values.size() < i) {
                    values.add(-1);
                }
                values.add(angelicCall.valueChosen);
            }
        }

        // pad all the lists so they are the same size
        for (DynAngel dynamicAngel : dynamicAngelsToValues.keySet()) {
            ArrayList<Integer> values = dynamicAngelsToValues.get(dynamicAngel);
            while (values.size() < traces.size()) {
                values.add(-1);
            }
        }

        return dynamicAngelsToValues;
    }

    public void compareAllDynamicAngels() {
        // list of all dynamic angels
        ArrayList<DynAngel> dynamicAngels = new ArrayList<DynAngel>();
        dynamicAngels.addAll(angelsToValueMap.keySet());

        // map from index to list of values (the index corresponds to the index
        // in dynamicAngels
        ArrayList<ArrayList<Integer>> indexToValues = new ArrayList<ArrayList<Integer>>();

        for (int i = 0; i < dynamicAngels.size(); i++) {
            DynAngel dynamicAngel = dynamicAngels.get(i);
            ArrayList<Integer> angelValues = angelsToValueMap.get(dynamicAngel);
            indexToValues.add(angelValues);
        }

        for (int i = 0; i < indexToValues.size(); i++) {
            for (int j = 0; j < indexToValues.size(); j++) {
                if (i == j) {
                    continue;
                }

                if (compareTwoDynamicAngel(indexToValues.get(i), indexToValues.get(j))) {
                    DebugOut.print("correlation between " + dynamicAngels.get(i) +
                            " and " + dynamicAngels.get(j) + ".");
                }
            }
        }

        compareDynamicAngelsConsole();
    }

    public void compareDynamicAngelsConsole() {
        Scanner in = new Scanner(System.in);
        while (true) {
            System.out.print("<<<");
            boolean matched = false;

            String line = in.nextLine();
            try {
                StringTokenizer tokens = new StringTokenizer(line);
                String command = tokens.nextToken();

                if ("compare".equals(command)) {
                    int holeId1 = Integer.parseInt(tokens.nextToken());
                    int numExec1 = Integer.parseInt(tokens.nextToken());
                    int holeId2 = Integer.parseInt(tokens.nextToken());
                    int numExec2 = Integer.parseInt(tokens.nextToken());
                    DynAngel loc1 = new DynAngel(holeId1, numExec1);
                    DynAngel loc2 = new DynAngel(holeId2, numExec2);

                    ArrayList<Integer> values1 = angelsToValueMap.get(loc1);
                    ArrayList<Integer> values2 = angelsToValueMap.get(loc2);
                    compareTwoDynamicAngel(values1, values2, true);
                }

            } catch (Exception e) {
                e.printStackTrace();
            } finally {
                if (!matched) {
                    System.out.println("Unknown command.");
                }
            }
        }
    }

    public boolean compareTwoDynamicAngel(ArrayList<Integer> angel1,
            ArrayList<Integer> angel2)
    {
        return compareTwoDynamicAngel(angel1, angel2, false);
    }

    public boolean compareTwoDynamicAngel(ArrayList<Integer> angel1,
            ArrayList<Integer> angel2, boolean verbose)
    {
        assert angel1.size() == angel2.size();

        int max1 = 0, max2 = 0;
        for (Integer v : angel1) {
            max1 = Math.max(v, max1);
        }

        for (Integer v : angel2) {
            max2 = Math.max(v, max2);
        }

        max1++;
        max2++;

        // default value is false
        boolean[][] correlationMap = new boolean[max1][max2];
        boolean[] usedValues1 = new boolean[max1];
        boolean[] usedValues2 = new boolean[max2];
        for (int i = 0; i < angel1.size(); i++) {
            if (angel1.get(i) == -1 || angel2.get(i) == -1) {
                continue;
            }
            correlationMap[angel1.get(i)][angel2.get(i)] = true;
            usedValues1[angel1.get(i)] = true;
            usedValues2[angel2.get(i)] = true;
        }

        if (verbose) {
            printCorrelationMap(usedValues1, usedValues2, correlationMap);
        }

        for (int i = 0; i < correlationMap.length; i++) {
            if (!usedValues1[i]) {
                continue;
            }
            boolean allTrue = true;
            for (int j = 0; j < correlationMap[0].length; j++) {
                if (!usedValues2[j]) {
                    continue;
                }
                if (!correlationMap[i][j]) {
                    allTrue = false;
                }
            }

            if (!allTrue) {
                return true;
            }
        }
        return false;
    }

    private void printCorrelationMap(boolean[] usedValues1, boolean[] usedValues2,
            boolean[][] correlationMap)
    {
        int numColumns = 1;
        int numRows = 1;
        for (int i = 0; i < usedValues1.length; i++) {
            if (usedValues1[i]) {
                numRows++;
            }
        }
        for (int i = 0; i < usedValues2.length; i++) {
            if (usedValues2[i]) {
                numColumns++;
            }
        }
        String[][] grid = new String[numRows][numColumns];
        for (int i = 0; i < grid.length; i++) {
            for (int j = 0; j < grid[0].length; j++) {
                grid[i][j] = "-";
            }
        }

        int maxStringSize = 0;

        int index = 1;
        for (int i = 0; i < usedValues1.length; i++) {
            if (usedValues1[i]) {
                grid[index][0] = "" + i;
                maxStringSize = Math.max(maxStringSize, grid[index][0].length());
                index++;
            }
        }

        index = 1;
        for (int i = 0; i < usedValues2.length; i++) {
            if (usedValues2[i]) {
                grid[0][index] = "" + i;
                maxStringSize = Math.max(maxStringSize, grid[0][index].length());
                index++;
            }
        }

        int indexI = 0;
        for (int i = 0; i < correlationMap.length; i++) {
            if (usedValues1[i]) {
                indexI++;
            } else {
                continue;
            }
            int indexJ = 0;
            for (int j = 0; j < correlationMap[0].length; j++) {
                if (usedValues2[j]) {
                    indexJ++;
                } else {
                    continue;
                }
                if (correlationMap[i][j]) {
                    grid[indexI][indexJ] = "X";
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
    }

    public void findConstantElements() {
        List<DynAngel> locations = new ArrayList<DynAngel>();

        for (DynAngel dynamicAngel : angelsToValueMap.keySet()) {
            ArrayList<Integer> values = angelsToValueMap.get(dynamicAngel);
            int firstValue = -1;
            boolean isConstant = true;
            for (int value : values) {
                if (value != -1) {
                    if (firstValue == -1) {
                        firstValue = value;
                    } else {
                        if (firstValue != value) {
                            isConstant = false;
                        }
                    }
                }
            }
            if (isConstant) {
                locations.add(dynamicAngel);
            }
        }

        Collections.sort(locations);
        DebugOut.print("Constant elements");
        String locationString = "";
        for (DynAngel location : locations) {
            locationString += ": " + location.staticAngelId + "[" + location.execNum + "]";
        }
        DebugOut.print("Location" + locationString);
    }
}
