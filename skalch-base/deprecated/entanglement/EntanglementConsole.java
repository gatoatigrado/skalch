package sketch.ui.entanglement;

import java.util.ArrayList;
import java.util.Collections;
import java.util.HashSet;
import java.util.List;
import java.util.Scanner;
import java.util.Set;
import java.util.StringTokenizer;
import java.util.Vector;

public class EntanglementConsole {

    private ArrayList<Trace> traces;
    private ArrayList<List<Trace>> partitions;
    private EntanglementAnalysis ea;

    public EntanglementConsole(Vector<Trace> traces) {
        this.traces = new ArrayList<Trace>(traces);

        partitions = new ArrayList<List<Trace>>();
        partitions.add(new ArrayList<Trace>(traces));

        ea = new EntanglementAnalysis(this.traces);
    }

    public void startConsole() {
        Scanner in = new Scanner(System.in);
        while (true) {
            System.out.print("<<<");
            boolean matched = false;

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
                } else if ("exit".equals(command)) {
                    break;
                } else {
                    System.out.println("Unknown command: " + command);
                }

            } catch (Exception e) {
                e.printStackTrace();
            }
        }
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
        boolean[][] correlationMap = result.correlationMap;

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
                if (correlationMap[i][j]) {
                    grid[i + 1][j + 1] = "X";
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
            System.out.println("Location: " + dynAngel.staticAngelId + "[" +
                    dynAngel.execNum + "]");

        }
    }
}
