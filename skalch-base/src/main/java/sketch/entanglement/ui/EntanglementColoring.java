package sketch.entanglement.ui;

import java.awt.Color;
import java.util.HashMap;
import java.util.Map;
import java.util.Set;
import java.util.Map.Entry;

import sketch.entanglement.DynAngel;
import sketch.entanglement.SimpleEntanglementAnalysis;
import sketch.entanglement.sat.SATEntanglementAnalysis;

public class EntanglementColoring {
    private SimpleEntanglementAnalysis ea;
    private SATEntanglementAnalysis satEA;

    public EntanglementColoring(SimpleEntanglementAnalysis ea,
            SATEntanglementAnalysis satEA)
    {
        this.ea = ea;
        this.satEA = satEA;
    }

    public Color[][] getColorMatrix() {
        int[][] partitionMatrix = getPartitionMatrix();

        // find the max partition number and which partitions only have one member
        HashMap<Integer, Integer> partitions = new HashMap<Integer, Integer>();

        for (int i = 0; i < partitionMatrix.length; i++) {
            for (int j = 0; j < partitionMatrix[i].length; j++) {
                int val = partitionMatrix[i][j];
                if (partitions.containsKey(val)) {
                    partitions.put(val, partitions.get(val) + 1);
                } else {
                    partitions.put(val, 1);
                }
            }
        }

        // give each partition a color
        int numLargePartitions = 0;
        for (Entry<Integer, Integer> entry : partitions.entrySet()) {
            if (entry.getValue() > 1) {
                numLargePartitions++;
            }
        }

        Map<Integer, Color> colors = new HashMap<Integer, Color>();

        int numSeen = 0;
        for (Entry<Integer, Integer> entry : partitions.entrySet()) {
            if (entry.getValue() == 1) {
                colors.put(entry.getKey(), null);
            } else {
                Color color =
                        Color.getHSBColor(numSeen * 1.0f / numLargePartitions, .7f, .7f);
                colors.put(entry.getKey(), color);
                numSeen += 1;
            }
        }

        Color[][] color = new Color[partitionMatrix.length][];
        for (int i = 0; i < color.length; i++) {
            color[i] = new Color[partitionMatrix[i].length];
            for (int j = 0; j < color[i].length; j++) {
                color[i][j] = colors.get(partitionMatrix[i][j]);
            }
        }
        return color;
    }

    private int[][] getPartitionMatrix() {
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

        int[][] partitionMatrix = new int[maxStaticAngel + 1][];
        for (int i = 0; i < partitionMatrix.length; i++) {
            partitionMatrix[i] = new int[maxExecNum[i] + 1];
        }

        int val = 1;
        int inc = 1;
        Set<Set<DynAngel>> partitions = satEA.getEntangledPartitions();
        for (Set<DynAngel> partition : partitions) {
            for (DynAngel d : partition) {
                partitionMatrix[d.staticAngelId][d.execNum] = val;
            }
            val += inc;
        }
        return partitionMatrix;
    }

}
