package sketch.entanglement;

import java.util.List;
import java.util.Set;

public class EntanglementComparison {

    final public Set<DynAngel> proj1, proj2;
    final public List<Trace> proj1Values, proj2Values;
    final public int[][] correlationMap;
    final public boolean isEntangled;

    public EntanglementComparison(Set<DynAngel> proj1, Set<DynAngel> proj2,
            List<Trace> proj1Values, List<Trace> proj2Values, int[][] correlationMap)
    {
        super();
        this.proj1 = proj1;
        this.proj2 = proj2;
        this.proj1Values = proj1Values;
        this.proj2Values = proj2Values;
        this.correlationMap = correlationMap;

        isEntangled = checkEntanglement(correlationMap);
    }

    private boolean checkEntanglement(int[][] correlationMap2) {
        for (int i = 0; i < correlationMap.length; i++) {
            for (int j = 0; j < correlationMap[0].length; j++) {
                if (correlationMap[i][j] == 0) {
                    return true;
                }
            }
        }
        return false;
    }

}
