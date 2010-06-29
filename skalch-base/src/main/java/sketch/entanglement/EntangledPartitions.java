package sketch.entanglement;

import java.util.Set;

public class EntangledPartitions {
    final public Set<Set<DynAngel>> unentangledSubsets;
    final public Set<Set<DynAngel>> entangledSubsets;

    public EntangledPartitions(Set<Set<DynAngel>> unentangledSubsets,
            Set<Set<DynAngel>> entangledSubsets)
    {
        this.unentangledSubsets = unentangledSubsets;
        this.entangledSubsets = entangledSubsets;
    }
}
