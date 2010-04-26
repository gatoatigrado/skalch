package sketch.entanglement;

import java.util.Set;

public class EntanglementSubsets {
    final public Set<Set<DynAngel>> unentangledSubsets;
    final public Set<Set<DynAngel>> entangledSubsets;

    public EntanglementSubsets(Set<Set<DynAngel>> unentangledSubsets,
            Set<Set<DynAngel>> entangledSubsets)
    {
        this.unentangledSubsets = unentangledSubsets;
        this.entangledSubsets = entangledSubsets;
    }
}
