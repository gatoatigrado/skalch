/**
 * 
 */
package sketch.dyn.stats;

public class ArtificalStatEntry extends ScStatEntry {
    public ArtificalStatEntry(float value, String name, String shortName) {
        super(name, shortName);
        this.value = value;
    }

    @Override
    public float getValue() {
        return value;
    }
}