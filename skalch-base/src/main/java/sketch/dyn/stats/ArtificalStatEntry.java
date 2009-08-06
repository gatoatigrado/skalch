/**
 * 
 */
package sketch.dyn.stats;

public class ArtificalStatEntry extends ScStatEntry {
    public ArtificalStatEntry(float value, String name, String short_name) {
        super(name, short_name);
        this.value = value;
    }

    @Override
    public float get_value() {
        return value;
    }
}