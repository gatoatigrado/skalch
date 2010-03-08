package sketch.util;

import java.util.HashMap;

/**
 * Java defaultdict
 *
 * @author gatoatigrado (nicholas tung) [email: ntung at ntung]
 * @license This file is licensed under BSD license, available at
 *          http://creativecommons.org/licenses/BSD/. While not required, if you make
 *          changes, please consider contributing back!
 */
public class DefaultHashMap<K, V> {
    HashMap<K, V> base = new HashMap<K, V>();
    private static final long serialVersionUID = -8338536913857704341L;
    public DefValueGenerator defvalue;

    public V get(final K key) {
        V result = this.base.get(key);
        if (result == null) {
            result = this.defvalue.getValue();
            if (result == null) {
                DebugOut.assertFalse("defvalue() gave a null value");
            }
            this.base.put(key, result);
        }
        if (result != this.base.get(key)) {
            DebugOut.assertFalse("didn't correctly retrieve same keyed object.");
        }
        return result;
    }

    public V put(final K key, final V value) {
        this.base.put(key, value);
        return value;
    }

    public abstract class DefValueGenerator {
        public abstract V getValue();
    }
}
