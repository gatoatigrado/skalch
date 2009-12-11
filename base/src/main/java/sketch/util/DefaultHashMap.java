package sketch.util;

import java.util.HashMap;

/**
 * Java defaultdict
 * @author gatoatigrado (nicholas tung) [email: ntung at ntung]
 * @license This file is licensed under BSD license, available at
 *          http://creativecommons.org/licenses/BSD/. While not required, if you
 *          make changes, please consider contributing back!
 */
public class DefaultHashMap<K, V> extends HashMap<K, V> {
    private static final long serialVersionUID = -8338536913857704341L;
    public DefValueGenerator defvalue;

    @SuppressWarnings("unchecked")
    @Override
    public V get(Object key) {
        V result = super.get(key);
        if (result == null) {
            result = defvalue.get_value();
            super.put((K) key, result);
        }
        return result;
    }

    public abstract class DefValueGenerator {
        public abstract V get_value();
    }
}
