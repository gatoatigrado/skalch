package sketch.util;

import java.util.Vector;

/**
 * Java defaultdict with Vector constructor (b/c java is too verbose...)
 *
 * @author gatoatigrado (nicholas tung) [email: ntung at ntung]
 * @license This file is licensed under BSD license, available at
 *          http://creativecommons.org/licenses/BSD/. While not required, if you make
 *          changes, please consider contributing back!
 */
public class DefaultVectorHashMap<K, T> extends DefaultHashMap<K, Vector<T>> {
    private static final long serialVersionUID = -1714071437364151873L;

    public DefaultVectorHashMap() {
        this.defvalue = new VectorGenerator();
    }

    public class VectorGenerator extends DefValueGenerator {
        @Override
        public Vector<T> getValue() {
            return new Vector<T>();
        }
    }
}
