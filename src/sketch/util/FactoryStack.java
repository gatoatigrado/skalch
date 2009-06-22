package sketch.util;

/**
 * stack whose objects are always allocated using a factory
 * @author gatoatigrado (nicholas tung) [email: ntung at ntung]
 * @license This file is licensed under BSD license, available at
 *          http://creativecommons.org/licenses/BSD/. While not required, if you
 *          make changes, please consider contributing back!
 */
public class FactoryStack<T extends ScCloneable<T>> {
    public Object[] array;
    public int next = 0;
    public ObjectFactory<T> factory;

    public FactoryStack(int initial_size, ObjectFactory<T> factory) {
        this(initial_size, factory, false);
    }

    protected FactoryStack(int initial_size, ObjectFactory<T> factory,
            boolean no_create)
    {
        this.factory = factory;
        initial_size = Math.max(1, initial_size);

        array = new Object[initial_size];
        if (!no_create) {
            for (int a = 0; a < array.length; a++) {
                array[a] = factory.create();
            }
        }
    }

    /** get the next object, which supposedly has a set() method */
    @SuppressWarnings("unchecked")
    public T push() {
        if (next >= array.length) {
            realloc();
        }
        T result = (T) array[next];
        next += 1;
        return result;
    }

    @SuppressWarnings("unchecked")
    public T pop() {
        next -= 1;
        return (T) array[next];
    }

    @SuppressWarnings("unchecked")
    public T peek() {
        return (T) array[next - 1];
    }

    public int size() {
        return next;
    }

    @SuppressWarnings("unchecked")
    public T get(int a) {
        return (T) array[a];
    }

    @SuppressWarnings("unchecked")
    public FactoryStack<T> clone() {
        FactoryStack<T> result =
                new FactoryStack<T>(array.length, factory, true);
        for (int a = 0; a < next; a++) {
            result.array[a] = ((T) array[a]).clone();
        }
        for (int a = next; a < array.length; a++) {
            result.array[a] = factory.create();
        }
        return result;
    }

    protected void realloc() {
        Object[] next_array = new Object[2 * array.length];
        System.arraycopy(array, 0, next_array, 0, array.length);
        for (int a = array.length; a < next_array.length; a++) {
            next_array[a] = factory.create();
        }
    }
}
