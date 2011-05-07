package sketch.entanglement.console;

import java.util.HashMap;

public class SparseMatrix<T> {

    class ColValue {
        private int col;
        private T value;

        public ColValue(int col, T value) {
            this.col = col;
            this.value = value;
        }
    }

    class Row {

        private HashMap<Integer, T> cols;

        public Row() {
            cols = new HashMap<Integer, T>();
        }

        public void put(int c, T val) {
            cols.put(c, val);
        }

        public T get(int c) {
            if (cols.containsKey(c)) {
                return cols.get(c);
            }
            return null;
        }
    }

    private int numRows;
    private int numCols;
    private HashMap<Integer, Row> rows;

    public SparseMatrix(int rows, int cols) {
        numRows = rows;
        numCols = cols;
        this.rows = new HashMap<Integer, Row>();
    }

    public void put(int r, int c, T val) {
        if (r < 0 || r >= numRows || c < 0 || c >= numCols) {
            return;
        }
        Row row = null;
        if (rows.containsKey(r)) {
            row = rows.get(r);
        } else {
            row = new Row();
            rows.put(r, row);
        }
        row.put(c, val);
    }

    public T get(int r, int c) {
        if (rows.containsKey(r)) {
            Row row = rows.get(r);
            return row.get(c);
        }
        return null;
    }

    public int getNumRows() {
        return numRows;
    }

    public int getNumCols() {
        return numCols;
    }
}
