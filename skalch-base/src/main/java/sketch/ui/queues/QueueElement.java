package sketch.ui.queues;

public class QueueElement {

    final private int queueNum;
    final private Object value;

    public QueueElement(int queueNum, Object value) {
        this.queueNum = queueNum;
        this.value = value;
    }

    public int getQueueNum() {
        return queueNum;
    }

    public Object getValue() {
        return value;
    }

}
