package sketch.ui.queues;

public interface QueueIterator {
    boolean checkValue(Object value);

    boolean canFinish();
}
