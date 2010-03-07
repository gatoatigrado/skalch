package sketch.dyn.main;

import java.util.Vector;

import sketch.dyn.constructs.ctrls.ScCtrlConf;
import sketch.dyn.constructs.inputs.ScInputConf;
import sketch.ui.queues.QueueIterator;
import sketch.ui.sourcecode.ScConstructValueString;
import sketch.ui.sourcecode.ScNoValueStringException;
import sketch.ui.sourcecode.ScSourceConstruct;

/**
 * provide common functionality (run, etc.) for sketches; mostly a way to abstract how
 * counterexamples are provided for now.
 * 
 * @author gatoatigrado (nicholas tung) [email: ntung at ntung]
 * @license This file is licensed under BSD license, available at
 *          http://creativecommons.org/licenses/BSD/. While not required, if you make
 *          changes, please consider contributing back!
 */
public interface ScDynamicSketchCall<T> {
    public int getNumCounterexamples();

    public void initializeBeforeAllTests(ScCtrlConf ctrl_conf, ScInputConf oracle_conf,
            QueueIterator queueIterator);

    public boolean runTest(int idx);

    public int getSolutionCost();

    public T getSketch();

    public ScConstructValueString getHoleValueString(int uid)
            throws ScNoValueStringException;

    public Vector<ScConstructValueString> getOracleValueString(int uid)
            throws ScNoValueStringException;

    public void addSourceInfo(ScSourceConstruct info);

}
