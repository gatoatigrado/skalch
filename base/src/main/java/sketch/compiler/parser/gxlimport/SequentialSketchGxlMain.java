package sketch.compiler.parser.gxlimport;

import sketch.compiler.ast.core.Program;
import sketch.compiler.main.seq.SequentialSketchMain;
import sketch.compiler.solvers.constructs.StaticHoleTracker;
import sketch.compiler.solvers.constructs.ValueOracle;

/**
 * run an existing program object.
 *
 * @author gatoatigrado (nicholas tung) [email: ntung at ntung]
 * @license This file is licensed under BSD license, available at
 *          http://creativecommons.org/licenses/BSD/. While not required, if you make
 *          changes, please consider contributing back!
 */
public class SequentialSketchGxlMain extends SequentialSketchMain {
    public SequentialSketchGxlMain(final String[] args, final Program prog) {
        super(args);
        this.prog = prog;
    }

    @Override
    public void run() {
        this.log(1, "Benchmark = " + this.benchmarkName());
        this.preprocAndSemanticCheck();
        this.prog.debugDump("After preprocessing");

        this.oracle = new ValueOracle(new StaticHoleTracker(this.varGen)/*
                                                                         * new
                                                                         * SequentialHoleTracker
                                                                         * (varGen)
                                                                         */);
        this.partialEvalAndSolve();
        this.eliminateStar();

        this.generateCode();
        this.log(1, "[SKETCH] DONE");
    }
}
