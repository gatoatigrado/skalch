package sketch.compiler.parser.gxlimport;

import sketch.compiler.ast.core.Program;
import sketch.compiler.main.seq.SequentialSketchMain;
import sketch.compiler.passes.cuda.CopyCudaMemTypeToFcnReturn;
import sketch.compiler.passes.cuda.GenerateAllOrSomeThreadsFunctions;
import sketch.compiler.passes.cuda.SplitAssignFromVarDef;
import sketch.compiler.passes.lowering.FunctionParamExtension;
import sketch.compiler.passes.preprocessing.ConvertArrayAssignmentsToInout;
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
    protected final GxlSketchOptions args;

    public SequentialSketchGxlMain(final GxlSketchOptions args, final Program prog) {
        super(args);
        this.args = args;
        this.prog = prog;
    }

    public class GxlPreProcStage1 extends PreProcStage1 {
        public GxlPreProcStage1() {
            super();
            this.passes.add(new ConvertArrayAssignmentsToInout());
            this.passes.add(new CopyCudaMemTypeToFcnReturn());
        }
    }

    public class GxlIRStage2 extends IRStage2 {
        public GxlIRStage2() {
            super();
            this.passes.add(new SplitAssignFromVarDef());

            // custom dependencies using old stages
            final GenerateAllOrSomeThreadsFunctions genThreadsFcns =
                    new GenerateAllOrSomeThreadsFunctions(
                            SequentialSketchGxlMain.this.options,
                            SequentialSketchGxlMain.this.varGen);
            final SemanticCheckPass semanticCheck = new SemanticCheckPass();
            final FunctionParamExtension paramExt = new FunctionParamExtension();
            this.passes.add(genThreadsFcns);
            this.passes.add(paramExt);
            this.passes.add(semanticCheck);
            this.stageRequires.append(paramExt, genThreadsFcns);
            this.stageRequires.append(semanticCheck, paramExt);
        }
    }

    @Override
    public PreProcStage1 getPreProcStage1() {
        return new GxlPreProcStage1();
    }

    @Override
    public IRStage2 getIRStage2() {
        return new GxlIRStage2();
    }

    @Override
    public void run() {
        this.log(1, "Benchmark = " + this.benchmarkName());
        this.preprocAndSemanticCheck();
        this.prog.debugDump("After preprocessing");

        this.oracle = new ValueOracle(new StaticHoleTracker(this.varGen));
        this.partialEvalAndSolve();
        this.eliminateStar();

        this.generateCode();
        this.log(1, "[SKETCH] DONE");
    }
}
