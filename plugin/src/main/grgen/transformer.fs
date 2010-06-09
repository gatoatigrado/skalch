module edu.berkeley.cs.grgenmods.fsharp.main
(*
    Copyright 2010 gatoatigrado (nicholas tung) [ntung at ntung]

    Licensed under the Apache License, Version 2.0 (the "License"); you may
    not use this file except in compliance with the License. You may obtain a
    copy of the License at http://www.apache.org/licenses/LICENSE-2.0 .

    Note that the rest of GrGen is licensed under LGPL v.3, so the effective
    license of this library becomes LGPL 3.
*)

open System
open de.unika.ipd.grGen.libGr
open de.unika.ipd.grGen.grShell
(* override GrGen's Set class *)
open Microsoft.FSharp.Collections
open edu.berkeley.cs.grgenmods.fsharp.util_fcns
open edu.berkeley.cs.grgenmods.fsharp.graph
open edu.berkeley.cs.grgenmods.fsharp.cmdline
open edu.berkeley.cs.grgenmods.fsharp.stages
open edu.berkeley.cs.grgenmods.fsharp.dependencies

(* rewrite rules *)
let WarnUnsupportedRules = [Xgrs "unsupportedWarnAll"]

let DeleteMarkedIgnoreRules = [Xgrs "setIgnoreAnnotationType*";
    Xgrs "deleteIgnoreAnnotated* & deleteDangling*"]

let DecorateNodesRules = [Xgrs "replaceAngelicSketchSymbol*";
    Xgrs "runAllSymbolRetypes";
    Xgrs "setStaticAnnotationTypes* & [setOuterSymbol]";
    Xgrs "setScalaRoot & setScalaSubtypes*";
    Xgrs "setSketchClasses*"]

let ConvertThisRules = [Xgrs "setEnclosingFunctionInitial+";
    Xgrs "deleteBridgeFunctions";
    Validate "testNoBridgeFunctions";
    Xgrs "[transformFcnWrapper]";
    Validate "testNoThisNodes";
    Xgrs "removeEnclosingLinks* & deleteDangling*";
    Xgrs "setSketchMainFcn*"]

(*let CleanupAccessorsRules = [Xgrs "[markGetterFcns]";
    Xgrs "markGetterCalls*";
    Xgrs "[setGetterFcnField]";
    Xgrs "deleteUnusedFcnDefs* & deleteDangling*"]*)

let BlockifyFcndefsRules = [Xgrs "removeFcnTarget*";
    Xgrs "(deleteDangling+ | removeNopTypeCast)*";
    Xgrs "createFunctionBlocks* & retypeBlockSKBlock*";
    Xgrs "checkOnlyFcnBlocks"]

let NiceListsRules = [Xgrs "listBlockInit*";
    Xgrs "listClassDefsInit*";
    Xgrs "listInitAllOrdered";
    Xgrs "listAddClassField*";
    Xgrs "listSetNext*";
    Xgrs "listCompleteLast*";
    Xgrs "listCompleteBlockLast*"]

let ProcessAnnotationsRules = [Xgrs "cleanupTmpTypedBlock*";
    Xgrs "deleteAnnotationLink";
    Xgrs "deleteDangling*";
    Xgrs "replacePrimitiveRanges* & decrementUntilValues*";
    Xgrs "markAnnotsWithNewSym*";
    Xgrs "deleteDangling*";
    Validate "! existsDanglingAnnotation"]

let ArrayLoweringRules = [Xgrs "replaceArrayInit+"]

let EmitRequiredImportsRules = [Xgrs "[setEnclosingFunctionInitial]";
    Xgrs "setCalledMethods* & emitRequiredImports* & emitProvides*";
    Xgrs "removeEnclosingLinks* & deleteDangling* & cleanupTmpEdges*"]

let LossyReplacementsRules = [Xgrs "replaceThrowWithAssertFalse*";
    Xgrs "deleteObjectInitCall*";
    Xgrs "retypeWeirdInits*";
    Xgrs "deleteUnitConstants*"]

let NewInitializerFcnStubsRules = [
    Xgrs "markInitializerFunctions+ && createInitializerFunctions+ && replaceConstructors+"]

let CstyleStmtsRules = [Xgrs "deleteDangling*";
    Xgrs "cfgInit";
    Xgrs "cfgSkipIf*";
    Validate "! cfgExistsIncomplete";
    Xgrs "setAttachableMemberFcns*";
    Xgrs "setAttachableBlocks*";
    Xgrs "forwardNonblockifyIntermediatePrologue*";
    Xgrs "setBlockifyNextForAlreadyBlockified*";
    Xgrs "(propagateBlockifyUnsafe+ | propagateBlockifyMarkSafe+)*";
    Xgrs "setBlockifyChain*";
    Xgrs "checkBlockifyLinks";
    Xgrs "forwardBlockifySkip*";
    Xgrs "addDummyBlockifyChainEndNodes*";
    Xgrs "deleteCfgNode*";
    Xgrs "createTemporaryAssign*";
    Xgrs "attachNodesToBlockList*";
    Xgrs "deleteLastAttachables* & deleteLastAttachables2*";
    Validate "testNoBlockify"]

let CstyleAssnsRules = [Xgrs "makeValDefsEmpty*";
    Xgrs "cstyleAssignToIfs+ | cstyleAssignToBlocks+"]

let SketchFinalMinorCleanupRules = [Xgrs "removeEmptyTrees";
    Xgrs "setSymbolNames* & deletePrintRenamer*";
    Xgrs "setSymbolSketchType*";
    Xgrs "setSketchTypeInt & setSketchTypeBoolean & setSketchTypeUnit";
    Xgrs "connectFunctions*";
    Xgrs "removeEmptyChains*";
    Xgrs "setAssertCalls* & deleteDangling*";
    Xgrs "setFcnBinaryCallBaseType*";
    Xgrs "setFcnBinaryCallArgs*";
    Xgrs "setSymbolBaseType*";
    Xgrs "setValDefBaseType*";
    Xgrs "setVarRefBaseType*";
    Xgrs "setFcnDefBaseType*";
    Xgrs "addSkExprStmts*"]



(* rewrite stages *)
let WarnUnsupported = {
    stageDefault with
        name = "WarnUnsupported";
        description = "check for unsupported features";
        stage = RewriteStage WarnUnsupportedRules }

let DeleteMarkedIgnore = {
    stageDefault with
        name = "DeleteMarkedIgnore";
        description = "delete classes marked as ignore";
        stage = RewriteStage DeleteMarkedIgnoreRules }

let DecorateNodes = {
    stageDefault with
        name = "DecorateNodes";
        description = "retype certain skalch constructs and function symbols";
        stage = RewriteStage DecorateNodesRules }

let ConvertThis = {
    stageDefault with
        name = "ConvertThis";
        description = "delete bridge functions, convert $this to a parameter";
        stage = RewriteStage ConvertThisRules }

(*let CleanupAccessors = {
    stageDefault with
        name = "CleanupAccessors";
        description = "";
        stage = RewriteStage CleanupAccessorsRules }*)

let BlockifyFcndefs = {
    stageDefault with
        name = "BlockifyFcndefs";
        description = "Convert function bodies to SKBlock(s)";
        stage = RewriteStage BlockifyFcndefsRules }

let NiceLists = {
    stageDefault with
        name = "NiceLists";
        description = "Convert all lists to nice ('generic') lists";
        stage = RewriteStage NiceListsRules }

let ProcessAnnotations = {
    stageDefault with
        name = "ProcessAnnotations";
        description = "Annotation processing";
        stage = RewriteStage ProcessAnnotationsRules }

let ArrayLowering = {
    stageDefault with
        name = "ArrayLowering";
        description = "Remove sugar from array calls, and retype boxed arrays";
        stage = RewriteStage ArrayLoweringRules }

let EmitRequiredImports = {
    stageDefault with
        name = "EmitRequiredImports";
        description = "print discrete union requirements";
        stage = RewriteStage EmitRequiredImportsRules }

let LossyReplacements = {
    stageDefault with
        name = "LossyReplacements";
        description = "ignore try / catch for now, fix some Scala weirdnesses (bugs?)";
        stage = RewriteStage LossyReplacementsRules }

let NewInitializerFcnStubs = {
    stageDefault with
        name = "NewInitializerFcnStubs";
        description = "create a function that rewrites";
        stage = RewriteStage NewInitializerFcnStubsRules }

let CstyleStmts = {
    stageDefault with
        name = "CstyleStmts";
        description = "move Scala-style expression-statements to blocks";
        stage = RewriteStage CstyleStmtsRules }

let CstyleAssns = {
    stageDefault with
        name = "CstyleAssns";
        description = "replace x = { block; expr } with block; x = expr";
        stage = RewriteStage CstyleAssnsRules }

let SketchFinalMinorCleanup = {
    stageDefault with
        name = "SketchFinalMinorCleanup";
        description = "Final minor cleanup (information loss stage)";
        stage = RewriteStage SketchFinalMinorCleanupRules }

let all_stages = [ WarnUnsupported; DeleteMarkedIgnore; DecorateNodes;
    ConvertThis; BlockifyFcndefs; NiceLists; ProcessAnnotations;
    ArrayLowering; EmitRequiredImports; LossyReplacements; NewInitializerFcnStubs;
    CstyleStmts; CstyleAssns; SketchFinalMinorCleanup]

(* Dependencies. If the "?" side (left or right) is present, the "+" stage is added *)
let deps = [
    WarnUnsupported <?? DecorateNodes;
    DeleteMarkedIgnore <+? DecorateNodes;
    DecorateNodes <+? ConvertThis;
    DecorateNodes <+? BlockifyFcndefs;
    BlockifyFcndefs <?? NiceLists;
    NiceLists <+? ProcessAnnotations;
    NiceLists <+? ArrayLowering;
    NiceLists <+? CstyleStmts;
    ProcessAnnotations <?? ArrayLowering;
    ArrayLowering <?? EmitRequiredImports;
    EmitRequiredImports <?? LossyReplacements;
    LossyReplacements <?? NewInitializerFcnStubs;
    NewInitializerFcnStubs <?? CstyleStmts;
    CstyleStmts <+? CstyleAssns;
    CstyleStmts <?? SketchFinalMinorCleanup
    ]

(* Goals -- sets of stages *)
let sketch = { name = "sketch"; stages=[ WarnUnsupported;
    DeleteMarkedIgnore; DecorateNodes; ConvertThis;
    BlockifyFcndefs; NiceLists; ProcessAnnotations;
    ArrayLowering; EmitRequiredImports; LossyReplacements;
    NewInitializerFcnStubs; CstyleStmts; CstyleAssns;
    SketchFinalMinorCleanup ] }

let all_goals = [sketch]

(* Functions for command line parsing. Exposed so you can add aliases, etc. *)
let goalMap, stageMap = (defaultGoalMap all_goals, defaultStageMap all_stages)

[<EntryPoint>]
let main(args:string[]) =
    (* Basic parsing of the command line *)
    let cmdline = args |> Seq.toList |> parseCommandLine

    (* This transformation system doesn't generate a graph, so we expect an initial import *)
    if cmdline |> List.filter (function | CmdSourceFile _ -> true | _ -> false) |> List.isEmpty then
        failwith "No source graph specified!"

    (* Load the actual stages and dependencies specified by the command line.
     If necessary, it's probably cleaner to modify these later (e.g. if custom dep rules are necessary) *)
    let stages, deps = loadCmdLine goalMap stageMap deps cmdline

    (* Since actions are necessary everywhere, they are not currently regarded as a stage.
     This could be changed in the future if necessary *)
    let initialgraph = new Graph("~/sandbox/skalch/plugin/src/main/grgen/AllRules_0.grg")
    initialgraph.Impl.SetDebugLayout "Compilergraph"
    initialgraph.Impl.SetDebugLayoutOption ("CREATE_LOOP_TREE", "false")
    initialgraph.SetNodeColors [
        "ScalaExprStmt", "lilac";
        "FcnDef", "green";
        "ClassDef", "red";
        "SketchConstructSymbol", "gold";
        "SketchConstructCall", "orange";
        "Symbol", "blue";
        "Annotation", "orchid";
        "TmpSymbol", "LightRed";
        "BlockifyValDef", "LightBlue";
        "TmpVarRef", "LightCyan";
        "CfgAbstractNode", "LightGreen";
        "HighlightValDef", "Black";
        "PrintNode", "DarkBlue";
        "DebugBadNode", "Red"
        ]
    initialgraph.SetEdgeColors [
        "CfgAbstractNext", "DarkGreen"
        "AbstractBlockify", "DarkRed"
        ]

    (* Each step of this loop processes one stage *)
    let rec mainLoop stages deps results graph =
        let nextStages = getNextStages stages deps
        match ((Set.isEmpty stages), nextStages) with
        | (true, []) -> ()

        (* Error case -- no next stages were found, despite having a nonempty stage set *)
        | (_, []) ->
            printfn "\n\n[ERROR] Remaining stages:\n    %s" (printstages "\n    " stages)
            failwith "[ERROR] no next stage -- maybe there is a cyclic dependency?"

        (* Process a stage *)
        | (_, hd :: tail) ->
            let results, graph = processStage hd results graph
            mainLoop (Set.remove hd stages) deps results graph

    mainLoop (getAllStages stages deps) deps [] initialgraph

    0
