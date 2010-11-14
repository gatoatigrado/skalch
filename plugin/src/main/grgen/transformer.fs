module edu.berkeley.cs.skalch.transformer.main
(*
    Copyright 2010 gatoatigrado (nicholas tung) [ntung at ntung]

    Licensed under the Apache License, Version 2.0 (the "License"); you may
    not use this file except in compliance with the License. You may obtain a
    copy of the License at http://www.apache.org/licenses/LICENSE-2.0 .

    Note that the rest of GrGen is licensed under LGPL v.3, so the effective
    license of this library becomes LGPL 3.
*)

open System
open System.Windows.Forms
open de.unika.ipd.grGen.libGr
open de.unika.ipd.grGen.grShell
(* override GrGen's Set class *)
open Microsoft.FSharp.Collections
open edu.berkeley.cs.grgenmods.fsharp.util_fcns
open edu.berkeley.cs.grgenmods.fsharp.graph
open edu.berkeley.cs.grgenmods.fsharp.cmdline
open edu.berkeley.cs.grgenmods.fsharp.stages
(*--------------------------------------------------
* open edu.berkeley.cs.grgenmods.fsharp.modular_compile_rules
*--------------------------------------------------*)
open edu.berkeley.cs.grgenmods.fsharp.dependencies
open edu.berkeley.cs.skalch.transformer.rewrite_rules
open edu.berkeley.cs.skalch.transformer.rewrite_stage_info

module Config =
    let templates = Path("~/.sketch/skalch-templates").create_dir
    let libraries = Path("~/.sketch/skalch-libraries").create_dir
    let template name = templates.subpath (sprintf "%s.gxl.gz" name)
    let library name = libraries.subpath (sprintf "%s.gxl.gz" name)



(* specialized stages for IO, not just rewrites *)
let create_template_rewrite_stage name =
    let rewrites = [ Xgrs (sprintf "[IsolateTemplate(\"%s\")]" name);
        Xgrs "[deletePackageDefs] & deleteDangling*";
        Xgrs "[removeToExport]" ]
    { stageDefault with
        name = (sprintf "ExportTemplateInner[%s]" name);
        description = (sprintf "export the template %s" name);
        stage = rewriteStage rewrites }

let create_libraries_rewrite_stage name =
    let rewrites = [ Xgrs (sprintf "[IsolateClassDef(\"%s\")]" name);
        Xgrs "[deletePackageDefs] & deleteDangling*";
        Xgrs "[removeToExport]";
        DebugXgrs "deleteDangling" ]
    { stageDefault with
        name = (sprintf "ExportLibraryInner[%s]" name);
        description = (sprintf "add the class %s to known libraries" name);
        stage = rewriteStage rewrites }

(* export each template individually. The export functions "destroy" the graph
 for any futher exports, so a graph stack is used to revert stage *)
let export_templates (results:RewriteResult list) graph =
    let exportfcn (results, (graph:Graph)) name =
        let graphstack = graph.Impl.GetGraphStack()
        graphstack.PushClone()
        let results, e_g = processStage (create_template_rewrite_stage name) results graph
        printfn "exporting template %s" (Config.template name).value
        let r_exp = e_g.Export (Config.template name)
        graphstack.Pop() |> ignore
        (r_exp :: results, graph)
    (getAllStageEmit results).Split '\n'
    |> Array.map (fun x -> x.Split())
    |> Array.filter (fun x -> (x.Length = 2) && "Template" = x.[0])
    |> Array.map (fun x -> x.[1])
    |> Array.fold exportfcn (results, graph)

(* TODO -- this is very stupidly duplicated code, I just don't want to re-refactor
 * the above at the moment *)
let export_libraries (results:RewriteResult list) graph =
    let exportfcn (results, (graph:Graph)) name =
        let graphstack = graph.Impl.GetGraphStack()
        graphstack.PushClone()
        let results, e_g = processStage (create_libraries_rewrite_stage name) results graph
        printfn "exporting class to library file '%s'" (Config.library name).value
        let r_exp = e_g.Export (Config.library name)
        graphstack.Pop() |> ignore
        (r_exp :: results, graph)
    (getAllStageEmit results).Split '\n'
    |> Array.map (fun x -> x.Split())
    |> Array.filter (fun x -> (x.Length = 2) && "Library" = x.[0])
    |> Array.map (fun x -> x.[1])
    |> Array.fold exportfcn (results, graph)

let import_templates (results:RewriteResult list) (graph:Graph) =
    let importfcn (results, (graph:Graph)) name =
        printfn "importing template %s" (Config.template name).value
        let r_exp = graph.ImportDUnion (Config.template name)
        (r_exp :: results, graph)
    (getLastStageEmit results).Split '\n'
    |> Array.map (fun x -> x.Split())
    |> Array.filter (fun x -> (x.Length = 3) && ([|"Request"; "template"|]) = x.[0 .. 1])
    |> Array.map (fun x -> x.[2])
    |> Array.fold importfcn (results, graph)

let resolve_gt (results:RewriteResult list) (graph:Graph) =
    let processRewrite = (fun x -> processRewrite false graph (Xgrs x)) >> (fun (x, y) -> x)
    let getGTInst instname basename () =
        let rewrites = [
            Xgrs (sprintf "isolateGT(\"%s\", \"%s\")" instname basename)
            Xgrs "deleteDangling*"
            Xgrs "emitGTParamsAndValues" ]
        let r, _ = processRewriteLst graph rewrites

        graph.Debug true
        r, graph.CurrentGraph

    let r2 = processRewrite "deleteGTInstanceTypeDefinition* && deleteDangling*"
    let results = r2 :: results

    RegexMatchAll "TemplateInstance ([\\s\\w$\\.\\_]+) of ([\\s\\w$\\.\\_]+)\\n" r2.emitout
    |> List.fold (fun (results, retv) ({groups=g}) -> 
        let r3, instgraph = graph.withstack(getGTInst g.[0] g.[1])
        (r3 @ results, (g.[0], g.[1], instgraph) :: retv)) (results, [])
    |> fun (results, _) -> results, graph

let ExportTemplates = {
    stageDefault with
        name = "ExportTemplates"
        description = "Export each template specified by the CreateTemplates stage"
        stage = CustomStage export_templates }

let ExportLibraries = {
    stageDefault with
        name = "ExportLibraries"
        description = "Export each class specified by the CreateTemplates stage"
        stage = CustomStage export_libraries }

let ImportTemplates = {
    stageDefault with
        name = "ImportTemplates"
        description = "Import any templates, required by the last stage of processing"
        (* run it before others, since it uses the last emitout values. *)
        priority = -0.1f
        stage = CustomStage import_templates }

let ResolveGT = {
    stageDefault with
        name = "ResolveGT"
        description = "Logic stage of specializing generics (i.e. using them as C++ templates)"
        stage = CustomStage resolve_gt }

let ProcessAnnotations = {
    stageDefault with
        name = "ProcessAnnotations"
        description = "(metastage) process annotations, esp. those for sketch constructs"
        stage = MetaStage [ProcessAnnotations1; ImportTemplates; ProcessAnnotations2] }

let CstyleMain = {
    stageDefault with
        name = "CstyleMain"
        description = "convert AST to C-style syntax (e.g. if statements aren't expressions)"
        stage = MetaStage [CstyleStmts; CstyleAssns; CstyleMinorCleanup] }



(* zones -- abstraction of dependencies
 read concrete dependencies below first

 For each zone, nodes in the list require the first tuple element,
 and are ordered before all next zones.

 In general, graphs in a zone will have similar types (nodes / edges) *)
let zone_initial = [
    DeleteMarkedIgnore
    WarnUnsupported ]
let zone_decorated =
    DecorateNodes,
    [
        CleanSketchConstructs
        RedirectAccessorsToFields
        ConvertThis
        BlockifyFcndefs ]
let zone_nice_lists =
    NiceLists,
    [
        LowerTprint
        SimplifyConstants
        RaiseSpecialGotos
        ProcessAnnotations
        ResolveGT
        CreateTemplates
        ExportTemplates
        CreateLibraries
        ExportLibraries
        LossyReplacements
        NewInitializerFcnStubs
        RewriteObjects
        CreateSpecialCudaNodesForSketch
        SpecializeCudaFcnCalls
        ]
let zone_ssa =
    nullStage "ssa",
    [
        ArrayLowering
        ConvertVLArraysToFixed
        EmitRequiredImports
        SketchNospec
        ]
let zone_cmemtypes =
    CMemTypeValueOrReference,
    [
        CMemTypes
        SketchCudaMemTypes
        ]
(*let zone_ll_ssa =
    LowerPhiFunctions, []*)
let zone_cstyle =
    CstyleMain,
    [
        SketchFinalMinorCleanup
        CudaCleanup
        CudaGenerateCode ]

let rec depsFromZones (prevDeps:NamedStage list) = function
    | [] -> []
    | ((init:NamedStage), (deps:NamedStage list)) :: tail ->
        (List.map ((<+?) init) deps) @
        (List.map (fun x -> x <?? init) prevDeps) @
        (depsFromZones (prevDeps @ (init :: deps)) tail)
let zone_deps =
    depsFromZones zone_initial [
        zone_decorated
        zone_nice_lists
        zone_ssa
        zone_cmemtypes 
        zone_cstyle ]

(* Dependencies. If the "?" side (left or right) is present, the "+" stage is added *)
let deps =
    zone_deps @
    [
        WarnUnsupported <?? DecorateNodes
        DecorateNodes <+? CreateTemplates (* unique sym names *)
        DeleteMarkedIgnore <+? CreateTemplates

        (* interzone dependencies *)
        BlockifyFcndefs <+? CreateTemplates
        BlockifyFcndefs <?? CleanSketchConstructs
        SimplifyConstants <+? ProcessAnnotations
        ProcessAnnotations <+? CstyleMain
        ProcessAnnotations <+? ResolveGT
        ArrayLowering <+? CMemTypes
        CMemTypes <+? SketchCudaMemTypes

        (* most later stages require decorate... *)
        DecorateNodes <+? NiceLists

        (* zone_decorated intradependencies *)
        RedirectAccessorsToFields <?? BlockifyFcndefs
        RedirectAccessorsToFields <?? ConvertThis

        (* zone_nice_lists intradependencies *)
        ProcessAnnotations <?? LossyReplacements
        CreateTemplates <?? ExportTemplates
        CreateLibraries <?? ExportLibraries
        LossyReplacements <?? NewInitializerFcnStubs
        CreateSpecialCudaNodesForSketch <?? SpecializeCudaFcnCalls

        (* SSA form intradependencies *)
        ArrayLowering <?? EmitRequiredImports
        ArrayLowering <?? ConvertVLArraysToFixed
        EmitRequiredImports <?? SketchNospec

        (* Code generation should be last *)
        CudaCleanup <?? CudaGenerateCode
    ]



(* metastages *)
let innocuous_meta = [ (*SetSymbolLabels*) DeleteMarkedIgnore; WarnUnsupported ]
let no_oo_meta = [ ConvertThis ]
(* add ssa to below *)
let optimize_meta = [ ArrayLowering ]
let sketch_base_meta = [ CleanSketchConstructs; DecorateNodes;
    RewriteObjects; SpecializeCudaFcnCalls;
    LossyReplacements; LowerTprint; NiceLists (*; ResolveGT *) ]
let sketch_meta = sketch_base_meta @ [ ProcessAnnotations;
    SketchFinalMinorCleanup; SketchNospec;
    ConvertVLArraysToFixed;
    SketchCudaMemTypes;
    CMemTypeValueOrReference;
    CreateSpecialCudaNodesForSketch ]
let cuda_meta = sketch_base_meta @ [ ProcessAnnotations;
    CMemTypes; CudaCleanup; CudaGenerateCode ]
let cstyle_meta = [ RaiseSpecialGotos; NewInitializerFcnStubs;
    BlockifyFcndefs; CstyleMain ]
let library_meta = [ EmitRequiredImports ]
let create_templates_meta = sketch_base_meta @ [ CreateTemplates;
    ExportTemplates;
    DecorateNodes; CleanSketchConstructs; RedirectAccessorsToFields ]
let create_libraries_meta = sketch_base_meta @ [ CreateLibraries;
    ExportLibraries;
    DecorateNodes; CleanSketchConstructs; RedirectAccessorsToFields ]

(* Goals -- sets of stages *)
let create_templates = { name = "create_templates";
    stages=innocuous_meta @ create_templates_meta }
let create_libraries = { name = "create_libraries";
    stages=innocuous_meta @ create_libraries_meta }
let test = { name = "test"; stages=innocuous_meta @ [NiceLists] }
let sketch = { name = "sketch";
    stages=innocuous_meta @ no_oo_meta @ optimize_meta @
        sketch_meta @ cstyle_meta @ library_meta }
let cuda = { name = "cuda";
    stages=innocuous_meta @ no_oo_meta @ optimize_meta @
        cstyle_meta @ library_meta @ cuda_meta }

let all_goals = [create_templates; create_libraries; sketch; test; cuda]
let all_stages = List.fold (fun x y -> x @ (y.stages)) [] all_goals

(* Functions for command line parsing. Exposed so you can add aliases, etc. *)
let goalMap, stageMap = (defaultGoalMap all_goals, defaultStageMap all_stages)



[<EntryPoint>]
let transformerMain(args : string[]) =
    (* first, compile the actions *)
    let grgendir = (Path Application.ExecutablePath).parent
    let grgenmodel = grgendir.subpath("ScalaAstModel.gm")
    let grgenAllRules = grgendir.subpath("AllRules_0.grg")
    assert grgenAllRules.isfile
    let rulesPath = grgendir.subpath("rules").subpath("gen")
    (*--------------------------------------------------
    * modular compilation extension not yet ready
    * let xgrsActionLoad = compileRules rulesPath grgenmodel all_stages true
    *--------------------------------------------------*)

    (* Basic parsing of the command line *)
    let cmdline = args |> Seq.toList |> parseCommandLine

    (* This transformation system doesn't generate a graph, so we expect an initial import *)
    if cmdline |> List.filter (function | CmdSourceFile _ -> true | _ -> false) |> List.isEmpty then
        uifail "No source graph specified!"

    (* Load the actual stages and dependencies specified by the command line.
     If necessary, it's probably cleaner to modify these later (e.g. if custom dep rules are necessary) *)
    let stages, deps = loadCmdLine goalMap stageMap deps cmdline

    (* Since actions are necessary everywhere, they are not currently regarded as a stage.
     This could be changed in the future if necessary *)

    (*--------------------------------------------------
    * modular compilation extension not yet ready
    * let initialgraph = new Graph(grgenmodel, xgrsActionLoad)
    *--------------------------------------------------*)

    let initialgraph = new Graph(grgenAllRules.value, Some("[nameAstNodeMain]"))
    initialgraph.Impl.SetDebugLayout "Compilergraph"
    initialgraph.Impl.SetDebugLayoutOption ("CREATE_LOOP_TREE", "false")
    initialgraph.SetNodeColors [
        "ScalaExprStmt", "lilac"
        "FcnDef", "green"
        "ClassDef", "red"
        "SketchConstructSymbol", "gold"
        "SketchConstructCall", "orange"
        "Symbol", "blue"
        "Annotation", "orchid"
        "TmpSymbol", "LightRed"
        "BlockifyValDef", "LightBlue"
        "TmpVarRef", "LightCyan"
        "CfgAbstractNode", "LightGreen"
        "PrintNode", "DarkBlue"
        "DebugBadNode", "Red"
        "List", "Grey"
        "ListNode", "Grey"
        "ListFirstNode", "LightGrey"
        "ListLastNode", "LightGrey"
        "SKWhileLoop", "DarkBlue"
        "CfgAssign", "LightRed"
        ]

    initialgraph.SetEdgeColors [
        "CfgAbstractNext", "DarkGreen"
        "AbstractBlockify", "DarkRed"
        "ListElt", "LightBlue"
        "ListValue", "Lilac"
        "ScTermSymbol", "gold"
        "ScTypeSymbol", "khaki"
        "CfgPrologue", "black"
        "CfgAssignThisStep", "LightRed"
        "CfgAssignPrevStep", "orange"
        "CfgAssignPossibleSymbol", "LightRed"
        "StringRep", "green"
        "MTypeAbstractEdge", "orange"
        "SketchType", "red"
        ]
    initialgraph.SetNodeLabel ""  "ListAbstractNode"
    initialgraph.SetNodeShape "circle" "ListAbstractNode"
    initialgraph.SetNodeNamesFromAttr "symbolName" "Symbol"
    initialgraph.SetNodeNamesFromAttr "typename" "Annotation"
    initialgraph.SetNodeNamesFromAttr "label" "ScAstNode"
    List.map (initialgraph.SetNodeNamesFromAttr "value") [ "BooleanConstant"; "CharConstant"; "LongConstant"; "IntConstant"; "StringConstant"; "NStringRepConstant" ] |> ignore
    List.iter (initialgraph.SetEdgeLabel "") [ "ListElt"; "ListNext"; "ListValue" ]

    (* Each step of this loop processes one stage *)
    let rec mainLoop (stages:StageSet) deps results graph =
        let nextStages = getNextStages stages deps
        match ((Set.isEmpty stages), nextStages) with
        | (true, []) -> ()

        (* Error case -- no next stages were found, despite having a nonempty stage set *)
        | (_, []) ->
            printfn "\n\n[ERROR] Remaining stages:\n    %s" (printstagedeps "\n    " stages deps)
            failwith "[ERROR] no next stage -- maybe there is a cyclic dependency?"

        (* Process a stage *)
        | (_, hd :: tail) ->
            let results, graph = processStage hd results graph
            let nextset = (Set.remove hd stages)
            (stages.Count - nextset.Count) = 1 |> assert1
            mainLoop (Set.remove hd stages) deps results graph

    mainLoop stages deps [] initialgraph

    0
