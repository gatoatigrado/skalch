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
open de.unika.ipd.grGen.libGr
open de.unika.ipd.grGen.grShell
(* override GrGen's Set class *)
open Microsoft.FSharp.Collections
open edu.berkeley.cs.grgenmods.fsharp.util_fcns
open edu.berkeley.cs.grgenmods.fsharp.graph
open edu.berkeley.cs.grgenmods.fsharp.cmdline
open edu.berkeley.cs.grgenmods.fsharp.stages
open edu.berkeley.cs.grgenmods.fsharp.dependencies
open edu.berkeley.cs.skalch.transformer.rewrite_rules
open edu.berkeley.cs.skalch.transformer.rewrite_stage_info

module Config =
    let templates = Path("~/.sketch/skalch-templates").create_dir
    let libraries = Path("~/.sketch/skalch-libraries").create_dir
    let template name = templates.subpath (sprintf "%s.gxl.gz" name)
    let library name = libraries.subpath (sprintf "%s.gxl.gz" name)

let create_rewrite_stage name =
    let rewrites = [ Xgrs (sprintf "[IsolateTemplate(\"%s\")]" name);
        Xgrs "[deletePackageDefs] & deleteDangling*";
        Xgrs "[removeToExport]" ]
    { stageDefault with
        name = (sprintf "ExportTemplateInner[%s]" name);
        description = (sprintf "export the template %s" name);
        stage = rewriteStage rewrites }

(* export each template individually. The export functions "destroy" the graph
 for any futher exports, so a graph stack is used to revert stage *)
let export_templates (results:RewriteResult list) graph =
    let exportfcn (results, (graph:Graph)) name =
        let graphstack = graph.Impl.getGraphStack()
        graphstack.PushClone()
        let results, e_g = processStage (create_rewrite_stage name) results graph
        printfn "exporting template %s" (Config.template name).value
        let r_exp = e_g.Export (Config.template name)
        graphstack.Pop() |> ignore
        (r_exp :: results, graph)
    (getAllStageEmit results).Split '\n'
    |> Array.map (fun x -> x.Split())
    |> Array.filter (fun x -> (x.Length = 2) && "Template" = x.[0])
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

let ExportTemplates = {
    stageDefault with
        name = "ExportTemplates"
        description = "Export each template specified by the CreateTemplates stage"
        stage = CustomStage export_templates }

let ImportTemplates = {
    stageDefault with
        name = "ImportTemplates"
        description = "Import any templates, required by the last stage of processing"
        (* run it before others, since it uses the last emitout values. *)
        priority = -0.1f
        stage = CustomStage import_templates }

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

(* metastages *)
let innocuous_meta = [ (*SetSymbolLabels*) DeleteMarkedIgnore; WarnUnsupported ]
let no_oo_meta = [ ConvertThis ]
let optimize_meta = [ ArrayLowering; SSAForm ]
let sketch_meta = [ ProcessAnnotations; LossyReplacements;
    CleanSketchConstructs; SketchFinalMinorCleanup; SketchNospec ]
let cstyle_meta = [ RaiseSpecialGotos; NewInitializerFcnStubs; BlockifyFcndefs; CstyleMain ]
let library_meta = [ EmitRequiredImports ]
let create_templates_meta = [ NiceLists; CreateTemplates; ExportTemplates;
    DecorateNodes; CleanSketchConstructs; RedirectAccessorsToFields ]

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
        SimplifyConstants
        RaiseSpecialGotos
        ProcessAnnotations
        ArrayLowering
        CreateTemplates
        ExportTemplates
        SketchNospec
        LossyReplacements
        EmitRequiredImports
        NewInitializerFcnStubs ]
let zone_ssa =
    SSAForm,
    [
        ]
let zone_cstyle =
    CstyleMain,
    [
        SketchFinalMinorCleanup ]

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

        (* most later stages require decorate... *)
        DecorateNodes <+? NiceLists

        (* zone_decorated intradependencies *)
        RedirectAccessorsToFields <?? BlockifyFcndefs
        RedirectAccessorsToFields <?? ConvertThis

        (* zone_nice_lists intradependencies *)
        ProcessAnnotations <?? ArrayLowering
        ProcessAnnotations <?? LossyReplacements
        ArrayLowering <?? EmitRequiredImports
        CreateTemplates <?? ExportTemplates
        EmitRequiredImports <?? SketchNospec
        LossyReplacements <?? EmitRequiredImports
        LossyReplacements <?? NewInitializerFcnStubs
    ]

(* Goals -- sets of stages *)
let create_templates = { name = "create_templates";
    stages=innocuous_meta @ create_templates_meta }
let test = { name = "test"; stages=innocuous_meta @ [NiceLists] }
let sketch = { name = "sketch";
    stages=innocuous_meta @ no_oo_meta @ optimize_meta @
        sketch_meta @ cstyle_meta @ library_meta }
let all_goals = [create_templates; sketch; test]
let all_stages = List.fold (fun x y -> x @ (y.stages)) [] all_goals

(* Functions for command line parsing. Exposed so you can add aliases, etc. *)
let goalMap, stageMap = (defaultGoalMap all_goals, defaultStageMap all_stages)

[<EntryPoint>]
let transformerMain(args:string[]) =
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
    let initialgraph = new Graph("~/sandbox/skalch/plugin/src/main/grgen/AllRules_0.grg")
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
        ]
    initialgraph.SetNodeLabel ""  "ListAbstractNode"
    initialgraph.SetNodeShape "circle" "ListAbstractNode"
    List.iter (initialgraph.SetEdgeLabel "") [ "ListElt"; "ListNext"; "ListValue" ]

    (* Each step of this loop processes one stage *)
    let rec mainLoop (stages:StageSet) deps results graph =
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
            let nextset = (Set.remove hd stages)
            (stages.Count - nextset.Count) = 1 |> assert1
            mainLoop (Set.remove hd stages) deps results graph

    mainLoop stages deps [] initialgraph

    0
