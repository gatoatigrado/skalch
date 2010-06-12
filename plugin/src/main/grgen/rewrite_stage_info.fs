module edu.berkeley.cs.skalch.transformer.rewrite_stage_info
(*
    Copyright 2010 gatoatigrado (nicholas tung) [ntung at ntung]

    Licensed under the Apache License, Version 2.0 (the "License"); you may
    not use this file except in compliance with the License. You may obtain a
    copy of the License at http://www.apache.org/licenses/LICENSE-2.0 .

    Note that the rest of GrGen is licensed under LGPL v.3, so the effective
    license of this library becomes LGPL 3.
*)

open edu.berkeley.cs.skalch.transformer.rewrite_rules
open edu.berkeley.cs.grgenmods.fsharp.stages

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

let RedirectAccessorsToFields = {
    stageDefault with
        name = "RedirectAccessorsToFields";
        description = "";
        stage = RewriteStage RedirectAccessorsToFieldsRules }

let CleanSketchConstructs = {
    stageDefault with
        name = "CleanSketchConstructs";
        description = "replace assert calls, clean calls to !! / ??";
        stage = RewriteStage CleanSketchConstructsRules }

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

let CreateTemplates = {
    stageDefault with
        name = "CreateTemplates";
        description = "transform RewriteTemplates.Template into useable templates"
        stage = RewriteStage CreateTemplatesRules }

let all_stages = [ WarnUnsupported; DeleteMarkedIgnore; DecorateNodes;
    ConvertThis; BlockifyFcndefs; NiceLists; ProcessAnnotations;
    ArrayLowering; EmitRequiredImports; LossyReplacements; NewInitializerFcnStubs;
    CstyleStmts; CstyleAssns; SketchFinalMinorCleanup]
