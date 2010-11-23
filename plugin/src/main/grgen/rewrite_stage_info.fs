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

let rewriteStage rules = RewriteStage (rules, false)

let WarnUnsupported = {
    stageDefault with
        name = "WarnUnsupported";
        description = "Check for unsupported features";
        stage = rewriteStage WarnUnsupportedRules }

let DeleteMarkedIgnore = {
    stageDefault with
        name = "DeleteMarkedIgnore";
        description = "Delete classes marked as ignore";
        stage = rewriteStage DeleteMarkedIgnoreRules }

let DecorateNodes = {
    stageDefault with
        name = "DecorateNodes";
        description = "Retype certain skalch constructs and function symbols";
        stage = rewriteStage DecorateNodesRules }

let DeleteFcnTarget = {
    stageDefault with
        name = "DeleteFcnTarget";
        description = "Change function targets to arguments";
        stage = rewriteStage DeleteFcnTargetRules }

let ConvertThis = {
    stageDefault with
        name = "ConvertThis";
        description = "Delete bridge functions, convert $this to a parameter";
        stage = rewriteStage ConvertThisRules }

let RedirectAccessorsToFields = {
    stageDefault with
        name = "RedirectAccessorsToFields";
        description = "Redirect obj.field() to obj.field; useful for template parameters";
        stage = rewriteStage RedirectAccessorsToFieldsRules }

let CleanSketchConstructs = {
    stageDefault with
        name = "CleanSketchConstructs";
        description = "Replace assert calls, clean calls to !! / ??";
        stage = rewriteStage CleanSketchConstructsRules }

let BlockifyFcndefs = {
    stageDefault with
        name = "BlockifyFcndefs";
        description = "Function target -> arg, convert function bodies to SKBlock(s)";
        stage = rewriteStage BlockifyFcndefsRules }

let NiceLists = {
    stageDefault with
        name = "NiceLists";
        description = "Convert all lists to nice ('generic') lists";
        stage = rewriteStage NiceListsRules }

let SimplifyConstants = {
    stageDefault with
        name = "SimplifyConstants"
        description = "Simplify any constants Scala hasn't"
        stage = rewriteStage SimplifyConstantsRules }

let LowerTprint = {
    stageDefault with
        name = "LowerTprint"
        description = "Desugar tprint() calls (test printout commands)"
        stage = rewriteStage LowerTprintRules }

let RaiseSpecialGotos = {
    stageDefault with
        name = "RaiseSpecialGotos"
        description = "Convert certain label-goto patterns into familiar for loops"
        stage = rewriteStage RaiseSpecialGotosRules }

let ProcessAnnotations1 = {
    stageDefault with
        name = "ProcessAnnotations1"
        description = "Array / hole / retype annotation pre-processing"
        stage = rewriteStage ProcessAnnotationsRules1 }

let ProcessAnnotations2 = {
    stageDefault with
        name = "ProcessAnnotations2"
        description = "Replace annotations with their literal"
        stage = rewriteStage ProcessAnnotationsRules2 }

let ResolveTemplates = {
    stageDefault with
        name = "ResolveTemplates"
        description = "Use annotated generic classes as C++ templates"
        stage = rewriteStage ResolveTemplatesRules }

let EmitRequiredImports = {
    stageDefault with
        name = "EmitRequiredImports";
        description = "Print discrete union requirements";
        stage = rewriteStage EmitRequiredImportsRules }

let SketchNospec = {
    stageDefault with
        name = "SketchNospec"
        description = "Create nospec functions for the sketch"
        stage = rewriteStage SketchNospecRules }

let LossyReplacements = {
    stageDefault with
        name = "LossyReplacements";
        description = "Ignore try / catch for now, fix some Scala weirdnesses (bugs?)";
        stage = rewriteStage LossyReplacementsRules }

let RewriteObjects = {
    stageDefault with
        name = "RewriteObjects";
        description = "Rewrite object functions by deleting $this variables";
        stage = rewriteStage RewriteObjectsRules }

let ConvertVLArraysToFixed = {
    stageDefault with
        name = "ConvertVLArraysToFixed";
        description = "Convert variable length arrays to fixed";
        stage = rewriteStage ConvertVLArraysToFixedRules }

let CreateSpecialCudaNodesForSketch = {
    stageDefault with
        name = "CreateSpecialCudaNodesForSketch";
        description = "Convert __syncthreads, etc. into special nodes";
        stage = rewriteStage CreateSpecialCudaNodesForSketchRules }

let SpecializeCudaFcnCalls = {
    stageDefault with
        name = "SpecializeCudaFcnCalls";
        description = "make threadIdx, etc. special syms that won't get renamed.";
        stage = rewriteStage SpecializeCudaFcnCallsRules }

let CMemTypeValueOrReference = {
    stageDefault with
        name = "CMemTypeValueOrReference";
        description = "Denote types of nodes as value or reference, but don't generate type structure";
        stage = rewriteStage (SetTypeValueOrReferenceRules) }

let CMemTypes = {
    stageDefault with
        name = "CMemTypes";
        description = "Set C memory types based on annotations, and add relevant address-of and dereference code";
        stage = rewriteStage (CMemTypesRules) }

let SketchCudaMemTypes = {
    stageDefault with
        name = "SketchCudaMemTypes";
        description = "Denote variables as being local or global";
        stage = rewriteStage SketchCudaMemTypesRules }

let NewInitializerFcnStubs = {
    stageDefault with
        name = "NewInitializerFcnStubs";
        description = "Create a function that both calls \"new obj()\" and \"obj.<init>()\"";
        stage = rewriteStage NewInitializerFcnStubsRules }

let SSAForm = {
    stageDefault with
        name = "SSAForm";
        description = "Convert assignments to SSA form";
        stage = rewriteStage SSAFormRules }

let ArrayLowering = {
    stageDefault with
        name = "ArrayLowering";
        description = "Remove sugar from array calls, and retype boxed arrays";
        stage = rewriteStage ArrayLoweringRules }

let CstyleStmts = {
    stageDefault with
        name = "CstyleStmts";
        description = "Create tempvars for Scala-style expression-statements";
        stage = rewriteStage CstyleStmtsRules }

let CstyleAssns = {
    stageDefault with
        name = "CstyleAssns";
        description = "Replace x = { block; expr } with block; x = expr";
        stage = rewriteStage CstyleAssnsRules }

let CstyleMinorCleanup = {
    stageDefault with
        name = "CstyleMinorCleanup"
        description = "change unit blocks to skblocks"
        stage = rewriteStage CstyleMinorCleanupRules }

let CudaCleanup = {
    stageDefault with
        name = "CudaCleanup";
        description = "Final cuda cleanup";
        stage = rewriteStage CudaCleanupRules }

let CudaGenerateCode = {
    stageDefault with
        name = "CudaGenerateCode";
        description = "Cuda code generation";
        stage = rewriteStage CudaGenerateCodeRules }

let SketchFinalMinorCleanup = {
    stageDefault with
        name = "SketchFinalMinorCleanup";
        description = "Final minor cleanup (information loss stage)";
        stage = rewriteStage SketchFinalMinorCleanupRules }

let CreateTemplates = {
    stageDefault with
        name = "CreateTemplates";
        description = "Transform RewriteTemplates.Template into useable templates"
        stage = rewriteStage CreateTemplatesRules }

let CreateLibraries = {
    stageDefault with
        name = "CreateLibraries";
        description = "Minor cleanup for exporting classes"
        stage = rewriteStage CreateLibrariesRules }
