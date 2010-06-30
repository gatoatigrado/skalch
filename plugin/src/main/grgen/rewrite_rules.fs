module edu.berkeley.cs.skalch.transformer.rewrite_rules
(*
    Copyright 2010 gatoatigrado (nicholas tung) [ntung at ntung]

    Licensed under the Apache License, Version 2.0 (the "License"); you may
    not use this file except in compliance with the License. You may obtain a
    copy of the License at http://www.apache.org/licenses/LICENSE-2.0 .

    Note that the rest of GrGen is licensed under LGPL v.3, so the effective
    license of this library becomes LGPL 3.
*)

open edu.berkeley.cs.grgenmods.fsharp.stages

let WarnUnsupportedRules = [Xgrs "unsupportedWarnAll"]

let DeleteMarkedIgnoreRules = [Xgrs "setIgnoreAnnotationType*";
    Xgrs "deleteIgnoreAnnotated* & deleteDangling*"]

let CleanupNonuniqueSymbolsRules = [
    Xgrs "[setDefaultSymbolUniqueName]"
    Xgrs "[setTypeSymbolUniqueName]"
    Xgrs "[setFieldSymbolUniqueName]"
    Xgrs "[setPackageDefSymbolUniqueName]"
    Xgrs "[setFcnSymbolUniqueName]"
    Xgrs "[setObjectSymbolUniqueName]"
    Xgrs "[warnSameSymbolName]"
    Xgrs "unionSubSymbol*" ]

let DecorateNodesRules = [
    Xgrs "[setRootSymbol]"
    ValidateXgrs "existsRootSymbol && ! multipleRootSymbols"
    Xgrs "replaceAngelicSketchSymbol*"
    Xgrs "runAllSymbolRetypes"
    Xgrs "setStaticAnnotationTypes* & [setOuterSymbol]"
    Xgrs "setScalaRoot & setScalaSubtypes*"
    Xgrs "setSketchClasses*"
    (* see note in grg file; ValidateXgrs "! existSymbolsWithSameUniqueName" *)
    Xgrs "deleteBridgeFunctions*"
    Xgrs "deleteDangling*"
    ValidateNeg "existsBridgeFcnSymbol" ] @ CleanupNonuniqueSymbolsRules

let ConvertThisRules = [Xgrs "setEnclosingFunctionInitial+";
    Xgrs "[transformFcnWrapper]";
    ValidateXgrs "testNoThisNodes";
    Xgrs "removeEnclosingLinks* & deleteDangling*";
    Xgrs "setSketchMainFcn*"]

let RedirectAccessorsToFieldsRules = [
    Xgrs "[markGetterCalls]";
    Xgrs "[setGetterFcnFieldEdges]";
    Xgrs "replaceGetterFcnCalls*";
    Xgrs "[deleteGetterEdges]";
    Xgrs "deleteDangling*" ]

let CleanSketchConstructsRules = [
    Xgrs "replaceAssertCalls* & deleteAssertElidableAnnot*"
    Xgrs "(valueConstructAssigned+ | classConstructAssigned+ | valueConstructAssigned2+ | valueConstructAssigned3+)*"
    Xgrs "replaceConstructCalls*"
    Xgrs "unboxConstructCalls*"
    Xgrs "simplifyClassConstruction*"
    Xgrs "deleteDangling*"]

let BlockifyFcndefsRules = [
    Xgrs "removeEmptyFcnTarget* & removeFcnTarget*"
    ValidateXgrs "! existsFcnTarget"
    Xgrs "(deleteDangling+ | removeNopTypeCast)*"
    Xgrs "createFunctionBlocks* & retypeBlockSKBlock*"
    Xgrs "deleteFcnBlockEmptyTrees*"
    ValidateXgrs "! existsFcnNonBlockBody" ]

let NiceListsRules = [Xgrs "listBlockInit*";
    Xgrs "listClassDefsInit*";
    Xgrs "listInitAllOrdered";
    Xgrs "listAddClassField*";
    Xgrs "listSetNext*";
    Xgrs "listCompleteLast*";
    Xgrs "listCompleteBlockLast*"]

(* TODO (x+ | y+)+ *)
let SimplifyConstantsRules = [
    Xgrs "replaceUnaryNeg+" ]

let RaiseSpecialGotosRules = [
    Xgrs "raiseWhileLoopGotos*"
    Xgrs "deleteDangling*"
    ValidateXgrs "! existsLabelDef" ]

let CleanTypedTmpBlockRules = [
    Xgrs "deleteDangling*";
    Xgrs "cleanupDummyVarBlocks*";
    Xgrs "deleteAnnotationLink";
    ValidateXgrs "! existsDanglingAnnotation";
    Xgrs "deleteDangling*"]

let ProcessAnnotationsRules1 =
    CleanTypedTmpBlockRules @
    [
        Xgrs "replacePrimitiveRanges* & decrementUntilValues* & deleteDangling*"
        Xgrs "deleteDangling*"
        ValidateXgrs "! existsDanglingAnnotation"
        Xgrs "[requestIntHoleTemplate]" ]

let PostImportUnionRules = [
    Xgrs "instantiateTemplateSymbols*"
    Xgrs "unionRootSymbol*"
    Xgrs "unionSubSymbol*"
    ValidateXgrs "! twoFcnsForSameSymbol" ]

let ProcessAnnotationsRules2 =
    PostImportUnionRules @
    [ Xgrs "attachAnnotationsToTemplates*";
    Xgrs "setTemplateParameter*";
    Xgrs "createFcnCallTemplates*";
    ValidateXgrs "! existsUnreplacedCall";
    ValidateXgrs "! existsDanglingTemplateFcn" ]

let EmitRequiredImportsRules = [
    Xgrs "[setEnclosingFunctionInitial]"
    Xgrs "setCalledMethods* & unsetDefinedCalledMethods*"
    Xgrs "emitRequiredImports*"
    Xgrs "emitProvides*"
    Xgrs "removeEnclosingLinks* & deleteDangling* & cleanupTmpEdges*"
    ]

let SketchNospecRules = [
    Xgrs "removeThisVarFromMain*";
    Xgrs "generateNospec*"
    Xgrs "doCopy"
    Xgrs "[setCopySymbolNames]"
    Xgrs "cleanupCopyTo*"
    Xgrs "checkNoMainFcns"
    Xgrs "deleteDangling*" ]

let LossyReplacementsRules = [
    Xgrs "replaceThrowWithAssertFalse*"
    Xgrs "deleteObjectInitCall*"
    Xgrs "retypeWeirdInits*"
    Xgrs "deleteUnitConstants*"
    Xgrs "detachBinaryFcns* & deleteAssertSymbolEdges* & deleteDangling*"
    Xgrs "deleteSketchThisSymbol* & deleteDangling*"
    ]

let NewInitializerFcnStubsRules = [
    Xgrs "markInitializerFunctions+ && createInitializerFunctions+ && replaceConstructors+"]

let CfgInitRules = [
    Xgrs "deleteDangling*"
    Xgrs "cfgInit"
    ValidateNeg "existsCfgNextFromAst"
    ValidateNeg "existsCfgNextToAst"
    Xgrs "deleteCfgNodesOnClassFields*"
    ValidateXgrs "! cfgExistsIncomplete" ]

let SSAFormRules =
    CfgInitRules @ [
        Xgrs "initCfgPossibleAssign*"
        Xgrs "propagateCfgPossibleAssignment*"
        Xgrs "createNewAssignSymbols*"
        Xgrs "uniquelyNameSymbols*"
        Xgrs "[setUniqueSSANames]"
        ValidateConnect

        Xgrs "createHLPhiFcns*"
        Xgrs "addAdditionalSymbolToPhiFcn*"
        Xgrs "redirectSingleVarRefs*"

        Xgrs "convertValDefsToInits_0*"
        Xgrs "convertFirstVDToSSAAssign*"
        Xgrs "convertValDefsToInits_1*"
        ValidateNeg "existsSSAParentNotOkToDelete"
        Xgrs "deleteCfgNode*"
        Xgrs "deleteDangling*" ]
        (* todo convertFirstVDToSSAAssign *)

let ArrayLoweringRules = [
    Xgrs "countNewArrayElts*"
    Xgrs "deleteWrapNewArray*"
    Xgrs "simplifyArrayConstructors*"
    Xgrs "decorateArrayGet*"
    Xgrs "deleteDangling*"
    Xgrs "createArrayLengthSyms*"
    Xgrs "typifyConstLenArrays*" ]

let LowerPhiFunctionsRules = [
    Xgrs "markPhiSyms* & createCounterVars*"
    ]

let CstyleStmtsRules =
    CfgInitRules @ [
        Xgrs "setAttachableMemberFcns*"
        Xgrs "setAttachableBlocks*"
        Xgrs "blockifyDefault* & blockifyLists*"
        Xgrs "forwardNonblockifyIntermediatePrologue*"
        Xgrs "setBlockifyNextForAlreadyBlockified*"
        Xgrs "(propagateBlockifyUnsafe+ | propagateBlockifyMarkSafe+)*"
        Xgrs "convertNodesAlreadyStmtsToBlockifySafe*"
        Xgrs "setBlockifyChain*"
        Xgrs "checkBlockifyLinks"
        Xgrs "forwardBlockifySkip*"
        Xgrs "addDummyBlockifyChainEndNodes*"
        Xgrs "deleteCfgNode*"
        Xgrs "createTemporaryAssign*"
        Xgrs "attachNodesToBlockList*"
        Xgrs "deleteLastAttachables* & deleteLastAttachables2*"
        ValidateXgrs "! existsBlockify"]

let CstyleAssnsRules = [Xgrs "makeValDefsEmpty*";
    Xgrs "cstyleAssignToIfs+ | cstyleAssignToBlocks+"]

let CstyleMinorCleanupRules = [
    Xgrs "unitBlocksToSKBlocks" ]

let SketchFinalMinorCleanupRules = [
    Xgrs "removeEmptyTrees"
    Xgrs "[setTmpSymbolNames]"
    Xgrs "[initSymbolNames]"
    Xgrs "uniqueSymbolNames*"
    Xgrs "[finalizeSymbolNames]"
    Xgrs "setSymbolSketchType*"
    Xgrs "setSketchTypeInt & setSketchTypeBoolean & setSketchTypeUnit"
    Xgrs "connectFunctions*"
    Xgrs "removeEmptyChains*"
    Xgrs "setAssertCalls* & deleteDangling*"
    Xgrs "setFcnBinaryCallBaseType*"
    Xgrs "setFcnBinaryCallArgs*"
    Xgrs "setSymbolBaseType*"
    Xgrs "setValDefBaseType*"
    Xgrs "setVarRefBaseType*"
    Xgrs "setFcnDefBaseType*"
    Xgrs "addSkExprStmts*" ]

let CreateTemplatesRules = [
    Xgrs "[markTemplates] & [deleteNonTemplates] & deleteDangling*"
    Xgrs "convertFieldsToTmplParams*"
    Xgrs "[deleteUnnecessaryTemplateFcns] & deleteDangling*"
    Xgrs "[printAndRetypeTemplates]" ] @ CleanTypedTmpBlockRules
