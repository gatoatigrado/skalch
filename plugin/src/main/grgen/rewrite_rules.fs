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

let WarnUnsupportedRules = [
    ValidateNeg ("existsUnsupportedAssignToNonvar", "assignment left-hand sides must be variables or field accesses");
    ValidateNeg ("existsUnsupportedArrayLength", "array length expressions are not supported.")
    ]

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
    (* misc cleanup *)
    Xgrs "cleanupLiteralTypeOnClassOf*"

    Xgrs "[setRootSymbol]"
    ValidateXgrs ("existsRootSymbol && ! multipleRootSymbols", "ASG has multiple root symbols")
    Xgrs "replaceAngelicSketchSymbol*"
    Xgrs "runAllSymbolRetypes"
    Xgrs "setRangeAnnotations*"
    Xgrs "setArrayLenAnnotations*"
    Xgrs "setGeneratorAnnotations*"
    Xgrs "[setOuterSymbol]"
    Xgrs "[setAngelicSketchSymbol]"
    Xgrs "setScalaRoot & setScalaSubtypes*"

    (* see note in grg file; ValidateXgrs "! existSymbolsWithSameUniqueName" *)
    Xgrs "deleteBridgeFunctions*"
    Xgrs "deleteDangling*"
    ValidateNeg ("existsBridgeFcnSymbol", "failed to cleanup Java API bridge functions") ] @ CleanupNonuniqueSymbolsRules

let ConvertThisRules = [
    Xgrs "setEnclosingFunctionInitial+"
    Xgrs "[transformFcnWrapper]"
    ValidateXgrs ("testNoThisNodes", "failed to convert all $this variables to parameters")
    Xgrs "removeEnclosingLinks* & deleteDangling*"
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
    Xgrs "removeInstanceOf*"
    Xgrs "simplifyClassConstruction*"
    Xgrs "deleteDangling*"]


let BlockifyFcndefsRules = [
    Xgrs "removeEmptyFcnTarget* & removeFcnTarget*"
    ValidateXgrs ("! existsFcnTarget", "failed to rewrite function targets (x.f()) into arguments (f(x))")
    Xgrs "(deleteDangling+ | removeNopTypeCast)*"
    Xgrs "createFunctionBlocks* & retypeBlockSKBlock*"
    Xgrs "deleteFcnBlockEmptyTrees*"
    ValidateNeg ("existsFcnNonBlockBody", "failed to convert all function bodies into blocks") ]

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

let LowerTprintRules = [
    Xgrs "(convertTprintArrayToArgList+ | deleteDangling+)+"
    ValidateNeg ("existsTprintWithoutTuple", "failed to convert a tprint argument (usage: tprint(\"const-str\" -> value))")
    ]

let RaiseSpecialGotosRules = [
    Xgrs "raiseWhileLoopGotos*"
    Xgrs "deleteDangling*"
    ValidateNeg ("existsLabelDef", "Could not recognize control flow structure (unknown goto)!") ]

let CleanTypedTmpBlockRules = [
    Xgrs "deleteDangling*"
    Xgrs "cleanupDummyVarBlocks*"
    Xgrs "deleteDuplicateAnnotations*"
    Xgrs "deleteAnnotationLink"
    ValidateNeg ("existsDanglingAnnotation", "Failed to convert annotations to relevant functions")

    Xgrs "deleteDangling*"]



let ProcessAnnotationsRules1 =
    CleanTypedTmpBlockRules @
    [
        (* gt-independent retype annotations *)
        Xgrs "retypeSymbolsAnnotations*"

        (* static array length annotations *)
        Xgrs "createAnnotArrayLengthSyms*"
        Xgrs "setAnnotArrayTypeSymbols*"
        Xgrs "updateAssignLhsTypes*"
        Xgrs "updateValDefSymbolTypes*"
        Xgrs "deleteDanglingArrayLenAnnotations*"
        Xgrs "deleteStaticArrayConstructInfo*"

        (* cuda function annotations *)
        Xgrs "setCudaFunction*"

        (* integer range annotations *)
        Xgrs "replacePrimitiveRanges* & decrementUntilValues* & deleteDangling*"
        Xgrs "deleteDangling*"
        ValidateNeg ("existsDanglingAnnotation", "Failed to convert annotations to relevant functions")
        (* NOTE -- only the last command is scanned! Don't add rules after this one *)
        Xgrs "[requestIntHoleTemplate] && [requestStaticIntArrayHoleTemplate]" ]

let PostImportUnionRules = [
    Xgrs "instantiateTemplateSymbols*"
    Xgrs "unionRootSymbol*"
    Xgrs "unionSubSymbol*"
    ValidateNeg ("twoFcnsForSameSymbol", "Failed to merge template/library -- two functions defined for the same symbol") ]

let ProcessAnnotationsRules2 =
    PostImportUnionRules @
    [ Xgrs "attachAnnotationsToTemplates*";
    Xgrs "setTemplateParameter*";
    Xgrs "createFcnCallTemplates*";
    ValidateNeg ("existsUnreplacedCall", "A requested template was not imported or correctly connected");
    ValidateNeg ("existsDanglingTemplateFcn", "An template imported was not connected") ]

let ResolveTemplatesRules = [
    Xgrs "deleteGTInstanceTypeDefinition*"
    Xgrs "deleteDangling*"
    ]

let EmitRequiredImportsRules = [
    Xgrs "[setEnclosingFunctionInitial]"
    Xgrs "setCalledMethods* & unsetDefinedCalledMethods*"
    Xgrs "emitRequiredImports*"
    Xgrs "emitProvides*"
    Xgrs "removeEnclosingLinks* & deleteDangling* & cleanupTmpEdges*"
    ]

let SketchNospecRules = [
    Xgrs "removeThisVarFromMain*"
    Xgrs "setHarnessAnnotatedAsMain*"
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

let SpecializeCudaFcnCallsRules =
    [
        for name in [ "threadIdx"; "blockDim"; "blockIdx"; "gridIdx" ] do
            yield Xgrs (sprintf "setFixedPrintName(\"skalch.CudaKernel.%s\", \"%s\")" name name)
    ] @
    [
        for name in [ "x"; "y"; "z" ] do
            yield Xgrs (sprintf "setFixedPrintName(\"skalch.CudaKernel$ParallelIndex.%s\", \"%s\")" name name)
    ] @

    (* very much C names... change if translating to other languages *)
    [
        Xgrs "setFixedPrintName(\"scala.Unit\", \"void\")"
        Xgrs "setFixedPrintName(\"scala.Int\", \"int\")"
        Xgrs "convertParallelIdxToCudaSpecial*"
        Xgrs "convertParallelIndexVecToField*"
        Xgrs "deleteDangling*"
        Xgrs "[createObjDefs]"
        Xgrs "rewriteSyncthreadsCall*" ]

let RewriteObjectsRules = [
    (* delete $this variables from object functions *)
    Xgrs "removeObjArgFromFcnCalls*"
    Xgrs "removeObjArgFromFcnDefs*"
    Xgrs "removeSuperCallFromInit*"
    Xgrs "deleteDangling*"
    Xgrs "addObjFieldAsGlobal*"
    Xgrs "convertObjFieldAccessToGlobalRef*" ]

let ConvertVLArraysToFixedRules = [
    Xgrs "convertVLArraySymbolToFixed*"
    Xgrs "setNewAssignedArrayCalls*"
    Xgrs "setNewAnonymousArrayCalls*"
    ]

let CreateSpecialCudaNodesForSketchRules = [
    Xgrs "convertParallelIdxToCudaSpecial*"
    Xgrs "convertParallelIndexVecToField*"
    Xgrs "createSketchThreadIdxNodes*"
    Xgrs "createSyncthreadsNodes*" ]

(* CMemTypes rules *)
let SetTypeValueOrReferenceRules =
    [
        for typ in [ "Byte"; "Short"; "Int"; "Long"; "Float"; "Double"; "Boolean"; "Char"; "Unit" ] do
            yield Xgrs (sprintf "setSpecificValueType(\"scala.%s\")" typ)
    ] @
    [
        Xgrs "setAnnotatedValueTypes*"
        Xgrs "deleteDangling*"
        Xgrs "setReferenceTypes*" ]

let CMemTypesRules = [
    Xgrs "[addMNodesForValueTypes]"
    Xgrs "addMNodesForVariableArrays*"
    Xgrs "[addMNodesForReferenceTypes]"

    (* link term symbols to memory types *)
    Xgrs "setMTermArrayInlinedTypes*"
    Xgrs "setMTermRegalarTypes*" ]

let FinalSetCudaMemTypesRules = [
    Xgrs "setDefaultAsLocal*"
    Xgrs "createDefaultMemLocation*"
    ValidateNeg ("existsConflictingMemTypes", "Annotations for type of variable conflict") ]

let SketchCudaMemTypesRules = [
    Xgrs "[setSharedForKernelParameters]"
    Xgrs "setAnnotatedShared*"
    Xgrs "setAnnotatedGlobal*"
    (* NOTE -- continued after other stages *) ] @ FinalSetCudaMemTypesRules

let CfgInitRules = [
    Xgrs "deleteDangling*"
    Xgrs "cfgInit"
    ValidateNeg ("existsCfgNextFromAst", "Failed to convert CFG->AST edges to CFG->CFG edges")
    ValidateNeg ("existsCfgNextToAst", "Failed to convert CFG->AST edges to CFG->CFG edges")
    Xgrs "deleteCfgNodesOnClassFields*"
    ValidateNeg ("cfgExistsIncomplete", "Failed to create a complete CFG; perhaps you added an AST node without adding CFG rules?") ]

let SSAFormRules =
    CfgInitRules @ [
        Xgrs "initCfgPossibleAssign*"
        Xgrs "propagateCfgPossibleAssignment*"
        Xgrs "createNewAssignSymbols*"
        Xgrs "uniquelyNameSymbols*"
        Xgrs "[setUniqueSSANames]"
        (* ValidateConnect *)

        Xgrs "createHLPhiFcns*"
        Xgrs "addAdditionalSymbolToPhiFcn*"
        Xgrs "redirectSingleVarRefs*"

        Xgrs "convertValDefsToInits_0*"
        Xgrs "convertFirstVDToSSAAssign*"
        Xgrs "convertValDefsToInits_1*"
        ValidateNeg ("existsSSAParentNotOkToDelete", "Failed to create and connect all SSA instance nodes (e.g. a_0, a_1, etc.)")
        Xgrs "deleteCfgNode*"
        Xgrs "deleteDangling*" ]
        (* todo convertFirstVDToSSAAssign *)

let ArrayLoweringRules = [
    (* a = Array(1, 2, 3, 4) *)
    Xgrs "countNewArrayElts*"
    Xgrs "deleteWrapNewArray*"
    Xgrs "simplifyArrayConstructors*"

    (* a = new Array(len)     for now, only constants supported *)
    Xgrs "simplifyVarLenArrayCtors*"

    (* fcns to sketch specials *)
    Xgrs "decorateArrayGet*"
    Xgrs "decorateArraySet*"
    Xgrs "deleteDangling*"

    (* create fixed array symbols *)
    Xgrs "createArrayLengthSyms*"
    Xgrs "typifyConstLenArrays*"

    (* create variable length array symbols *)
    Xgrs "createVariableArraySyms*"
    Xgrs "typifyVariableLenArrays*"

    (* propagate types to further variables *)
    Xgrs "updateAssignLhsTypes*"
    Xgrs "updateValDefSymbolTypes*"
    Xgrs "updateVarRefTypes*"

    ValidateNeg ("existsAssignToDiffLenArray", "no alternating/variable array lengths yet, sorry")

    (* re-update types / FIXME, hack *)
    Xgrs "deleteDangling*"
    Xgrs "typifyConstLenArrays*"
    Xgrs "typifyVariableLenArrays*"
    ]

let LowerPhiFunctionsRules = [
    Xgrs "markPhiSyms* & createCounterVars*"
    ]

let CstyleStmtsRules =
    CfgInitRules @ [
        Xgrs "setAttachableMemberFcns*"
        Xgrs "setAttachableBlocks*"
        Xgrs "blockifyDefault* & blockifyLists*"
        Xgrs "forwardNonblockifyIntermediatePrologue*"
        Xgrs "(setBlockifyNextForAlreadyBlockified+| propagateBlockifyUnsafe+ | propagateBlockifyMarkSafe+)*"
        Xgrs "convertNodesAlreadyStmtsToBlockifySafe*"
        Xgrs "setBlockifyChain*"
        Xgrs "checkBlockifyLinks"
        Xgrs "forwardBlockifySkip*"
        Xgrs "addDummyBlockifyChainEndNodes*"
        Xgrs "deleteCfgNode*"
        Xgrs "createTemporaryAssign*"
        Xgrs "attachNodesToBlockList*"
        Xgrs "deleteLastAttachables* & deleteLastAttachables2*"
        ValidateXgrs ("! existsBlockify", "Failed to convert all statements to C-style statements") ]

let CstyleAssnsRules = [Xgrs "makeValDefsEmpty*";
    Xgrs "cstyleAssignToIfs+ | cstyleAssignToBlocks+"]

let CstyleMinorCleanupRules = [
    Xgrs "unitBlocksToSKBlocks" ] @ FinalSetCudaMemTypesRules

let NameSymbolsRules = [
    Xgrs "[setTmpSymbolNames]"
    Xgrs "[initSymbolNames]"
    Xgrs "uniqueSymbolNames*"
    Xgrs "[finalizeSymbolNames]"
    Xgrs "setSymbolSketchType*"
    Xgrs "setSketchTypeInt & setSketchTypeBoolean & setSketchTypeUnit"
    Xgrs "setSketchTypeArray*"
    ]

let SketchAndCudaCleanupRules =
    NameSymbolsRules @
    [
        Xgrs "removeEmptyTrees"
        Xgrs "connectFunctions*" (* set PackageDefFcn edges *)
        Xgrs "removeEmptyChains*" ]

let CudaCleanupRules =
    SketchAndCudaCleanupRules @
    [
        (* Xgrs "removeThisVarFromCudaKernel*" *)
        ]

let SketchFinalMinorCleanupRules =
    SketchAndCudaCleanupRules  @
    [
        (* Attach more specific type information to terms *)
        Xgrs "copySketchTypesToTerms*"
        ValidateNeg ("existsVDSymWithoutTerm", "Didn't assign symbol a sketch type")

        (* Set arrays to be reference types *)
        Xgrs "setArrayReferenceParamInout*"
        Xgrs "setOtherParamIn*"

        Xgrs "convertArrayAssignForSketch*"
        Xgrs "setAssertCalls* & deleteDangling*"
        Xgrs "setFcnBinaryCallBaseType*"
        Xgrs "setFcnBinaryCallArgs*"
        Xgrs "setSymbolBaseType*"
        Xgrs "setValDefBaseType*"
        Xgrs "setVarRefBaseType*"
        Xgrs "setFcnDefBaseType*"
        Xgrs "setClassDefBaseType*"
        Xgrs "[setGeneratorFcn] & [setNonGeneratorFcn]"
        Xgrs "addSkExprStmts*" ]

let CudaGenerateCodeRules = [
    (* basic string representation *)
    Xgrs "setStringRepFcnDef*"
    Xgrs "[setStringRepCudaKernelFcn]"

    Xgrs "setStringRepBlock*"

    (* function calls; start with specialized ones first *)
    Xgrs "setStringRepCudaParIdxCall*"
    Xgrs "setStringRepFcnCallBinary*"
    Xgrs "setStringRepFcnCall*"

    (* fields, run first since they are valdefs *)
    Xgrs "setStringRepClassDef*"

    (* other nodes *)
    Xgrs "setStringRepFieldAccess*"
    Xgrs "setStringRepEmptyValDef*"
    Xgrs "setStringRepSketchArrayAccess*"
    Xgrs "setStringRepSketchArrayAssign*"
    Xgrs "setStringRepReturn*"
    Xgrs "setStringRepAssign*"



    (* TEMP DEBUG *)
    Xgrs "dummySetVarArraysToPtrs*"

    Xgrs "setStringRepVarRef*"
    Xgrs "setStringRepSymbol*"

    (* Convert higher-level nodes to basic strings *)
    (*--------------------------------------------------
    * Xgrs "deleteUnnecessaryParens_VarRef*"
    *--------------------------------------------------*)
    Xgrs "expandNSRSurround*"
    Xgrs "expandNSRSpaces*"
    Xgrs "convertParaListsToBasic*"
    Xgrs "expandStringRepSepList_Copy*"
    Xgrs "expandStringRepSepList_InsertSep*"
    Xgrs "expandStringRepSepList_DeleteNode*"

    (* TEMP DEBUG *)
    Xgrs "deleteDangling*"
    Xgrs "testAppendDummyStringRep*"

    (* linearization *)
    Xgrs "(forwardStringReps+ | forwardStringRepsSingleChild+ | linearizeStringReps+)+"




    (* TEMP DEBUG -- redo above *)
    Xgrs "deleteDangling*"
    Xgrs "testAppendDummyStringRep*"
    Xgrs "(forwardStringReps+ | forwardStringRepsSingleChild+ | linearizeStringReps+)+"




    (* newlines and indentation *)
    Xgrs "deleteDangling*"
    Xgrs "handleAdjacentNewlines*"
    Xgrs "propagateIndent*"
    Xgrs "propagateDeindent*"
    Xgrs "handleNewline*"

    (* collapse to a single string *)
    Xgrs "deleteDangling*"
    Xgrs "collapseStringRepAdjacentLiterals*"
    Xgrs "collapseStringRep*"
    Xgrs "deleteDangling*"
    Xgrs "collapseSingletonLists*"
    Xgrs "deleteDangling*"

    Xgrs "[printMethodStringRep]"
    ]

let CreateTemplatesRules = [
    Xgrs "[markTemplates] & [deleteNonTemplates] & deleteDangling*"
    Xgrs "convertFieldsToTmplParams*"
    Xgrs "[deleteUnnecessaryTemplateFcns] & deleteDangling*"
    Xgrs "[printAndRetypeTemplates]" ] @ CleanTypedTmpBlockRules

let CreateLibrariesRules = [
    Xgrs "[printClassNames]" ] @ CleanTypedTmpBlockRules

