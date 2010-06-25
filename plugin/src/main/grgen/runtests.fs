module edu.berkeley.cs.skalch.tests.main
(*
    Copyright 2010 gatoatigrado (nicholas tung) [ntung at ntung]

    Licensed under the Apache License, Version 2.0 (the "License"); you may
    not use this file except in compliance with the License. You may obtain a
    copy of the License at http://www.apache.org/licenses/LICENSE-2.0 .

    Note that the rest of GrGen is licensed under LGPL v.3, so the effective
    license of this library becomes LGPL 3.
*)

open System
open System.IO
open edu.berkeley.cs.grgenmods.fsharp.util_fcns
open edu.berkeley.cs.grgenmods.fsharp.graph
open edu.berkeley.cs.grgenmods.fsharp.cmdline
open edu.berkeley.cs.grgenmods.fsharp.stages
open edu.berkeley.cs.grgenmods.fsharp.dependencies
open edu.berkeley.cs.skalch.transformer.main

let (|ScalaFile|) (arg:string) = (Path arg).isfile && arg.EndsWith ".scala"

type TestRule =
    | XmloutContains of string

type TestFile(scalaname:string) =
    let scalaname = Path scalaname
    let basename =
        match scalaname.value with
        | RegexMatch "\.scala$" {left=left} -> left
        | _ -> uifail "internal err"
    let tests_lst =
        let tr = "// TEST RULE"
        scalaname.lines
        |> List.filter (fun x -> x.StartsWith tr)
        |> List.map (function
            | RegexMatch (sprintf "^%s XMLOUT CONTAINS " tr) {right=right} ->
                XmloutContains right
            | x -> uifail (sprintf "unknown test rule '%s'" x))

    member self.scalapath = scalaname
    member self.name = (Path basename).basename
    member self.scalagxlpath = Path (scalaname.value + ".ast.gxl")
    member self.sketchgxlpath = Path (basename + ".sketch.ast.gxl")
    member self.testoutpath = Path (basename + ".testout")
    member self.tests = tests_lst

let rec parseCommandLine (files : TestFile list, options) = function
    | [] -> (files, options)
    | (y & "--verbose") :: tail ->
        let files, options = parseCommandLine (files, options) tail
        files, y :: options
    | (y & ("--debugrule" | "--debugafter" | "--debugbefore")) :: x :: tail ->
        let files, options = parseCommandLine (files, options) tail
        files, y :: x :: options
    | (hd & ScalaFile true) :: tail ->
        let files, options = parseCommandLine (files, options) tail
        ((TestFile hd) :: files, options)
    | hd :: tail -> uifail (sprintf "unknown command %s (maybe you forgot an argument? try --help)" hd)

[<EntryPoint>]
let testsMain(args:string[]) =
    let files, options = args |> Array.toList |> parseCommandLine ([], [])
    let verbose = options |> List.filter (fun x -> x = "--verbose") |> List.isEmpty |> not

    (* scala to scala gxl *)
    SubProc("make", ["py-fsc-compile"]).wait

    (* scala gxl to sketch gxl *)
    let getopts (x:TestFile) =
        options @ [
            "--goal"; "sketch";
            "--export"; x.sketchgxlpath.value;
            x.scalagxlpath.value] |> Array.ofList
    files |> List.map (fun x -> transformerMain (getopts x)) |> ignore

    let execJava (x:TestFile) =
        let args = sprintf "-Dexec.args=--fe-output-xml %s %s" x.testoutpath.value x.sketchgxlpath.value
        let main = "-Dexec.mainClass=sketch.compiler.parser.gxlimport.GxlImport"
        SubProc("mvn", ["-e"; "compile"; "exec:java"; main; args]).wait
    Directory.SetCurrentDirectory "base"
    files |> List.map execJava |> ignore

    let runtests (x:TestFile) =
        let testout_lines = x.testoutpath.lines
        let testout_text = List.fold (fun a b -> a + b + "\n") "" testout_lines
        let runTest (y:TestRule) =
            let test (succeeded:bool) =
                if not succeeded then
                    uifail (sprintf "failed running test on '%s': %A" x.name y)
            if verbose then
                printfn "running test: %A ..." y
            match y with
            | XmloutContains z -> test (testout_text.Contains z)
            if verbose then
                printfn "... succeeded"
        x.tests |> List.iter runTest
    files |> List.iter runtests
    0
