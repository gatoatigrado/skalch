let SessionLoad = 1
if &cp | set nocp | endif
noremap  "_d
noremap <silent>  :call UnComment()
map 	 oimport ipdb; ipdb.set_trace()
let s:cpo_save=&cpo
set cpo&vim
map <NL> ok
noremap <silent>  :call Comment()
map  o
nnoremap  :set paste:read !xclip_paste:set nopaste
map OE i
map O5C w
map O5D b
map O5B ^F
map O5A ^B
map O2C w
map O2D b
map O2B ^F
map O2A ^B
map [3;5~ x
map [2;5~ i
map [3;2~ x
map [2;2~ i
map O5F $
map O5H 0
map O2F $
map O2H 0
map OF $
map OH 0
map [E i
map [D h
map [C l
map [B j
map [A k
map [4~ $
map [1~ 0
map [F $
map [H 0
map On .
map Op 0
map Os 3
map Or 2
map Oq 1
map Ov 6
map Ou 5
map Ot 4
map Oy 9
map Ox 8
map Ow 7
map OM 
map Ol ,
map Ok +
map Om -
map Oj *
map Oo :
map [3~ x
xmap <silent> ,e <Plug>CamelCaseMotion_e
xmap <silent> ,b <Plug>CamelCaseMotion_b
xmap <silent> ,w <Plug>CamelCaseMotion_w
omap <silent> ,e <Plug>CamelCaseMotion_e
omap <silent> ,b <Plug>CamelCaseMotion_b
omap <silent> ,w <Plug>CamelCaseMotion_w
nmap <silent> ,e <Plug>CamelCaseMotion_e
nmap <silent> ,b <Plug>CamelCaseMotion_b
nmap <silent> ,w <Plug>CamelCaseMotion_w
xmap S <Plug>VSurround
nmap cs <Plug>Csurround
nmap ds <Plug>Dsurround
nmap gx <Plug>NetrwBrowseX
xmap gS <Plug>VgSurround
xmap <silent> i,e <Plug>CamelCaseMotion_ie
xmap <silent> i,b <Plug>CamelCaseMotion_ib
xmap <silent> i,w <Plug>CamelCaseMotion_iw
omap <silent> i,e <Plug>CamelCaseMotion_ie
omap <silent> i,b <Plug>CamelCaseMotion_ib
omap <silent> i,w <Plug>CamelCaseMotion_iw
xmap s <Plug>Vsurround
nmap ySS <Plug>YSsurround
nmap ySs <Plug>YSsurround
nmap yss <Plug>Yssurround
nmap yS <Plug>YSurround
nmap ys <Plug>Ysurround
nnoremap <silent> <Plug>NetrwBrowseX :call netrw#NetrwBrowseX(expand("<cWORD>"),0)
vnoremap <Plug>CamelCaseMotion_ie :call camelcasemotion#InnerMotion('e',v:count1)
vnoremap <Plug>CamelCaseMotion_ib :call camelcasemotion#InnerMotion('b',v:count1)
vnoremap <Plug>CamelCaseMotion_iw :call camelcasemotion#InnerMotion('w',v:count1)
onoremap <Plug>CamelCaseMotion_ie :call camelcasemotion#InnerMotion('e',v:count1)
onoremap <Plug>CamelCaseMotion_ib :call camelcasemotion#InnerMotion('b',v:count1)
onoremap <Plug>CamelCaseMotion_iw :call camelcasemotion#InnerMotion('w',v:count1)
vnoremap <Plug>CamelCaseMotion_e :call camelcasemotion#Motion('e',v:count1,'v')
vnoremap <Plug>CamelCaseMotion_b :call camelcasemotion#Motion('b',v:count1,'v')
vnoremap <Plug>CamelCaseMotion_w :call camelcasemotion#Motion('w',v:count1,'v')
onoremap <Plug>CamelCaseMotion_e :call camelcasemotion#Motion('e',v:count1,'o')
onoremap <Plug>CamelCaseMotion_b :call camelcasemotion#Motion('b',v:count1,'o')
onoremap <Plug>CamelCaseMotion_w :call camelcasemotion#Motion('w',v:count1,'o')
nnoremap <Plug>CamelCaseMotion_e :call camelcasemotion#Motion('e',v:count1,'n')
nnoremap <Plug>CamelCaseMotion_b :call camelcasemotion#Motion('b',v:count1,'n')
nnoremap <Plug>CamelCaseMotion_w :call camelcasemotion#Motion('w',v:count1,'n')
map <F1> :w | silent execute "! ./run_jinja2.py || sleep 1" | redraw!
map <F4> dd][jdd
map <F3> }{o{#][o#}
map <F5> :e
noremap <F2> :FufBuffer
imap S <Plug>ISurround
imap s <Plug>Isurround
imap  <Plug>Isurround
map! OE <Insert>
map! O5D <S-Left>
map! O5C <S-Right>
map! O5B <PageDown>
map! O5A <PageUp>
map! O2D <S-Left>
map! O2C <S-Right>
map! O2B <PageDown>
map! O2A <PageUp>
map! [3;5~ <Del>
map! [2;5~ <Insert>
map! [3;2~ <Del>
map! [2;2~ <Insert>
map! O5F <End>
map! O5H <Home>
map! O2F <End>
map! O2H <Home>
map! OF <End>
map! OH <Home>
map! [E <Insert>
map! [D <Left>
map! [C <Right>
map! [B <Down>
map! [A <Up>
map! [4~ <End>
map! [1~ <Home>
map! [F <End>
map! [H <Home>
map! On .
map! Op 0
map! Os 3
map! Or 2
map! Oq 1
map! Ov 6
map! Ou 5
map! Ot 4
map! Oy 9
map! Ox 8
map! Ow 7
map! OM 
map! Ol ,
map! Ok +
map! Om -
map! Oj *
map! Oo :
map! [3~ <Del>
map á ea
map é bi
map ï oko
let &cpo=s:cpo_save
unlet s:cpo_save
set autoindent
set backspace=indent,eol,start
set expandtab
set fileencodings=ucs-bom,utf-8,default,latin1
set guifont=CMU\ Typewriter\ Text\ Medium\ 10
set helplang=en
set ignorecase
set iminsert=0
set imsearch=0
set nomodeline
set mouse=a
set operatorfunc=<SNR>27_opfunc
set ruler
set shiftwidth=4
set showmatch
set smartindent
set softtabstop=4
set tabstop=4
set termencoding=utf-8
set window=95
let s:so_save = &so | let s:siso_save = &siso | set so=0 siso=0
let v:this_session=expand("<sfile>:p")
silent only
cd ~/sandbox/skalch
if expand('%') == '' && !&modified && line('$') <= 1 && getline(1) == ''
  let s:wipebuf = bufnr('%')
endif
set shortmess=aoO
badd +7 base/src/test/scala/angelic/simple/SugaredTest.scala
badd +12 plugin/src/main/grgen/unified/common.unified.grg
badd +46 plugin/src/main/grgen/unified/lower_tprint.unified.grg
badd +6 plugin/src/main/grgen/build_templates.py
badd +61 plugin/src/main/grgen/ScalaAstModel.gm.jinja2
badd +14 plugin/src/main/grgen/rewrite_rules.fs
badd +150 base/src/codegen/gxltosketch/gxltosketch.py
badd +13 base/src/main/java/sketch/compiler/parser/gxlimport/GxlHandleNodes.java.jinja2
badd +15 ~/sandbox/grgen/engine-net-2/FSharpBindings/cmdline.fs
badd +54 ~/sandbox/grgen/engine-net-2/FSharpBindings/stages.fs
badd +3 base/src/test/scala/angelic/simple/Test0003_WhileLoops.scala
badd +8 base/src/test/scala/angelic/simple/Test0005_tprint.scala
badd +19 base/src/main/scala/skalch/AngelicSketch.scala
badd +68 ~/sandbox/grgen/engine-net-2/FSharpBindings/graph.fs
badd +106 plugin/src/main/grgen/unified/decorate_nodes.unified.grg
badd +44 plugin/src/main/grgen/unified/process_annotations.unified.grg
badd +48 plugin/src/main/grgen/AllRules_0.grg
badd +1 plugin/src/main/grgen/rules/simplify_sketch_constructs.grg
badd +42 plugin/src/main/grgen/unified/macros.grg
badd +29 base/src/main/scala/skalch/RewriteTemplates.scala
badd +52 plugin/src/main/grgen/unified/sketch_final_minor_cleanup.unified.grg
badd +282 plugin/src/main/grgen/transformer.fs
badd +1 ~/sandbox/sketch-frontend/src/main/java/sketch/compiler/ast/core/Function.java
badd +155 base/src/main/java/sketch/compiler/parser/gxlimport/GxlHandleNodesBase.java
badd +48 plugin/src/main/grgen/unified/array_lowering.unified.grg
badd +1 plugin/src/main/scala/skalch/plugins/ScalaGxlNodeMap.scala
badd +21 base/src/test/scala/cuda/VectorAdd.scala
badd +2 base/src/main/java/sketch/compiler/parser/gxlimport/GxlSketchOptions.java
badd +4 base/src/main/scala/skalch/cuda/CudaKernel.scala
badd +41 plugin/src/main/grgen/rules/print_graph/sym_names.grg
badd +159 plugin/src/main/grgen/rewrite_stage_info.fs
badd +31 plugin/src/main/grgen/unified/cuda_specials.unified.grg
badd +198 plugin/src/main/grgen/unified/cuda_generate_code.unified.grg
badd +142 plugin/src/main/grgen/unified/generate_cfg.unified.grg
badd +14 plugin/src/main/grgen/unified/cstyle_stmts.unified.grg
badd +1 plugin/src/main/grgen/unified/nice_lists.unified.grg
badd +36 plugin/src/main/grgen/unified/sketch_nospec.unified.grg
badd +20 base/src/main/scala/skalch/cuda/annotations/CudaAnnotations.scala
badd +119 ~/sandbox/grgen/engine-net-2/FSharpBindings/util_fcns.fs
badd +40 base/src/test/scala/cuda/Histogram.scala
badd +93 plugin/src/main/grgen/unified/create_templates.unified.grg
badd +22 base/src/main/scala/skalch/cuda/ScIArray1D.scala
badd +35 plugin/src/main/grgen/unified/create_libraries.unified.grg
badd +39 plugin/src/main/grgen/unified/postimport_union.unified.grg
badd +142 plugin/src/main/scala/skalch/plugins/GraphTypes.scala
badd +16 base/src/main/scala/skalch/cuda/ScIArray1D.scala.jinja2
badd +27 base/src/main/scala/skalch/cuda/PrimitiveVariables.scala
badd +33 base/src/main/scala/skalch/cuda/PrimitiveVariables.scala.jinja2
badd +165 ~/sandbox/grgen/engine-net-2/FSharpBindings/modular_compile_rules.fs
badd +1 modular_rules_compile.fs
badd +25 base/src/test/scala/cuda/Histogram.scala.html
badd +1 base/src/main/scala/skalch/cuda/ScIArray1D.scala.jinja2.html
badd +1 base/src/main/scala/skalch/cuda/PrimitiveVariables.scala.jinja2.html
badd +21 ~/sandbox/grgen/engine-net-2/FSharpBindings/dependencies.fs
badd +152 ~/sandbox/grgen/engine-net-2/src/lgspBackend/lgspGrGen.cs
args modular_rules_compile.fs
edit ~/sandbox/grgen/engine-net-2/FSharpBindings/graph.fs
set splitbelow splitright
wincmd _ | wincmd |
vsplit
1wincmd h
wincmd w
set nosplitbelow
set nosplitright
wincmd t
set winheight=1 winwidth=1
exe 'vert 1resize ' . ((&columns * 84 + 82) / 165)
exe 'vert 2resize ' . ((&columns * 80 + 82) / 165)
argglobal
setlocal keymap=
setlocal noarabic
setlocal autoindent
setlocal balloonexpr=
setlocal nobinary
setlocal bufhidden=
setlocal buflisted
setlocal buftype=
setlocal nocindent
setlocal cinkeys=0{,0},0),:,0#,!^F,o,O,e
setlocal cinoptions=
setlocal cinwords=if,else,while,do,for,switch
setlocal comments=s1:/*,mb:*,ex:*/,://,b:#,:%,:XCOMM,n:>,fb:-
setlocal commentstring=/*%s*/
setlocal complete=.,w,b,u,t,i
setlocal completefunc=
setlocal nocopyindent
setlocal nocursorcolumn
setlocal nocursorline
setlocal define=
setlocal dictionary=
setlocal nodiff
setlocal equalprg=
setlocal errorformat=
setlocal expandtab
if &filetype != 'ocaml'
setlocal filetype=ocaml
endif
setlocal foldcolumn=0
setlocal foldenable
setlocal foldexpr=0
setlocal foldignore=#
setlocal foldlevel=0
setlocal foldmarker={{{,}}}
setlocal foldmethod=manual
setlocal foldminlines=1
setlocal foldnestmax=20
setlocal foldtext=foldtext()
setlocal formatexpr=
setlocal formatoptions=tcq
setlocal formatlistpat=^\\s*\\d\\+[\\]:.)}\\t\ ]\\s*
setlocal grepprg=
setlocal iminsert=0
setlocal imsearch=0
setlocal include=
setlocal includeexpr=
setlocal indentexpr=
setlocal indentkeys=0{,0},:,0#,!^F,o,O,e
setlocal noinfercase
setlocal iskeyword=@,48-57,_,192-255
setlocal keywordprg=
setlocal nolinebreak
setlocal nolisp
setlocal nolist
setlocal makeprg=
setlocal matchpairs=(:),{:},[:]
setlocal nomodeline
setlocal modifiable
setlocal nrformats=octal,hex
setlocal nonumber
setlocal numberwidth=4
setlocal omnifunc=
setlocal path=
setlocal nopreserveindent
setlocal nopreviewwindow
setlocal quoteescape=\\
setlocal noreadonly
setlocal norightleft
setlocal rightleftcmd=search
setlocal noscrollbind
setlocal shiftwidth=4
setlocal noshortname
setlocal smartindent
setlocal softtabstop=4
setlocal nospell
setlocal spellcapcheck=[.?!]\\_[\\])'\"\	\ ]\\+
setlocal spellfile=
setlocal spelllang=en
setlocal statusline=
setlocal suffixesadd=
setlocal swapfile
setlocal synmaxcol=3000
if &syntax != 'ocaml'
setlocal syntax=ocaml
endif
setlocal tabstop=4
setlocal tags=
setlocal textwidth=0
setlocal thesaurus=
setlocal nowinfixheight
setlocal nowinfixwidth
setlocal wrap
setlocal wrapmargin=0
silent! normal! zE
let s:l = 66 - ((65 * winheight(0) + 47) / 94)
if s:l < 1 | let s:l = 1 | endif
exe s:l
normal! zt
66
normal! 029l
lcd ~/sandbox/skalch
wincmd w
argglobal
edit ~/sandbox/skalch/plugin/src/main/grgen/AllRules_0.grg
setlocal keymap=
setlocal noarabic
setlocal autoindent
setlocal balloonexpr=
setlocal nobinary
setlocal bufhidden=
setlocal buflisted
setlocal buftype=
setlocal nocindent
setlocal cinkeys=0{,0},0),:,0#,!^F,o,O,e
setlocal cinoptions=
setlocal cinwords=if,else,while,do,for,switch
setlocal comments=s1:/*,mb:*,ex:*/,://,b:#,:%,:XCOMM,n:>,fb:-
setlocal commentstring=/*%s*/
setlocal complete=.,w,b,u,t,i
setlocal completefunc=
setlocal nocopyindent
setlocal nocursorcolumn
setlocal nocursorline
setlocal define=
setlocal dictionary=
setlocal nodiff
setlocal equalprg=
setlocal errorformat=
setlocal expandtab
if &filetype != 'conf'
setlocal filetype=conf
endif
setlocal foldcolumn=0
setlocal foldenable
setlocal foldexpr=0
setlocal foldignore=#
setlocal foldlevel=0
setlocal foldmarker={{{,}}}
setlocal foldmethod=manual
setlocal foldminlines=1
setlocal foldnestmax=20
setlocal foldtext=foldtext()
setlocal formatexpr=
setlocal formatoptions=tcq
setlocal formatlistpat=^\\s*\\d\\+[\\]:.)}\\t\ ]\\s*
setlocal grepprg=
setlocal iminsert=0
setlocal imsearch=0
setlocal include=
setlocal includeexpr=
setlocal indentexpr=
setlocal indentkeys=0{,0},:,0#,!^F,o,O,e
setlocal noinfercase
setlocal iskeyword=@,48-57,_,192-255
setlocal keywordprg=
setlocal nolinebreak
setlocal nolisp
setlocal nolist
setlocal makeprg=
setlocal matchpairs=(:),{:},[:]
setlocal nomodeline
setlocal modifiable
setlocal nrformats=octal,hex
setlocal nonumber
setlocal numberwidth=4
setlocal omnifunc=
setlocal path=
setlocal nopreserveindent
setlocal nopreviewwindow
setlocal quoteescape=\\
setlocal noreadonly
setlocal norightleft
setlocal rightleftcmd=search
setlocal noscrollbind
setlocal shiftwidth=4
setlocal noshortname
setlocal smartindent
setlocal softtabstop=4
setlocal nospell
setlocal spellcapcheck=[.?!]\\_[\\])'\"\	\ ]\\+
setlocal spellfile=
setlocal spelllang=en
setlocal statusline=
setlocal suffixesadd=
setlocal swapfile
setlocal synmaxcol=3000
if &syntax != 'conf'
setlocal syntax=conf
endif
setlocal tabstop=4
setlocal tags=
setlocal textwidth=0
setlocal thesaurus=
setlocal nowinfixheight
setlocal nowinfixwidth
setlocal wrap
setlocal wrapmargin=0
silent! normal! zE
let s:l = 17 - ((16 * winheight(0) + 47) / 94)
if s:l < 1 | let s:l = 1 | endif
exe s:l
normal! zt
17
normal! 0
lcd ~/sandbox/skalch
wincmd w
exe 'vert 1resize ' . ((&columns * 84 + 82) / 165)
exe 'vert 2resize ' . ((&columns * 80 + 82) / 165)
tabnext 1
if exists('s:wipebuf')
  silent exe 'bwipe ' . s:wipebuf
endif
unlet! s:wipebuf
set winheight=1 winwidth=20 shortmess=filnxtToO
let s:sx = expand("<sfile>:p:r")."x.vim"
if file_readable(s:sx)
  exe "source " . fnameescape(s:sx)
endif
let &so = s:so_save | let &siso = s:siso_save
doautoall SessionLoadPost
unlet SessionLoad
" vim: set ft=vim :
