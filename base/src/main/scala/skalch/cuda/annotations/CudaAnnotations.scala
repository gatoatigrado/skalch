package skalch.cuda.annotations

/**
 * Annotations for how to convert Java to C/CUDA when there
 * are many possible
 * @author gatoatigrado (nicholas tung) [email: ntung at ntung]
 * @license This file is licensed under BSD license, available at
 *          http://creativecommons.org/licenses/BSD/. While not required,
 *          if you make changes, please consider contributing back!
 */
class scKernel extends StaticAnnotation
class scCopy extends StaticAnnotation
class scSpecialFcn extends StaticAnnotation
class scIField extends StaticAnnotation
class scInlineArray extends StaticAnnotation
class scPtr extends StaticAnnotation
class scRawArray extends StaticAnnotation
class scDefaultCopyInline extends StaticAnnotation
class scPointerObject(c_name : String) extends StaticAnnotation
class scCTypeNameOverride(c_name : String) extends StaticAnnotation

class scTemplateClass(params : String*) extends StaticAnnotation
class scRetype(typ : Class[_]) extends StaticAnnotation
class scRetypeTemplate(name : String) extends StaticAnnotation
class scRetypeTemplateInner(name : String) extends StaticAnnotation
class scTemplateInstanceType(paramValues : Class[_]*) extends StaticAnnotation
