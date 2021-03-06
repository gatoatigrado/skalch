#!/usr/bin/env zsh
echo -n "name? "
read name
[ "$name" ] || exit 1
cp unified-template.txt unified/$name.unified.grg
sed -i -r "s/\\/\\/ add unified here/#include \"rules\\/gen\\/$name.grg\"\n\\0/g" AllRules_0.grg
sed -i -r "s/\\/\\/ add unified here/{% include \"nodes\\/gen\\/$name.gm\" %}\n\\0/g" ScalaAstModel.gm.jinja2
readlink -f unified/$name.unified.grg
