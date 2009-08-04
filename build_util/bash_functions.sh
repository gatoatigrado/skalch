#/bin/bash

# mostly run from the makefile, so I don't have to excessively escape $'s --gatoatigrado

function trim_whitespace() {
    [ "$1" ] || { echo "usage: set_package_decls target"; return 1; }
    for i in $(find "$1" -type f); do
        [ "$(file "$i" | grep text)" ] && {
            sed -i -r "s/\s+\$//g" "$i"
        }
    done
}

function set_package_decls() {
    [ "$1" ] || { echo "usage: set_package_decls target, where target != \".\"";
        return 1; }
    for i in $(find "$1" -iname "*.scala"); do
        nicename="$(echo "$(dirname "$i")" | sed 's/\//./g')";
        echo "setting package for $i to $nicename"
        sed -i -r "s/^package.+\$/package $nicename/g" "$i"
    done
}
