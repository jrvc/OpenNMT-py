#!/bin/bash

# Usage: ./demo-translation-client.sh --from en [cs de fr] --to cs [fr de en] text to translate
# or ./demo-translation-client.sh -f en text to translate
# (if no --from or --to options are selected, then all the directions are
# used)


SERVER=u-pl0.ms.mff.cuni.cz:5000

# the server has encode-decoder pairs loaded in this order, indexing from 0:
#		de-cs en-cs fr-cs cs-en de-en fr-en cs-de en-de fr-de cs-fr de-fr en-fr

lang_pair_to_id() {
	src=$1
	tgt=$2
	case $tgt in
	cs) case $src in 
		de)
			id=0
			;;
		en) id=1
			;;
		fr) id=2
		;;
		esac
	;;
	en) case $src in 
		cs)
			id=3
			;;
		de) id=4
			;;
		fr) id=5
		;;
		esac
	;;
	de) case $src in 
		cs)
			id=6
			;;
		en) id=7
			;;
		fr) id=8
		;;
		esac
	;;
	fr) case $src in 
		cs)
			id=9
			;;
		de) id=10
			;;
		en) id=11
		;;
		esac
	;;
	esac
	echo $id

}
#		de-cs en-cs fr-cs cs-en de-en fr-en cs-de en-de fr-de cs-fr de-fr en-fr

translate() {
	src_lang="$1"
	tgt_lang="$2"
	src_text="$3"
	id=`lang_pair_to_id $src_lang $tgt_lang`
	CMD="curl -X POST -i http://$SERVER/translator/translate --data ""'[{\"src\": \"$src_text\", \"id\":$id}]'"
	translation=`echo $CMD | bash  2>/dev/null | tail -n 1 | sed 's/.*tgt":"//;s/"}]].*//'`
	echo -e "$translation"
}


# demo
translate_ahoj() {
	for t in cs en de fr; do
		for s in cs en de fr; do
			[ $t == $s ] && continue
			echo -n $s ' ' $t ' '
			translate $s $t "hello world"
		done
	done
}

#translate_ahoj

source_langs="en de fr cs"
target_langs="en de fr cs"

is_lang() {
	echo $1 | egrep 'en|de|fr|cs' > /dev/null
}

while [ "$#" -gt 0 ]; do
	case $1 in
	-f|--from)
		shift
		source_langs=""
		while is_lang $1; do
			source_langs="$1 $source_langs"
			shift
		done
		;;
	-t|--to)
		shift
		target_langs=""
		while is_lang $1; do
			target_langs="$1 $target_langs"
			shift
		done
		;;
	--)
		shift
		break
		;;
	*)
		break
		;;
	esac
done
	
for f in $source_langs; do
	for t in $target_langs; do
		[ $f == $t ] && continue
		echo $f '->' $t"":
		translate $f $t "$@"
	done
done

