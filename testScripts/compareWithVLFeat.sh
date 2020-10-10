#!/bin/bash

# for TESTNAME in default new twobins vlfeatdesc vlfeat
for TESTNAME in new vlfeatdesc
do
    # VLF=/home/griff/Downloads/vlfeat-0.9.20/bin/glnxa64/sift
    VLF=/home/griff/GIT/vlfeat/bin/glnxa64/sift
    POP=/home/griff/GIT/popsift/build/Linux-x86_64/popsift-demo

    rm -f hash.sift*
    rm -f level1.sift*
    rm -f coord-*
    rm -f sum-*

    # FILES="hash level1 boat"
    # FILES="level1"
    # FILES="hash"
    # FILES="boat"
    FILES="boat1"

    for file in ${FILES} ; do
	echo ${VLF} -v ${file}.pgm
	${VLF} -v ${file}.pgm
	sort -n ${file}.sift > ${file}-vlfeat.sift
	rm ${file}.sift
	awk -e '{printf("%f %f %f %f\n",$1,$2,$3,$4);}' < ${file}-vlfeat.sift > coord-${file}-vlfeat.txt
	awk -f compute.awk ${file}-vlfeat.sift > sum-${file}-vlfeat.txt
    done

    if [ "$TESTNAME" = "default" ]
    then
        echo "Test is default"
        PAR1=" "
    elif [ "$TESTNAME" = "new" ]
    then
        echo "Test is new: loopdescriptor  and BestBin"
        PAR1="--desc-mode=loop --ori-mode=BestBin"
    elif [ "$TESTNAME" = "twobins" ]
    then
        echo "Test is twobins: loop descriptor and InterpolatedBin"
        PAR1="--desc-mode=loop --ori-mode=InterpolatedBin"
    elif [ "$TESTNAME" = "vlfeatdesc" ]
    then
        echo "Test is vlfeatdesc: vlfeat descriptor and BestBin"
        PAR1="--desc-mode=vlfeat --ori-mode=BestBin"
    elif [ "$TESTNAME" = "vlfeat" ]
    then
        echo "Test is vlfeat: vlfeat descriptor and InterpolatedBin"
        PAR1="--desc-mode=vlfeat --ori-mode=InterpolatedBin"
    else
        echo "Test is undefined, $TESTNAME"
        exit
    fi

    # VLFeat multiplies the descriptors with 512 before converting to uchar,
    # so we use 9 for a 2**9 multiplier as well.
    PAR0="--pgmread-loading --norm-mode=classic --norm-multi 9 --write-as-uchar --write-with-ori"
    PAR2="${PAR0} --initial-blur 0.5 --sigma 1.6 --threshold 0 --edge-threshold 10.0"

    for file in ${FILES} ; do
	echo ${POP} ${PAR1} ${PAR2} -i ${file}.pgm
	${POP} ${PAR1} ${PAR2} -i ${file}.pgm
	sort -n output-features.txt > ${file}-popsift_mod.sift
	rm output-features.txt
	awk -e '{printf("%f %f %f %f\n",$1,$2,$3,$4);}' < ${file}-popsift_mod.sift > coord-${file}-popsift_mod.txt
	awk -f compute.awk ${file}-popsift_mod.sift > sum-${file}-popsift_mod.txt
    done

    for file in ${FILES} ; do
	echo "Perform brute force matching"
	echo ./compareSiftFiles ${file}-popsift_mod.sift ${file}-vlfeat.sift
	./compareSiftFiles ${file}-popsift_mod.sift ${file}-vlfeat.sift > UML-${file}.txt

	echo "Sorting"
	sort -k3 -g UML-${file}.txt > sort-${file}-by-1st-match.txt
	sort -k6 -g UML-${file}.txt > sort-${file}-by-pixdist.txt
	sort -k10 -g UML-${file}.txt > sort-${file}-by-2nd-match.txt
	sort -k8 -g UML-${file}.txt > sort-${file}-by-angle.txt

	echo "Calling gnuplot (pixdist)"
	echo "set title \"L2 distance between pixels, PopSift ${TESTNAME} vs VLFeat" > cmd.gp
	echo "set xlabel \"Keypoint index sorted by closest best match\"" >> cmd.gp
	echo "set logscale y" >> cmd.gp
	echo "set terminal png" >> cmd.gp
	echo "set output \"sort-${file}-by-pixdist-${TESTNAME}.png\"" >> cmd.gp
	echo "plot \"sort-${file}-by-pixdist.txt\" using (\$6+0.00001) notitle" >> cmd.gp
	gnuplot cmd.gp

	echo "Calling gnuplot (1st dist)"
	echo "set title \"L2 distance between descriptors, PopSift ${TESTNAME} vs VLFeat" > cmd.gp
	echo "set xlabel \"Keypoint index sorted by closest best match\"" >> cmd.gp
	echo "set terminal png" >> cmd.gp
	echo "set output \"/dev/null\"" >> cmd.gp
	echo "plot   \"sort-${file}-by-1st-match.txt\" using 3 title \"best distance\"" >> cmd.gp
	echo "replot \"sort-${file}-by-1st-match.txt\" using 10 title \"2nd best distance\"" >> cmd.gp
	echo "set output \"sort-${file}-by-1st-match-${TESTNAME}.png\"" >> cmd.gp
	echo "replot" >> cmd.gp
	gnuplot cmd.gp

	echo "Calling gnuplot for angular diff (1st dist)"
	echo "set title \"Distance in degree between orientations, PopSift ${TESTNAME} vs VLFeat" > cmd.gp
	echo "set ylabel \"Difference (degree)\"" >> cmd.gp
	echo "set xlabel \"Keypoint index sorted by orientation difference\"" >> cmd.gp
	echo "set grid" >> cmd.gp
	echo "set logscale y" >> cmd.gp
	echo "set yrange [0.001:*]" >> cmd.gp
	echo "set style data lines" >> cmd.gp
	echo "set terminal png" >> cmd.gp
	echo "set output \"sort-${file}-by-angle-${TESTNAME}.png\"" >> cmd.gp
	echo "plot \"sort-${file}-by-angle.txt\" using 8 notitle" >> cmd.gp
	cp cmd.gp bak.gp
	gnuplot cmd.gp

	echo "Calling gnuplot (2nd dist)"
	echo "set title \"L2 distance between descriptors, PopSift ${TESTNAME} vs VLFeat" > cmd.gp
	echo "set xlabel \"Keypoint index sorted by 2nd best match\"" >> cmd.gp
	echo "set terminal png" >> cmd.gp
	echo "set output \"/dev/null\"" >> cmd.gp
	echo "plot   \"sort-${file}-by-2nd-match.txt\" using 3 title \"best distance\"" >> cmd.gp
	echo "replot \"sort-${file}-by-2nd-match.txt\" using 10 title \"2nd best distance\"" >> cmd.gp
	echo "set output \"sort-${file}-by-2nd-match-${TESTNAME}.png\"" >> cmd.gp
	echo "replot" >> cmd.gp
	gnuplot cmd.gp

	rm -f cmd.gp
	rm -f UML-${file}.txt
	# rm -f sort-${file}-by-pixdist.txt
	# rm -f sort-${file}-by-1st-match.txt
	# rm -f sort-${file}-by-2nd-match.txt
    done
done
