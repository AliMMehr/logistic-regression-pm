mkdir results
mkdir stdouts

j=0
numConcur=$3
for i in $(seq $1 $2)
do
    export THEANO_FLAGS="base_compiledir=/home/MINERVA/amehr/.theano/$j-1/"
    (echo "d_$i: start"; python ground-truth-d-full-CNF.py $i > ./stdouts/d_$i.out; echo "d_$i: end")&

    j=$((j+1))
    if [ $j -ge $numConcur ]
    then
        wait
        j=0
    fi

    export THEANO_FLAGS="base_compiledir=/home/MINERVA/amehr/.theano/$j-1/"
    (echo "c_full_$i: start"; python ground-truth-c-full-CNF.py $i > ./stdouts/c_full_$i.out; echo "c_full_$i: end")&

    j=$((j+1))
    if [ $j -ge $numConcur ]
    then
        wait
        j=0
    fi

    export THEANO_FLAGS="base_compiledir=/home/MINERVA/amehr/.theano/$j-1/"
    (echo "dLR_$i: start"; python ground-truth-d-LR.py $i > ./stdouts/dLR_$i.out; echo "dLR_$i: end")&

    j=$((j+1))
    if [ $j -ge $numConcur ]
    then
        wait
        j=0
    fi
    
done 

python Read-test-results-and-plot-graphs.py