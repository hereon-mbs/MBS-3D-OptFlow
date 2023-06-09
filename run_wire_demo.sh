##############################################################################################

DVCPROGRAM=./mbsoptflow
ANALYSISPROGRAM=./voxel2mesh

##############################################################################################
GPU=0

ALPHA=0.05
NORMALIZATION="histogram_independent"
LEVEL=10
SCALE=0.8
PREFILTER=0.5

LOCALGLOBAL=4
GRADIENTMASK="-gradientmask 0 0.05"
ADDITIONAL_ARGS="-prestrain_ref 0.00475 0.5"

MEMORY_LIMITER=100000

SIGMA_RENDERING=7
STRAIN_OUTPUTS="Ezz"

ROOTPATH="/Demos/WireDemo/"
OUTPATH="/Demos/WireDemo/optflow/"

##############################################################################################

EXPERIMENT_LIST="
Frame06
"

############################# Execute #############################
ARGUMENTS="-alpha "$ALPHA" -norm "$NORMALIZATION" -level "$LEVEL" -scale "$SCALE" -prefilter gaussian "$PREFILTER" -gpu0 "$GPU" -memlimit "$MEMORY_LIMITER" -localglobal "$LOCALGLOBAL" "$ADDITIONAL_ARGS
DISPLAY_ARGUMENTS="--taubin --vertices "$STRAIN_OUTPUTS" -sigma "$SIGMA_RENDERING

for i in $EXPERIMENT_LIST; do
    INPATH=$i
    $DVCPROGRAM -i0 $ROOTPATH"/Frame00/" -i1 $ROOTPATH$INPATH -o $OUTPATH $ARGUMENTS
    $ANALYSISPROGRAM -i_mesh /Demos/WireDemo/wire_mesh.vtk -i_disp $OUTPATH $DISPLAY_ARGUMENTS
done
###################################################################
