##############################################################################################

DVCPROGRAM=./mbsoptflow
ANALYSISPROGRAM=./voxel2mesh

##############################################################################################
GPU=0

ALPHA=0.1
NORMALIZATION="histogram"
LEVEL=10
SCALE=0.9
PREFILTER=0.5

ADDITIONAL_ARGS="--median"

ROOTPATH="/Demos/RayDemo/"
MASK="/Demos/RayDemo/Mask01/"
OUTPATH="/Demos/RayDemo/optflow/"

##############################################################################################

EXPERIMENT_LIST="
Frame02
"

############################# Execute #############################
ARGUMENTS="-alpha "$ALPHA" -norm "$NORMALIZATION" -level "$LEVEL" -scale "$SCALE" -prefilter gaussian "$PREFILTER" -gpu0 "$GPU" "$ADDITIONAL_ARGS

for i in $EXPERIMENT_LIST; do
    INPATH=$i
    $DVCPROGRAM -i0 $ROOTPATH"Frame01/" -i1 $ROOTPATH$INPATH -m $MASK -o $OUTPATH $ARGUMENTS
    $ANALYSISPROGRAM -i_mesh /Demos/RayDemo/raymesh.vtk -i_disp $OUTPATH --taubin --vertices
done
###################################################################
